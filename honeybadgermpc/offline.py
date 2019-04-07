import asyncio
import logging
from honeybadgermpc.hbavss import HbAvssBatch
from honeybadgermpc.avss_value_processor import AvssValueProcessor
from honeybadgermpc.protocols.crypto.boldyreva import dealer
from honeybadgermpc.betterpairing import G1, ZR
from honeybadgermpc.field import GF
from honeybadgermpc.elliptic_curve import Subgroup
from honeybadgermpc.batch_reconstruction import subscribe_recv, wrap_send
from abc import ABC, abstractmethod


def get_avss_params(n, t, my_id):
    g, h = G1.rand(seed=[0, 0, 0, 1]), G1.rand(seed=[0, 0, 0, 2])
    public_keys, private_keys = [None]*n, [None]*n
    for i in range(n):
        private_keys[i] = ZR.random(seed=17+i)
        public_keys[i] = pow(g, private_keys[i])
    return g, h, public_keys, private_keys[my_id]


class PreProcessingBase(ABC):
    def __init__(self, n, t, my_id, send, recv, tag, batch_size, max_iterations):
        self.n, self.t, self.my_id = n, t, my_id
        self.tag = tag

        # Batch size of values to AVSS from a node
        self.batch_size = batch_size

        self.output_queue = asyncio.Queue()

        # Create a mechanism to split the `send` and `recv` channels based on `tag`
        subscribe_recv_task, subscribe = subscribe_recv(recv)
        self.recv_task = subscribe_recv_task
        self.await_tasks = []
        self.max_iterations = max_iterations

        def _get_send_recv(tag):
            return wrap_send(tag, send), subscribe(tag)
        self.get_send_recv = _get_send_recv
        self.field = GF(Subgroup.BLS12_381)
        self.done = False

    async def get(self):
        return await self.output_queue.get()

    @abstractmethod
    def _get_input_batch(self):
        raise NotImplementedError

    async def _trigger_and_wait_for_avss(self, avss_id):
        inputs = self._get_input_batch()
        assert type(inputs) in [tuple, list]
        avss_tasks = [None]*self.n
        for i in range(self.n):
            if i != self.my_id:
                avss_tasks[i] = asyncio.create_task(
                    self.avss_instance.avss(avss_id, dealer_id=i))
        avss_tasks[self.my_id] = asyncio.create_task(
            self.avss_instance.avss(avss_id, values=inputs, dealer_id=self.my_id))
        # TODO: WAITING HERE FOR ALL TASKS. ONLY FOR BENCHMARKING.
        await asyncio.gather(*avss_tasks)

    async def _runner(self):
        logging.debug("[%d] Starting preprocessing runner: %s", self.my_id, self.tag)
        for i in range(self.max_iterations):
            logging.debug("[%d] Start AVSS. id: %d", self.my_id, i)
            await self._trigger_and_wait_for_avss(i)
            logging.debug("[%d] AVSS Completed. Start ACS. id: %d", self.my_id, i)
            # stime = time()
            await self.avss_value_processor.run_acs(i)
            # logging.info("ACS Time: %f", time()-stime)
            logging.debug("[%d] ACS Completed. id: %d", self.my_id, i)
        self.done = True

    async def _get_output_batch(self):
        while not self.done or self.avss_value_processor.output_queue.qsize() > 0:
            batch = []
            while True:
                values = await self.avss_value_processor.get()
                if values is None:
                    break
                batch.append(values)
            assert len(batch) >= self.n - self.t
            assert len(batch) <= self.n
            yield batch

    async def _extract(self):
        raise NotImplementedError

    async def __aenter__(self):
        n, t, my_id = self.n, self.t, self.my_id
        send, recv = self.get_send_recv(f'{self.tag}-AVSS')
        g, h, pks, sk = get_avss_params(n, t, my_id)
        self.avss_instance = HbAvssBatch(pks, sk, g, h, n, t, my_id, send, recv)
        self.avss_instance.__enter__()
        self.await_tasks.append(asyncio.create_task(self._runner()))

        send, recv = self.get_send_recv(f'{self.tag}-AVSS_VALUE_PROCESSOR')
        pk, sks = dealer(n, t+1, seed=17)
        self.avss_value_processor = AvssValueProcessor(
            pk, sks[my_id],
            n, t, my_id,
            send, recv,
            self.avss_instance.output_queue.get)
        self.avss_value_processor.__enter__()
        self.await_tasks.append(asyncio.create_task(self._extract()))
        return self

    async def __aexit__(self, *args):
        await asyncio.gather(*self.await_tasks)
        self.recv_task.cancel()
        self.avss_instance.__exit__(*args)
        self.avss_value_processor.__exit__(*args)
