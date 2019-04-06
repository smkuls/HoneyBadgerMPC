import asyncio
import logging
from honeybadgermpc.hbavss import HbAvssBatch
from honeybadgermpc.avss_value_processor import AvssValueProcessor
from honeybadgermpc.protocols.crypto.boldyreva import dealer
from honeybadgermpc.betterpairing import G1, ZR
from progs.random_refinement import refine_randoms
from honeybadgermpc.field import GF
from honeybadgermpc.elliptic_curve import Subgroup
from honeybadgermpc.batch_reconstruction import subscribe_recv, wrap_send
from abc import ABC, abstractmethod
from honeybadgermpc.asyncio_wrapper import create_background_task


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
        self.tasks = [subscribe_recv_task]
        self.max_iterations = max_iterations

        def _get_send_recv(tag):
            return wrap_send(tag, send), subscribe(tag)
        self.get_send_recv = _get_send_recv

    async def get(self):
        return await self.output_queue.get()

    @abstractmethod
    def _get_input_batch(self):
        raise NotImplementedError

    async def _trigger_and_wait_for_avss(self, avss_id):
        inputs = self._get_input_batch()
        assert type(inputs) in [tuple, list]
        avss_tasks = [None]*self.n
        avss_tasks[self.my_id] = asyncio.create_task(
            self.avss_instance.avss(avss_id, values=inputs, dealer_id=self.my_id))
        for i in range(self.n):
            if i != self.my_id:
                avss_tasks[i] = asyncio.create_task(
                    self.avss_instance.avss(avss_id, dealer_id=i))
        await asyncio.gather(*avss_tasks)

    async def _runner(self):
        logging.debug("[%d] Starting preprocessing runner: %s", self.my_id, self.tag)
        for i in range(self.max_iterations):
            logging.debug("[%d] Start AVSS. id: %d", self.my_id, i)
            await self._trigger_and_wait_for_avss(i)
            logging.debug("[%d] AVSS Completed. Start ACS. id: %d", self.my_id, i)
            await self.avss_value_processor.run_acs(i)
            logging.debug("[%d] ACS Completed. id: %d", self.my_id, i)

    async def _get_output_batch(self):
        for i in range(self.batch_size):
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

    def __enter__(self):
        n, t, my_id = self.n, self.t, self.my_id
        send, recv = self.get_send_recv(f'{self.tag}-AVSS')
        g, h, pks, sk = get_avss_params(n, t, my_id)
        self.avss_instance = HbAvssBatch(pks, sk, g, h, n, t, my_id, send, recv)
        self.avss_instance.__enter__()
        self.tasks.append(create_background_task(self._runner()))

        send, recv = self.get_send_recv(f'{self.tag}-AVSS_VALUE_PROCESSOR')
        pk, sks = dealer(n, t+1, seed=17)
        self.avss_value_processor = AvssValueProcessor(
            pk, sks[my_id],
            n, t, my_id,
            send, recv,
            self.avss_instance.output_queue.get)
        self.avss_value_processor.__enter__()
        self.tasks.append(create_background_task(self._extract()))
        return self

    def __exit__(self, *args):
        for task in self.tasks:
            task.cancel()
        self.avss_instance.__exit__(*args)
        self.avss_value_processor.__exit__(*args)


class RandomGenerator(PreProcessingBase):
    def __init__(self, n, t, my_id, send, recv, batch_size=16, max_iterations=10):
        super(RandomGenerator, self).__init__(
            n, t, my_id, send, recv, "rand", batch_size, max_iterations)
        self.field = GF(Subgroup.BLS12_381)

    def _get_input_batch(self):
        return [self.field.random().value for _ in range(self.batch_size)]

    async def _extract(self):
        while True:
            async for futures_list_per_party in self._get_output_batch():
                shares_list_per_party = await asyncio.gather(*futures_list_per_party)
                for all_party_shares in zip(*shares_list_per_party):
                    refined_shares = refine_randoms(
                        self.n, self.t, self.field, list(all_party_shares))
                    for share in refined_shares:
                        self.output_queue.put_nowait(share)


class TripleGenerator(PreProcessingBase):
    def __init__(self, n, t, my_id, send, recv, batch_size=16, max_iterations=10):
        super(TripleGenerator, self).__init__(
            n, t, my_id, send, recv, "triple", batch_size, max_iterations)
        self.field = GF(Subgroup.BLS12_381)

    def _get_input_batch(self):
        inputs = []
        for _ in range(self.batch_size):
            a, b = self.field.random(), self.field.random()
            ab = a*b
            inputs += [a.value, b.value, ab.value]
        return inputs

    async def _extract(self):
        while True:
            async for futures_list_per_party in self._get_output_batch():
                shares_list_per_party = await asyncio.gather(*futures_list_per_party)
                assert (all(len(i) == 3*self.batch_size) for i in shares_list_per_party)
                for j in range(0, self.batch_size*3, 3):
                    for i in range(len(shares_list_per_party)):
                        a, b, ab = shares_list_per_party[i][j:j+3]
                        self.output_queue.put_nowait((int(a), int(b), int(ab)))


async def get_random(n, t, my_id, send, recv):
    with RandomGenerator(n, t, my_id, send, recv) as random_generator:
        while True:
            yield await random_generator.get()


async def _mpc_prog(context, **kwargs):
    randoms = kwargs["randoms"]
    n, i = 10, 0
    async for random in randoms:
        logging.info("i: %d => %d", i, await context.Share(random).open())
        i += 1
        if i == n:
            break
    await randoms.aclose()


async def _prog(peers, n, t, my_id):
    async with ProcessProgramRunner(peers, n, t, my_id) as runner:
        send, recv = runner.get_send_recv(0)
        runner.execute(1, _mpc_prog, randoms=get_random(n, t, my_id, send, recv))


if __name__ == "__main__":
    from honeybadgermpc.config import HbmpcConfig
    from honeybadgermpc.ipc import ProcessProgramRunner

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    try:
        loop.run_until_complete(_prog(
            HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id))
    finally:
        loop.close()
