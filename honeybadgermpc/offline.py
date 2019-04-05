import asyncio
import logging
from abc import ABC, abstractmethod
from time import time
import random
from honeybadgermpc.avss_value_processor import AvssValueProcessor
from honeybadgermpc.batch_reconstruction import subscribe_recv, wrap_send
from honeybadgermpc.betterpairing import G1, ZR
from honeybadgermpc.elliptic_curve import Subgroup
from honeybadgermpc.field import GF
from honeybadgermpc.hbavss import HbAvssBatch
from honeybadgermpc.protocols.crypto.boldyreva import dealer
from progs.random_refinement import refine_randoms


def get_avss_params(n, t, my_id):
    g, h = G1.rand(seed=[0, 0, 0, 1]), G1.rand(seed=[0, 0, 0, 2])
    public_keys, private_keys = [None]*n, [None]*n
    for i in range(n):
        private_keys[i] = ZR.random(seed=17+i)
        public_keys[i] = pow(g, private_keys[i])
    return g, h, public_keys, private_keys[my_id]


max_iterations = 7


class PreProcessingBase(ABC):
    def __init__(self, n, t, my_id, send, recv, tag,
                 batch_size=10, avss_value_processor_chunk_size=1):
        self.n, self.t, self.my_id = n, t, my_id
        self.tag = tag
        self.avss_value_processor_chunk_size = avss_value_processor_chunk_size

        # Batch size of values to AVSS from a node
        self.batch_size = batch_size

        self.output_queue = asyncio.Queue()

        # Create a mechanism to split the `send` and `recv` channels based on `tag`
        subscribe_recv_task, subscribe = subscribe_recv(recv)
        self.tasks = [subscribe_recv_task]

        def _get_send_recv(tag):
            return wrap_send(tag, send), subscribe(tag)
        self.get_send_recv = _get_send_recv
        self.my_rand = random.Random(my_id)
        logging.debug("[%d] Seed: %d", my_id, my_id)

    async def get(self):
        return await self.output_queue.get()

    @abstractmethod
    def _get_input_batch(self):
        raise NotImplementedError

    async def _trigger_and_wait_for_avss(self, avss_id):
        inputs = self._get_input_batch()
        assert type(inputs) in [tuple, list]
        avss_tasks = [None] * self.n

        for i in range(self.n):
            if i != self.my_id:
                values, dealer_id = None, i
            else:
                values, dealer_id = inputs, self.my_id
            avss_tasks[i] = asyncio.create_task(
                self.avss_instance.avss(avss_id, values=values, dealer_id=dealer_id))

        # TODO: Only wait for the AVSS task in which you are the dealer since an
        # AVSS task started by another node might not finish if the node is corrupt.
        # THIS IS ONLY FOR THE EASE OF BENCHMARKING.
        await asyncio.gather(*avss_tasks)

    async def _runner(self):
        counter = 0
        logging.debug("[%d] Starting preprocessing runner: %s", self.my_id, self.tag)

        while True:
            logging.debug("[%d] AVSS started: %d", self.my_id, counter)
            await asyncio.create_task(self._trigger_and_wait_for_avss(counter))
            logging.debug("[%d] AVSS completed. ACS started: %d", self.my_id, counter)
            await asyncio.create_task(self.avss_value_processor.run_acs(counter))
            global max_iterations
            if counter == max_iterations-1:
                # TODO: THIS IS ONLY FOR THE EASE OF BENCHMARKING.
                break
            counter += 1

    async def _get_output_batch(self, group_size=1):
        for i in range(self.batch_size):
            batches = []
            while True:
                values = await self.avss_value_processor.get()
                if values is None:
                    break
                batches.append(values)
            assert len(batches) / group_size >= self.n - self.t
            assert len(batches) / group_size <= self.n
            yield batches

    async def _extract(self):
        raise NotImplementedError

    def __enter__(self):
        n, t, my_id = self.n, self.t, self.my_id
        send, recv = self.get_send_recv(f'{self.tag}-AVSS')
        g, h, pks, sk = get_avss_params(n, t, my_id)
        self.avss_instance = HbAvssBatch(pks, sk, g, h, n, t, my_id, send, recv)
        self.avss_instance.__enter__()
        self.tasks.append(asyncio.create_task(self._runner()))

        send, recv = self.get_send_recv(f'{self.tag}-AVSS_VALUE_PROCESSOR')
        pk, sks = dealer(n, t+1, seed=17)
        self.avss_value_processor = AvssValueProcessor(
            pk, sks[my_id],
            n, t, my_id,
            send, recv,
            self.avss_instance.output_queue.get,
            self.avss_value_processor_chunk_size)
        self.avss_value_processor.__enter__()
        self.tasks.append(asyncio.create_task(self._extract()))
        self.executor = self.avss_instance.executor
        return self

    def __exit__(self, *args):
        logging.info("Pending processed value count: %d", self.output_queue.qsize())
        for task in self.tasks:
            task.cancel()
        self.avss_instance.__exit__(*args)
        self.avss_value_processor.__exit__(*args)


class RandomGenerator(PreProcessingBase):
    def __init__(self, n, t, my_id, send, recv, batch_size=10):
        super(RandomGenerator, self).__init__(
            n, t, my_id, send, recv, "rand", batch_size)
        self.field = GF(Subgroup.BLS12_381)

    def _get_input_batch(self):
        return [self.my_rand.randint(
            0, self.field.modulus-1) for _ in range(self.batch_size)]

    async def _extract(self):
        while True:
            async for batches in self._get_output_batch():
                random_share_batches = await asyncio.gather(*batches)
                assert len(set([len(batch) for batch in random_share_batches])) == 1
                refine_args = [
                    [self.n, self.t, self.field, list(batch)] for batch in zip(
                        *random_share_batches)]
                refined_batches = self.executor.map(
                    refine_randoms, *zip(*refine_args))
                for batch in refined_batches:
                    for value in batch:
                        self.output_queue.put_nowait(self.field(value))


class TripleGenerator(PreProcessingBase):
    def __init__(self, n, t, my_id, send, recv, batch_size=10):
        super(TripleGenerator, self).__init__(n, t, my_id, send, recv, "triple",
                                              batch_size,
                                              avss_value_processor_chunk_size=3)
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
            async for batch in self._get_output_batch(3):
                triple_shares_int = await asyncio.gather(*batch)
                # Number of nodes which have contributed values to this batch
                n = len(triple_shares_int)
                assert n % 3 == 0

                for i in range(0, n, 3):
                    a, b, ab = triple_shares_int[i:i+3]
                    self.output_queue.put_nowait((a, b, ab))


async def get_random(n, t, b, my_id, send, recv):
    with RandomGenerator(n, t, my_id, send, recv, b) as random_generator:
        while True:
            yield await random_generator.get()


async def _mpc_prog(context, b, randoms):
    global max_iterations
    n, i = max_iterations * b * (context.N - context.t), 0
    logging.info("Waiting for n: %d", n)
    stime = time()
    vals = []
    async for rand in randoms:
        vals.append(rand)
        i += 1
        logging.info("[%d] => %s", i, await context.Share(rand).open())
        if i == n:
            break
    await randoms.aclose()
    total_time = time()-stime
    logging.info("Total time: %s", total_time)
    logging.info("Batch size: %d, %f per second.", b, n/total_time)
    # logging.info("Unique values: %d", len(set(vals)))
    opened_shares = await context.ShareArray(vals).open()
    logging.info("Unique openings: %d", len(set(opened_shares)))


async def _prog(peers, n, t, my_id):
    # import cProfile, pstats, io
    # from pstats import SortKey

    # pr = cProfile.Profile()
    # pr.enable()
    b = HbmpcConfig.extras["k"]
    b = 2**8
    async with ProcessProgramRunner(peers, n, t, my_id) as runner:
        send, recv = runner.get_send_recv("rand-gen")
        runner.execute("mpc-prog", _mpc_prog, b=b, randoms=get_random(n, t, b, my_id, send, recv))

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())


if __name__ == "__main__":
    from honeybadgermpc.config import HbmpcConfig
    from honeybadgermpc.ipc import ProcessProgramRunner

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(_prog(
            HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id))
    finally:
        loop.close()
