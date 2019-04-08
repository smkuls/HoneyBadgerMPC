import asyncio
import logging
from time import time
from honeybadgermpc.offline import PreProcessingBase
from honeybadgermpc.config import HbmpcConfig
from honeybadgermpc.ipc import ProcessProgramRunner
from progs.random_refinement import refine_randoms


class RandomGenerator(PreProcessingBase):
    def __init__(self, n, t, my_id, send, recv, batch_size=16, max_iterations=10):
        super(RandomGenerator, self).__init__(
            n, t, my_id, send, recv, "rand", batch_size, max_iterations)

    def _get_input_batch(self):
        return [self.field.random().value for _ in range(self.batch_size)]

    async def _extract(self):
        async for futures_list_per_party in self._get_output_batch():
            shares_list_per_party = await asyncio.gather(*futures_list_per_party)
            for all_party_shares in zip(*shares_list_per_party):
                refined_shares = refine_randoms(
                    self.n, self.t, self.field, list(all_party_shares))
                for share in refined_shares:
                    self.output_queue.put_nowait(share)


async def get_randoms(n, t, my_id, send, recv):
    iterations = HbmpcConfig.extras.get("iterations", 3)
    b = HbmpcConfig.extras.get("b", 2**10)
    logging.info("[RANDOM GENERATION] N: %d, T: %d, ITERATIONS: %d, AVSS BATCH SIZE: %d",
                 n, t, iterations, b)
    e = iterations*b*(n-t)
    k = (iterations)*b*(n-t)
    randoms = [None]*k
    start_time = time()
    async with RandomGenerator(n, t, my_id, send, recv, b, iterations) as rand_generator:
        for i in range(k):
            randoms[i] = await rand_generator.get()
    total_time = time()-start_time
    logging.info("[PREPROCESSING] COUNT: %d/%d. TIME: %f. PER SECOND: %f/second.",
                 k, e, total_time, k/total_time)
    logging.info("[PREPROCESSING] Unique values: %d/%d.", len(set(randoms)), k)
    return randoms


async def _mpc_prog(context, randoms, get_bytes):
    values = await randoms
    sent_bytes = get_bytes()
    logging.info("[PREPROCESSING] BYTES: %d", sent_bytes)
    assert all(v is not None for v in values)
    stime = time()
    await context.ShareArray(values).open()
    logging.info("[BATCH OPENING] COUNT: %d. TIME: %f. BYTES: %d.",
                 len(values), time()-stime, get_bytes()-sent_bytes)


async def _prog(peers, n, t, my_id):
    async with ProcessProgramRunner(peers, n, t, my_id) as runner:
        send, recv = runner.get_send_recv(0)
        task = asyncio.create_task(get_randoms(n, t, my_id, send, recv))
        runner.execute(1, _mpc_prog, randoms=task,
                       get_bytes=runner.node_communicator.get_sent_bytes)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_prog(
            HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id))
    finally:
        loop.close()
    logging.info("-"*40)
