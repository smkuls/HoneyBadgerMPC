import asyncio
import logging
from time import time
from honeybadgermpc.offline import PreProcessingBase
from honeybadgermpc.config import HbmpcConfig
from honeybadgermpc.ipc import ProcessProgramRunner
from progs.triple_refinement import refine_triples


class TripleGenerator(PreProcessingBase):
    def __init__(self, n, t, my_id, send, recv, batch_size=16, max_iterations=10):
        super(TripleGenerator, self).__init__(
            n, t, my_id, send, recv, "triple", batch_size, max_iterations)

    def _get_input_batch(self):
        inputs = []
        for _ in range(self.batch_size):
            a, b = self.field.random(), self.field.random()
            ab = a*b
            inputs += [a.value, b.value, ab.value]
        return inputs

    async def _extract(self):
        async for futures_list_per_party in self._get_output_batch():
            shares_list_per_party = await asyncio.gather(*futures_list_per_party)
            assert (all(len(i) == 3*self.batch_size) for i in shares_list_per_party)
            for j in range(0, self.batch_size*3, 3):
                for i in range(len(shares_list_per_party)):
                    a, b, ab = shares_list_per_party[i][j:j+3]
                    self.output_queue.put_nowait((int(a), int(b), int(ab)))
                self.output_queue.put_nowait(None)


async def get_triples(n, t, my_id, send, recv):
    iterations = 4
    b = 2**8
    c = 0
    e = iterations*b*n
    stime = time()
    batches = [None]*b*(iterations)
    async with TripleGenerator(
            n, t, my_id, send, recv, b, iterations) as triple_generator:
        for i in range(b*(iterations)):
            batch = [[], [], []]
            while True:
                triple = await triple_generator.get()
                if triple is None:
                    break
                assert len(triple) == 3
                batch[0].append(triple[0])
                batch[1].append(triple[1])
                batch[2].append(triple[2])
                c += 1
            batches[i] = batch
    total_time = time()-stime
    logging.info("Batch size: %d, Iterations: %s, n: %d, t: %d", b, iterations, b, t)
    logging.info("Generated [%d/%d] in %f [%f/second].", c, e, total_time, c/total_time)
    return batches


async def _mpc_prog(context, triples):
    values = await triples
    assert all(v is not None for v in values)
    p, q, pq = list(zip(*values))
    tasks = [None]*len(values)
    for i, v in enumerate(values):
        p, q, pq = v
        tasks[i] = asyncio.create_task(refine_triples(context, p, q, pq))
    stime = time()
    refined_triples = await asyncio.gather(*tasks)
    etime = time()-stime
    to_open = []
    for triple in refined_triples:
        for j in zip(*triple):
            assert len(j) == 3
            p, q, pq = j
            to_open.append(p)
            to_open.append(q)
            to_open.append(pq)
    stime = time()
    opened_vals = await context.ShareArray(to_open).open()
    logging.info("Triple refinement - %d. Time : %f", len(to_open)/3, etime)
    logging.info("Batch opening - %d. Time: %f", len(to_open), time()-stime)
    logging.info("Number of unique values: %d/%d",
                 len(set(opened_vals)), len(opened_vals))
    for i in range(0, len(opened_vals), 3):
        p = opened_vals[i]
        q = opened_vals[i+1]
        pq = opened_vals[i+2]
        if p*q != pq:
            raise Exception("NOT A TRIPLE!")


async def _prog(peers, n, t, my_id):
    async with ProcessProgramRunner(peers, n, t, my_id) as runner:
        send, recv = runner.get_send_recv(0)
        task = asyncio.create_task(get_triples(n, t, my_id, send, recv))
        runner.execute(31, _mpc_prog, triples=task)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_prog(
            HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id))
    finally:
        loop.close()
    logging.info("-"*40)
