import asyncio
import logging
from honeybadgermpc.preprocessing import PreProcessedElements
from honeybadgermpc.preprocessing import wait_for_preprocessing, preprocessing_done
from honeybadgermpc.ipc import ProcessProgramRunner
from honeybadgermpc.config import HbmpcConfig


async def batch_opening(context):
    pp_elements = PreProcessedElements()
    k = HbmpcConfig.extras["k"]
    share_array = context.ShareArray([pp_elements.get_rand(context) for _ in range(k)])
    await share_array.open()
    logging.info("Batch opening finished.")


if __name__ == "__main__":
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    try:
        if not HbmpcConfig.skip_preprocessing:
            # Only one party needs to generate the initial shares
            if HbmpcConfig.my_id == 0:
                pp_elements = PreProcessedElements()
                logging.info('Generating randoms in sharedata/')
                pp_elements.generate_rands(1000, HbmpcConfig.N, HbmpcConfig.t)
                preprocessing_done()
            else:
                loop.run_until_complete(wait_for_preprocessing())
        program_runner = ProcessProgramRunner(
            HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id)
        loop.run_until_complete(program_runner.start())
        program_runner.add(1, batch_opening)
        loop.run_until_complete(program_runner.join())
        loop.run_until_complete(program_runner.close())
    finally:
        loop.close()
