import asyncio
import logging
import sys


def create_background_task(coroutine):
    task = asyncio.create_task(coroutine)

    def callback(future):
        if future.cancelled():
            logging.debug("Background task was cancelled.")
            return
        if future.done():
            try:
                result = future.result()
            except Exception:
                logging.exception("Background task threw exception.")
                sys.exit("Background task failed.")
            return result

    task.add_done_callback(callback)
    return task
