import asyncio
import logging
import sys
from inspect import getframeinfo, stack


def create_background_task(coroutine):
    task = asyncio.create_task(coroutine)

    call_stack = stack()[1]
    caller = getframeinfo(call_stack[0])
    line_number = caller.lineno
    file_name = caller.filename.split("/")[-1]
    method_name = call_stack[3]
    caller_details = "%s:%d:%s" % (file_name, line_number, method_name)

    def callback(future):
        if future.cancelled():
            logging.warning(
                "Background task at [%s] was cancelled.", caller_details)
            return
        if future.done():
            try:
                result = future.result()
            except Exception:
                logging.exception("Task at [%s] threw exception.", caller_details)
                sys.exit("Background task failed.")
            return result

    task.add_done_callback(callback)
    return task
