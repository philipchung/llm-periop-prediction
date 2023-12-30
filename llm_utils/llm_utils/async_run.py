import asyncio
import threading
import time
from typing import Any, Callable


def sidethread_event_loop_async_runner(async_function: Callable) -> Any:
    """Runs an async function on another thread, blocking until result complete.

    This creates a thread and event loop on that thread.  The async function
    is run on that event loop.  When the async function completes, it stops and closes
    the event loop and thread.

    Args:
        async_function (Callable): async function.  This should be defined with
            `async def`.

    Returns:
        Any: The return value of `async_function`.
    """
    # Create new event loop
    event_loop_a = asyncio.new_event_loop()

    def run_forever_safe(loop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()
        # NOTE: Event loop runs until `loop.stop()` is called.
        # This will also cause this `target` for thread to complete and stop the thread.

    # Place event loop on a dedicated thread & run the event loop
    thread = threading.Thread(target=lambda: run_forever_safe(event_loop_a))
    thread.start()

    # Execute asyncio function on the event loop on other thread (so we don't block main thread)
    future = asyncio.run_coroutine_threadsafe(async_function, event_loop_a)
    # Wait for the result
    result = future.result()
    # Stop event loop on other thread
    event_loop_a.call_soon_threadsafe(event_loop_a.stop)
    # Block main thread until event loop stopped
    while event_loop_a.is_running():
        time.sleep(0.1)
    # Close event loop now that it is no longer running
    event_loop_a.close
    return result
