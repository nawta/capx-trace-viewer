"""Distinguish the runner's trial alarm from file-system timeouts."""


def is_trial_alarm(exc: BaseException) -> bool:
    """The runner raises its alarm without an OS error number.

    ``socket.timeout`` is an alias of ``TimeoutError`` since Python 3.10,
    and ``asyncio.TimeoutError`` is an alias since Python 3.11. Instances
    without errno count as alarms too.
    """
    return isinstance(exc, TimeoutError) and exc.errno is None
