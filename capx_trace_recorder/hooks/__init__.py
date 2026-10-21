"""Runtime hooks H1-H16 (docs/trace-format.md, section 7.1).

Each module exposes ``install_*`` functions that ``capx_trace_recorder/bootstrap.py``
calls right after the target module has executed (or at once when it is
already imported). Every wrapper carries ``__capx_trace_wrapper__ = True`` and
is never applied twice to the same callable.
"""
