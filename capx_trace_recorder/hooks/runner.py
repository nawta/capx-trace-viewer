"""H15 and H16: the experiment runner (docs/trace-format.md, sections 3, 4.1, 4.2, 5.2).

H15 wraps ``capx.experiments.manifest.load_manifest`` (``run.py`` binds it at
import, after the patch) and ``capx.experiments.revision.fingerprint``
(``run.py`` calls it as a module attribute). ``load_manifest`` only keeps a
reference to the returned manifest, because ``run.py`` applies ``--run-id``
and ``--output-root`` to that object afterwards. ``fingerprint`` is called
after those overrides and before the preflight and the workers start: it adds
``"tracing": {"schema": "capx-trace/1"}`` to the fingerprint parts (in place,
so ``run.json`` shows the part and a drift names it), and exports
``CAPX_TRACE_RUN_ID`` (unless the user set it) and ``CAPX_TRACE_RUNNER_RUN_DIR``
so that spawned workers use the runner's run id and run dir.

H16 wraps ``capx.experiments.records.append_jsonl``; when the parent process
appends to the runner's ``trials.jsonl``, a copy of the record is appended to
``<trace root>/<run_id>/runner_records.jsonl`` by the writer thread. The calling
thread only takes a JSON-safe copy of the record; the writer redacts its error
texts (tracebacks of failed attempts) like every stored exception text
(``redact_record``, which lists the changed texts in ``trace_redacted_fields``)
and serialises it. All other fields are copied unchanged.

Related: ``capx_trace_recorder/hooks/trial.py`` (episode ids use the runner's run dir).
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

from capx_trace_recorder import SCHEMA_VERSION
from capx_trace_recorder import context as C
from capx_trace_recorder.config import ENV_RUN_ID, ENV_RUNNER_RUN_DIR
from capx_trace_recorder.hooks.common import guard, mark, patch_attribute
from capx_trace_recorder.ids import sanitize_id
from capx_trace_recorder.store import get_store


def _after_load_manifest(manifest: Any) -> None:
    C.REGISTRY.runner.manifest = manifest


def make_load_manifest_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def load_manifest(*args: Any, **kwargs: Any) -> Any:
        manifest = original(*args, **kwargs)
        guard(_after_load_manifest, manifest)
        return manifest

    return mark(load_manifest, original)


def _before_fingerprint(parts: Any) -> None:
    runner = C.REGISTRY.runner
    manifest = runner.manifest
    if manifest is None or not isinstance(parts, dict):
        return
    parts["tracing"] = {"schema": SCHEMA_VERSION}
    run_id = getattr(manifest, "run_id", None)
    run_dir = getattr(manifest, "run_dir", None)
    if run_id:
        runner.run_id = sanitize_id(str(run_id))
        if not os.environ.get(ENV_RUN_ID):
            os.environ[ENV_RUN_ID] = runner.run_id
    if run_dir is not None:
        try:
            runner.run_dir = Path(str(run_dir)).resolve()
            os.environ[ENV_RUNNER_RUN_DIR] = str(runner.run_dir)
        except TimeoutError:  # an OSError subclass: the trial alarm (section 2)
            raise
        except OSError:
            pass
    runner.exported = True
    from capx_trace_recorder import bootstrap

    bootstrap.log_failures_to_run()  # failures recorded before the run id was known


def make_fingerprint_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def fingerprint(*args: Any, **kwargs: Any) -> Any:
        parts = args[0] if args else kwargs.get("parts")
        guard(_before_fingerprint, parts)
        return original(*args, **kwargs)

    return mark(fingerprint, original)


def _after_append(path: Any, record: Any) -> None:
    runner = C.REGISTRY.runner
    store = get_store()
    if store is None or runner.run_dir is None:
        return
    try:
        target = Path(str(path))
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return
    if target.name != "trials.jsonl":
        return
    try:
        if target.resolve().parent != runner.run_dir:
            return
    except TimeoutError:
        raise
    except OSError:
        return
    run_id = os.environ.get(ENV_RUN_ID) or runner.run_id
    if not run_id:
        return
    from capx_trace_recorder.serialize import json_safe

    snapshot = json_safe(record)  # the runner may change its dict later

    def make_line(snapshot: Any = snapshot) -> bytes:  # on the writer thread
        from capx_trace_recorder.payload import redact_record

        return (json.dumps(redact_record(snapshot), ensure_ascii=True, separators=(",", ":"))
                + "\n").encode("ascii")

    store.append_run_line(store.config.trace_dir / sanitize_id(run_id), "runner_records.jsonl", make_line)


def make_append_jsonl_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def append_jsonl(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        path = args[0] if args else kwargs.get("path")
        record = args[1] if len(args) > 1 else kwargs.get("record")
        guard(_after_append, path, record)
        return result

    return mark(append_jsonl, original)


def install_manifest(module: Any) -> list[str]:
    err = patch_attribute(module, "load_manifest", make_load_manifest_wrapper)
    return [err] if err else []


def install_revision(module: Any) -> list[str]:
    err = patch_attribute(module, "fingerprint", make_fingerprint_wrapper)
    return [err] if err else []


def install_records(module: Any) -> list[str]:
    err = patch_attribute(module, "append_jsonl", make_append_jsonl_wrapper)
    return [err] if err else []


__all__ = ["install_manifest", "install_revision", "install_records"]
