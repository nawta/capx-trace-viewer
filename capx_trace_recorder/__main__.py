"""Recorder installation and single-process launcher."""

import argparse
import os
import runpy
import sys
import sysconfig
from pathlib import Path


PTH_LINE = (
    'import os,sys; exec("sys._capx_trace_pth_seen = True\\ntry:\\n import capx_trace_recorder.pth\\n'
    'except Exception as exc:\\n try:\\n  print(\'capx_trace_recorder: startup failed: \' + '
    'type(exc).__name__, file=sys.stderr)\\n except Exception:\\n  pass") if os.environ.get("CAPX_TRACE_DIR", "").strip() '
    'and not getattr(sys, "_capx_trace_pth_seen", False) else None\n'
)
LEGACY_PTH_LINE = (
    'import os,sys; exec("sys._capx_trace_pth_seen = True\\ntry:\\n import capx_trace_recorder.pth\\n'
    'except ImportError as exc:\\n print(\'capx_trace_recorder: \' + '
    'type(exc).__name__, file=sys.stderr)") if os.environ.get("CAPX_TRACE_DIR", "").strip() '
    'and not getattr(sys, "_capx_trace_pth_seen", False) else None\n'
)


def pth_path() -> Path:
    return Path(sysconfig.get_paths()["purelib"]) / "capx_trace_recorder.pth"


def _read_hook(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _read_hook_for_change(path: Path, parser: argparse.ArgumentParser) -> str | None:
    try:
        return _read_hook(path)
    except OSError as exc:
        parser.error(f"cannot read hook at {path}: {exc}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        words = sys.argv[2:]
        if words in (["-h"], ["--help"]):
            print("usage: capx-trace-recorder run [-m MODULE | SCRIPT] [args...]")
            return
        if words and words[0] == "-m":
            if len(words) < 2:
                raise SystemExit("run: -m requires a module")
            module, target, trailing = words[1], None, words[2:]
        elif words:
            module, target, trailing = None, words[0], words[1:]
        else:
            raise SystemExit("run: provide -m MODULE or SCRIPT")
        old_path = sys.path[0]
        old_argv = sys.argv
        try:
            if module:
                sys.path[0] = os.getcwd()
            else:
                target = str(Path(target).resolve())
                sys.path[0] = str(Path(target).parent)
            if module != "capx_trace_viewer.server" and not (
                target and target.replace("\\", "/").endswith("capx_trace_viewer/server.py")
            ):
                from capx_trace_recorder.bootstrap import activate

                activate()
            sys.argv = [module or target, *trailing]
            if module:
                runpy.run_module(module, run_name="__main__", alter_sys=True)
            else:
                runpy.run_path(target, run_name="__main__")
        finally:
            sys.path[0] = old_path
            sys.argv = old_argv
        return
    parser = argparse.ArgumentParser(prog="capx-trace-recorder")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("install-pth", "uninstall-pth", "status"):
        sub.add_parser(name)
    sub.add_parser("run")
    args = parser.parse_args()
    path = pth_path()
    detail = f"python={sys.executable} CAPX_TRACE_DIR={'set' if os.environ.get('CAPX_TRACE_DIR', '').strip() else 'unset'}"
    if args.command == "install-pth":
        try:
            if path.is_symlink():
                parser.error(f"refusing to overwrite a symlink: {path}")
            if path.is_dir():
                parser.error(f"hook path is a directory: {path}")
            if path.exists():
                existing = _read_hook_for_change(path, parser)
                if existing not in (PTH_LINE, LEGACY_PTH_LINE):
                    parser.error(f"refusing to overwrite a modified hook: {path}")
                if existing == PTH_LINE:
                    print(f"already installed: {path} ({detail})")
                    return
                path.write_text(PTH_LINE, encoding="utf-8")
                print(f"replaced legacy hook: {path} ({detail})")
                return
            with path.open("x", encoding="utf-8") as stream:
                stream.write(PTH_LINE)
        except FileExistsError:
            parser.error(f"hook appeared during installation; inspect it before retrying: {path}")
        except OSError as exc:
            parser.error(f"cannot install hook at {path}: {exc}. Use a writable Python environment")
        print(f"installed: {path} ({detail})")
    elif args.command == "uninstall-pth":
        try:
            if path.is_symlink():
                parser.error(f"refusing to remove a symlink: {path}")
            if path.is_dir():
                parser.error(f"hook path is a directory: {path}")
            if path.exists():
                if _read_hook_for_change(path, parser) not in (PTH_LINE, LEGACY_PTH_LINE):
                    parser.error(f"refusing to remove a modified hook: {path}")
                path.unlink()
                print(f"removed: {path} ({detail})")
            else:
                print(f"not present: {path} ({detail})")
        except OSError as exc:
            parser.error(f"cannot uninstall hook at {path}: {exc}")
    elif args.command == "status":
        try:
            if path.is_dir():
                parser.error(f"hook path is a directory: {path}")
            present = path.exists()
            existing = _read_hook(path) if present else None
            state = ("absent" if not present else "installed" if existing == PTH_LINE
                     else "legacy" if existing == LEGACY_PTH_LINE else "modified")
        except OSError:
            state = "unreadable"
        from capx_trace_recorder.bootstrap import recorder_status
        active, reason = recorder_status()
        print(f"{state}: {path} ({detail}); this process: active={active}; reason={reason or 'none'}")


if __name__ == "__main__":
    main()
