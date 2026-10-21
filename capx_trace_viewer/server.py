"""Read-only FastAPI server for CaP-X traces."""
from __future__ import annotations

import argparse
import os
import stat
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from capx_trace_viewer.trace_api import create_trace_router

_CWD = Path.cwd()
_DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[1] / 'web-ui' / 'dist'
_DEFAULT_RUNNER_ROOTS = ('outputs/runner',)
_VIEWER_ONLY_MISSING = ('api/', 'ws/', 'viser-proxy')


def default_trace_root() -> Path:
    raw = os.environ.get('CAPX_TRACE_DIR', '').strip()
    if not raw:
        return _CWD / 'outputs' / 'traces'
    path = Path(os.path.expanduser(raw))
    return Path(os.path.abspath(_CWD / path))


def _package_version() -> str:
    try:
        return version('capx-trace-viewer')
    except PackageNotFoundError:
        return '0.0.0'


def _cli_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(value))))


def _safe_dist_file(dist_root: Path, path: str) -> Path | None:
    if not path:
        return None
    try:
        candidate = (dist_root / path).resolve()
        if not candidate.is_relative_to(dist_root):
            return None
        st = candidate.stat()
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate if stat.S_ISREG(st.st_mode) else None


def _frontend_file_response(path: Path) -> FileResponse:
    headers = {'Cache-Control': 'no-store'} if path.suffix.lower() == '.html' else None
    return FileResponse(path, headers=headers)


def create_app(*, viewer_only: bool = True, trace_root: str | Path | None = None,
               expose_traces: bool = False, runner_roots: Sequence[str | Path] | None = None,
               frontend_dist: str | Path | None = None) -> FastAPI:
    if not viewer_only:
        raise ValueError('The standalone server supports viewer-only mode')
    app = FastAPI(title='CaP-X Trace Viewer', version=_package_version(),
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware,
                       allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000'],
                       allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    trace_dir = _cli_path(trace_root) if trace_root is not None else default_trace_root()
    roots = [_cli_path(r) for r in runner_roots] if runner_roots else [_CWD / r for r in _DEFAULT_RUNNER_ROOTS]
    dist = _cli_path(frontend_dist) if frontend_dist is not None else _DEFAULT_FRONTEND_DIST
    app.state.viewer_only = True
    app.state.trace_root = trace_dir
    app.include_router(create_trace_router(trace_dir, viewer_only=True, expose=expose_traces, runner_roots=roots))

    @app.get('/')
    async def redirect_to_traces():
        return RedirectResponse('/traces', status_code=307)

    if not (dist / 'index.html').is_file():
        @app.get('/traces')
        async def frontend_missing():
            return Response('Frontend dist is missing or has no index.html. Build the frontend (cd web-ui && npm ci && npm run build) or pass --frontend-dist. The trace API is served at /api/traces.\n',
                            status_code=503, media_type='text/plain')
        @app.get('/{path:path}')
        async def frontend_missing_file(path: str):
            return Response('Frontend dist is missing or has no index.html. Build the frontend or pass --frontend-dist.\n',
                            status_code=503, media_type='text/plain')
        return app

    if (dist / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')
    dist_root = dist.resolve()

    @app.get('/{path:path}')
    async def serve_viewer_routes(path: str):
        if path.startswith(_VIEWER_ONLY_MISSING):
            return JSONResponse({'detail': 'not available in --viewer-only mode'}, status_code=404)
        file_path = _safe_dist_file(dist_root, path)
        return _frontend_file_response(file_path or dist / 'index.html')

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description='CaP-X read-only trace viewer')
    parser.add_argument('--viewer-only', action='store_true', help='Accepted for compatibility; viewer-only is always on')
    parser.add_argument('--trace-root')
    parser.add_argument('--runner-root', nargs='+', action='append', default=[])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8290)
    parser.add_argument('--expose-traces', action='store_true')
    parser.add_argument('--frontend-dist', help='Directory containing the built frontend index.html and assets')
    args = parser.parse_args()
    dist = _cli_path(args.frontend_dist) if args.frontend_dist is not None else _DEFAULT_FRONTEND_DIST
    if not (dist / 'index.html').is_file():
        print(f'Frontend dist is missing or has no index.html: {dist}', file=sys.stderr)
    roots = [root for group in args.runner_root for root in group]
    app = create_app(trace_root=args.trace_root, runner_roots=roots or None,
                     expose_traces=args.expose_traces, frontend_dist=args.frontend_dist)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
