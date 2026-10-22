"""Linux-only tests declare their platform requirement at collection time."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LINUX_TESTS = {
    "tests/recorder/test_trial_videos.py": ("test_fifo_is_skipped_and_later_events_are_written",),
    "tests/recorder/test_writer_limits.py": (
        "test_run_json_short_writes_are_completed",
        "test_a_failed_run_json_write_is_recorded_and_retried",
        "test_a_failed_unscoped_log_is_recorded_outside_it",
    ),
    "tests/recorder/test_entrypoints.py": ("test_orphaned_pth_reports_once_through_lib64_symlink",),
    "tests/test_trace_reader.py": ("test_dir_rejects_fifo_without_blocking",),
}


def test_linux_only_cases_have_collection_marks():
    for filename, names in LINUX_TESTS.items():
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in names:
            decorators = [ast.unparse(node) for node in functions[name].decorator_list]
            assert any("pytest.mark.skipif(sys.platform != 'linux'" in item for item in decorators), (filename, name)

    alarm = (ROOT / "tests/recorder/test_alarm_windows.py").read_text(encoding="utf-8")
    assert 'pytestmark = pytest.mark.skipif(sys.platform != "linux"' in alarm
