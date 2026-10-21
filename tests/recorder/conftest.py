import pytest
import importlib.util


def pytest_collection_modifyitems(items):
    if importlib.util.find_spec("capx") is not None:
        return
    for item in items:
        if item.get_closest_marker("requires_capx"):
            item.add_marker(pytest.mark.skip(reason="requires an installed CaP-X checkout"))

@pytest.fixture
def work_dir(tmp_path):
    return tmp_path
