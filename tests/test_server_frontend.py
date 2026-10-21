"""Frontend selection and missing-build behavior of the standalone server."""

from fastapi.testclient import TestClient

from capx_trace_viewer.server import create_app


def test_selected_frontend_serves_index_and_static_files(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("viewer")
    (dist / "extra.html").write_text("extra")
    client = TestClient(create_app(trace_root=tmp_path / "traces", frontend_dist=dist))
    assert client.get("/traces").text == "viewer"
    assert client.get("/extra.html").text == "extra"


def test_missing_frontend_omits_selected_dist(tmp_path):
    dist = tmp_path / "missing-dist"
    client = TestClient(create_app(trace_root=tmp_path / "traces", frontend_dist=dist))
    for route in ("/traces", "/extra.html"):
        response = client.get(route)
        assert response.status_code == 503
        assert str(dist) not in response.text
