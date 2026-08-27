from fastapi.testclient import TestClient

from muillm.server.server import build_app


class _MemoryManager:
    tp_size = 2

    def memory_stats(self, collect: bool = False, reset_peak: bool = False):
        self.collect = collect
        self.reset_peak = reset_peak
        return [{"process_rss_bytes": 123}, {"process_rss_bytes": 456}]


def test_memory_endpoint_requests_worker_stats() -> None:
    manager = _MemoryManager()
    client = TestClient(build_app(manager))

    response = client.get("/debug/memory?collect=true&reset_peak=true")

    assert response.status_code == 200
    assert response.json() == {"workers": [{"process_rss_bytes": 123}, {"process_rss_bytes": 456}]}
    assert manager.collect is True
    assert manager.reset_peak is True