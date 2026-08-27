from muillm.modules.kvcache.cache_utils import MuiCache


class _CacheWithNativeModule(MuiCache):
    def __init__(self) -> None:
        self.deinit_calls = 0

    def _deinit_cpp_module(self) -> None:
        self.deinit_calls += 1


def test_cache_finalizer_releases_native_module() -> None:
    cache = _CacheWithNativeModule()

    cache.__del__()

    assert cache.deinit_calls == 1