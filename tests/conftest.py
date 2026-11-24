import pytest


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    for name in ("RCI_DATA", "RCI_RESULTS", "RCI_CKPT", "RCI_MODELS", "RCI_LOGS", "RCI_CACHE"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    for name in ("RCI_BASE_URLS", "RCI_GPTOSS_SYSTEM_DATE"):
        monkeypatch.delenv(name, raising=False)
    yield
