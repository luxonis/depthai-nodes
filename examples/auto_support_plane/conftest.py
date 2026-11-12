"""Keep native example checks separate from the library suite's global node mocks."""

from pathlib import Path

APP = Path(__file__).resolve().parent


def pytest_ignore_collect(collection_path, config):
    return (
        config.rootpath.resolve() != APP and collection_path.resolve() == APP / "tests"
    )
