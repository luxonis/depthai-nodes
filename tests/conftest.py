def pytest_addoption(parser):
    parser.addoption(
        "--nn_archive_path", action="store", default="", help="NN archive path"
    )
    parser.addoption(
        "--slug", action="store", default="", help="Model slug from the ZOO."
    )
    parser.addoption(
        "--platform",
        action="store",
        default="",
        help="RVC platform to run the tests on.",
    )
    parser.addoption(
        "--models", action="store", default="", help="Model slug from the ZOO."
    )
    parser.addoption("--parsers", action="store", default="", help="Parsers to test.")
