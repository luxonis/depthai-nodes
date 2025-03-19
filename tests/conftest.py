def pytest_addoption(parser):
    parser.addoption("--nn_archive_path", default="", type=str, help="NN archive path")
    parser.addoption("--model", default="", type=str, help="Model from the ZOO.")
    parser.addoption(
        "--platform",
        default="",
        type=str,
        help="RVC platform to run the tests on.",
    )
    parser.addoption("--models", default="", type=str, help="Model slug from the ZOO.")
    parser.addoption("--parsers", default="", type=str, help="Parsers to test.")
    parser.addoption(
        "--duration",
        default=None,
        type=int,
        help="Duration of the test in seconds.",
    )
