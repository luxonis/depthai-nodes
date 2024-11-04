def pytest_addoption(parser):
    parser.addoption(
        "--nn_archive_path", action="store", default="", help="NN archive path"
    )
    parser.addoption(
        "--slug", action="store", default="", help="Model slug from the ZOO."
    )
