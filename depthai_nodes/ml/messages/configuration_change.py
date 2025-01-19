import depthai as dai


class ConfigurationChange(dai.Buffer):
    def __init__(self, parameter: str, value_as_string: str):
        super().__init__()
        self.parameter = parameter
        self.value_as_string = value_as_string
