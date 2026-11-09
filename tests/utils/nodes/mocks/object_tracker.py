from .input import InputMock
from .output import OutputMock


class ObjectTrackerMock:
    def __init__(self):
        self.inputDetections = InputMock()
        self.inputTrackerFrame = InputMock()
        self._out = OutputMock()
        self._tracker_type = None
        self._id_assignment_policy = None

    @property
    def out(self):
        return self._out

    def setTrackerType(self, tracker_type):
        self._tracker_type = tracker_type

    def setTrackerIdAssignmentPolicy(self, policy):
        self._id_assignment_policy = policy
