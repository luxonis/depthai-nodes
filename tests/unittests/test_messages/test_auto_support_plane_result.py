import pytest

from depthai_nodes.message import AutoSupportPlaneResult, SupportPlane

PLANE = SupportPlane(normal=(0, 0, 1), offset_mm=-1000)


def test_result_rejects_passed_without_a_plane():
    with pytest.raises(ValueError):
        AutoSupportPlaneResult(True, None)


@pytest.mark.parametrize(
    "passed,plane",
    [(1, PLANE), ("yes", PLANE), (None, None), (True, (0, 0, 1)), (False, -1000.0)],
)
def test_result_rejects_wrong_types(passed, plane):
    with pytest.raises(TypeError):
        AutoSupportPlaneResult(passed, plane)


@pytest.mark.parametrize("field", ["passed", "plane"])
def test_result_fields_are_read_only(field):
    result = AutoSupportPlaneResult(True, PLANE)

    with pytest.raises(AttributeError):
        setattr(result, field, None)
