"""Record strict dependency and native/fallback parity from completed reports.

Run with ``python -m tests.parcel_reference.compare_reports LEFT RIGHT``.
A dependency difference remains a failed equality check at the original
1e-9 tolerance. This tool records every difference without changing that gate.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.verify_parcel_port import ABSOLUTE_TOLERANCE, vector


def compare_candidates(left, right, name):
    structural = []
    geometry = []
    maximum_mm = maximum_axis = 0.0
    if len(left) != len(right):
        return [{"name": name, "field": "candidate_count"}], geometry, 0.0, 0.0
    for number, (actual, expected) in enumerate(zip(left, right)):
        for field in ("decision", "indices_sha256", "points_sha256"):
            if actual[field] != expected[field]:
                structural.append({"name": name, "candidate": number, "field": field})
        if actual["decision"] != "eligible" or expected["decision"] != "eligible":
            continue
        a, b = actual["cuboid"], expected["cuboid"]
        if (a is None) != (b is None):
            structural.append({"name": name, "candidate": number, "field": "no_fit"})
            continue
        if a is None:
            continue
        if (a["height_mm"] is None) != (b["height_mm"] is None):
            structural.append(
                {"name": name, "candidate": number, "field": "unknown_height"}
            )
            continue
        difference = np.abs(vector(SimpleNamespace(**a)) - vector(SimpleNamespace(**b)))
        maximum_mm = max(
            maximum_mm, float(np.max(np.concatenate((difference[:3], difference[6:]))))
        )
        maximum_axis = max(maximum_axis, float(np.max(difference[3:6])))
        if np.max(difference) > ABSOLUTE_TOLERANCE:
            geometry.append(
                {
                    "name": name,
                    "candidate": number,
                    "base_center_absolute_difference_mm": difference[:3].tolist(),
                    "length_axis_absolute_difference": difference[3:6].tolist(),
                    "dimension_absolute_difference_mm": difference[6:].tolist(),
                }
            )
    return structural, geometry, maximum_mm, maximum_axis


def compare_records(left, right, label):
    if left.keys() != right.keys():
        raise AssertionError("Compared scene/path coverage differs")
    structural = []
    geometry = []
    maximum_mm = maximum_axis = 0.0
    for key in left:
        a, b, drift_mm, drift_axis = compare_candidates(left[key], right[key], key)
        maximum_mm = max(maximum_mm, drift_mm)
        maximum_axis = max(maximum_axis, drift_axis)
        structural.extend(a)
        geometry.extend(b)
    return {
        "name": label,
        "result": "passed" if not structural and not geometry else "different",
        "compared_record_count": len(left),
        "structural_differences": structural,
        "geometry_differences": geometry,
        "maximum_geometry_drift_mm": maximum_mm,
        "maximum_length_axis_component_drift": maximum_axis,
    }


def scene_records(report, mode=None):
    return {
        (item["mode"] + "/" if mode is None else "") + item["name"]: item["candidates"]
        for item in report["records"]
        if mode is None or item["mode"] == mode
    }


def measurement_records(report, mode=None):
    return {
        (group["mode"] + "/" if mode is None else "") + case["name"]: [
            {
                "decision": "eligible",
                "indices_sha256": None,
                "points_sha256": None,
                "cuboid": case["cuboid"],
            }
        ]
        for group in report["standalone_measurements"]
        if mode is None or group["mode"] == mode
        for case in group["cases"]
    }


def compare_reports(left, right):
    for field in (
        "candidate_id",
        "reference_source_commit",
        "scoped_settings_sha256",
        "synthetic_scene_count",
        "validation_scene_count",
        "validation_manifest_sha256",
    ):
        if left[field] != right[field]:
            raise AssertionError(f"Report {field} differs")
    cross_dependency = compare_records(
        {**scene_records(left), **measurement_records(left)},
        {**scene_records(right), **measurement_records(right)},
        "dependencies",
    )
    native_fallback = []
    for label, report in (("left", left), ("right", right)):
        if report["native_available"]:
            native_fallback.append(
                compare_records(
                    {
                        **scene_records(report, "native"),
                        **measurement_records(report, "native"),
                    },
                    {
                        **scene_records(report, "fallback"),
                        **measurement_records(report, "fallback"),
                    },
                    label,
                )
            )
    return {
        "schema_version": "parcel-environment-comparison/1",
        "result": "passed"
        if cross_dependency["result"] == "passed"
        and all(item["result"] == "passed" for item in native_fallback)
        else "different",
        "tolerance": {"absolute": ABSOLUTE_TOLERANCE, "relative": 0},
        "candidate_id": left["candidate_id"],
        "environments": {"left": left["environment"], "right": right["environment"]},
        "cross_dependency": cross_dependency,
        "native_fallback": native_fallback,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_reports(
        json.loads(args.left.read_text()), json.loads(args.right.read_text())
    )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
