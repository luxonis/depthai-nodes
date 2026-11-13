const get = (id) => document.getElementById(id);
const canvas = get("depth");
const context = canvas.getContext("2d");
const reasonLabels = {
  insufficient_sample: "Insufficient samples",
  coverage_below_minimum: "Coverage below minimum",
  tilt_above_maximum: "Tilt above maximum",
  plane_not_below_camera: "Plane must be below the camera",
  vertical_distance_unavailable: "No intersection along gravity",
  vertical_distance_out_of_range: "Vertical distance outside range",
};

function measurements(a) {
  const coverage = a.on_plane_ratio === null ? "unavailable" : `${(100 * a.on_plane_ratio).toFixed(1)}%`;
  const vertical = a.vertical_distance_mm === null ? "unavailable" : `${a.vertical_distance_mm.toFixed(0)} mm`;
  return `${vertical} along gravity · ${a.tilt_deg.toFixed(1)}° tilt · ${coverage} coverage (${a.on_plane_point_count}/${a.sampled_point_count})`;
}

function reasons(a) {
  return a.reasons.map(({code, measured, limit}) => {
    const value = typeof measured === "number" ? Number(measured.toPrecision(4)) : measured;
    const bound = Array.isArray(limit) ? limit.join("–") : limit;
    return (reasonLabels[code] ?? code)
      + (value == null ? "" : ` · measured ${value}`)
      + (bound == null ? "" : ` · limit ${bound}`);
  });
}

function draw(preview, plane, tolerance) {
  if (!preview) return;
  const {width, height, points, sequence} = preview;
  canvas.width = width;
  canvas.height = height;
  const frame = context.createImageData(width, height);
  const normal = plane?.normal;
  for (let i = 0; i < width * height; i++) {
    const x = points[3 * i];
    const y = points[3 * i + 1];
    const z = points[3 * i + 2];
    const distance = normal && Math.abs(normal[0] * x + normal[1] * y + normal[2] * z + plane.offset_mm);
    const light = Math.max(0, 1 - z / 2400);
    let color = [12, 17, 20];
    if (z > 0) {
      color = normal && distance <= tolerance
        ? [255, 192, 119]
        : [36 + light * 132, 58 + light * 174, 80 + light * 158];
    }
    frame.data.set([...color, 255], i * 4);
  }
  context.putImageData(frame, 0, 0);
  get("frame").textContent = `Cloud #${sequence}`;
}
