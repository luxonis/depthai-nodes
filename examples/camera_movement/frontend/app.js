import { CameraConnection, CameraVideo } from "./stream.js";

const settings = [
  ["Tilt alerts", [
    ["tilt_threshold_deg", "Tilt threshold", "°", "Angle from the calibrated pose that must be reached before an alert can fire."],
    ["tilt_rearm_deg", "Rearm threshold", "°", "The tilt must fall to this angle or lower before another tilt alert can fire. Keep it below the tilt threshold."],
    ["tilt_confirmation_s", "Confirmation time", "s", "How long tilt must stay across the alert or rearm threshold."],
  ]],
  ["Horizontal turns", [
    ["turn_threshold_deg", "Turn threshold", "°", "Minimum net turn angle. The turn must reach 3°/s and settle for 0.5 s within 15 s. Leave blank to disable.", true],
  ]],
  ["Calibration", [
    ["calibration_window_s", "Stationary window", "s", "How long the camera must be still to accept a reference pose."],
    ["calibration_timeout_s", "Calibration timeout", "s", "How long to wait for a stationary window. A timeout keeps any existing baseline."],
  ]],
  ["Status and IMU", [
    ["status_interval_s", "Status interval", "s", "How often the node sends a status heartbeat. Leave blank to disable heartbeat reporting.", true],
    ["max_sample_gap_s", "Maximum sample gap", "s", "A larger gap in either IMU stream resets pending calibration, tilt, and turn evidence."],
    ["imu_stale_after_s", "IMU stale after", "s", "Age of an accelerometer or gyro sample before the status marks the IMU stale."],
  ]],
];

const $ = (id) => document.getElementById(id);
const inputs = new Map();
for (const [title, fields] of settings) {
  const group = document.createElement("fieldset");
  const legend = document.createElement("legend");
  legend.textContent = title;
  group.append(legend);
  for (const [name, label, unit, help, optional] of fields) {
    const row = document.createElement("div");
    row.className = "field";
    const element = document.createElement("label");
    element.htmlFor = name;
    element.textContent = label;
    const wrap = document.createElement("div");
    wrap.className = "input-wrap";
    const input = document.createElement("input");
    input.type = "number";
    input.id = name;
    input.name = name;
    input.min = "0";
    input.step = "any";
    input.required = !optional;
    input.placeholder = optional ? "Off" : "";
    input.setAttribute("aria-describedby", `${name}-help`);
    const hint = document.createElement("p");
    hint.id = `${name}-help`;
    hint.className = "field-help";
    hint.textContent = help;
    const units = document.createElement("span");
    units.textContent = unit;
    wrap.append(input, units);
    row.append(element, wrap, hint);
    group.append(row);
    inputs.set(name, input);
  }
  $("fields").append(group);
}

const calibrationNames = {
  not_requested: "Not requested",
  collecting: "Collecting",
  succeeded: "Succeeded",
  timed_out: "Timed out",
};
const reasonNames = { tilt_changed: "Tilt changed", turn_detected: "Turn detected" };
const video = new CameraVideo($("video"), $("video-status"));
let state = null;
let connected = false;
let busy = false;
let awaitingApply = false;
let eventSignature;
let commandVersion = 0;

function showFeedback(id, message, error = false) {
  const element = $(id);
  element.textContent = message;
  element.dataset.error = error;
}

function updateButtons() {
  $("defaults").disabled = !state;
  $("apply").disabled = !connected || !state?.ready || busy || state.applying;
  $("calibrate").disabled = $("apply").disabled || state.status?.calibration === "collecting";
}

function fillForm(config) {
  for (const [name, input] of inputs) input.value = config[name] ?? "";
}

function showStatus(data) {
  const status = data.status;
  const age = (value) => value == null ? "—" : `${value.toFixed(3)} s`;
  const metrics = {
    calibration: calibrationNames[status?.calibration],
    baseline_ready: status && (status.baseline_ready ? "Ready" : "Unset"),
    tilt_deg: status?.tilt_deg == null ? "—" : `${status.tilt_deg.toFixed(2)}°`,
    tilt_changed: status && (status.tilt_changed ? "Active" : "Clear"),
    imu_stale: status && (status.imu_stale ? "Stale" : "Fresh"),
    accelerometer_age_s: age(status?.accelerometer_age_s),
    gyroscope_age_s: age(status?.gyroscope_age_s),
  };
  for (const metric of $("metrics").querySelectorAll("dd")) {
    const key = metric.dataset.status;
    metric.textContent = metrics[key] ?? "—";
    metric.dataset.alert = (key === "tilt_changed" && status?.tilt_changed) || (key === "imu_stale" && status?.imu_stale) || false;
  }
  $("status-note").textContent = data.config.status_interval_s === null
    ? "Heartbeat reporting is off. Set a status interval to see live node status."
    : status ? `Heartbeat every ${data.config.status_interval_s} s` : "Waiting for a heartbeat.";
  if (status?.calibration === "collecting") showFeedback("calibration-feedback", "");
}

function showEvents(events) {
  const signature = JSON.stringify(events);
  if (signature === eventSignature) return;
  eventSignature = signature;
  const rows = events.map((event) => {
    const row = document.createElement("tr");
    const time = new Date(event.time * 1000).toLocaleTimeString();
    for (const value of [reasonNames[event.reason] ?? event.reason, `${event.angle_deg.toFixed(1)}°`, time]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  });
  $("events").replaceChildren(...rows);
  $("no-events").hidden = events.length > 0;
  $("events-table").hidden = events.length === 0;
}

function update(data) {
  if (!state) fillForm(data.config);
  state = data;
  showStatus(data);
  showEvents(data.events);
  if (awaitingApply && !data.applying) {
    awaitingApply = false;
    showFeedback("feedback", "Settings applied.");
  }
  updateButtons();
}

const connection = new CameraConnection(
  (online) => {
    connected = online;
    $("connection").textContent = online ? "Connected" : "Reconnecting";
    $("connection").dataset.connected = online;
    if (!online) $("status-note").textContent = "Disconnected. Showing the last heartbeat.";
    updateButtons();
  },
  (data, timestamp) => video.push(data, timestamp),
  () => video.reset(),
);

async function command(action, config, feedback) {
  busy = true;
  commandVersion++;
  updateButtons();
  try {
    const data = await connection.call(action, config);
    if (action === "apply") {
      awaitingApply = true;
      showFeedback(feedback, "Applying settings…");
    } else {
      showFeedback(feedback, "Calibration requested.");
    }
    update(data);
  } catch (error) {
    showFeedback(feedback, error.message, true);
  } finally {
    busy = false;
    updateButtons();
  }
}

$("settings-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const config = Object.fromEntries(Array.from(inputs, ([name, input]) =>
    [name, input.value === "" ? null : input.valueAsNumber]
  ));
  command("apply", config, "feedback");
});
$("calibrate").addEventListener("click", () => command("calibrate", undefined, "calibration-feedback"));
$("defaults").addEventListener("click", () => {
  fillForm(state.defaults);
  showFeedback("feedback", "Defaults loaded. Apply settings to use them.");
});

async function poll() {
  if (connected && !busy) {
    try {
      const version = commandVersion;
      const data = await connection.call("state");
      if (version === commandVersion && !busy && connected) update(data);
    } catch {
      if (connected) $("status-note").textContent = "Waiting for the camera to respond.";
    }
  }
  setTimeout(poll, 250);
}
poll();
