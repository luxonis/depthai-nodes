let state;
let displayedBatch;

function render() {
  const {result, check, batch, mode, distance, preview, synthetic} = state;
  const plane = result?.plane;
  get("activity").textContent = plane ? "Plane held" : "Waiting for plane";
  get("mode").textContent = mode === "continuous" ? "Continuous" : "On start";
  get("scene").hidden = !synthetic;
  get("active").textContent = plane
    ? `Normal (${plane.normal.map((n) => n.toFixed(3)).join(", ")}) · offset ${plane.offset_mm.toFixed(1)} mm · cloud #${result.cloud_sequence}`
    : "Waiting for a plane.";
  get("verification").textContent = check
    ? [
      `Check ${check.request_id} · cloud #${check.cloud_sequence}`,
      `offset ${check.plane.offset_mm.toFixed(1)} mm`,
      check.accepted ? "Passed" : "Rejected",
      measurements(check),
      ...reasons(check),
    ].join(" · ")
    : mode === "on_start" ? "On-start mode holds the first plane." : "Waiting for a check.";
  get("verification").className = check ? (check.accepted ? "pass" : "reject") : "";
  if (batch && displayedBatch !== batch.cloud_sequence) {
    displayedBatch = batch.cloud_sequence;
    get("fit").textContent = `Cloud #${displayedBatch}`;
    get("candidates").replaceChildren();
    if (batch.assessments.length === 0) get("candidates").textContent = "No plane equations.";
    for (const assessment of batch.assessments) {
      const item = document.createElement("p");
      item.className = assessment.accepted ? "pass" : "reject";
      item.textContent = [
        assessment.accepted ? "Passed" : "Rejected",
        measurements(assessment),
        ...reasons(assessment),
      ].join(" · ");
      get("candidates").append(item);
    }
  }
  draw(preview, plane, distance);
}

async function poll() {
  try {
    const response = await fetch("api/state", {cache: "no-store"});
    if (!response.ok) throw new Error("Unable to read application state.");
    state = await response.json();
    render();
  } catch {
    get("activity").textContent = "Disconnected";
  }
  setTimeout(poll, 250);
}

for (const button of document.querySelectorAll("[data-depth]")) {
  button.addEventListener("click", async () => {
    try {
      const response = await fetch("api/action", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({action: "scene", depth_mm: Number(button.dataset.depth)}),
      });
      if (!response.ok) throw new Error((await response.json()).error);
      get("error").hidden = true;
    } catch (error) {
      get("error").textContent = error.message;
      get("error").hidden = false;
    }
  });
}
poll();
