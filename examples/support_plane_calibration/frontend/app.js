let state;
let selected = null;
let shownAttempt = -1;
let busy = false;
let online = false;
let commandError = "";

function render() {
  const disabled = busy || !online;
  if (!state) {
    get("start").disabled = true;
    get("activity").textContent = online ? "Connecting" : "Disconnected";
    return;
  }
  const {batch, attempt, searching, pending, verification, active, saved_at, error} = state;
  if (shownAttempt !== attempt || !batch) {
    shownAttempt = attempt;
    selected = null;
  }
  const assessments = batch?.assessments ?? [];
  const approvedCheck = verification?.candidate?.[0] === attempt
    && verification.candidate[1] === selected && verification.accepted;
  get("activity").textContent = !online ? "Disconnected" : searching ? "Searching" : pending ? "Verifying" : "Idle";
  get("start").disabled = disabled || searching || pending;
  get("stop").disabled = disabled || !searching;
  get("verify").disabled = disabled || searching || selected === null || pending;
  get("approve").disabled = disabled || searching || pending || !approvedCheck;
  get("restore").disabled = disabled || !saved_at || pending;
  get("invalidate").disabled = disabled || (!saved_at && !error);
  get("error").textContent = commandError || error || "";
  get("error").hidden = !get("error").textContent;
  get("fit").textContent = batch ? `Fit ${attempt} · cloud #${batch.cloud_sequence}` : "";
  get("batch-info").textContent = !batch
    ? searching ? "Waiting for a fitting attempt." : "Start a search to fit candidate planes."
    : assessments.length === 0 ? "This fit produced no plane equations."
    : searching ? "Select a plane to preview. Stop the search to verify it."
    : "Select a plane to preview and verify.";
  const candidates = get("candidates");
  if (candidates.dataset.attempt !== String(attempt) || !batch) {
    candidates.replaceChildren();
    candidates.dataset.attempt = String(attempt);
    assessments.forEach((a, index) => {
      const button = document.createElement("button");
      button.className = "candidate";
      const title = document.createElement("strong");
      title.className = a.accepted ? "pass" : "reject";
      title.textContent = `Plane ${index + 1} · ${a.accepted ? "Passed" : "Rejected"}`;
      button.append(title);
      for (const line of [measurements(a), ...reasons(a)]) {
        const detail = document.createElement("small");
        detail.textContent = line;
        button.append(detail);
      }
      button.addEventListener("click", () => {
        selected = index;
        render();
      });
      candidates.append(button);
    });
  }
  Array.from(candidates.children).forEach((button, index) => {
    button.setAttribute("aria-pressed", String(selected === index));
  });
  get("verification").textContent = pending ? "Waiting for a usable cloud and gravity."
    : verification ? [
      `Check ${verification.request_id} · cloud #${verification.cloud_sequence}`,
      verification.accepted ? "Passed" : "Rejected",
      measurements(verification),
      ...reasons(verification),
    ].join(" · ") : "";
  get("active").textContent = active
    ? `Normal (${active.normal.map((n) => n.toFixed(3)).join(", ")}) · offset ${active.offset_mm.toFixed(1)} mm`
    : saved_at ? `Approval saved ${saved_at}. Verify it to activate.` : "No active plane.";
  get("overlay").textContent = selected !== null ? `Showing Plane ${selected + 1} on live depth.`
    : active ? "Showing the active plane on live depth." : "";
  draw(state.preview, assessments[selected]?.plane ?? active, state.distance);
}

async function refresh() {
  try {
    const response = await fetch("api/state", {cache: "no-store"});
    if (!response.ok) throw new Error("Unable to read the application state.");
    state = await response.json();
    online = true;
  } catch {
    online = false;
  }
  render();
}

async function action(name, fields = {}) {
  if (busy || !online) return;
  busy = true;
  commandError = "";
  render();
  try {
    const response = await fetch("api/action", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action: name, ...fields}),
    });
    if (!response.ok) throw new Error((await response.json()).error);
  } catch (error) {
    commandError = error.message;
  } finally {
    busy = false;
    await refresh();
  }
}

for (const name of ["start", "stop", "restore", "invalidate"]) {
  get(name).addEventListener("click", () => action(name));
}
get("verify").addEventListener("click", () => action("verify", {index: selected, attempt: state.attempt}));
get("approve").addEventListener("click", () => action("approve", {index: selected, request_id: state.verification.request_id}));

async function poll() {
  if (!busy) await refresh();
  setTimeout(poll, 250);
}
poll();
