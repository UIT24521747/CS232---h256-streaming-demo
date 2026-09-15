// Plain JS, no framework: poll /api/state + /api/users every ~250ms and
// redraw. Each user cell is created once (see createCell) and only ever
// updated in place afterward -- a bandwidth slider mid-drag or an
// in-flight image load is never disrupted by the next poll tick.
//
// Every user cell's "Received + SR" pane is a real per-user player: it
// only advances once that user's frame image has actually been written by
// the backend (see on_frame in web/app.py), so a low-bandwidth user
// visibly stalls (BUFFERING) independently of every other user.

const STAGE_STEPS = [
  { key: "source", label: "SOURCE" },
  { key: "encode", label: "H256 ENCODER" },
  { key: "streaming", label: "STREAMING NODES" },
  { key: "lb", label: "LB + CLIENT + DECODE + SR (per user)" },
  { key: "final", label: "FINAL" },
];

const STAGE_MAP = {
  starting: "source",
  "video loaded": "source",
  "H256 encoding": "encode",
  "starting streaming nodes": "streaming",
  "client streaming": "lb",
  done: "final",
};

const QUALITY_LADDER = [[0, "240p"], [1200, "360p"], [2500, "540p"]];

function chooseQuality(bandwidth) {
  let chosen = QUALITY_LADDER[0][1];
  for (const [minKbps, name] of QUALITY_LADDER) {
    if (bandwidth >= minKbps) chosen = name;
  }
  return chosen;
}

const el = (id) => document.getElementById(id);

const cells = new Map(); // user_id -> DOM refs
const players = new Map(); // user_id -> playback state
let selectedUserId = null;
let currentVideoId = null;

// ---------------------------------------------------------------- player --

// A gap between "ready" frames shorter than this is just normal poll
// jitter (frames arrive in bursts once per segment, but we only learn
// about them once every POLL_MS) -- not a real stall. Only flag BUFFERING
// once we've been stuck on the same frame for longer than that, so the
// overlay doesn't flash on/off every poll cycle.
const STALL_GRACE_MS = 300;

function ensurePlayer(userId, runId, totalFrames, fps) {
  let p = players.get(userId);
  if (!p || p.runId !== runId) {
    if (p && p.timer) clearInterval(p.timer);
    p = { runId, totalFrames, fps: fps || 30, playhead: 0, readySet: new Set(), timer: null, stalledSince: null };
    p.timer = setInterval(() => tickPlayer(userId), 1000 / p.fps);
    players.set(userId, p);
  }
  if (totalFrames) p.totalFrames = totalFrames;
  return p;
}

function tickPlayer(userId) {
  const p = players.get(userId);
  const cell = cells.get(userId);
  if (!p || !cell || !p.totalFrames || !currentVideoId) return;

  if (p.playhead >= p.totalFrames) {
    cell.overlay.hidden = true;
    return;
  }

  if (p.readySet.has(p.playhead)) {
    const fid = p.playhead;
    cell.imgFinal.src = `/outputs/${currentVideoId}/users/${userId}/live/frame_${fid}_final.png?run=${p.runId}`;
    cell.overlay.hidden = true;
    p.stalledSince = null;
    p.playhead += 1;
  } else {
    if (p.stalledSince === null) p.stalledSince = performance.now();
    if (performance.now() - p.stalledSince > STALL_GRACE_MS) {
      cell.overlay.hidden = false;
    }
  }
}

// -------------------------------------------------------------- user grid --

function createCell(userId) {
  const root = document.createElement("div");
  root.className = "user-cell";
  root.innerHTML = `
    <div class="user-cell-head">
      <span class="user-id">${userId}</span>
      <button type="button" class="user-remove" title="remove user">&times;</button>
    </div>
    <div class="user-bandwidth-row">
      <input type="range" class="user-bandwidth" min="200" max="6000" step="50" value="1800">
      <span class="user-bandwidth-value">1800 kbps</span>
    </div>
    <div class="user-player-row">
      <img class="user-player-image" data-role="final" alt="received + SR">
      <div class="player-overlay user-overlay" hidden>BUFFERING</div>
    </div>
    <div class="user-frame-label">waiting for frames&hellip;</div>
    <div class="buffer-meter-track"><div class="buffer-meter-fill"></div></div>
    <div class="user-stats"></div>
  `;

  const cell = {
    root,
    remove: root.querySelector(".user-remove"),
    bandwidth: root.querySelector(".user-bandwidth"),
    bandwidthValue: root.querySelector(".user-bandwidth-value"),
    imgFinal: root.querySelector('[data-role="final"]'),
    overlay: root.querySelector(".user-overlay"),
    frameLabel: root.querySelector(".user-frame-label"),
    stats: root.querySelector(".user-stats"),
    bufferFill: root.querySelector(".buffer-meter-fill"),
  };

  cell.bandwidth.addEventListener("input", () => {
    cell.bandwidthValue.textContent = `${cell.bandwidth.value} kbps (${chooseQuality(Number(cell.bandwidth.value))})`;
    fetch(`/api/users/${userId}/bandwidth`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bandwidth: Number(cell.bandwidth.value) }),
    }).catch(() => {});
  });

  cell.remove.addEventListener("click", (e) => {
    e.stopPropagation();
    fetch(`/api/users/${userId}`, { method: "DELETE" })
      .then(() => removeCell(userId))
      .catch(() => {});
  });

  root.addEventListener("click", () => selectUser(userId));

  cells.set(userId, cell);
  el("user-grid").appendChild(root);
  return cell;
}

function removeCell(userId) {
  const cell = cells.get(userId);
  if (cell) cell.root.remove();
  cells.delete(userId);
  const p = players.get(userId);
  if (p && p.timer) clearInterval(p.timer);
  players.delete(userId);
  if (selectedUserId === userId) {
    selectedUserId = null;
    el("user-detail-card").hidden = true;
    el("trace-card").hidden = true;
  }
  el("user-grid-hint").hidden = cells.size > 0;
}

function renderUsers(state, users) {
  currentVideoId = state.video_id;

  const seen = new Set(users.map((u) => u.user_id));
  for (const userId of Array.from(cells.keys())) {
    if (!seen.has(userId)) removeCell(userId);
  }
  el("user-grid-hint").hidden = users.length > 0;

  users.forEach((row) => {
    const cell = cells.get(row.user_id) || createCell(row.user_id);

    if (document.activeElement !== cell.bandwidth) {
      cell.bandwidth.value = row.bandwidth_kbps;
      cell.bandwidthValue.textContent = `${row.bandwidth_kbps.toFixed(0)} kbps (${chooseQuality(row.bandwidth_kbps)})`;
    }

    const last = row.last_frame;
    cell.frameLabel.textContent = last
      ? `#${last.frame_id} · ${last.type} · ${last.quality} · node ${last.node}`
      : "waiting for frames…";

    const psnr = row.avg_psnr !== null && row.avg_psnr !== undefined ? `${row.avg_psnr} dB` : "-";
    cell.stats.textContent =
      `${row.frames_received}/${row.total_frames || state.num_frames || "?"} frames · ` +
      `avg PSNR ${psnr} · ${row.total_bytes} B received · buffer ${row.buffer_level_s.toFixed(2)}s` +
      (row.rebuffering ? " · REBUFFERING" : "");
    cell.root.classList.toggle("rebuffering", !!row.rebuffering);
    cell.bufferFill.style.width = `${Math.min(100, (row.buffer_level_s / 2) * 100)}%`;

    const p = ensurePlayer(row.user_id, state.run_id, row.total_frames || state.num_frames, state.playback_fps);
    (row.frames_ready || []).forEach((fid) => p.readySet.add(fid));
  });
}

// ------------------------------------------------------------ frame table --

async function selectUser(userId) {
  selectedUserId = userId;
  el("user-detail-card").hidden = false;
  el("detail-user-label").textContent = userId;
  el("trace-card").hidden = true;
  try {
    const res = await fetch(`/api/users/${userId}/frames`);
    const history = await res.json();
    renderFrameTable(userId, history);
  } catch (err) {
    // transient -- next click retries
  }
}

function renderFrameTable(userId, history) {
  const tbody = document.querySelector("#frame-table tbody");
  tbody.innerHTML = history
    .map((r) => {
      const ref = (r.refs || []).join(", ") || "-";
      const psnr = r.psnr_db !== null && r.psnr_db !== undefined ? `${r.psnr_db} dB` : "inf";
      return `<tr data-frame="${r.frame_id}">
        <td>${r.recv_order}</td>
        <td>${r.frame_id}</td>
        <td><span class="tag ${r.type}">${r.type}</span></td>
        <td>${ref}</td>
        <td>${r.quality}</td>
        <td>${r.node}</td>
        <td>${r.encoded_bytes}</td>
        <td>${psnr}</td>
      </tr>`;
    })
    .join("");
  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => loadTrace(userId, Number(tr.dataset.frame)));
  });
}

async function loadTrace(userId, frameId) {
  try {
    const res = await fetch(`/api/users/${userId}/frames/${frameId}`);
    const trace = await res.json();
    if (trace.error) return;
    renderTrace(trace);
  } catch (err) {
    // transient -- next click retries
  }
}

function renderTrace(t) {
  el("trace-card").hidden = false;
  el("trace-user-label").textContent = `${t.user_id} · frame ${t.frame_id}`;

  const rows = [
    ["stage-title", "[1] SOURCE"],
    ["k", "resolution"], ["v", t.source.resolution],
    ["k", "original size"], ["v", `${t.source.original_size_bytes} B`],

    ["stage-title", "[2] FRAME TYPE"],
    ["k", "type"], ["v", t.frame_type.type],
    ["k", "reference"], ["v", [t.frame_type.ref1, t.frame_type.ref2].filter((x) => x !== null && x !== undefined).join(", ") || "-"],

    ["stage-title", "[3] PREDICTION"],
    ["k", "method"], ["v", t.prediction.description],

    ["stage-title", "[4] RESIDUAL"],
    ["k", "range"], ["v", `[${t.residual.min}, ${t.residual.max}]`],
    ["k", "MSE"], ["v", t.residual.mse.toFixed(2)],

    ["stage-title", "[5] QUANTIZATION"],
    ["k", "QP"], ["v", t.quantization.qp],

    ["stage-title", "[6] H256 ENCODE"],
    ["k", "raw -> encoded"], ["v", `${t.h256_encode.raw_size} -> ${t.h256_encode.encoded_size} B`],
    ["k", "compression"], ["v", `${t.h256_encode.compression_pct.toFixed(1)}%`],

    ["stage-title", "[7] BITSTREAM"],
    ["k", "segment"], ["v", `${t.bitstream.quality}/${t.bitstream.segment}`],

    ["stage-title", "[8] LOAD BALANCER"],
    ["k", "selected node"], ["v", `${t.load_balancer.selected_node} (counter=${t.load_balancer.lb_counter})`],
    ["k", "policy"], ["v", t.load_balancer.policy],

    ["stage-title", "[9-10] CLIENT + DECODE"],
    ["k", "received bytes"], ["v", t.client.received_bytes],
    ["k", "decoded type"], ["v", t.h256_decode.frame_type],

    ["stage-title", "[11] RECONSTRUCTION"],
    ["k", "resolution"], ["v", t.reconstruction.resolution],
    ["k", "PSNR / SSIM"], ["v", `${t.reconstruction.psnr.toFixed(2)} dB / ${t.reconstruction.ssim.toFixed(4)}`],

    ["stage-title", "[12] SUPER RESOLUTION"],
    ["k", "input -> output"], ["v", `${t.super_resolution.input} -> ${t.super_resolution.output}`],
    ["k", "method"], ["v", t.super_resolution.method],
  ];

  let html = "";
  for (let i = 0; i < rows.length; i++) {
    const [cls, text] = rows[i];
    if (cls === "stage-title") {
      html += `<div class="stage-title">${text}</div>`;
    } else if (cls === "k") {
      const [, val] = rows[i + 1];
      html += `<div class="k">${text}</div><div>${val}</div>`;
      i++;
    }
  }
  el("trace-grid").innerHTML = html;

  const base = t.image_base;
  const bust = Date.now();
  el("trace-reconstructed").src = `${base}/04_reconstructed.png?t=${bust}`;
  el("trace-mv").src = `${base}/mv_overlay.png?t=${bust}`;
  el("trace-summary").src = `${base}/summary.png?t=${bust}`;
}

// ------------------------------------------------------------- misc panels --

function renderPipelineStrip(currentStage) {
  const container = el("pipeline-strip");
  const currentKey = STAGE_MAP[currentStage] || null;
  const currentIndex = STAGE_STEPS.findIndex((s) => s.key === currentKey);

  container.innerHTML = "";
  STAGE_STEPS.forEach((step, i) => {
    const span = document.createElement("span");
    span.className = "step";
    if (currentIndex >= 0) {
      if (i < currentIndex) span.className += " done";
      else if (i === currentIndex) span.className += " active";
    }
    span.textContent = step.label;
    container.appendChild(span);
    if (i < STAGE_STEPS.length - 1) {
      const arrow = document.createElement("span");
      arrow.className = "step-arrow";
      arrow.textContent = "→";
      container.appendChild(arrow);
    }
  });
}

function renderVideoMeta(state) {
  const rows = [
    ["video_id", state.video_id || "-"],
    ["resolution", state.resolution || "-"],
    ["frames", state.num_frames || "-"],
    ["native fps", state.native_fps ? state.native_fps.toFixed(1) : "-"],
    ["QP (last run)", state.qp],
  ];
  el("video-meta").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  const runBtn = el("run-btn");
  const statusLine = el("status-line");
  const errorBanner = el("error-banner");
  runBtn.disabled = state.status === "running";
  if (state.status === "error") {
    statusLine.textContent = `error: ${state.error}`;
    statusLine.className = "status-line error";
    errorBanner.textContent = `Pipeline gặp lỗi, mọi ô user sẽ đứng ở "waiting for frames": ${state.error}`;
    errorBanner.hidden = false;
  } else {
    const time = state.run_seconds !== null && state.run_seconds !== undefined ? ` (${state.run_seconds}s)` : "";
    statusLine.textContent = `${state.status}${state.stage ? " - " + state.stage : ""}${time}`;
    statusLine.className = "status-line";
    errorBanner.hidden = true;
  }
}

function renderLbTable(state) {
  const tbody = document.querySelector("#lb-table tbody");
  tbody.innerHTML = state.lb_log
    .slice()
    .reverse()
    .map((e) => `<tr><td>${e.user || "-"}</td><td>${e.request}</td><td>${e.selected_node}</td><td>${e.lb_counter}</td></tr>`)
    .join("");
}

function renderLog(state) {
  const box = el("log-box");
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
  box.textContent = state.log.join("\n");
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

// --------------------------------------------------------------- polling --

async function poll() {
  try {
    const [stateRes, usersRes] = await Promise.all([fetch("/api/state"), fetch("/api/users")]);
    const state = await stateRes.json();
    const users = await usersRes.json();

    renderPipelineStrip(state.stage);
    renderVideoMeta(state);
    renderLbTable(state);
    renderLog(state);
    renderUsers(state, users);
  } catch (err) {
    // transient network hiccup during a page reload -- ignore, next poll retries
  }
}

function setupControls() {
  const newBandwidth = el("new-bandwidth");
  const newBandwidthValue = el("new-bandwidth-value");
  newBandwidth.addEventListener("input", () => {
    newBandwidthValue.textContent = `${newBandwidth.value} kbps`;
  });

  el("add-user-btn").addEventListener("click", () => {
    fetch("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bandwidth: Number(newBandwidth.value) }),
    })
      .then(poll)
      .catch(() => {});
  });

  const qp = el("qp");
  const qpValue = el("qp-value");
  qp.addEventListener("input", () => {
    qpValue.textContent = qp.value;
  });

  el("run-btn").addEventListener("click", () => {
    fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ qp: Number(qp.value) }),
    }).catch(() => {});
  });

  el("upload-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("video", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      currentVideoId = data.video_id;
      poll();
    } catch (err) {
      alert("upload failed");
    }
  });
}

setupControls();
poll();
setInterval(poll, 250);
