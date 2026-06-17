/**
 * app.js – Employee Face Recognition Dashboard
 * ─────────────────────────────────────────────
 * Connects to the FastAPI backend via:
 *   • WebSocket /ws/events   – real-time detection events
 *   • REST GET /api/stats    – polled every 10 s
 *   • REST GET /api/cameras  – polled to update camera tiles
 *   • MJPEG  /video/{id}     – live camera streams
 */

(function () {
  "use strict";

  // ── Config ────────────────────────────────────────────────
  const API_BASE = "";              // same origin
  const WS_EVENTS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/events`;
  const STATS_POLL_MS  = 10_000;
  const MAX_FEED_ITEMS = 80;

  // ── Auth state ────────────────────────────────────────────
  let _authToken = sessionStorage.getItem("frs_token") || "";

  async function authFetch(url, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (_authToken) headers["Authorization"] = `Bearer ${_authToken}`;
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401) {
      _authToken = "";
      sessionStorage.removeItem("frs_token");
      showLoginOverlay();
    }
    return res;
  }

  // Patch window.location-based CSV downloads to include token via query param
  function downloadWithAuth(path) {
    const sep = path.includes("?") ? "&" : "?";
    window.location.href = _authToken ? `${path}${sep}_token=${encodeURIComponent(_authToken)}` : path;
  }

  // ── Login overlay ─────────────────────────────────────────
  const $loginOverlay  = document.getElementById("login-overlay");
  const $loginForm     = document.getElementById("login-form");
  const $loginPassword = document.getElementById("login-password");
  const $loginError    = document.getElementById("login-error");

  function showLoginOverlay() { $loginOverlay.classList.remove("hidden"); }
  function hideLoginOverlay() { $loginOverlay.classList.add("hidden"); }

  $loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const password = $loginPassword.value;
    try {
      const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const data = await res.json();
      if (res.ok) {
        _authToken = data.token || "";
        sessionStorage.setItem("frs_token", _authToken);
        hideLoginOverlay();
        bootApp();
      } else {
        $loginError.classList.remove("hidden");
        $loginError.textContent = "✗ " + (data.detail || "Incorrect password.");
      }
    } catch {
      $loginError.classList.remove("hidden");
      $loginError.textContent = "✗ Network error – please try again.";
    }
  });

  async function checkAuthAndBoot() {
    try {
      const cfgRes = await fetch(`${API_BASE}/api/auth/config`);
      const cfg = cfgRes.ok ? await cfgRes.json() : { require_auth: false };
      if (!cfg.require_auth) {
        bootApp();
        return;
      }
      if (_authToken) {
        const chk = await authFetch(`${API_BASE}/api/auth/check`);
        if (chk.ok) { bootApp(); return; }
      }
      showLoginOverlay();
    } catch {
      bootApp();   // if API is unreachable, show the UI anyway
    }
  }

  // ── State ─────────────────────────────────────────────────
  let ws = null;
  let wsReconnectTimer = null;
  let filterUnknown = false;
  let knownCameraIds = new Set();

  // ── DOM refs ───────────────────────────────────────────────
  const $wsIndicator  = document.getElementById("ws-indicator");
  const $clock        = document.getElementById("clock");
  const $cameraGrid   = document.getElementById("camera-grid");
  const $eventList    = document.getElementById("event-list");
  const $activeIds    = document.getElementById("active-ids");
  const $filterCb     = document.getElementById("filter-unknown");
  const $toast        = document.getElementById("toast");
  const $toastCam     = document.getElementById("toast-cam");
  const $statEmp      = document.getElementById("stat-employees");
  const $statTotal    = document.getElementById("stat-total");
  const $statKnown    = document.getElementById("stat-known");
  const $statUnknown  = document.getElementById("stat-unknown");
  const $openModal       = document.getElementById("open-modal");
  const $openModalEmp    = document.getElementById("open-modal-emp");
  const $closeModal      = document.getElementById("close-modal");
  const $modalOverlay    = document.getElementById("modal-overlay");
  const $regForm         = document.getElementById("register-form");
  const $formFeedback    = document.getElementById("form-feedback");
  const $empList         = document.getElementById("emp-list");
  const $empDetailOverlay = document.getElementById("emp-detail-overlay");
  const $empDetailContent = document.getElementById("emp-detail-content");
  const $closeEmpDetail   = document.getElementById("close-emp-detail");

  // ── Clock ─────────────────────────────────────────────────
  function updateClock() {
    const now = new Date();
    $clock.textContent = now.toLocaleTimeString([], { hour12: false });
  }
  setInterval(updateClock, 1000);
  updateClock();

  // ── WebSocket connection ──────────────────────────────────
  function connectWS() {
    if (ws) {
      ws.close();
    }
    ws = new WebSocket(WS_EVENTS_URL);

    ws.onopen = () => {
      setIndicator("connected");
      clearTimeout(wsReconnectTimer);
    };

    ws.onmessage = (evt) => {
      try {
        const event = JSON.parse(evt.data);
        handleEvent(event);
      } catch (e) {
        console.warn("WS parse error:", e);
      }
    };

    ws.onclose = () => {
      setIndicator("disconnected");
      wsReconnectTimer = setTimeout(connectWS, 3000);
    };

    ws.onerror = () => {
      setIndicator("disconnected");
    };
  }

  function setIndicator(state) {
    $wsIndicator.className = "badge";
    if (state === "connected") {
      $wsIndicator.classList.add("badge-success");
      $wsIndicator.textContent = "● Live";
    } else {
      $wsIndicator.classList.add("badge-warning");
      $wsIndicator.textContent = "Reconnecting…";
    }
  }

  // Keep alive ping every 20 s
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send("ping");
    }
  }, 20_000);

  // ── Handle incoming detection event ───────────────────────
  function handleEvent(event) {
    addFeedItem(event);
    updateActiveIds(event);

    if (!event.is_known) {
      showUnknownToast(event.camera_id);
    }
  }

  // ── Feed list ──────────────────────────────────────────────
  function addFeedItem(event) {
    const ts = new Date(event.timestamp);
    const timeStr = ts.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
    const confidencePct = Math.round((event.confidence || 0) * 100);
    const isUnknown = !event.is_known;

    const li = document.createElement("li");
    li.className = `event-item ${isUnknown ? "unknown" : "known"}`;
    if (filterUnknown && !isUnknown) li.classList.add("hidden");

    li.innerHTML = `
      <div class="event-name">${escHtml(event.employee_name || "Unknown")}</div>
      <div class="event-meta">
        <span>${escHtml(event.camera_id || "")}</span>
        <span class="event-confidence">${confidencePct}%</span>
        <span>${timeStr}</span>
      </div>
    `;

    $eventList.insertBefore(li, $eventList.firstChild);

    // Prune old items
    while ($eventList.children.length > MAX_FEED_ITEMS) {
      $eventList.removeChild($eventList.lastChild);
    }
  }

  // ── Active IDs bar ─────────────────────────────────────────
  const activeMap = new Map();   // employee_id → { chip, timer }

  function updateActiveIds(event) {
    const key = event.is_known ? event.employee_id : `unknown-${event.camera_id}`;
    const label = event.is_known
      ? `${escHtml(event.employee_name)}  (${Math.round(event.confidence * 100)}%)`
      : "⚠ Unknown";

    // Remove existing chip for this key
    const existing = activeMap.get(key);
    if (existing) {
      clearTimeout(existing.timer);
      existing.chip.remove();
    }

    const chip = document.createElement("div");
    chip.className = `id-chip ${event.is_known ? "known" : "unknown"}`;
    chip.innerHTML = label;
    $activeIds.appendChild(chip);

    // Auto-remove after 5 s
    const timer = setTimeout(() => {
      chip.remove();
      activeMap.delete(key);
    }, 5000);

    activeMap.set(key, { chip, timer });
  }

  // ── Unknown person toast ───────────────────────────────────
  let toastTimer = null;

  function showUnknownToast(cameraId) {
    $toastCam.textContent = ` – Camera: ${cameraId}`;
    $toast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $toast.classList.add("hidden"), 4000);
  }

  // ── Camera grid ────────────────────────────────────────────
  function refreshCameras(cameraStats) {
    Object.entries(cameraStats).forEach(([camId, stats]) => {
      if (!knownCameraIds.has(camId)) {
        addCameraCard(camId, stats);
        knownCameraIds.add(camId);
      } else {
        const fpsEl = document.getElementById(`cam-fps-${camId}`);
        if (fpsEl) fpsEl.textContent = `${stats.fps} fps`;
        const resEl = document.getElementById(`cam-res-${camId}`);
        if (resEl) resEl.textContent = resBadge(stats);
        // Keep latest stats on the settings button so modal sees fresh values
        const btn = document.querySelector(`#cam-tile-${camId} .cam-settings-btn`);
        if (btn) btn._stats = stats;
      }
    });
  }

  function resBadge(stats) {
    if (stats.resize_width && stats.resize_height)
      return `${stats.resize_width}×${stats.resize_height}`;
    return "";
  }

  function addCameraCard(camId, stats) {
    const tile = document.createElement("div");
    tile.className = "camera-tile";
    tile.id = `cam-tile-${camId}`;
    const res = resBadge(stats);
    tile.innerHTML = `
      <div class="camera-tile-header">
        <span class="camera-name">${escHtml(camId)}</span>
        <span class="camera-fps" id="cam-fps-${camId}">${stats.fps} fps</span>
        ${res ? `<span class="camera-res" id="cam-res-${escHtml(camId)}">${escHtml(res)}</span>` : `<span class="camera-res" id="cam-res-${escHtml(camId)}"></span>`}
        <button class="cam-settings-btn" data-id="${escHtml(camId)}" title="Camera settings">⚙</button>
        <button class="cam-remove-btn" data-id="${escHtml(camId)}" title="Remove camera">✕</button>
      </div>
      <img
        class="camera-feed"
        src="${API_BASE}/video/${encodeURIComponent(camId)}"
        alt="Live feed – ${escHtml(camId)}"
        onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22640%22 height=%22360%22%3E%3Crect width=%22100%25%22 height=%22100%25%22 fill=%22%231a1d27%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22%238899aa%22 font-size=%2218%22%3ENo signal%3C/text%3E%3C/svg%3E'"
      />
    `;
    tile.querySelector(".cam-remove-btn").addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = e.currentTarget.dataset.id;
      if (!confirm(`Remove camera "${id}" from the system?`)) return;
      const res = await authFetch(`${API_BASE}/api/cameras/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (res.ok) {
        document.getElementById(`cam-tile-${id}`)?.remove();
        knownCameraIds.delete(id);
      } else {
        const j = await res.json().catch(() => ({}));
        alert(`Failed to remove: ${j.detail || "Unknown error"}`);
      }
    });
    tile.querySelector(".cam-settings-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      openCamSettings(e.currentTarget.dataset.id, stats);
    });
    $cameraGrid.appendChild(tile);
  }

  // ── Add Camera modal ───────────────────────────────────────
  const $addCameraOverlay  = document.getElementById("add-camera-overlay");
  const $closeAddCamera    = document.getElementById("close-add-camera");
  const $addCameraForm     = document.getElementById("add-camera-form");
  const $addCameraFeedback = document.getElementById("add-camera-feedback");
  const $openAddCameraBtn  = document.getElementById("open-add-camera-btn");

  function openAddCameraModal()  { if ($addCameraOverlay) $addCameraOverlay.classList.remove("hidden"); }
  function closeAddCameraModal() {
    if ($addCameraOverlay) $addCameraOverlay.classList.add("hidden");
    if ($addCameraForm)    $addCameraForm.reset();
    if ($addCameraFeedback) $addCameraFeedback.className = "form-feedback hidden";
  }

  if ($openAddCameraBtn) $openAddCameraBtn.addEventListener("click", openAddCameraModal);
  if ($closeAddCamera)   $closeAddCamera.addEventListener("click",   closeAddCameraModal);
  if ($addCameraOverlay) {
    $addCameraOverlay.addEventListener("click", (e) => {
      if (e.target === $addCameraOverlay) closeAddCameraModal();
    });
  }

  if ($addCameraForm) {
    $addCameraForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const fd = new FormData($addCameraForm);
      const payload = {
        id:     (fd.get("id") || "").trim(),
        name:   (fd.get("name") || "").trim() || (fd.get("id") || "").trim(),
        source: (fd.get("source") || "").trim(),
        fps:    parseInt(fd.get("fps") || "25", 10),
      };
      if ($addCameraFeedback) $addCameraFeedback.className = "form-feedback hidden";
      try {
        const res = await authFetch(`${API_BASE}/api/cameras`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const json = await res.json();
        if (res.ok) {
          if ($addCameraFeedback) {
            $addCameraFeedback.className = "form-feedback success";
            $addCameraFeedback.textContent = `✓ ${json.message}`;
          }
          fetchStatsThrottled();
          setTimeout(closeAddCameraModal, 1200);
        } else {
          if ($addCameraFeedback) {
            $addCameraFeedback.className = "form-feedback error";
            $addCameraFeedback.textContent = `✗ ${json.detail || "Failed to add camera."}`;
          }
        }
      } catch {
        if ($addCameraFeedback) {
          $addCameraFeedback.className = "form-feedback error";
          $addCameraFeedback.textContent = "✗ Network error — please try again.";
        }
      }
    });
  }

  // But we also need the remove button in the tile header — add it to innerHTML:
  // (handled above via tile.querySelector inside addCameraCard)

  // ── Stats polling (debounced for performance) ──────────────
  let _statsTimeout = null;
  function fetchStatsThrottled() {
    if (_statsTimeout) return;
    _statsTimeout = setTimeout(async () => {
      _statsTimeout = null;
      try {
        const [statsRes, camRes] = await Promise.all([
          authFetch(`${API_BASE}/api/stats`),
          authFetch(`${API_BASE}/api/cameras`),
        ]);

        if (statsRes.ok) {
          const data = await statsRes.json();
          const db = data.database || {};
          // Batch DOM updates to reduce reflow
          requestAnimationFrame(() => {
            if ($statEmp && ($statEmp.textContent !== String(db.registered_employees ?? "–")))
              $statEmp.textContent = db.registered_employees ?? "–";
            if ($statTotal && ($statTotal.textContent !== String(db.total_detections ?? "–")))
              $statTotal.textContent = db.total_detections ?? "–";
            if ($statKnown && ($statKnown.textContent !== String(db.known_detections ?? "–")))
              $statKnown.textContent = db.known_detections ?? "–";
            if ($statUnknown && ($statUnknown.textContent !== String(db.unknown_detections ?? "–")))
              $statUnknown.textContent = db.unknown_detections ?? "–";
          });
        }

        if (camRes.ok) {
          const cams = await camRes.json();
          requestAnimationFrame(() => refreshCameras(cams));
        }
      } catch (e) {
        console.warn("Stats fetch error:", e);
      }
    }, 500);  // Debounce rapid updates
  }

  setInterval(fetchStatsThrottled, STATS_POLL_MS);

  // ── Filter toggle ──────────────────────────────────────────
  $filterCb.addEventListener("change", () => {
    filterUnknown = $filterCb.checked;
    document.querySelectorAll(".event-item").forEach(item => {
      if (filterUnknown) {
        item.classList.toggle("hidden", !item.classList.contains("unknown"));
      } else {
        item.classList.remove("hidden");
      }
    });
  });

  // ── Webcam capture state ──────────────────────────────────
  const CAPTURE_POSES = [
    { label: "Look Straight", arrow: "⬤" },
    { label: "Look Left",     arrow: "←" },
    { label: "Look Right",    arrow: "→" },
    { label: "Look Up",       arrow: "↑" },
    { label: "Look Down",     arrow: "↓" },
  ];
  let _webcamStream  = null;
  let _captureStep   = 0;
  let _capturedBlobs = new Array(CAPTURE_POSES.length).fill(null);

  const $webcamWrap      = document.getElementById("webcam-wrap");
  const $regWebcam       = document.getElementById("reg-webcam");
  const $regCanvas       = document.getElementById("reg-canvas");
  const $webcamDirection = document.getElementById("webcam-direction");
  const $stepDots        = document.getElementById("step-dots");
  const $captureInstr    = document.getElementById("capture-instruction");
  const $captureThumbs   = document.getElementById("capture-thumbs");
  const $startWebcamBtn  = document.getElementById("start-webcam-btn");
  const $captureFrameBtn = document.getElementById("capture-frame-btn");
  const $retakeStepBtn   = document.getElementById("retake-step-btn");
  const $regSubmitBtn    = document.getElementById("reg-submit-btn");

  // ── Modal (register) ──────────────────────────────────────
  function openRegisterModal() { $modalOverlay.classList.remove("hidden"); }
  function closeRegisterModal() {
    $modalOverlay.classList.add("hidden");
    _stopWebcam();
    _resetCaptureState();
  }

  $openModal.addEventListener("click", openRegisterModal);
  if ($openModalEmp) $openModalEmp.addEventListener("click", openRegisterModal);
  $closeModal.addEventListener("click", closeRegisterModal);
  $modalOverlay.addEventListener("click", (e) => {
    if (e.target === $modalOverlay) closeRegisterModal();
  });

  $regForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const captured = _capturedBlobs.filter(Boolean);

    if (captured.length < 1) {
      showFeedback("error", "✗ Please capture at least 1 face photo using the camera.");
      return;
    }

    const formData = new FormData();
    formData.append("employee_id", ($regForm.querySelector('input[name="employee_id"]') || {}).value || "");
    formData.append("name", ($regForm.querySelector('input[name="name"]') || {}).value || "");
    formData.append("department", ($regForm.querySelector('input[name="department"]') || {}).value || "");
    captured.forEach((blob, idx) => formData.append("photos", blob, `face_${idx + 1}.jpg`));
    showFeedback("", "");

    try {
      const res = await authFetch(`${API_BASE}/api/employees`, {
        method: "POST",
        body: formData,
      });
      const json = await res.json();

      if (res.ok) {
        showFeedback("success", `✓ ${json.message}`);
        _stopWebcam();
        $regForm.reset();
        fetchStatsThrottled();
        fetchEmployees();
        showFeedback("error", `✗ ${json.detail || "Registration failed."}`);
      }
    } catch (err) {
      showFeedback("error", "Network error – please try again.");
    }
  });

  function showFeedback(type, message) {
    $formFeedback.className = "form-feedback";
    if (!type) {
      $formFeedback.classList.add("hidden");
      return;
    }
    $formFeedback.classList.add(type);
    $formFeedback.textContent = message;
  }

  // ── Webcam capture functions ──────────────────────────────
  function _initCaptureDots() {
    if (!$stepDots || !$captureThumbs) return;
    $stepDots.innerHTML = "";
    $captureThumbs.innerHTML = "";
    CAPTURE_POSES.forEach((_, i) => {
      const dot = document.createElement("span");
      dot.className = "step-dot" + (i === 0 ? " active" : "");
      dot.id = `step-dot-${i}`;
      $stepDots.appendChild(dot);
      const empty = document.createElement("div");
      empty.className = "capture-thumb-empty";
      empty.id = `thumb-${i}`;
      empty.textContent = String(i + 1);
      $captureThumbs.appendChild(empty);
    });
  }

  function _showCaptureStep(idx) {
    const pose = CAPTURE_POSES[idx];
    if ($captureInstr)    $captureInstr.textContent = `Step ${idx + 1} / ${CAPTURE_POSES.length} — ${pose.label}`;
    if ($webcamDirection) $webcamDirection.textContent = `${pose.arrow}  ${pose.label}`;
    CAPTURE_POSES.forEach((_, i) => {
      const dot = document.getElementById(`step-dot-${i}`);
      if (!dot) return;
      dot.className = "step-dot" + (_capturedBlobs[i] ? " done" : i === idx ? " active" : "");
    });
    if ($captureFrameBtn) $captureFrameBtn.classList.remove("hidden");
    if ($retakeStepBtn) {
      _capturedBlobs[idx]
        ? $retakeStepBtn.classList.remove("hidden")
        : $retakeStepBtn.classList.add("hidden");
    }
  }

  async function _startWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if ($captureInstr) $captureInstr.textContent = "Camera API not available in this browser.";
      return;
    }
    try {
      _webcamStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      const track = _webcamStream.getVideoTracks()[0];
      try {
        await track.applyConstraints({ width: 320, height: 240, frameRate: 30 });
      } catch (_) { /* camera may not support exact values — continue anyway */ }
      if ($regWebcam) {
        $regWebcam.srcObject = _webcamStream;
        await $regWebcam.play().catch(() => {});
      }
      if ($webcamWrap) $webcamWrap.classList.remove("hidden");
      if ($startWebcamBtn) $startWebcamBtn.classList.add("hidden");
      _captureStep = 0;
      _capturedBlobs = new Array(CAPTURE_POSES.length).fill(null);
      _initCaptureDots();
      _showCaptureStep(0);
    } catch (_err) {
      if ($captureInstr) $captureInstr.textContent = "Camera access denied — allow camera permission and try again.";
    }
  }

  function _stopWebcam() {
    if (_webcamStream) {
      _webcamStream.getTracks().forEach(t => t.stop());
      _webcamStream = null;
    }
    if ($regWebcam) $regWebcam.srcObject = null;
  }

  function _captureFrame() {
    if (!$regWebcam || !$regCanvas) return;
    $regCanvas.width  = $regWebcam.videoWidth  || 640;
    $regCanvas.height = $regWebcam.videoHeight || 480;
    const ctx = $regCanvas.getContext("2d");
    ctx.drawImage($regWebcam, 0, 0);
    $regCanvas.toBlob((blob) => {
      if (!blob) return;
      _capturedBlobs[_captureStep] = blob;

      const prev = document.getElementById(`thumb-${_captureStep}`);
      if (prev) {
        if (prev.tagName === "IMG") URL.revokeObjectURL(prev.src);
        const img = document.createElement("img");
        img.className = "capture-thumb";
        img.id = `thumb-${_captureStep}`;
        img.src = URL.createObjectURL(blob);
        prev.replaceWith(img);
      }
      const dot = document.getElementById(`step-dot-${_captureStep}`);
      if (dot) dot.className = "step-dot done";

      const next = _capturedBlobs.findIndex((b, i) => b === null);
      if (next !== -1) {
        _captureStep = next;
        _showCaptureStep(_captureStep);
      } else {
        if ($captureInstr)    $captureInstr.textContent = "All 5 poses captured — ready to register!";
        if ($webcamDirection) $webcamDirection.textContent = "✓ All done";
        if ($captureFrameBtn) $captureFrameBtn.classList.add("hidden");
        if ($retakeStepBtn)   $retakeStepBtn.classList.remove("hidden");
      }
      if ($regSubmitBtn) $regSubmitBtn.disabled = _capturedBlobs.filter(Boolean).length < 1;
    }, "image/jpeg", 0.92);
  }

  function _retakeStep() {
    const prev = document.getElementById(`thumb-${_captureStep}`);
    if (prev && prev.tagName === "IMG") {
      URL.revokeObjectURL(prev.src);
      const empty = document.createElement("div");
      empty.className = "capture-thumb-empty";
      empty.id = `thumb-${_captureStep}`;
      empty.textContent = String(_captureStep + 1);
      prev.replaceWith(empty);
    }
    _capturedBlobs[_captureStep] = null;
    _showCaptureStep(_captureStep);
    if ($regSubmitBtn) $regSubmitBtn.disabled = _capturedBlobs.filter(Boolean).length < 1;
  }

  function _resetCaptureState() {
    _capturedBlobs.forEach((blob, i) => {
      if (!blob) return;
      const el = document.getElementById(`thumb-${i}`);
      if (el && el.tagName === "IMG") URL.revokeObjectURL(el.src);
    });
    _captureStep   = 0;
    _capturedBlobs = new Array(CAPTURE_POSES.length).fill(null);
    if ($webcamWrap)      $webcamWrap.classList.add("hidden");
    if ($startWebcamBtn)  $startWebcamBtn.classList.remove("hidden");
    if ($captureFrameBtn) $captureFrameBtn.classList.add("hidden");
    if ($retakeStepBtn)   $retakeStepBtn.classList.add("hidden");
    if ($captureInstr)    $captureInstr.textContent = 'Press "Start Camera" to begin';
    if ($stepDots)        $stepDots.innerHTML = "";
    if ($captureThumbs)   $captureThumbs.innerHTML = "";
    if ($regSubmitBtn)    $regSubmitBtn.disabled = true;
  }

  if ($startWebcamBtn)  $startWebcamBtn.addEventListener("click",  _startWebcam);
  if ($captureFrameBtn) $captureFrameBtn.addEventListener("click", _captureFrame);
  if ($retakeStepBtn)   $retakeStepBtn.addEventListener("click",   _retakeStep);

  if ($stepDots) {
    $stepDots.addEventListener("click", (e) => {
      const dot = e.target.closest(".step-dot");
      if (!dot || !_webcamStream) return;
      const idx = parseInt(dot.id.replace("step-dot-", ""), 10);
      if (!isNaN(idx)) { _captureStep = idx; _showCaptureStep(idx); }
    });
  }

  // ── Employee detail modal ──────────────────────────────────
  $closeEmpDetail.addEventListener("click", () => $empDetailOverlay.classList.add("hidden"));
  $empDetailOverlay.addEventListener("click", (e) => {
    if (e.target === $empDetailOverlay) $empDetailOverlay.classList.add("hidden");
  });

  function showEmpDetail(emp) {
    const initials = getInitials(emp.name);
    const avatarColor = stringToColor(emp.employee_id);
    const registered = emp.registered_at
      ? new Date(emp.registered_at).toLocaleDateString()
      : "–";
    $empDetailContent.innerHTML = `
      <div class="emp-detail">
        <div class="emp-detail-header">
          <div class="emp-detail-avatar" style="background:${avatarColor}">${escHtml(initials)}</div>
          <div>
            <div class="emp-detail-title">${escHtml(emp.name)}</div>
            <div class="emp-detail-subtitle">${escHtml(emp.department || "No department")}</div>
          </div>
        </div>
        <div class="emp-detail-grid">
          <div class="emp-detail-field">
            <div class="emp-detail-field-label">Employee ID</div>
            <div class="emp-detail-field-value">${escHtml(emp.employee_id)}</div>
          </div>
          <div class="emp-detail-field">
            <div class="emp-detail-field-label">Department</div>
            <div class="emp-detail-field-value">${escHtml(emp.department || "–")}</div>
          </div>
          <div class="emp-detail-field">
            <div class="emp-detail-field-label">Full Name</div>
            <div class="emp-detail-field-value">${escHtml(emp.name)}</div>
          </div>
          <div class="emp-detail-field">
            <div class="emp-detail-field-label">Registered</div>
            <div class="emp-detail-field-value">${escHtml(registered)}</div>
          </div>
        </div>
        <div class="emp-detail-actions">
          <button class="btn" id="detail-edit-btn"
            data-id="${escHtml(emp.employee_id)}"
            data-name="${escHtml(emp.name)}"
            data-dept="${escHtml(emp.department || "")}"
            style="background:var(--primary);color:#fff">Edit</button>
          <button class="btn btn-danger" id="detail-delete-btn" data-id="${escHtml(emp.employee_id)}">Remove</button>
        </div>
      </div>
    `;
    $empDetailOverlay.classList.remove("hidden");
    document.getElementById("detail-edit-btn").addEventListener("click", (e) => {
      const btn = e.currentTarget;
      $empDetailOverlay.classList.add("hidden");
      openEditModal(btn.dataset.id, btn.dataset.name, btn.dataset.dept);
    });
    document.getElementById("detail-delete-btn").addEventListener("click", async (e) => {
      const id = e.currentTarget.dataset.id;
      if (!confirm(`Remove ${emp.name} from the system?`)) return;
      await deleteEmployee(id);
      $empDetailOverlay.classList.add("hidden");
    });
  }

  // ── Employee Edit modal ────────────────────────────────────
  const $empEditOverlay   = document.getElementById("emp-edit-overlay");
  const $closeEmpEdit     = document.getElementById("close-emp-edit");
  const $cancelEmpEdit    = document.getElementById("cancel-emp-edit");
  const $empEditForm      = document.getElementById("emp-edit-form");
  const $editEmpId        = document.getElementById("edit-emp-id");
  const $editEmpName      = document.getElementById("edit-emp-name");
  const $editEmpDept      = document.getElementById("edit-emp-dept");
  const $editFormFeedback = document.getElementById("edit-form-feedback");

  function openEditModal(id, name, dept) {
    $editEmpId.value   = id;
    $editEmpName.value = name;
    $editEmpDept.value = dept;
    $editFormFeedback.className = "form-feedback hidden";
    $empEditOverlay.classList.remove("hidden");
  }

  [$closeEmpEdit, $cancelEmpEdit].forEach(btn =>
    btn.addEventListener("click", () => $empEditOverlay.classList.add("hidden"))
  );
  $empEditOverlay.addEventListener("click", (e) => {
    if (e.target === $empEditOverlay) $empEditOverlay.classList.add("hidden");
  });

  $empEditForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id   = $editEmpId.value;
    const name = $editEmpName.value.trim();
    const dept = $editEmpDept.value.trim();
    const photoInput = $empEditForm.querySelector('input[name="photos"]');
    const files = photoInput ? Array.from(photoInput.files || []) : [];

    if (!name && !dept && files.length === 0) {
      showEditFeedback("error", "✗ Enter at least one field to update.");
      return;
    }
    if (files.length > 5) {
      showEditFeedback("error", "✗ Maximum 5 photos allowed.");
      return;
    }

    const formData = new FormData();
    if (name) formData.append("name", name);
    if (dept) formData.append("department", dept);
    files.forEach(f => formData.append("photos", f));

    try {
      const res = await authFetch(`${API_BASE}/api/employees/${encodeURIComponent(id)}`, {
        method: "PUT",
        body: formData,
      });
      const json = await res.json();
      if (res.ok) {
        showEditFeedback("success", `✓ ${json.message}`);
        fetchEmployees();
        setTimeout(() => $empEditOverlay.classList.add("hidden"), 1200);
      } else {
        showEditFeedback("error", `✗ ${json.detail || "Update failed."}`);
      }
    } catch {
      showEditFeedback("error", "Network error – please try again.");
    }
  });

  function showEditFeedback(type, message) {
    $editFormFeedback.className = `form-feedback ${type}`;
    $editFormFeedback.textContent = message;
  }

  // ── Tab navigation ─────────────────────────────────────────
  const tabBtns   = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.add("hidden"));
      btn.classList.add("active");
      document.getElementById(`tab-${target}`).classList.remove("hidden");

      if (target === "attendance") loadAttendance();
      if (target === "monthly")    loadMonthly();
      if (target === "settings" && !_settingsLoaded) loadSettings();
    });
  });

  // ── Attendance tab ─────────────────────────────────────────
  const $attDate          = document.getElementById("att-date");
  const $attRefresh       = document.getElementById("att-refresh");
  const $attDownload      = document.getElementById("att-download");
  const $attDownloadXl    = document.getElementById("att-download-xl");
  const $attTbody         = document.getElementById("att-tbody");
  const $attEmpty         = document.getElementById("att-empty");
  const $attPresentCount  = document.getElementById("att-present-count");
  const $attAbsentCount   = document.getElementById("att-absent-count");
  const $absentChips      = document.getElementById("absent-chips");
  const $absentEmpty      = document.getElementById("absent-empty");

  // Default date to today
  (function () {
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm   = String(today.getMonth() + 1).padStart(2, "0");
    const dd   = String(today.getDate()).padStart(2, "0");
    $attDate.value = `${yyyy}-${mm}-${dd}`;
  })();

  $attRefresh.addEventListener("click", loadAttendance);
  $attDate.addEventListener("change", loadAttendance);
  $attDownload.addEventListener("click", () => {
    downloadWithAuth(`${API_BASE}/api/attendance/export/${$attDate.value}`);
  });
  $attDownloadXl.addEventListener("click", () => {
    downloadWithAuth(`${API_BASE}/api/attendance/export/${$attDate.value}/excel`);
  });

  async function loadAttendance() {
    const d = $attDate.value;
    if (!d) return;
    try {
      const [presRes, absRes] = await Promise.all([
        authFetch(`${API_BASE}/api/attendance/${d}`),
        authFetch(`${API_BASE}/api/attendance/absent/${d}`),
      ]);
      const presData = presRes.ok ? await presRes.json() : { records: [] };
      const absData  = absRes.ok  ? await absRes.json()  : { absent: [] };

      renderAttendanceTable(presData.records || []);
      renderAbsentChips(absData.absent || []);
      $attPresentCount.textContent = `${(presData.records || []).length} present`;
      $attAbsentCount.textContent  = `${(absData.absent  || []).length} absent`;
    } catch (e) {
      console.warn("Attendance fetch error:", e);
    }
  }

  function renderAttendanceTable(records) {
    $attTbody.innerHTML = "";
    if (records.length === 0) {
      $attEmpty.classList.remove("hidden");
      return;
    }
    $attEmpty.classList.add("hidden");
    records.forEach(r => {
      const first = fmtTime(r.first_seen);
      const last  = fmtTime(r.last_seen);
      const conf  = `${Math.round((r.confidence || 0) * 100)}%`;
      const wm    = r.work_duration_minutes;
      const wh    = wm != null ? `${Math.floor(wm/60)}h ${String(wm%60).padStart(2,"0")}m` : "–";
      const late  = r.is_late;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escHtml(r.employee_name || "–")}</td>
        <td>${escHtml(r.department   || "–")}</td>
        <td>${escHtml(first)}</td>
        <td>${escHtml(last)}</td>
        <td>${escHtml(wh)}</td>
        <td><span class="status-badge ${late ? "late" : "ontime"}">${late ? "Late" : "On Time"}</span></td>
        <td>${escHtml(conf)}</td>
      `;
      $attTbody.appendChild(tr);
    });
  }

  function renderAbsentChips(absent) {
    $absentChips.innerHTML = "";
    if (absent.length === 0) {
      $absentEmpty.classList.remove("hidden");
      return;
    }
    $absentEmpty.classList.add("hidden");
    absent.forEach(emp => {
      const chip = document.createElement("div");
      chip.className = "absent-chip";
      const initials = getInitials(emp.name);
      const color = stringToColor(emp.employee_id);
      chip.innerHTML = `
        <div class="absent-avatar" style="background:${color}">${escHtml(initials)}</div>
        <div>
          <div class="absent-name">${escHtml(emp.name)}</div>
          <div class="absent-dept">${escHtml(emp.department || emp.employee_id)}</div>
        </div>
      `;
      $absentChips.appendChild(chip);
    });
  }

  // ── Monthly Report tab ─────────────────────────────────────
  const $monthlyMonth      = document.getElementById("monthly-month");
  const $monthlyYear       = document.getElementById("monthly-year");
  const $monthlyLoad       = document.getElementById("monthly-load");
  const $monthlyDownload   = document.getElementById("monthly-download");
  const $monthlyDownloadXl = document.getElementById("monthly-download-xl");
  const $monthlyTbody      = document.getElementById("monthly-tbody");
  const $monthlyEmpty    = document.getElementById("monthly-empty");

  (function initMonthlyPickers() {
    const now = new Date();
    const months = ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"];
    months.forEach((m, i) => {
      const opt = document.createElement("option");
      opt.value = i + 1;
      opt.textContent = m;
      if (i + 1 === now.getMonth() + 1) opt.selected = true;
      $monthlyMonth.appendChild(opt);
    });
    for (let y = now.getFullYear(); y >= now.getFullYear() - 3; y--) {
      const opt = document.createElement("option");
      opt.value = y;
      opt.textContent = y;
      $monthlyYear.appendChild(opt);
    }
  })();

  $monthlyLoad.addEventListener("click", loadMonthly);
  $monthlyDownload.addEventListener("click", () => {
    const y = $monthlyYear.value;
    const m = $monthlyMonth.value;
    downloadWithAuth(`${API_BASE}/api/attendance/export/monthly?year=${y}&month=${m}`);
  });
  $monthlyDownloadXl.addEventListener("click", () => {
    const y = $monthlyYear.value;
    const m = $monthlyMonth.value;
    downloadWithAuth(`${API_BASE}/api/attendance/export/monthly/excel?year=${y}&month=${m}`);
  });

  async function loadMonthly() {
    const year  = $monthlyYear.value;
    const month = $monthlyMonth.value;
    try {
      const res = await authFetch(`${API_BASE}/api/attendance/monthly?year=${year}&month=${month}`);
      if (!res.ok) return;
      const data = await res.json();
      renderMonthlyTable(data.summary || []);
    } catch (e) {
      console.warn("Monthly fetch error:", e);
    }
  }

  function renderMonthlyTable(summary) {
    $monthlyTbody.innerHTML = "";
    if (summary.length === 0) {
      $monthlyEmpty.classList.remove("hidden");
      return;
    }
    $monthlyEmpty.classList.add("hidden");
    summary.forEach(r => {
      const rate = r.attendance_rate || 0;
      const rateClass = rate >= 80 ? "rate-good" : rate >= 50 ? "rate-mid" : "rate-low";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escHtml(r.name)}</td>
        <td>${escHtml(r.department || "–")}</td>
        <td class="num">${r.days_present}</td>
        <td class="num">${r.days_absent}</td>
        <td class="num">${r.late_days}</td>
        <td class="num">${r.total_days}</td>
        <td><span class="rate-badge ${rateClass}">${rate.toFixed(1)}%</span></td>
      `;
      $monthlyTbody.appendChild(tr);
    });
  }

  // ── Employee roster ────────────────────────────────────────
  async function fetchEmployees() {
    try {
      const res = await authFetch(`${API_BASE}/api/employees`);
      if (!res.ok) return;
      const data = await res.json();
      renderEmployees(data.employees || []);
    } catch (e) {
      console.warn("Employee fetch error:", e);
    }
  }

  function renderEmployees(employees) {
    if (!$empList) return;
    if (employees.length === 0) {
      $empList.innerHTML = '<p class="emp-empty">No employees registered.</p>';
      return;
    }
    $empList.innerHTML = "";
    employees.forEach(emp => {
      const initials    = getInitials(emp.name);
      const avatarColor = stringToColor(emp.employee_id);
      const photoSrc    = emp.has_embedding
        ? `${API_BASE}/api/employees/${encodeURIComponent(emp.employee_id)}/photo?_t=${Date.now()}`
        : "";
      const avatarHtml  = photoSrc
        ? `<img class="emp-avatar emp-avatar-photo" src="${photoSrc}"
               onerror="this.outerHTML='<div class=\\"emp-avatar\\" style=\\"background:${avatarColor}\\">${escHtml(initials)}</div>'"
               alt="${escHtml(emp.name)}" />`
        : `<div class="emp-avatar" style="background:${avatarColor}">${escHtml(initials)}</div>`;

      const card = document.createElement("div");
      card.className = "emp-card";
      card.innerHTML = `
        ${avatarHtml}
        <div class="emp-info">
          <div class="emp-name">${escHtml(emp.name)}</div>
          <div class="emp-meta">${escHtml(emp.employee_id)}${emp.department ? " · " + escHtml(emp.department) : ""}</div>
        </div>
        <button class="emp-del" data-id="${escHtml(emp.employee_id)}" title="Remove employee">✕</button>
      `;
      card.addEventListener("click", () => showEmpDetail(emp));
      card.querySelector(".emp-del").addEventListener("click", async (e) => {
        e.stopPropagation();
        const id = e.currentTarget.dataset.id;
        if (!confirm(`Remove ${emp.name}?`)) return;
        await deleteEmployee(id);
      });
      $empList.appendChild(card);
    });
  }

  async function deleteEmployee(id) {
    try {
      const res = await authFetch(`${API_BASE}/api/employees/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        fetchEmployees();
        fetchStats();
      } else {
        const json = await res.json();
        alert(`Failed to remove: ${json.detail || "Unknown error"}`);
      }
    } catch (e) {
      alert("Network error – could not remove employee.");
    }
  }

  // ── Helpers ────────────────────────────────────────────────
  function fmtTime(isoStr) {
    if (!isoStr) return "–";
    const t = isoStr.slice(11, 19);           // "HH:MM:SS"
    if (!t || t.length < 5) return "–";
    let [h, m] = t.split(":").map(Number);
    const ampm = h >= 12 ? "PM" : "AM";
    h = h % 12 || 12;
    return `${h}:${String(m).padStart(2, "0")} ${ampm}`;
  }

  function getInitials(name) {
    const parts = String(name).trim().split(/\s+/);
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }

  function stringToColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
    const h = Math.abs(hash) % 360;
    return `hsl(${h}, 55%, 42%)`;
  }

  // ── Security: HTML escape ──────────────────────────────────
  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#x27;");
  }

  // ── Settings tab ──────────────────────────────────────────
  let _settingsLoaded = false;

  function syncRange(rangeId, numId) {
    const range = document.getElementById(rangeId);
    const num   = document.getElementById(numId);
    if (!range || !num) return;
    range.addEventListener("input", () => { num.value = range.value; });
    num.addEventListener("input",   () => { range.value = num.value; });
  }

  syncRange("set-rec-threshold",  "set-rec-threshold-num");
  syncRange("set-det-thresh",     "set-det-thresh-num");
  syncRange("set-att-confidence", "set-att-confidence-num");

  async function loadSettings() {
    try {
      const res = await authFetch(`${API_BASE}/api/settings`);
      if (!res.ok) return;
      const s = await res.json();

      // Detection & Recognition
      const rec = s.recognition || {};
      const det = s.detection   || {};
      const trk = s.tracking    || {};
      setVal("set-rec-threshold",     rec.threshold          ?? 0.45);
      setVal("set-rec-threshold-num", rec.threshold          ?? 0.45);
      setVal("set-det-thresh",        det.det_thresh         ?? 0.45);
      setVal("set-det-thresh-num",    det.det_thresh         ?? 0.45);
      setVal("set-min-face",          det.min_face_size      ?? 30);
      setVal("set-track-cooldown",    trk.cooldown_seconds   ?? 30);
      setVal("set-track-dist",        trk.max_distance       ?? 80);

      // Alarm & Alerts
      const al  = s.alarm  || {};
      const alt = s.alerts || {};
      setChecked("set-alarm-enabled",  al.enabled          ?? false);
      setChecked("set-alert-unknown",  alt.unknown_person  ?? true);
      setVal("set-alarm-cooldown",     al.cooldown_seconds ?? 30);
      setVal("set-alarm-sound",        al.sound            ?? "voice");
      setVal("set-alarm-output",       al.output           ?? "local");
      setVal("set-alarm-voice-text",   al.voice_text       ?? "Intruder alert!");
      setVal("set-webhook-url",        alt.webhook_url     ?? "");

      // Camera defaults
      const perf = s.performance || {};
      // rtsp_transport lives on each camera; show first camera's value or default
      setVal("set-frame-skip",  perf.frame_skip  ?? 1);
      setVal("set-batch-size",  perf.batch_size  ?? 8);

      // System
      const log = s.logging    || {};
      const att = s.attendance || {};
      setVal("set-log-level",          log.level                  ?? "INFO");
      setChecked("set-log-unknown-frames", log.log_unknown_frames ?? false);
      setVal("set-att-confidence",     att.confidence_threshold   ?? 0.45);
      setVal("set-att-confidence-num", att.confidence_threshold   ?? 0.45);
      setVal("set-shift-start",        att.shift_start            ?? "");
      setVal("set-shift-end",          att.shift_end              ?? "");

      _settingsLoaded = true;
    } catch (e) {
      console.warn("Settings load error:", e);
    }
  }

  function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val ?? "";
  }
  function setChecked(id, val) {
    const el = document.getElementById(id);
    if (el) el.checked = Boolean(val);
  }

  function showSettingsFeedback(id, type, msg) {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = `settings-feedback ${type}`;
    el.textContent = msg;
    setTimeout(() => { el.className = "settings-feedback hidden"; }, 3500);
  }

  async function saveSettings(payload, feedbackId) {
    try {
      const res = await authFetch(`${API_BASE}/api/settings`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      if (res.ok) {
        const liveMsg = json.live_updated?.length
          ? ` (live: ${json.live_updated.join(", ")})`
          : "";
        const restartMsg = json.needs_restart?.length
          ? ` · restart required for: ${json.needs_restart.join(", ")}`
          : "";
        showSettingsFeedback(feedbackId, "success", `✓ Saved${liveMsg}${restartMsg}`);
      } else {
        showSettingsFeedback(feedbackId, "error", `✗ ${json.detail || "Save failed."}`);
      }
    } catch {
      showSettingsFeedback(feedbackId, "error", "✗ Network error — please try again.");
    }
  }

  // Recognition & Detection form
  const $sfRecognition = document.getElementById("settings-form-recognition");
  if ($sfRecognition) {
    $sfRecognition.addEventListener("submit", async (e) => {
      e.preventDefault();
      await saveSettings({
        recognition: { threshold:    parseFloat(document.getElementById("set-rec-threshold").value) },
        detection:   { det_thresh:   parseFloat(document.getElementById("set-det-thresh").value),
                       min_face_size: parseInt(document.getElementById("set-min-face").value, 10) },
        tracking:    { cooldown_seconds: parseFloat(document.getElementById("set-track-cooldown").value),
                       max_distance:     parseInt(document.getElementById("set-track-dist").value, 10) },
      }, "settings-feedback-recognition");
    });
  }

  // Alarm & Alerts form
  const $sfAlarm = document.getElementById("settings-form-alarm");
  if ($sfAlarm) {
    $sfAlarm.addEventListener("submit", async (e) => {
      e.preventDefault();
      await saveSettings({
        alarm: {
          enabled:          document.getElementById("set-alarm-enabled").checked,
          cooldown_seconds: parseFloat(document.getElementById("set-alarm-cooldown").value),
          sound:            document.getElementById("set-alarm-sound").value,
          output:           document.getElementById("set-alarm-output").value,
          voice_text:       document.getElementById("set-alarm-voice-text").value,
        },
        alerts: {
          unknown_person: document.getElementById("set-alert-unknown").checked,
          webhook_url:    document.getElementById("set-webhook-url").value.trim(),
        },
      }, "settings-feedback-alarm");
    });
  }

  // Camera defaults form
  const $sfCamera = document.getElementById("settings-form-camera");
  if ($sfCamera) {
    $sfCamera.addEventListener("submit", async (e) => {
      e.preventDefault();
      await saveSettings({
        performance: {
          frame_skip: parseInt(document.getElementById("set-frame-skip").value, 10),
          batch_size: parseInt(document.getElementById("set-batch-size").value, 10),
        },
      }, "settings-feedback-camera");
    });
  }

  // System form
  const $sfSystem = document.getElementById("settings-form-system");
  if ($sfSystem) {
    $sfSystem.addEventListener("submit", async (e) => {
      e.preventDefault();
      const shiftStart = document.getElementById("set-shift-start").value || null;
      const shiftEnd   = document.getElementById("set-shift-end").value   || null;
      await saveSettings({
        logging: {
          level:              document.getElementById("set-log-level").value,
          log_unknown_frames: document.getElementById("set-log-unknown-frames").checked,
        },
        attendance: {
          confidence_threshold: parseFloat(document.getElementById("set-att-confidence").value),
          shift_start:          shiftStart,
          shift_end:            shiftEnd,
        },
      }, "settings-feedback-system");
    });
  }

  // ── Camera Settings modal ──────────────────────────────────
  const PRESETS = {
    "480":  { w: 854,  h: 480  },
    "720":  { w: 1280, h: 720  },
    "1080": { w: 1920, h: 1080 },
  };

  const $camSettingsOverlay  = document.getElementById("cam-settings-overlay");
  const $closeCamSettings    = document.getElementById("close-cam-settings");
  const $cancelCamSettings   = document.getElementById("cancel-cam-settings");
  const $camSettingsForm     = document.getElementById("cam-settings-form");
  const $camSettingsId       = document.getElementById("cam-settings-id");
  const $camSettingsName     = document.getElementById("cam-settings-cam-name");
  const $camSettingsPreset   = document.getElementById("cam-settings-preset");
  const $camSettingsCustomRow = document.getElementById("cam-settings-custom-row");
  const $camSettingsW        = document.getElementById("cam-settings-w");
  const $camSettingsH        = document.getElementById("cam-settings-h");
  const $camSettingsFeedback = document.getElementById("cam-settings-feedback");

  function openCamSettings(camId, stats) {
    $camSettingsId.value = camId;
    $camSettingsName.textContent = `Camera: ${camId}`;

    // Pre-select the closest preset
    const w = stats.resize_width  || 0;
    const h = stats.resize_height || 0;
    let matched = "custom";
    for (const [key, p] of Object.entries(PRESETS)) {
      if (p.w === w && p.h === h) { matched = key; break; }
    }
    $camSettingsPreset.value = matched;
    $camSettingsW.value = w || 1280;
    $camSettingsH.value = h || 720;
    $camSettingsCustomRow.classList.toggle("hidden", matched !== "custom");
    $camSettingsFeedback.className = "form-feedback hidden";
    $camSettingsOverlay.classList.remove("hidden");
  }

  function closeCamSettings() {
    $camSettingsOverlay.classList.add("hidden");
    $camSettingsFeedback.className = "form-feedback hidden";
  }

  if ($closeCamSettings)  $closeCamSettings.addEventListener("click",  closeCamSettings);
  if ($cancelCamSettings) $cancelCamSettings.addEventListener("click", closeCamSettings);
  if ($camSettingsOverlay) {
    $camSettingsOverlay.addEventListener("click", (e) => {
      if (e.target === $camSettingsOverlay) closeCamSettings();
    });
  }

  if ($camSettingsPreset) {
    $camSettingsPreset.addEventListener("change", () => {
      const v = $camSettingsPreset.value;
      if (v === "custom") {
        $camSettingsCustomRow.classList.remove("hidden");
      } else {
        $camSettingsCustomRow.classList.add("hidden");
        const p = PRESETS[v];
        $camSettingsW.value = p.w;
        $camSettingsH.value = p.h;
      }
    });
  }

  if ($camSettingsForm) {
    $camSettingsForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const camId = $camSettingsId.value;
      const w = parseInt($camSettingsW.value, 10);
      const h = parseInt($camSettingsH.value, 10);

      if (!w || !h || w < 160 || h < 120) {
        $camSettingsFeedback.className = "form-feedback error";
        $camSettingsFeedback.textContent = "✗ Enter a valid resolution (min 160 × 120).";
        return;
      }

      $camSettingsFeedback.className = "form-feedback hidden";
      try {
        const res = await authFetch(`${API_BASE}/api/cameras/${encodeURIComponent(camId)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ width: w, height: h }),
        });
        const json = await res.json();
        if (res.ok) {
          $camSettingsFeedback.className = "form-feedback success";
          $camSettingsFeedback.textContent = `✓ Resolution set to ${w}×${h}`;
          const resEl = document.getElementById(`cam-res-${camId}`);
          if (resEl) resEl.textContent = `${w}×${h}`;
          setTimeout(closeCamSettings, 1200);
        } else {
          $camSettingsFeedback.className = "form-feedback error";
          $camSettingsFeedback.textContent = `✗ ${json.detail || "Failed to update."}`;
        }
      } catch {
        $camSettingsFeedback.className = "form-feedback error";
        $camSettingsFeedback.textContent = "✗ Network error — please try again.";
      }
    });
  }

  // ── Initialise ─────────────────────────────────────────────
  function bootApp() {
    fetchEmployees();
    connectWS();
    fetchStats();
  }

  checkAuthAndBoot();

})();
