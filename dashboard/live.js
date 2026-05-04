// Saamid live-mode WebSocket client.
//
// Tries to open ws://<host>/ws on startup.  If the backend is reachable,
// `useSaamidLive()` returns { live: true, ... } and the dashboard drives
// itself off real detection events instead of the SaamidFlightPath demo.
// If the connection fails or `?demo=1` is set in the URL, returns
// { live: false } and the dashboard falls back to its built-in simulator.

(function () {
  const FORCE_DEMO = new URLSearchParams(location.search).has("demo");

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    // Allow ?ws=ws://other.host:8000/ws for split deployments.
    const override = new URLSearchParams(location.search).get("ws");
    if (override) return override;
    return `${proto}//${location.host}/ws`;
  }

  // ----- low-level singleton connection -----
  let socket = null;
  let listeners = new Set();
  let helloSeen = null;
  let lastFrameSeen = null;
  let lastSourceSeen = null;
  let connected = false;
  let everConnected = false;
  let attempt = 0;
  let stopped = false;

  function broadcast(ev) {
    for (const fn of listeners) {
      try { fn(ev); } catch (e) { console.warn("[live] listener error", e); }
    }
  }

  function connect() {
    if (FORCE_DEMO) return;
    if (stopped) return;
    try {
      socket = new WebSocket(wsUrl());
    } catch (e) {
      console.warn("[live] socket construct failed", e);
      scheduleRetry();
      return;
    }
    socket.onopen = () => {
      attempt = 0;
      connected = true;
      everConnected = true;
      broadcast({ type: "_status", connected: true });
    };
    socket.onmessage = (msg) => {
      let ev;
      try { ev = JSON.parse(msg.data); }
      catch { return; }
      if (ev.type === "hello") helloSeen = ev;
      if (ev.type === "frame") lastFrameSeen = ev;
      if (ev.type === "source") lastSourceSeen = ev;
      broadcast(ev);
    };
    socket.onclose = () => {
      connected = false;
      broadcast({ type: "_status", connected: false });
      scheduleRetry();
    };
    socket.onerror = () => {
      // onerror fires before onclose; let onclose handle the reconnect
    };
  }

  function scheduleRetry() {
    if (stopped) return;
    attempt++;
    // exponential backoff capped at 5s
    const delay = Math.min(5000, 500 * 2 ** Math.min(attempt - 1, 3));
    setTimeout(connect, delay);
  }

  function subscribe(fn) {
    listeners.add(fn);
    if (helloSeen) fn(helloSeen);
    if (lastSourceSeen) fn(lastSourceSeen);
    if (lastFrameSeen) fn(lastFrameSeen);
    return () => listeners.delete(fn);
  }

  // ----- REST helpers -----
  async function fetchJSON(path, init) {
    const r = await fetch(path, init);
    if (!r.ok) {
      let detail = "";
      try { detail = (await r.json()).detail || ""; } catch {}
      throw new Error(`${path} → ${r.status} ${detail}`);
    }
    return r.json();
  }

  async function fetchDevices() {
    return fetchJSON("/devices");
  }

  async function setSource(kind, opts = {}) {
    return fetchJSON("/control/source", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ kind, ...opts }),
    });
  }

  async function setMics(mics) {
    return fetchJSON("/control/mics", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ mics }),
    });
  }

  async function setThreshold(threshold) {
    return fetchJSON("/control/threshold", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ threshold }),
    });
  }

  async function uploadAudio(file) {
    const fd = new FormData();
    fd.append("file", file, file.name);
    const r = await fetch("/control/upload", { method: "POST", body: fd });
    if (!r.ok) {
      let detail = `${r.status}`;
      try { detail = (await r.json()).detail || detail; } catch {}
      throw new Error(`upload failed: ${detail}`);
    }
    return r.json();
  }

  async function uploadMicArray(files) {
    // files: array of File objects, in mic-index order (M1, M2, ...).
    const fd = new FormData();
    for (const f of files) fd.append("files", f, f.name);
    const r = await fetch("/control/upload_array", { method: "POST", body: fd });
    if (!r.ok) {
      let detail = `${r.status}`;
      try { detail = (await r.json()).detail || detail; } catch {}
      throw new Error(`array upload failed: ${detail}`);
    }
    return r.json();
  }

  async function resetDefault() {
    return fetchJSON("/control/reset_default", { method: "POST" });
  }

  // Boot the connection ASAP — react hook attaches later.
  if (!FORCE_DEMO) connect();

  // ----- React hook -----
  function useSaamidLive() {
    const [state, setState] = React.useState({
      live: false,
      connected: false,
      hello: helloSeen,
      frame: lastFrameSeen,
      source: lastSourceSeen,
      events: [],
      lastCue: null,
    });

    React.useEffect(() => {
      if (FORCE_DEMO) return;
      const off = subscribe((ev) => {
        setState((prev) => {
          if (ev.type === "_status") {
            return { ...prev, live: everConnected, connected: ev.connected };
          }
          if (ev.type === "hello") {
            return { ...prev, live: true, connected: true, hello: ev };
          }
          if (ev.type === "source") {
            return { ...prev, live: true, connected: true, source: ev };
          }
          if (ev.type === "frame") {
            return { ...prev, live: true, connected: true, frame: ev };
          }
          if (ev.type === "log") {
            return {
              ...prev,
              live: true,
              connected: true,
              events: [
                {
                  time: (ev.ts_iso || "").slice(11, 19) || nowTimeStr(),
                  key: ev.key,
                  tone: ev.tone,
                  conf: ev.confidence,
                },
                ...prev.events,
              ].slice(0, 60),
            };
          }
          if (ev.type === "cue") {
            return {
              ...prev,
              live: true,
              connected: true,
              lastCue: {
                time: (ev.ts_iso || "").slice(11, 19) || nowTimeStr(),
                payload: ev.cueing,
              },
            };
          }
          return prev;
        });
      });
      return off;
    }, []);

    return state;
  }

  function nowTimeStr() {
    const d = new Date();
    return d.toTimeString().slice(0, 8);
  }

  // ----- map a live frame -> the {x, y, prob, type, ...} shape that the
  // existing demo SaamidSampleTrack returns, so dashboard.jsx can swap
  // sources with a single line. -----
  function frameToSample(frame, fallback) {
    if (!frame) return fallback;
    const pos = frame.drone_position_m;
    const vel = frame.drone_velocity_m_s;
    return {
      t: frame.t_s,
      x: pos ? pos[0] : (fallback ? fallback.x : 0),
      y: pos ? pos[1] : (fallback ? fallback.y : 0),
      alt: frame.altitude_m != null ? frame.altitude_m : (fallback ? fallback.alt : 0),
      prob: frame.p_drone,
      type: frame.threat,
      vx: vel ? vel[0] : 0,
      vy: vel ? vel[1] : 0,
      speed: vel ? Math.hypot(vel[0], vel[1]) : 0,
      bearing: frame.bearing_deg,
      range: frame.range_m,
      open_set_unknown: !!frame.open_set_unknown,
      decoy_label: frame.decoy_label,
      longest_run: frame.longest_run,
      hasPosition: !!pos,
    };
  }

  // RMS → 0..100 percent for VU bars.  Logarithmic so a quiet drone
  // (RMS ~0.001) reads ~35% and a loud drone (RMS ~0.05) reads ~95%.
  function rmsToPercent(rms) {
    if (!isFinite(rms) || rms <= 0) return 2;
    const db = 20 * Math.log10(rms);
    const pct = ((db + 80) / 60) * 100;
    return Math.max(2, Math.min(100, pct));
  }

  window.SaamidLive = {
    useSaamidLive,
    frameToSample,
    fetchDevices,
    setSource,
    setMics,
    setThreshold,
    uploadAudio,
    uploadMicArray,
    resetDefault,
    rmsToPercent,
    isForcedDemo: () => FORCE_DEMO,
  };
})();
