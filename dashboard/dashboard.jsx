// Saamid Dashboard — root app

// React hooks accessed via React.* to avoid global-scope name clashes across Babel files

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "lang": "auto",
  "theme": "dark",
  "mapStyle": "hybrid",
  "headerStyle": "command",
  "alertIntensity": "moderate",
  "showCompass": true,
  "showMicLabels": true,
  "panelDensity": "comfortable"
}/*EDITMODE-END*/;

function detectLang() {
  const browser = (navigator.language || "en").toLowerCase();
  if (browser.startsWith("ar")) return "ar";
  // user requested: detect, default Arabic if unclear (treat any non-en as ar fallback)
  return "ar";
}

function nowTimeStr() {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}

// VU bar mapping lives in live.js so panels.jsx can share it.
const rmsToPercent = (window.SaamidLive && window.SaamidLive.rmsToPercent)
  ? window.SaamidLive.rmsToPercent
  : (v) => Math.max(2, Math.min(100, Math.sqrt(+v) * 200));

function App() {
  const [T_, setTweak] = window.useTweaks
    ? window.useTweaks(TWEAK_DEFAULTS)
    : [TWEAK_DEFAULTS, () => {}];

  // Live backend state — empty/false in demo mode.
  const live = window.SaamidLive
    ? window.SaamidLive.useSaamidLive()
    : { live: false, connected: false, hello: null, frame: null, events: [], lastCue: null };

  // Resolve language
  const [lang, setLang] = React.useState(() => {
    if (T_.lang === "ar" || T_.lang === "en") return T_.lang;
    return detectLang();
  });
  React.useEffect(() => {
    if (T_.lang === "ar" || T_.lang === "en") setLang(T_.lang);
  }, [T_.lang]);

  const [theme, setTheme] = React.useState(T_.theme || "dark");
  React.useEffect(() => { setTheme(T_.theme); }, [T_.theme]);

  const T = window.SaamidI18N[lang];
  const dir = T.dir;

  React.useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = dir;
    document.documentElement.classList.remove("theme-light", "theme-dark");
    document.documentElement.classList.add(theme === "light" ? "theme-light" : "theme-dark");
  }, [lang, dir, theme]);

  // Mic positions — seeded from defaults, overwritten by backend `hello` if live
  const [mics, setMics] = React.useState(window.SaamidDefaultMics);
  const [hoveredMicId, setHoveredMicId] = React.useState(null);

  React.useEffect(() => {
    if (live.live && live.hello && Array.isArray(live.hello.mics)) {
      const fromHello = live.hello.mics.map((m, i) => ({
        id: i + 1, x: +m[0], y: +m[1], z: +(m[2] || 0),
      }));
      // only overwrite once per hello — operator may still edit afterwards
      setMics((prev) => (prev === window.SaamidDefaultMics ? fromHello : prev));
    }
  }, [live.live, live.hello]);

  // Live detection threshold — owned at App level so the gauge and the
  // source-bar slider stay in sync, and so we can broadcast it back to the
  // backend via /control/threshold.
  const [liveThreshold, setLiveThreshold] = React.useState(0.6);
  React.useEffect(() => {
    if (live.live && live.hello && typeof live.hello.threshold === "number") {
      setLiveThreshold(live.hello.threshold);
    }
  }, [live.live, live.hello]);

  // Push mic edits to the backend so triangulation actually uses them.
  // Debounced — judges may scrub a number while typing.
  const micsPushRef = React.useRef(null);
  const micsLastSentRef = React.useRef("");
  React.useEffect(() => {
    if (!live.live || !window.SaamidLive) return;
    const helloN = (live.hello && live.hello.n_channels) || mics.length;
    if (mics.length !== helloN) return; // count mismatch — backend will reject
    const payload = mics.map((m) => [+m.x || 0, +m.y || 0, +m.z || 0]);
    const key = JSON.stringify(payload);
    if (key === micsLastSentRef.current) return;
    if (micsPushRef.current) clearTimeout(micsPushRef.current);
    micsPushRef.current = setTimeout(() => {
      micsLastSentRef.current = key;
      window.SaamidLive.setMics(payload).catch((e) => console.warn("[mics] push failed", e));
    }, 350);
    return () => micsPushRef.current && clearTimeout(micsPushRef.current);
  }, [mics, live.live, live.hello]);

  // Sim state — frozen when live mode is connected
  const [simT, setSimT] = React.useState(0.0);
  const [simPlaying, setSimPlaying] = React.useState(true);
  const [simSpeed, setSimSpeed] = React.useState(1);
  const lastTickRef = React.useRef(performance.now());

  React.useEffect(() => {
    if (live.live) return; // backend is the source of truth
    let raf;
    const tick = (now) => {
      const dt = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      if (simPlaying) {
        setSimT((t) => {
          let next = t + (dt * simSpeed) / 24; // ~24s loop at 1×
          if (next > 1) next = 0;
          return next;
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [simPlaying, simSpeed, live.live]);

  // Sample current drone state — from backend frame in live mode, simulator otherwise.
  const demoSample = React.useMemo(() => window.SaamidSampleTrack(simT), [simT]);
  const sample = React.useMemo(() => {
    if (live.live && live.frame && window.SaamidLive) {
      return window.SaamidLive.frameToSample(live.frame, demoSample);
    }
    return demoSample;
  }, [live.live, live.frame, demoSample]);

  // In live mode the backend already classified the threat — trust it.
  const liveThreat = live.live && live.frame ? live.frame.threat : null;

  // Drone-type classification with sticky hysteresis.  Backend confidence
  // sits right around the prototype threshold for real Shahed audio
  // (0.965-0.973), which makes the per-frame `drone_class` field flicker
  // between "shahed-136" and "unknown" every 0.5 s.  We latch the
  // classification for the duration of a detection burst: once we've seen
  // shahed-136 in this burst, keep showing it until threat goes clear.
  const stickyClassRef = React.useRef("unknown");
  const stickyClassConfRef = React.useRef(0);
  const liveDroneClass = live.live && live.frame ? live.frame.drone_class : "unknown";
  const liveClassConf = live.live && live.frame ? (live.frame.class_confidence || 0) : 0;
  React.useEffect(() => {
    if (!liveThreat || liveThreat === "clear") {
      stickyClassRef.current = "unknown";
      stickyClassConfRef.current = 0;
    } else if (liveDroneClass !== "unknown") {
      stickyClassRef.current = liveDroneClass;
      // Track the highest confidence we've seen this burst — stable readout.
      if (liveClassConf > stickyClassConfRef.current) {
        stickyClassConfRef.current = liveClassConf;
      }
    }
  }, [liveThreat, liveDroneClass, liveClassConf]);
  const aboveThreshold = liveThreat
    ? liveThreat !== "clear"
    : sample.prob > 0.6;
  const drone = aboveThreshold && (!live.live || sample.hasPosition) ? sample : null;

  // Trajectory-derived speed — read directly off the keyframe segment slope.
  // Robust against scrubbing / wrap-around (no EMA needed).
  const speedNow = sample.speed != null ? sample.speed : 0;

  // Longest consecutive windows above threshold (rolling, mirrors triangulate.py)
  const runRef = React.useRef({ cur: 0, longest: 0 });
  React.useEffect(() => {
    if (live.live && live.frame) {
      // Backend reports authoritative longest_run.
      runRef.current.longest = live.frame.longest_run || 0;
      runRef.current.cur = aboveThreshold ? runRef.current.cur + 1 : 0;
      return;
    }
    if (aboveThreshold) {
      runRef.current.cur += 1;
      runRef.current.longest = Math.max(runRef.current.longest, runRef.current.cur);
    } else {
      runRef.current.cur = 0;
    }
  }, [aboveThreshold, simT, live.live, live.frame]);

  // Threat type — backend-supplied in live mode, derived from p(drone) in demo.
  const threatType = React.useMemo(() => {
    if (liveThreat) return liveThreat;
    if (!aboveThreshold) return "clear";
    if (sample.prob < 0.78) return "unknown";
    return "detected";
  }, [sample.prob, aboveThreshold, liveThreat]);

  // No trail.  GCC-PHAT positions wander frame-to-frame even on a clean
  // recording, and a multi-point trail visually amplifies that wander —
  // it makes the system look unreliable even when detection itself is
  // confident.  We just show the current locked-in position marker
  // instead, which the backend only emits once it has 3 consecutive
  // nearby triangulations.
  const trailRef = React.useRef([]);
  const trailVersion = 0;

  // Sampled predicted path (down-sampled trail, in metres) — after trail refs
  const predictedPath = React.useMemo(() => {
    if (!drone) return [];
    return trailRef.current.slice(0, 6).map(p => [+p.x.toFixed(2), +p.y.toFixed(2), 0.0]);
  }, [drone, trailVersion]);

  // Event log
  const [demoEvents, setDemoEvents] = React.useState([]);
  const lastTypeRef = React.useRef("clear");
  const lastCueTimeRef = React.useRef(null);

  React.useEffect(() => {
    if (live.live) return; // events stream from backend
    if (threatType !== lastTypeRef.current) {
      const time = nowTimeStr();
      const push = (key, tone, conf, suffix) => setDemoEvents(es => [{ time, key, tone, conf, suffix }, ...es].slice(0, 60));
      if (threatType === "detected") {
        push("ev_first_detect", "alert", sample.prob);
        push("ev_decoy_check", "ok", null);
        push("ev_handoff", "info", null);
        lastCueTimeRef.current = time;
      } else if (threatType === "unknown") {
        push("ev_path_lock", "warn", sample.prob);
      } else if (threatType === "decoy") {
        push("ev_decoy_check", "warn", sample.prob, " ✗");
      } else if (threatType === "clear" && lastTypeRef.current !== "clear") {
        push("ev_lost", "info", null);
      }
      lastTypeRef.current = threatType;
    }
  }, [threatType, live.live]);

  // Choose event source.  Live mode uses backend log events; demo mode uses local.
  const events = live.live ? live.events : demoEvents;
  const lastCueTime = live.live
    ? (live.lastCue ? live.lastCue.time : null)
    : lastCueTimeRef.current;

  // Alert banner — shown on fresh transition into 'detected' until acknowledged
  const [alertVisible, setAlertVisible] = React.useState(false);
  const [muted, setMuted] = React.useState(false);
  const prevTypeForAlert = React.useRef("clear");
  React.useEffect(() => {
    if (threatType === "detected" && prevTypeForAlert.current !== "detected") {
      setAlertVisible(true);
      // auto-dismiss after 8s
      const id = setTimeout(() => setAlertVisible(false), 9000);
      return () => clearTimeout(id);
    }
    if (threatType === "clear") setAlertVisible(false);
    prevTypeForAlert.current = threatType;
  }, [threatType]);

  // Top-bar clock
  const [clock, setClock] = React.useState(nowTimeStr());
  React.useEffect(() => {
    const id = setInterval(() => setClock(nowTimeStr()), 1000);
    return () => clearInterval(id);
  }, []);

  // Splash hide
  React.useEffect(() => {
    const splash = document.getElementById("splash");
    if (splash) {
      setTimeout(() => splash.classList.add("hide"), 700);
      setTimeout(() => splash.remove(), 1400);
    }
  }, []);

  return (
    <div className={`app dir-${dir} hdr-${T_.headerStyle || "command"} density-${T_.panelDensity || "comfortable"}`} dir={dir}>
      <Header
        T={T} dir={dir} lang={lang} setLang={(L) => { setLang(L); setTweak("lang", L); }}
        theme={theme} setTheme={(t) => setTweak("theme", t)}
        clock={clock}
        threatType={threatType}
        style={T_.headerStyle || "command"}
        liveStatus={live.live ? (live.connected ? "live" : "reconnecting") : "demo"}
      />

      <AlertBanner
        visible={alertVisible}
        T={T} dir={dir}
        onAck={() => setAlertVisible(false)}
        intensity={T_.alertIntensity || "moderate"}
        muted={muted} setMuted={setMuted}
      />

      {live.live && (
        <SourceBar
          T={T} dir={dir}
          source={live.source}
          frame={live.frame}
          hello={live.hello}
          nChannels={(live.hello && live.hello.n_channels) || mics.length}
          threshold={liveThreshold}
          onThreshold={setLiveThreshold}
        />
      )}

      <main className="grid">
        {/* LEFT COLUMN */}
        <div className="col col-left">
          <div className="panel">
            <div className="panel-head">
              <h3>{T.panel_threat}</h3>
              <span className="t-eyebrow status-pill">
                <span className={`status-dot ${threatType !== "clear" ? "active" : ""}`} />
                {threatType !== "clear" ? T.online : T.online}
              </span>
            </div>
            <div className="panel-body">
              <ThreatPanel threatType={threatType} prob={sample.prob} T={T} dir={dir}
                droneClass={stickyClassRef.current}
                classConfidence={stickyClassConfRef.current} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head"><h3>{T.panel_prob}</h3></div>
            <div className="panel-body">
              <ProbabilityGauge prob={sample.prob} T={T} dir={dir}
                threshold={live.live ? liveThreshold : 0.6} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head"><h3>{T.panel_bearing}</h3></div>
            <div className="panel-body">
              <BearingReadout drone={drone} T={T} dir={dir}
                speed={speedNow}
                oodFlag={threatType === "unknown"} />
            </div>
          </div>
        </div>

        {/* CENTER — MAP */}
        <div className="col col-center">
          <div className="panel panel-map">
            <div className="panel-head">
              <h3>{T.panel_map}</h3>
              <div className="map-meta">
                <span className="t-eyebrow">{T.last_update}</span>
                <strong className="t-mono">{clock}</strong>
              </div>
            </div>
            <div className="panel-body map-body">
              <SaamidMap
                mics={mics}
                drone={drone}
                trail={trailRef.current}
                mapStyle={T_.mapStyle || "hybrid"}
                threatType={threatType}
                T={T} dir={dir}
                showCompass={T_.showCompass}
                showMicLabels={T_.showMicLabels}
                highlightedMicId={hoveredMicId}
              />
            </div>
            {!live.live && (
              <div className="panel-foot">
                <SimControls
                  playing={simPlaying} t={simT}
                  setT={setSimT} setPlaying={setSimPlaying}
                  speed={simSpeed} setSpeed={setSimSpeed}
                  T={T} dir={dir}
                />
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div className="col col-right">
          <div className="panel">
            <div className="panel-head"><h3>{T.panel_log}</h3></div>
            <div className="panel-body panel-body-flush">
              <EventLog events={events} T={T} dir={dir} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head"><h3>{T.panel_mics}</h3></div>
            <div className="panel-body">
              <MicEditor
                mics={mics} setMics={setMics} T={T} dir={dir}
                onHover={setHoveredMicId}
                levels={live.frame ? live.frame.rms_per_channel : null}
                silentChannels={live.frame ? live.frame.silent_channels : null}
                sourceLive={live.live && live.source && (live.source.state === "live" || live.source.state === "simulate")}
              />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head"><h3>{T.panel_cue}</h3></div>
            <div className="panel-body">
              <CueingPanel
                drone={drone} threatType={threatType}
                mics={mics} T={T} dir={dir}
                lastCueTime={lastCueTime}
                longestRun={runRef.current.longest}
                sampledPath={predictedPath}
                velocity={drone ? [+(sample.vx||0).toFixed(2), +(sample.vy||0).toFixed(2), 0] : null}
                simTimeS={live.live && live.frame ? live.frame.t_s : simT * 24}
                liveCue={live.live && live.lastCue ? live.lastCue.payload : null}
              />
            </div>
          </div>
        </div>
      </main>

      <Footer T={T} dir={dir} />

      {window.TweaksPanel && (
        <window.TweaksPanel title="Tweaks" defaultPosition="bottom-right">
          <window.TweakSection title="Language & theme">
            <window.TweakRadio label="Language"
              value={T_.lang === "auto" ? lang : T_.lang}
              onChange={(v) => setTweak("lang", v)}
              options={[{ value: "ar", label: "AR" }, { value: "en", label: "EN" }]} />
            <window.TweakRadio label="Theme"
              value={T_.theme}
              onChange={(v) => setTweak("theme", v)}
              options={[{ value: "dark", label: "Dark" }, { value: "light", label: "Light" }]} />
          </window.TweakSection>

          <window.TweakSection title="Map style">
            <window.TweakRadio label="Map"
              value={T_.mapStyle}
              onChange={(v) => setTweak("mapStyle", v)}
              options={[
                { value: "hybrid", label: "Hybrid" },
                { value: "radar",  label: "Radar" },
                { value: "grid",   label: "Grid" },
                { value: "sat",    label: "Terrain" },
              ]} />
            <window.TweakToggle label="Compass overlay"
              value={!!T_.showCompass} onChange={(v) => setTweak("showCompass", v)} />
            <window.TweakToggle label="Mic labels"
              value={!!T_.showMicLabels} onChange={(v) => setTweak("showMicLabels", v)} />
          </window.TweakSection>

          <window.TweakSection title="Header & alerts">
            <window.TweakRadio label="Header"
              value={T_.headerStyle}
              onChange={(v) => setTweak("headerStyle", v)}
              options={[
                { value: "command", label: "Command" },
                { value: "minimal", label: "Minimal" },
                { value: "ribbon",  label: "Ribbon" },
              ]} />
            <window.TweakRadio label="Alert intensity"
              value={T_.alertIntensity}
              onChange={(v) => setTweak("alertIntensity", v)}
              options={[
                { value: "subtle",   label: "Subtle" },
                { value: "moderate", label: "Moderate" },
                { value: "loud",     label: "Loud" },
              ]} />
          </window.TweakSection>

          <window.TweakSection title="Density">
            <window.TweakRadio label="Panels"
              value={T_.panelDensity}
              onChange={(v) => setTweak("panelDensity", v)}
              options={[
                { value: "comfortable", label: "Comfortable" },
                { value: "compact",     label: "Compact" },
              ]} />
          </window.TweakSection>
        </window.TweaksPanel>
      )}
    </div>
  );
}

/* ---------------- Header ---------------- */
function Header({ T, dir, lang, setLang, theme, setTheme, clock, threatType, style, liveStatus }) {
  const otherLang = lang === "ar" ? "en" : "ar";
  const otherT = window.SaamidI18N[otherLang];

  const statusClass =
    liveStatus === "live" ? "live"
    : liveStatus === "reconnecting" ? "reconnecting"
    : "demo";
  const statusLabel =
    liveStatus === "live" ? "LIVE"
    : liveStatus === "reconnecting" ? "RECONNECTING"
    : "DEMO";

  return (
    <header className={`hdr style-${style}`}>
      <div className="hdr-left">
        <img className="hdr-logo" src="assets/logo.png" alt="Saamid logo" />
        <div className="hdr-id">
          <div className="hdr-name t-display">{T.project}</div>
          <div className="hdr-tagline">
            {style === "ribbon" ? null : T.tagline}
          </div>
        </div>
      </div>

      <div className="hdr-mid" />

      <div className="hdr-right">
        <div className={`hdr-mode ${statusClass}`} title={`Source: ${statusLabel}`}>
          <span className="mode-pip" />
          <span>{statusLabel}</span>
        </div>
        <div className="hdr-clock t-mono">
          <span className="clock-pip" />
          {clock}
        </div>
        <div className="hdr-status">
          <span className={`status-dot ${threatType !== "clear" ? "alert" : "ok"}`} />
          <span>{T.online}</span>
        </div>
        <div className="hdr-toggles">
          <div className="theme-toggle" role="group">
            <button className={theme === "light" ? "on" : ""} onClick={() => setTheme("light")}>
              <SunIcon />
              <span>{T.theme_light}</span>
            </button>
            <button className={theme === "dark" ? "on" : ""} onClick={() => setTheme("dark")}>
              <MoonIcon />
              <span>{T.theme_dark}</span>
            </button>
          </div>
          <button className="lang-btn" onClick={() => setLang(otherLang)}
            aria-label={`Switch to ${otherT.lang_btn_label}`}>
            <span className="lang-now">{lang === "ar" ? "AR" : "EN"}</span>
            <span className="lang-arrow">{dir === "rtl" ? "←" : "→"}</span>
            <span className="lang-other">{lang === "ar" ? "EN" : "AR"}</span>
          </button>
        </div>
      </div>
    </header>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="t-eyebrow">{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SunIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.4"/>
      {[0,45,90,135,180,225,270,315].map((a)=>{
        const r=(a-90)*Math.PI/180;
        return <line key={a} x1={8+Math.cos(r)*5} y1={8+Math.sin(r)*5}
          x2={8+Math.cos(r)*7} y2={8+Math.sin(r)*7}
          stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>;
      })}
    </svg>
  );
}
function MoonIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M12.5 9.5 A 5 5 0 1 1 6.5 3.5 A 4 4 0 0 0 12.5 9.5 Z"
        stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinejoin="round"/>
    </svg>
  );
}

/* ---------------- Source bar ----------------
   Live-mode-only banner under the header.  Shows the active audio source
   (live mic interface, replaying WAV, or none), inline VU bars across all
   channels, and a "Change" button that opens a device picker. */
function SourceBar({ T, dir, source, frame, nChannels, hello, threshold, onThreshold }) {
  const [open, setOpen] = React.useState(false);
  const [devices, setDevices] = React.useState(null);
  // Has the host got at least one input device with enough channels for our
  // mic array?  Probed once on mount.  When it's `false` (e.g. on the
  // cloud HF Space, no audio hardware), we hide the live-device picker
  // entirely — there's nothing to pick, and the empty list looks broken.
  const [hasDevices, setHasDevices] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState("");

  React.useEffect(() => {
    if (!window.SaamidLive) return;
    window.SaamidLive.fetchDevices()
      .then((d) => {
        setDevices(d);
        setHasDevices(Array.isArray(d.devices) && d.devices.length > 0);
      })
      .catch(() => setHasDevices(false));
  }, []);

  // Threshold owned by App; we just debounce the POST to the backend.
  const threshTimerRef = React.useRef(null);
  const onThresh = (v) => {
    onThreshold(v);
    if (threshTimerRef.current) clearTimeout(threshTimerRef.current);
    threshTimerRef.current = setTimeout(() => {
      window.SaamidLive.setThreshold(v).catch((e) => console.warn("[thresh] push failed", e));
    }, 150);
  };

  const state = source ? source.state : "none";
  const label = source ? (source.label || "") : "";

  const stateText = ({
    none:      T.src_state_none,
    opening:   T.src_state_opening,
    live:      T.src_state_live,
    simulate:  T.src_state_simulate,
    error:     T.src_state_error,
  })[state] || state;

  const stateClass = state === "error" ? "err"
    : state === "live" ? "ok"
    : state === "simulate" ? "warn"
    : state === "opening" ? "warn"
    : "muted";

  // Auto-open the picker if there's no source yet — judges shouldn't have
  // to hunt for the button.  Auto-close once the source comes up so the
  // picker doesn't keep covering the dashboard.  Skip auto-open entirely
  // if the host has no compatible audio hardware (cloud demo) — opening
  // an empty picker just looks broken.
  React.useEffect(() => {
    if (hasDevices === false) return;
    if (state === "none" || state === "error") setOpen(true);
    else if (state === "live" || state === "simulate") setOpen(false);
  }, [state, hasDevices]);

  async function refresh() {
    setBusy(true); setErr("");
    try {
      const d = await window.SaamidLive.fetchDevices();
      setDevices(d);
      setHasDevices(Array.isArray(d.devices) && d.devices.length > 0);
    } catch (e) {
      setErr(String(e.message || e));
    } finally { setBusy(false); }
  }
  React.useEffect(() => { if (open && !devices) refresh(); }, [open]);

  async function pick(idx) {
    setBusy(true); setErr("");
    try {
      await window.SaamidLive.setSource("live", { device_index: idx });
      setOpen(false);
    } catch (e) { setErr(String(e.message || e)); }
    finally { setBusy(false); }
  }
  async function disconnect() {
    setBusy(true); setErr("");
    try { await window.SaamidLive.setSource("none"); }
    catch (e) { setErr(String(e.message || e)); }
    finally { setBusy(false); }
  }

  // Audio upload — primary path is per-mic array (one file per microphone),
  // with a "single file" fallback for quick mono tests.
  const [uploadOpen, setUploadOpen] = React.useState(false);
  const [uploading, setUploading] = React.useState(false);

  async function resetToDefault() {
    setBusy(true); setErr("");
    try { await window.SaamidLive.resetDefault(); }
    catch (e) { setErr(String(e.message || e)); }
    finally { setBusy(false); }
  }

  const hasDefault = !!(hello && hello.has_default_simulate);
  const defaultName = (hello && hello.default_simulate_label) || "";
  const playingDefault = state === "simulate" && label && defaultName
    && label.endsWith(defaultName);

  const levels = frame ? (frame.rms_per_channel || []) : [];
  const silent = new Set(((frame && frame.silent_channels) || []).map(Number));

  return (
    <div className={`src-bar src-${stateClass}`}>
      <div className="src-row">
        <span className={`src-pip ${stateClass}`} />
        <span className="src-state">{stateText}</span>
        {label && <span className="src-label">{label}</span>}
        {source && source.detail && state === "error" && (
          <span className="src-detail">{source.detail}</span>
        )}
        {state === "live" && levels.length > 0 && (
          <span className="src-meters">
            {levels.slice(0, nChannels).map((v, i) => (
              <span key={i} className={`src-meter ${silent.has(i) ? "silent" : ""}`}
                    title={`M${i + 1} · RMS ${(+v).toFixed(4)}${silent.has(i) ? ` · ${T.src_silent}` : ""}`}>
                <span className="src-meter-fill"
                      style={{ width: `${rmsToPercent(+v)}%` }} />
              </span>
            ))}
          </span>
        )}
        <span className="src-spacer" />
        <span className="src-thresh" title="Detection threshold — lower if room is noisy">
          <span className="t-eyebrow">θ</span>
          <input type="range" min="0.1" max="0.6" step="0.01"
            value={threshold} onChange={(e) => onThresh(+e.target.value)} />
          <span className="src-thresh-num t-mono">{threshold.toFixed(2)}</span>
        </span>
        <button className="src-btn upload"
                disabled={busy || uploading}
                onClick={() => setUploadOpen(true)}
                title={T.src_upload_hint}>
          {uploading ? T.src_uploading : `↑ ${T.src_upload_array}`}
        </button>
        {hasDefault && !playingDefault && (
          <button className="src-btn ghost" disabled={busy || uploading}
                  onClick={resetToDefault} title={defaultName}>
            ↺ {T.src_reset_default}
          </button>
        )}
        {hasDevices !== false && (
          <button className="src-btn ghost" disabled={busy} onClick={() => setOpen(o => !o)}>
            {state === "live" || state === "simulate" ? T.src_change : T.src_pick_device}
          </button>
        )}
        {state === "live" && (
          <button className="src-btn ghost" disabled={busy} onClick={disconnect}>
            {T.src_disconnect}
          </button>
        )}
      </div>
      {open && (
        <div className="src-picker">
          <div className="src-picker-head">
            <span className="t-eyebrow">{T.src_pick_device}</span>
            <button className="src-btn ghost" disabled={busy} onClick={refresh}>{T.src_refresh}</button>
          </div>
          {err && <div className="src-err">{err}</div>}
          {!devices && !err && <div className="src-empty">…</div>}
          {devices && devices.devices.length === 0 && (
            <div className="src-empty">{T.src_no_compatible} (need ≥{devices.n_channels_required})</div>
          )}
          {devices && devices.devices.length > 0 && (
            <ul className="src-dev-list">
              {devices.devices.map((d) => (
                <li key={d.index}>
                  <button className="src-dev" disabled={busy} onClick={() => pick(d.index)}>
                    <span className="src-dev-name">{d.name}</span>
                    <span className="src-dev-meta">
                      {d.max_input_channels} ch · {Math.round(d.default_samplerate / 1000)} kHz
                      {d.host_api ? ` · ${d.host_api}` : ""}
                      {d.is_default ? " · default" : ""}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {uploadOpen && (
        <UploadModal
          T={T} dir={dir}
          nChannels={nChannels}
          uploading={uploading}
          setUploading={setUploading}
          onClose={() => setUploadOpen(false)}
          onError={setErr}
        />
      )}
    </div>
  );
}

/* ---------------- Upload modal ---------------- */
function UploadModal({ T, dir, nChannels, uploading, setUploading, onClose, onError }) {
  const [slots, setSlots] = React.useState(() => Array.from({ length: nChannels }, () => null));
  const inputRefs = React.useRef([]);
  const singleRef = React.useRef(null);
  const [localErr, setLocalErr] = React.useState("");

  const allFilled = slots.every((s) => s != null);

  const setSlot = (i, file) => {
    setSlots((prev) => {
      const next = prev.slice();
      next[i] = file;
      return next;
    });
  };

  async function submitArray() {
    if (!allFilled) return;
    setLocalErr(""); onError(""); setUploading(true);
    try {
      await window.SaamidLive.uploadMicArray(slots);
      onClose();
    } catch (e) {
      setLocalErr(String(e.message || e));
    } finally {
      setUploading(false);
    }
  }

  async function submitSingle(file) {
    setLocalErr(""); onError(""); setUploading(true);
    try {
      await window.SaamidLive.uploadAudio(file);
      onClose();
    } catch (e) {
      setLocalErr(String(e.message || e));
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="upload-modal-backdrop" onClick={onClose}>
      <div className="upload-modal" onClick={(e) => e.stopPropagation()} dir={dir}>
        <div className="upload-modal-head">
          <h3>{T.src_array_title}</h3>
          <button className="upload-close" onClick={onClose} aria-label={T.src_array_cancel}>×</button>
        </div>
        <p className="upload-help">{T.src_array_help}</p>

        <div className="upload-slots">
          {slots.map((file, i) => (
            <div key={i} className={`upload-slot ${file ? "filled" : ""}`}>
              <span className="upload-slot-tag">M{i + 1}</span>
              <button
                className="upload-slot-btn"
                disabled={uploading}
                onClick={() => inputRefs.current[i] && inputRefs.current[i].click()}
              >
                {file ? file.name : T.src_array_choose}
              </button>
              {file && (
                <span className="upload-slot-meta">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </span>
              )}
              <input
                type="file"
                ref={(el) => (inputRefs.current[i] = el)}
                accept=".wav,.mp3,.m4a,.flac,.ogg,.aac,.opus,.webm,audio/*"
                style={{ display: "none" }}
                onChange={(e) => {
                  const f = e.target.files && e.target.files[0];
                  if (f) setSlot(i, f);
                  e.target.value = "";
                }}
              />
            </div>
          ))}
        </div>

        {localErr && <div className="src-err">{localErr}</div>}

        <div className="upload-actions">
          <button className="src-btn ghost" onClick={onClose} disabled={uploading}>
            {T.src_array_cancel}
          </button>
          <button
            className="src-btn upload"
            disabled={!allFilled || uploading}
            onClick={submitArray}
          >
            {uploading ? T.src_uploading : T.src_array_submit}
          </button>
        </div>

        <div className="upload-divider">
          <button
            className="upload-single-link"
            disabled={uploading}
            onClick={() => singleRef.current && singleRef.current.click()}
          >
            {T.src_array_or_single}
          </button>
          <input
            type="file"
            ref={singleRef}
            accept=".wav,.mp3,.m4a,.flac,.ogg,.aac,.opus,.webm,audio/*"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files && e.target.files[0];
              if (f) submitSingle(f);
              e.target.value = "";
            }}
          />
        </div>
      </div>
    </div>
  );
}

/* ---------------- Footer ---------------- */
function Footer({ T, dir }) {
  return (
    <footer className="ftr">
      <div className="ftr-left">
        <img src="assets/favicon.png" alt="" className="ftr-logo" />
        <span>{T.project_short}</span>
      </div>
      <div className="ftr-right t-mono">
        SAAMID-EWS · v1.0 · GCC-PHAT + AST · 4-MIC ARRAY · 16 kHz
      </div>
    </footer>
  );
}

/* mount */
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
