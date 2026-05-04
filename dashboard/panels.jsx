// Saamid panels: threat, probability gauge, event log, mic editor, cue, sim controls

// React hooks accessed via React.* to avoid global-scope name clashes across Babel files

/* ---------------- Threat indicator ---------------- */
function ThreatPanel({ threatType, T, dir, prob, droneClass, classConfidence }) {
  const map = {
    clear:    { tone: "ok",     dot: "var(--green-500)",  label: T.th_clear,    sub: T.th_clear_sub,    code: "T-0" },
    detected: { tone: "alert",  dot: "var(--alert)",      label: T.th_detected, sub: T.th_detected_sub, code: "T-3" },
    decoy:    { tone: "warn",   dot: "var(--warn)",       label: T.th_decoy,    sub: T.th_decoy_sub,    code: "T-2" },
    unknown:  { tone: "warn",   dot: "var(--gold-400)",   label: T.th_unknown,  sub: T.th_unknown_sub,  code: "T-1" },
  };
  const cfg = map[threatType] || map.clear;
  const isShahed = droneClass === "shahed-136";
  const showClassChip = threatType !== "clear" && isShahed;
  return (
    <div className={`threat-card tone-${cfg.tone}`}>
      <div className="threat-row">
        <span className="threat-dot" style={{ background: cfg.dot }} />
        <span className="threat-code t-mono">{cfg.code}</span>
        <span className="threat-spacer" />
        <span className="threat-prob t-mono">p={prob.toFixed(2)}</span>
      </div>
      <div className="threat-label">{cfg.label}</div>
      {showClassChip && (
        <div className="threat-class">
          <span className="threat-class-tag">{T.cls_label}</span>
          <span className="threat-class-name">{T.cls_shahed}</span>
          <span className="threat-class-conf t-mono">
            {(classConfidence * 100).toFixed(1)}%
          </span>
        </div>
      )}
      <div className="threat-sub">{cfg.sub}</div>
      <div className="threat-bars">
        {["clear", "unknown", "decoy", "detected"].map((k, i) => (
          <span key={k} className={`bar ${k === threatType ? "on" : ""} tone-${map[k].tone}`} />
        ))}
      </div>
    </div>
  );
}

/* ---------------- Probability gauge ---------------- */
function ProbabilityGauge({ prob, T, dir, threshold = 0.6 }) {
  // semi-circular gauge
  const pct = Math.max(0, Math.min(1, prob));
  const ang = -90 + pct * 180; // -90 = left, +90 = right
  const r = 78;
  const cx = 110, cy = 100;
  const arc = (start, end, color, width) => {
    const s = (start - 90) * Math.PI / 180;
    const e = (end - 90) * Math.PI / 180;
    const x1 = cx + Math.cos(s) * r, y1 = cy + Math.sin(s) * r;
    const x2 = cx + Math.cos(e) * r, y2 = cy + Math.sin(e) * r;
    const large = end - start > 180 ? 1 : 0;
    return <path d={`M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`}
      fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" />;
  };
  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 220 130" className="gauge-svg">
        {/* track */}
        {arc(0, 180, "var(--surface-line)", 10)}
        {/* fill */}
        {pct > 0.001 && arc(0, pct * 180,
          pct > threshold ? "var(--alert)" : "var(--accent)", 10)}
        {/* end-cap dot at the leading edge of the fill — sits ON the arc,
            not over the centred numeric readout the way a centre-pivot
            needle does */}
        {pct > 0.001 && (() => {
          const a = ang * Math.PI / 180;
          const tx = cx + Math.cos(a) * r;
          const ty = cy + Math.sin(a) * r;
          return (
            <circle cx={tx} cy={ty} r="4.5"
              fill={pct > threshold ? "var(--alert)" : "var(--accent)"}
              stroke="var(--bg-elev)" strokeWidth="1.5" />
          );
        })()}
        {/* tick labels */}
        <text x="20" y="118" fill="var(--text-faint)" fontSize="9" fontFamily="var(--font-mono)">0</text>
        <text x={cx} y="20" fill="var(--text-faint)" fontSize="9" textAnchor="middle" fontFamily="var(--font-mono)">0.5</text>
        <text x="195" y="118" fill="var(--text-faint)" fontSize="9" textAnchor="end" fontFamily="var(--font-mono)">1.0</text>
      </svg>
      <div className="gauge-readout">
        <div className="gauge-num t-mono">{pct.toFixed(2)}</div>
        <div className="gauge-cap">{T.panel_prob}</div>
        <div className="gauge-thresh">
          <span className="thresh-pip" /> {dir === "rtl" ? "العتبة" : "threshold"} {threshold.toFixed(2)}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Event log ---------------- */
function EventLog({ events, T, dir }) {
  return (
    <div className="log-list">
      {events.length === 0 && (
        <div className="log-empty">{T.log_empty}</div>
      )}
      {events.map((e, i) => (
        <div key={i} className={`log-row tone-${e.tone}`}>
          <span className="log-time t-mono">{e.time}</span>
          <span className="log-dot" />
          <span className="log-text">{(T[e.key] || e.text || "") + (e.suffix || "")}</span>
          {e.conf != null && <span className="log-conf t-mono">{e.conf.toFixed(2)}</span>}
        </div>
      ))}
    </div>
  );
}

/* ---------------- Mic editor ---------------- */
function MicEditor({ mics, setMics, T, dir, onHover, levels, silentChannels, sourceLive }) {
  const update = (id, key, val) => {
    setMics(mics.map(m => m.id === id ? { ...m, [key]: parseFloat(val) || 0 } : m));
  };
  const add = () => {
    const id = (mics.length ? Math.max(...mics.map(m => m.id)) : 0) + 1;
    setMics([...mics, { id, x: 1, y: 1, z: 0 }]);
  };
  const remove = (id) => setMics(mics.filter(m => m.id !== id));
  const showLevels = sourceLive && Array.isArray(levels) && levels.length > 0;
  const silentSet = new Set((silentChannels || []).map(Number));
  return (
    <div className="mic-editor">
      <div className={`mic-grid-head ${showLevels ? "with-level" : ""}`}>
        <span className="t-eyebrow">{T.mic}</span>
        {showLevels && <span className="t-eyebrow mic-level-head">{T.src_levels || "level"}</span>}
        <span className="t-eyebrow">{T.coord_x}</span>
        <span className="t-eyebrow">{T.coord_y}</span>
        <span className="t-eyebrow">{T.coord_z}</span>
        <span />
      </div>
      {mics.map((m, idx) => {
        // levels are per audio-channel index (0..n-1), mic ids start at 1
        const lvl = showLevels ? levels[idx] || 0 : 0;
        const silent = showLevels && silentSet.has(idx);
        return (
          <div key={m.id} className={`mic-grid-row ${showLevels ? "with-level" : ""} ${silent ? "silent" : ""}`}
            onMouseEnter={() => onHover && onHover(m.id)}
            onMouseLeave={() => onHover && onHover(null)}>
            <span className="mic-tag">M{m.id}</span>
            {showLevels && (
              <span className="mic-level" title={silent ? T.src_silent : `RMS ${lvl.toFixed(4)}`}>
                <span
                  className={`mic-level-fill ${silent ? "silent" : ""}`}
                  style={{ width: `${(window.SaamidLive && window.SaamidLive.rmsToPercent) ? window.SaamidLive.rmsToPercent(lvl) : Math.min(100, Math.max(2, Math.sqrt(lvl) * 200))}%` }}
                />
              </span>
            )}
            <input type="number" step="0.1" value={m.x} onChange={e => update(m.id, "x", e.target.value)} />
            <input type="number" step="0.1" value={m.y} onChange={e => update(m.id, "y", e.target.value)} />
            <input type="number" step="0.1" value={m.z} onChange={e => update(m.id, "z", e.target.value)} />
            <button className="mic-remove" onClick={() => remove(m.id)} aria-label={T.remove}>×</button>
          </div>
        );
      })}
      <button className="mic-add" onClick={add}>＋ {T.add_mic}</button>
    </div>
  );
}

/* ---------------- Cueing handoff JSON ----------------
   Mirrors the `cueing` object emitted by scripts/triangulate.py exactly:
     drone_class, position_m, velocity_m_s, predicted_path_m,
     confidence, threat_level, is_decoy,
     longest_consecutive_windows_above_threshold, timestamp_s
   Wrapped in a transport envelope (protocol/timestamp/site_id) for the bus. */
function CueingPanel({ drone, threatType, mics, T, dir, lastCueTime, longestRun, sampledPath, velocity, simTimeS, liveCue }) {
  const json = React.useMemo(() => {
    // Live mode: trust the backend's cue payload verbatim.
    if (liveCue) {
      return {
        protocol: "saamid.cue/1",
        timestamp: new Date().toISOString(),
        site_id: "RUH-14",
        cueing: liveCue,
      };
    }
    // map UI threat type -> triangulate.py threat_level vocabulary
    const threatLevel =
      threatType === "decoy"    ? "decoy" :
      threatType === "unknown"  ? "unreliable" :
      threatType === "detected" ? (drone && drone.prob >= 0.7 ? "high" : "moderate") :
                                  "clear";
    const cueing = {
      drone_class: "unknown",            // model is binary today; always "unknown"
      position_m: drone ? [+drone.x.toFixed(3), +drone.y.toFixed(3), 0.0] : null,
      velocity_m_s: velocity || null,
      predicted_path_m: sampledPath || [],
      confidence: drone ? +drone.prob.toFixed(3) : 0.0,
      threat_level: threatLevel,
      is_decoy: threatType === "decoy",
      longest_consecutive_windows_above_threshold: longestRun || 0,
      timestamp_s: +(simTimeS || 0).toFixed(2),
    };
    return {
      protocol: "saamid.cue/1",
      timestamp: new Date().toISOString(),
      site_id: "RUH-14",
      cueing,
    };
  }, [drone, threatType, longestRun, sampledPath, velocity, simTimeS, liveCue]);
  return (
    <div className="cue-panel">
      <div className="cue-status-row">
        <span className="t-eyebrow">{T.cue_status}</span>
        <span className={`cue-pip ${threatType !== "clear" ? "live" : ""}`} />
        <span className="cue-status-text">
          {threatType !== "clear" ? `${T.cue_sent} ${lastCueTime || "—"}` : T.cue_idle}
        </span>
      </div>
      <pre className="cue-json t-mono">{JSON.stringify(json, null, 2)}</pre>
    </div>
  );
}

/* ---------------- Track simulator scrubber ---------------- */
function SimControls({ playing, t, setT, setPlaying, speed, setSpeed, T, dir }) {
  return (
    <div className="sim-controls">
      <button className={`sim-btn ${playing ? "playing" : ""}`} onClick={() => setPlaying(!playing)}>
        {playing ? (
          <svg width="14" height="14" viewBox="0 0 14 14"><rect x="3" y="2" width="3" height="10" fill="currentColor"/><rect x="8" y="2" width="3" height="10" fill="currentColor"/></svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14"><polygon points="3,2 12,7 3,12" fill="currentColor"/></svg>
        )}
        <span>{playing ? T.pause : T.play}</span>
      </button>
      <button className="sim-reset" onClick={() => { setT(0); setPlaying(false); }}>↺ {T.reset}</button>
      <div className="sim-scrub">
        <input type="range" min="0" max="1" step="0.001" value={t}
          onChange={e => { setT(parseFloat(e.target.value)); setPlaying(false); }} />
        <span className="t-mono sim-t">{(t * 100).toFixed(0)}%</span>
      </div>
      <div className="sim-speed">
        <span className="t-eyebrow">{T.speed_x}</span>
        {[0.5, 1, 2].map(s => (
          <button key={s} className={`speed-pill ${s === speed ? "on" : ""}`}
            onClick={() => setSpeed(s)}>{s}×</button>
        ))}
      </div>
    </div>
  );
}

/* ---------------- Bearing / range readout (large) ----------------
   The 4-mic 2m planar array has poor z-resolution — we deliberately omit
   altitude and GPS, and only show the things triangulate.py actually
   computes from TDoAs: planar position, bearing, range, and trajectory-
   derived speed. */
function BearingReadout({ drone, T, dir, speed, oodFlag }) {
  const bearing = drone
    ? (((Math.atan2(drone.x - 1, drone.y - 1) * 180 / Math.PI) + 360) % 360)
    : null;
  const range = drone ? Math.sqrt((drone.x - 1) ** 2 + (drone.y - 1) ** 2) : null;
  const stat = (label, value, unit, foot) => (
    <div className="stat-block">
      <span className="t-eyebrow stat-label">{label}</span>
      <span className="stat-value t-mono">{value}</span>
      <span className="stat-unit">{unit}</span>
      {foot && <span className="stat-foot">{foot}</span>}
    </div>
  );
  return (
    <div className="bearing-grid">
      {stat(T.bearing,   bearing != null ? Math.round(bearing).toString().padStart(3, "0") : "—", "°")}
      {stat(T.distance,  range   != null ? range.toFixed(1) : "—", "m")}
      {stat(T.speed,     drone   ? speed.toFixed(1) : "—", "m/s")}
      {stat(T.openset,   oodFlag ? T.openset_yes : T.openset_no, "", T.openset_foot)}
    </div>
  );
}

/* ---------------- Alert banner ---------------- */
function AlertBanner({ visible, T, dir, onAck, intensity = "moderate", muted, setMuted }) {
  if (!visible) return null;
  return (
    <div className={`alert-banner intensity-${intensity}`}>
      <div className="alert-pulse" />
      <div className="alert-content">
        <div className="alert-icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M10 2 L18 17 L2 17 Z" stroke="currentColor" strokeWidth="1.6" fill="rgba(255,255,255,0.08)"/>
            <line x1="10" y1="8" x2="10" y2="12" stroke="currentColor" strokeWidth="1.6"/>
            <circle cx="10" cy="14.5" r="1" fill="currentColor"/>
          </svg>
        </div>
        <div className="alert-text">
          <strong className="alert-title">{T.alert_title}</strong>
          <span className="alert-sub">{T.alert_sub}</span>
        </div>
      </div>
      <div className="alert-actions">
        <button className="alert-mute" onClick={() => setMuted(!muted)} title={muted ? T.unmute : T.mute}>
          {muted ? (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 6h2l3-3v10l-3-3H3V6z" fill="currentColor"/><line x1="11" y1="5" x2="14" y2="11" stroke="currentColor" strokeWidth="1.4"/><line x1="14" y1="5" x2="11" y2="11" stroke="currentColor" strokeWidth="1.4"/></svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 6h2l3-3v10l-3-3H3V6z" fill="currentColor"/><path d="M11 5 Q13 8 11 11" stroke="currentColor" strokeWidth="1.4" fill="none"/><path d="M13 3 Q16 8 13 13" stroke="currentColor" strokeWidth="1.4" fill="none"/></svg>
          )}
        </button>
        <button className="alert-ack" onClick={onAck}>{T.acknowledge}</button>
      </div>
    </div>
  );
}

Object.assign(window, {
  ThreatPanel, ProbabilityGauge, EventLog, MicEditor,
  CueingPanel, SimControls, BearingReadout, AlertBanner,
});
