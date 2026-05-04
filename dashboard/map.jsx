// Saamid overhead map
// Renders mic array, drone position, fading trail, range rings, grid, bearing ticks
// Map style: 'hybrid' (default), 'radar' (rings only), 'grid' (grid only), 'sat' (terrain blob)

// React hooks accessed via React.* to avoid global-scope name clashes across Babel files

function SaamidMap({
  mics,
  drone,           // {x, y, alt, prob, type} or null
  trail,           // array of {x, y, alpha}
  mapStyle,        // 'hybrid' | 'radar' | 'grid' | 'sat'
  threatType,      // 'clear' | 'detected' | 'decoy' | 'unknown'
  T,               // i18n
  dir,             // 'rtl' | 'ltr'
  onMicMove,
  showMicLabels,
  showCompass,
  highlightedMicId,
}) {
  // World bounds: -80..+80 metres in x and y, mic array centred near (1,1)
  const W = 100; // half-width metres
  const cx = 1, cy = 1; // map centre in metres (mic array centroid)
  const VIEW = 800; // px viewBox
  const scale = VIEW / (W * 2);
  const wx = (mx) => (mx - cx + W) * scale;
  const wy = (my) => VIEW - (my - cy + W) * scale; // y up

  const ringDistances = [10, 25, 50, 75, 100]; // metres

  // bearing & range from origin (mic centroid)
  const bearingDeg = drone
    ? (Math.atan2(drone.x - cx, drone.y - cy) * 180 / Math.PI + 360) % 360
    : null;
  const range = drone
    ? Math.sqrt((drone.x - cx) ** 2 + (drone.y - cy) ** 2)
    : null;

  // colors via CSS vars
  const cs = useThemeVars();

  return (
    <div className="map-wrap" data-map-style={mapStyle}>
      <svg
        className="map-svg"
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          {/* radial vignette */}
          <radialGradient id="mapVignette" cx="50%" cy="50%" r="65%">
            <stop offset="0%" stopColor={cs.mapInk} stopOpacity="0" />
            <stop offset="80%" stopColor={cs.mapInk} stopOpacity="0" />
            <stop offset="100%" stopColor={cs.mapInk} stopOpacity="0.18" />
          </radialGradient>

          <radialGradient id="droneGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(214,48,48,0.55)" />
            <stop offset="60%" stopColor="rgba(214,48,48,0.10)" />
            <stop offset="100%" stopColor="rgba(214,48,48,0)" />
          </radialGradient>

          <radialGradient id="protectedZone" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={cs.accent} stopOpacity="0.08" />
            <stop offset="80%" stopColor={cs.accent} stopOpacity="0.02" />
            <stop offset="100%" stopColor={cs.accent} stopOpacity="0" />
          </radialGradient>

          {/* satellite-style terrain blob */}
          <radialGradient id="terrain" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor={cs.accent} stopOpacity="0.18" />
            <stop offset="100%" stopColor={cs.accent} stopOpacity="0.02" />
          </radialGradient>

          <pattern id="dotGrid" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="0.7" fill={cs.gridLine} />
          </pattern>
        </defs>

        {/* background */}
        <rect x="0" y="0" width={VIEW} height={VIEW} fill={cs.mapBg} />

        {/* satellite-style terrain (only for sat) */}
        {mapStyle === "sat" && (
          <g>
            <circle cx={VIEW/2} cy={VIEW/2} r={VIEW*0.46} fill="url(#terrain)" />
            <path
              d={`M ${VIEW*0.18} ${VIEW*0.22} Q ${VIEW*0.4} ${VIEW*0.35}, ${VIEW*0.55} ${VIEW*0.18} T ${VIEW*0.85} ${VIEW*0.32}`}
              fill="none" stroke={cs.accent} strokeOpacity="0.18" strokeWidth="1"
            />
            <path
              d={`M ${VIEW*0.12} ${VIEW*0.78} Q ${VIEW*0.32} ${VIEW*0.65}, ${VIEW*0.5} ${VIEW*0.74} T ${VIEW*0.92} ${VIEW*0.6}`}
              fill="none" stroke={cs.accent} strokeOpacity="0.14" strokeWidth="1"
            />
          </g>
        )}

        {/* protected zone (always) */}
        <circle cx={wx(cx)} cy={wy(cy)} r={ringDistances[2] * scale} fill="url(#protectedZone)" />

        {/* GRID (hybrid + grid) */}
        {(mapStyle === "hybrid" || mapStyle === "grid") && (
          <g className="grid">
            {/* dotted base */}
            <rect x="0" y="0" width={VIEW} height={VIEW} fill="url(#dotGrid)" opacity="0.6" />
            {/* every-10m lines */}
            {Array.from({ length: 21 }).map((_, i) => {
              const v = -100 + i * 10;
              return (
                <g key={i}>
                  <line x1={wx(v + cx)} y1="0" x2={wx(v + cx)} y2={VIEW}
                    stroke={cs.gridLine} strokeWidth={v === 0 ? 1 : 0.5} />
                  <line x1="0" y1={wy(v + cy)} x2={VIEW} y2={wy(v + cy)}
                    stroke={cs.gridLine} strokeWidth={v === 0 ? 1 : 0.5} />
                </g>
              );
            })}
            {/* axis labels every 25m */}
            {[-50, -25, 25, 50].map((v) => (
              <g key={v}>
                <text x={wx(v + cx)} y={wy(cy) - 6}
                  fill={cs.textFaint} fontSize="9" textAnchor="middle"
                  fontFamily="var(--font-mono)">
                  {v}
                </text>
                <text x={wx(cx) + 6} y={wy(v + cy) + 3}
                  fill={cs.textFaint} fontSize="9"
                  fontFamily="var(--font-mono)">
                  {v}
                </text>
              </g>
            ))}
          </g>
        )}

        {/* RANGE RINGS (hybrid + radar + sat) */}
        {(mapStyle === "hybrid" || mapStyle === "radar" || mapStyle === "sat") && (
          <g className="rings">
            {ringDistances.map((d, i) => (
              <circle
                key={d}
                cx={wx(cx)} cy={wy(cy)}
                r={d * scale}
                fill="none"
                stroke={cs.gridLineStrong}
                strokeWidth={i === 2 ? 1 : 0.6}
                strokeDasharray={i === 2 ? "0" : "2 4"}
              />
            ))}
            {ringDistances.map((d) => (
              <text key={`l-${d}`}
                x={wx(cx) + d * scale + 4}
                y={wy(cy) - 3}
                fill={cs.textFaint}
                fontSize="9"
                fontFamily="var(--font-mono)">
                {d}m
              </text>
            ))}

            {/* bearing tick marks every 10°, labels every 30° */}
            {Array.from({ length: 36 }).map((_, i) => {
              const ang = i * 10;
              const rad = (ang - 90) * Math.PI / 180;
              const r1 = ringDistances[ringDistances.length - 1] * scale;
              const tickLen = ang % 30 === 0 ? 8 : 4;
              return (
                <line key={ang}
                  x1={wx(cx) + Math.cos(rad) * r1}
                  y1={wy(cy) + Math.sin(rad) * r1}
                  x2={wx(cx) + Math.cos(rad) * (r1 - tickLen)}
                  y2={wy(cy) + Math.sin(rad) * (r1 - tickLen)}
                  stroke={cs.gridLineStrong}
                  strokeWidth="0.7"
                />
              );
            })}
            {[0, 90, 180, 270].map((ang) => {
              const rad = (ang - 90) * Math.PI / 180;
              const r1 = ringDistances[ringDistances.length - 1] * scale + 14;
              const labels = { 0: "N", 90: "E", 180: "S", 270: "W" };
              return (
                <text key={ang}
                  x={wx(cx) + Math.cos(rad) * r1}
                  y={wy(cy) + Math.sin(rad) * r1 + 3}
                  fill={cs.text}
                  fontSize="11"
                  fontWeight="700"
                  textAnchor="middle"
                  fontFamily="var(--font-mono)">
                  {labels[ang]}
                </text>
              );
            })}

            {/* sweep beam if detecting */}
            {threatType !== "clear" && drone && (
              <g className="sweep">
                <line
                  x1={wx(cx)} y1={wy(cy)}
                  x2={wx(drone.x)} y2={wy(drone.y)}
                  stroke="rgba(214,48,48,0.35)"
                  strokeWidth="1"
                  strokeDasharray="3 4"
                />
              </g>
            )}
          </g>
        )}

        {/* vignette overlay */}
        <rect x="0" y="0" width={VIEW} height={VIEW} fill="url(#mapVignette)" pointerEvents="none" />

        {/* Trail (fading) */}
        <g className="trail">
          {trail.map((p, i) => (
            <circle key={i}
              cx={wx(p.x)} cy={wy(p.y)} r={3.6 - i * 0.18}
              fill="rgba(214,48,48,1)"
              opacity={p.alpha * 0.7}
            />
          ))}
          {trail.length > 1 && (
            <polyline
              points={trail.map((p) => `${wx(p.x)},${wy(p.y)}`).join(" ")}
              fill="none"
              stroke="rgba(214,48,48,0.45)"
              strokeWidth="1.2"
            />
          )}
        </g>

        {/* Drone marker */}
        {drone && (
          <g className="drone-marker" transform={`translate(${wx(drone.x)} ${wy(drone.y)})`}>
            <circle r="34" fill="url(#droneGlow)">
              <animate attributeName="r" values="22;38;22" dur="1.6s" repeatCount="indefinite" />
            </circle>
            <circle r="14" fill="none" stroke="rgba(214,48,48,0.6)" strokeWidth="1">
              <animate attributeName="r" values="14;28;14" dur="1.6s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.8;0;0.8" dur="1.6s" repeatCount="indefinite" />
            </circle>
            <circle r="6" fill="var(--alert)" stroke="var(--cream-50)" strokeWidth="1.5" />
            <circle r="2" fill="var(--cream-50)" />
            {/* crosshair ticks */}
            {[0, 90, 180, 270].map((a) => {
              const rad = a * Math.PI / 180;
              return (
                <line key={a}
                  x1={Math.cos(rad) * 10}
                  y1={Math.sin(rad) * 10}
                  x2={Math.cos(rad) * 16}
                  y2={Math.sin(rad) * 16}
                  stroke="var(--alert)" strokeWidth="1.2" />
              );
            })}
          </g>
        )}

        {/* Mic array — rendered as a SINGLE grouped marker at the centroid.
            The 4-mic 2m square would collapse to one pixel at this scale, and
            the main map's job is showing drone position relative to the array,
            not inter-mic geometry. Per-mic labels live in the inset only. */}
        <g className="mic-group">
          {(() => {
            const ax = mics.reduce((s, m) => s + m.x, 0) / Math.max(mics.length, 1);
            const ay = mics.reduce((s, m) => s + m.y, 0) / Math.max(mics.length, 1);
            const cxp = wx(ax), cyp = wy(ay);
            const isHi = highlightedMicId != null;
            return (
              <g transform={`translate(${cxp} ${cyp})`}>
                {isHi && (
                  <circle r="18" fill="none" stroke={cs.accent2} strokeWidth="1.5" opacity="0.85">
                    <animate attributeName="r" values="14;26;14" dur="1.4s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.85;0;0.85" dur="1.4s" repeatCount="indefinite" />
                  </circle>
                )}
                {/* diamond glyph: 4 mics → diamond shape; rotated square */}
                <rect x="-7" y="-7" width="14" height="14"
                  transform="rotate(45)"
                  fill={cs.bgElev} stroke={cs.accent} strokeWidth="1.6" />
                <rect x="-3" y="-3" width="6" height="6"
                  transform="rotate(45)"
                  fill={cs.accent} />
              </g>
            );
          })()}
        </g>

        {/* Mic-array zoom inset */}
        <MicArrayInset mics={mics} cs={cs} dir={dir} T={T} />
      </svg>

      {/* Compass overlay (top-end corner) */}
      {showCompass !== false && (
        <div className={"map-compass " + (dir === "rtl" ? "left" : "right")}>
          <div className="compass-ring">
            <div className="compass-needle" style={{ transform: `rotate(${bearingDeg ?? 0}deg)` }} />
            <span className="compass-n">N</span>
          </div>
          <div className="compass-readout">
            <div>
              <span className="t-eyebrow">{T.bearing}</span>
              <strong className="t-mono">{bearingDeg != null ? Math.round(bearingDeg) : "—"}°</strong>
            </div>
            <div>
              <span className="t-eyebrow">{T.distance}</span>
              <strong className="t-mono">{range != null ? range.toFixed(1) : "—"} {T.range_m}</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MicArrayInset({ mics, cs, dir, T }) {
  // Tiny inset showing the mic array enlarged (since it's only ~2m, hard to see at world scale)
  // Inset is bigger so 2m geometry is legible; pad symmetrically and use
  // the largest of (extent, 2m) so a single-mic edit doesn't blow up scale.
  const PAD = 0.6;
  const xs = mics.map(m => m.x), ys = mics.map(m => m.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const ext = Math.max(
    Math.max(...xs) - Math.min(...xs),
    Math.max(...ys) - Math.min(...ys),
    2.0,
  );
  const half = ext / 2 + PAD;
  const minX = cx - half, maxX = cx + half;
  const minY = cy - half, maxY = cy + half;
  const w = maxX - minX, h = maxY - minY;
  const SIZE = 180;
  const s = SIZE / Math.max(w, h);
  const tx = (m) => (m.x - minX) * s;
  const ty = (m) => SIZE - (m.y - minY) * s;
  // pin to bottom-end corner of the SVG viewBox (800)
  const x = dir === "rtl" ? 16 : 800 - SIZE - 16;
  const y = 800 - SIZE - 16;

  // Quadrant-aware label offset so labels don't overlap the marker glyph
  // or its neighbours: NE label sits NE of marker, etc.
  const labelOffset = (m) => {
    const dx = m.x - cx, dy = m.y - cy;
    const ox = dx >= 0 ? 9 : -9;
    const oy = dy >= 0 ? -9 : 13;
    const anchor = dx >= 0 ? "start" : "end";
    return { ox, oy, anchor };
  };

  const insetTitle = (T && T.inset_title) || "MIC ARRAY · 1:1m";

  // bounding 2m square outline (anchored at min corner of the actual array)
  const aMinX = Math.min(...xs), aMinY = Math.min(...ys);
  const aMaxX = Math.max(...xs), aMaxY = Math.max(...ys);
  const boxX1 = (aMinX - minX) * s, boxX2 = (aMaxX - minX) * s;
  const boxY1 = SIZE - (aMaxY - minY) * s, boxY2 = SIZE - (aMinY - minY) * s;

  return (
    <g transform={`translate(${x} ${y})`}>
      <rect width={SIZE} height={SIZE} rx="6"
        fill={cs.bgElev} stroke={cs.surfaceLineStrong} strokeWidth="0.6"
        opacity="0.96" />
      <text x={dir === "rtl" ? SIZE - 10 : 10} y={16}
        textAnchor={dir === "rtl" ? "end" : "start"}
        fill={cs.textMuted} fontSize="9"
        letterSpacing={dir === "rtl" ? 0 : 1.5}
        fontFamily="var(--font-sans)" fontWeight="700">
        {insetTitle}
      </text>

      {/* bounding 2m square outline */}
      <rect
        x={boxX1} y={boxY1}
        width={boxX2 - boxX1} height={boxY2 - boxY1}
        fill="none" stroke={cs.accent} strokeWidth="0.9"
        strokeDasharray="3 3" opacity="0.45" />

      {/* dimension labels on the bounding box */}
      <text x={(boxX1 + boxX2) / 2} y={boxY1 - 4}
        textAnchor="middle" fill={cs.textFaint} fontSize="8"
        fontFamily="var(--font-mono)">
        {(aMaxX - aMinX).toFixed(1)}m
      </text>
      <text x={boxX2 + 4} y={(boxY1 + boxY2) / 2 + 3}
        fill={cs.textFaint} fontSize="8"
        fontFamily="var(--font-mono)">
        {(aMaxY - aMinY).toFixed(1)}m
      </text>

      {mics.map((m) => {
        const lo = labelOffset(m);
        return (
          <g key={m.id} transform={`translate(${tx(m)} ${ty(m)})`}>
            <circle r="5.5" fill={cs.bgElev} stroke={cs.accent} strokeWidth="1.4" />
            <circle r="2.6" fill={cs.accent} />
            <text x={lo.ox} y={lo.oy} textAnchor={lo.anchor}
              fill={cs.text} fontSize="9" fontWeight="700"
              fontFamily="var(--font-mono)">
              M{m.id}
            </text>
          </g>
        );
      })}

      {/* scale bar */}
      <line x1="10" y1={SIZE - 10} x2={10 + s * 1} y2={SIZE - 10} stroke={cs.text} strokeWidth="1" />
      <line x1="10" y1={SIZE - 13} x2="10" y2={SIZE - 7} stroke={cs.text} strokeWidth="1" />
      <line x1={10 + s} y1={SIZE - 13} x2={10 + s} y2={SIZE - 7} stroke={cs.text} strokeWidth="1" />
      <text x={10 + s/2} y={SIZE - 14} textAnchor="middle"
        fill={cs.textMuted} fontSize="8"
        fontFamily="var(--font-mono)">1m</text>
    </g>
  );
}

// helper to read CSS vars at render time
function useThemeVars() {
  const [vars, setVars] = React.useState(() => readVars());
  React.useEffect(() => {
    const mo = new MutationObserver(() => setVars(readVars()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => mo.disconnect();
  }, []);
  return vars;
}
function readVars() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim() || "#000";
  return {
    mapBg: v("--map-bg"),
    mapInk: v("--map-ink"),
    bg: v("--bg"),
    bgElev: v("--bg-elev"),
    text: v("--text"),
    textMuted: v("--text-muted"),
    textFaint: v("--text-faint"),
    accent: v("--accent"),
    accent2: v("--accent-2"),
    gridLine: v("--grid-line"),
    gridLineStrong: v("--grid-line-strong"),
    surfaceLine: v("--surface-line"),
    surfaceLineStrong: v("--surface-line-strong"),
    charcoal: v("--charcoal-900"),
  };
}

Object.assign(window, { SaamidMap, useThemeVars });
