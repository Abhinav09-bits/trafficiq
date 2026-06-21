import { useEffect, useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import HeatMap from './HeatMap.jsx'
import Insights from './Insights.jsx'

const LEVEL_COLOR = { 1: '#34a853', 2: '#f9ab00', 3: '#ea8600', 4: '#ea4335' }
const LEVEL_NAME = { 1: 'Minor', 2: 'Building up', 3: 'Serious', 4: 'Critical' }
const Icon = ({ n }) => <span className="mi">{n}</span>

export default function App() {
  const [drawer, setDrawer] = useState(true)
  const [sheet, setSheet] = useState(false)
  const [q, setQ] = useState('')
  const [focus, setFocus] = useState(false)

  const [opts, setOpts] = useState(null)
  const [presets, setPresets] = useState([])
  const [event, setEvent] = useState({
    etype: 'unplanned', cause: 'public_event', corridor: 'Non-corridor',
    zone: 'North Zone 2', veh: 'bmtc_bus', priority: 'High', hour: 21, dow: 5, closure: true,
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const [levels, setLevels] = useState([1, 2, 3, 4])
  const [points, setPoints] = useState([])
  const [insights, setInsights] = useState(null)
  const mapRef = useRef(null)

  useEffect(() => {
    fetch('/api/options').then(r => r.json()).then(setOpts)
    fetch('/api/presets').then(r => r.json()).then(setPresets)
    fetch('/api/insights').then(r => r.json()).then(setInsights)
  }, [])

  useEffect(() => {
    fetch(`/api/heatmap?levels=${levels.join(',')}`).then(r => r.json())
      .then(d => setPoints(d.points))
  }, [levels])

  const predict = async (ev) => {
    setDrawer(false)            // collapse panel to reveal result on the map
    setLoading(true); setResult({})
    const r = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ev || event),
    })
    setResult(await r.json()); setLoading(false)
  }
  const applyPreset = (p) => { setEvent(p.event); predict(p.event) }
  const set = (k, v) => setEvent(e => ({ ...e, [k]: v }))

  // search suggestions from causes + corridors
  const suggestions = useMemo(() => {
    if (!opts || !q.trim()) return []
    const t = q.toLowerCase()
    const causes = opts.causes.filter(c => c.includes(t)).map(c => ({ t: 'cause', v: c }))
    const cors = opts.corridors.filter(c => c.toLowerCase().includes(t)).map(c => ({ t: 'corridor', v: c }))
    return [...causes, ...cors].slice(0, 6)
  }, [q, opts])

  const pickSuggest = (s) => {
    setEvent(e => ({ ...e, [s.t === 'cause' ? 'cause' : 'corridor']: s.v }))
    setQ(''); setFocus(false); setDrawer(true)
  }

  const zoom = (d) => { const m = mapRef.current; if (m) m.setZoom(m.getZoom() + d) }
  const recenter = () => { const m = mapRef.current; if (m) m.flyTo([12.97, 77.59], 11) }

  const lvl = result && result.level

  return (
    <div className={`app ${drawer ? 'drawer-open' : ''}`}>
      {/* FULL-SCREEN MAP */}
      <div className="mapfull">
        <HeatMap points={points} onReady={(m) => { mapRef.current = m }} />
      </div>

      <div className="overlay">
        {/* SEARCH BAR */}
        <div className="searchbar">
          <button className="icon-btn" onClick={() => setDrawer(d => !d)} title="Menu">
            <Icon n="menu" />
          </button>
          <input value={q} placeholder="Search a cause, corridor or zone…"
            onChange={e => { setQ(e.target.value); setFocus(true) }}
            onFocus={() => setFocus(true)}
            onBlur={() => setTimeout(() => setFocus(false), 150)} />
          <button className="search-pill" onClick={() => { setDrawer(true) }}>
            <Icon n="tune" /> Plan event
          </button>
          {focus && suggestions.length > 0 && (
            <div className="suggest">
              {suggestions.map((s, i) => (
                <div className="row" key={i} onMouseDown={() => pickSuggest(s)}>
                  <Icon n={s.t === 'cause' ? 'event' : 'route'} />
                  {s.v.replace(/_/g, ' ')}<small>{s.t}</small>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* STAT CHIPS */}
        {insights && (
          <div className="stats">
            <div className="statchip"><div className="v">{insights.total_events.toLocaleString()}</div>
              <div className="l">Events</div></div>
            <div className="statchip"><div className="v">{(insights.metrics.level.accuracy * 100).toFixed(0)}%</div>
              <div className="l">Accuracy</div></div>
            <div className="statchip"><div className="v" style={{ color: 'var(--red)' }}>
              {insights.corridors[0]?.events}</div>
              <div className="l">Top corridor</div></div>
          </div>
        )}

        {/* ANALYTICS FAB */}
        <div className="fab-analytics">
          <button className="fab" onClick={() => setSheet(true)}>
            <Icon n="insights" /> Analytics
          </button>
        </div>

        {/* LEFT DRAWER — EVENT PLANNING */}
        <div className={`drawer ${drawer ? '' : 'closed'}`}>
          <div className="drawer-head">
            <div className="logo"><Icon n="traffic" /></div>
            <div>
              <h1>TrafficIQ</h1>
              <p>Event-Driven Traffic Intelligence · Bengaluru</p>
            </div>
            <button className="icon-btn close" onClick={() => setDrawer(false)}>
              <Icon n="close" />
            </button>
          </div>
          <div className="drawer-body">
            <div className="section-label">Quick scenarios</div>
            <div className="preset-grid">
              {presets.map((p, i) => (
                <motion.button className="preset" key={i} whileTap={{ scale: .96 }}
                  onClick={() => applyPreset(p)}>
                  <span className="ico">{p.icon}</span>{p.name}
                </motion.button>
              ))}
            </div>

            <div className="section-label">Plan an event</div>
            {opts && <>
              <Sel label="Event type" v={event.etype} set={v => set('etype', v)} opts={['unplanned', 'planned']} />
              <Sel label="Cause" v={event.cause} set={v => set('cause', v)} opts={opts.causes} />
              <Sel label="Corridor / road" v={event.corridor} set={v => set('corridor', v)} opts={opts.corridors} />
              <Sel label="Zone" v={event.zone} set={v => set('zone', v)} opts={opts.zones} />
              <Sel label="Vehicle type" v={event.veh} set={v => set('veh', v)} opts={opts.vehs} />
              <div className="field"><label>Priority</label>
                <div className="seg">{['High', 'Low'].map(p => (
                  <button key={p} className={event.priority === p ? 'on' : ''} onClick={() => set('priority', p)}>{p}</button>
                ))}</div>
              </div>
              <div className="two">
                <Rng label="Hour" v={event.hour} min={0} max={23} set={v => set('hour', v)} />
                <Rng label="Day (0=Mon)" v={event.dow} min={0} max={6} set={v => set('dow', v)} />
              </div>
              <label className="switch">Requires road closure?
                <input type="checkbox" checked={event.closure} onChange={e => set('closure', e.target.checked)} />
              </label>
              <button className="btn-primary" onClick={() => predict()}>
                <Icon n="auto_awesome" /> Predict &amp; Plan
              </button>
            </>}
          </div>
        </div>

        {/* RESULT CARD */}
        <AnimatePresence>
          {result && (
            <motion.div className={`resultcard ${drawer ? '' : 'full-left'}`}
              initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 24 }} transition={{ duration: .3 }}>
              {loading ? <div className="rc-body"><div className="spinner" /></div> : (
                <>
                  <div className="rc-banner" style={{
                    background: `linear-gradient(135deg, ${LEVEL_COLOR[lvl]}, ${LEVEL_COLOR[lvl]}cc)`,
                  }}>
                    <button className="icon-btn rc-close" onClick={() => setResult(null)}><Icon n="close" /></button>
                    <div className="lvl">Congestion Level {lvl}</div>
                    <h2>{LEVEL_NAME[lvl]}</h2>
                    <div className="sub">{result.cause?.replace(/_/g, ' ')} on {result.corridor}</div>
                  </div>
                  <div className="rc-body">
                    <div className="rc-tiles">
                      <Tile lab="Block time" val={`${result.duration_hours}h`} sub="predicted" />
                      <Tile lab="Veh-hours" val={result.vehicle_hours.toLocaleString()} sub="lost" />
                      <Tile lab="Cooldown" val={`${result.cooldown_min}m`} sub="hold flow" />
                    </div>
                    <div className="money">💰 <b>₹{result.rupees.toLocaleString()}</b> estimated economic impact</div>
                    <div className="plan-h">Recommended action plan</div>
                    {result.barricades.map((b, i) => (
                      <div className="barr" key={i}><Icon n="block" />{b.point} <b>{b.officers} officers</b></div>
                    ))}
                    <div className="divert">🔀 Divert to: {result.divert_to.join(', ')} • {result.total_officers} officers total</div>
                    <details className="why"><summary>Why this plan?</summary>
                      <ul>{result.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
                    </details>
                  </div>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* LEVEL FILTER */}
        <div className="botbar">
          {[1, 2, 3, 4].map(l => (
            <button key={l} className={`lvlchip ${levels.includes(l) ? 'on' : ''}`}
              onClick={() => setLevels(levels.includes(l) ? levels.filter(x => x !== l) : [...levels, l].sort())}>
              <span className="dot" style={{ background: LEVEL_COLOR[l] }} />Level {l}
            </button>
          ))}
        </div>

        {/* ZOOM CONTROL */}
        <div className="zoomctl">
          <button onClick={() => zoom(1)}><Icon n="add" /></button>
          <button onClick={() => zoom(-1)}><Icon n="remove" /></button>
        </div>

        {/* LEGEND */}
        <div className="legend">
          <div className="t">Traffic density</div>
          <div className="bar" />
          <div className="scale"><span>Low</span><span>Moderate</span><span>Severe</span></div>
        </div>
      </div>

      {/* ANALYTICS SHEET */}
      <div className={`scrim ${sheet ? 'show' : ''}`} onClick={() => setSheet(false)} />
      <div className={`sheet ${sheet ? 'open' : ''}`}>
        <div className="sheet-head">
          <Icon n="insights" />
          <h2>Analytics &amp; Model</h2>
          <button className="icon-btn close" onClick={() => setSheet(false)}><Icon n="close" /></button>
        </div>
        <div className="sheet-body"><Insights data={insights} /></div>
      </div>
    </div>
  )
}

function Sel({ label, v, set, opts }) {
  return (
    <div className="field"><label>{label}</label>
      <select value={v} onChange={e => set(e.target.value)}>
        {opts.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  )
}
function Rng({ label, v, min, max, set }) {
  return (
    <div className="field"><label>{label}: <span className="rngv">{v}</span></label>
      <input type="range" min={min} max={max} value={v} onChange={e => set(Number(e.target.value))} />
    </div>
  )
}
function Tile({ lab, val, sub }) {
  return <div className="rc-tile"><div className="lab">{lab}</div>
    <div className="val">{val}</div><div className="sub">{sub}</div></div>
}
