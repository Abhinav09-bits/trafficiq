import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const axis = { stroke: '#80868b', fontSize: 11 }
const tip = {
  contentStyle: {
    background: '#fff', border: '1px solid #e3e6ea', borderRadius: 12,
    color: '#1f2125', fontSize: 12, boxShadow: '0 2px 8px rgba(60,64,67,.2)',
  },
}

export default function Insights({ data }) {
  if (!data) return <div className="spinner" />
  const m = data.metrics
  return (
    <div>
      <div className="kpis">
        <div className="kpi"><div className="v">{(m.level.accuracy * 100).toFixed(1)}%</div>
          <div className="l">Level Accuracy</div><div className="s">LightGBM classifier</div></div>
        <div className="kpi"><div className="v">{m.level.f1_macro}</div>
          <div className="l">F1 (macro)</div><div className="s">balanced score</div></div>
        <div className="kpi"><div className="v">{m.duration.MAE_hours} h</div>
          <div className="l">Block-time MAE</div><div className="s">regression error</div></div>
        <div className="kpi"><div className="v">{data.total_events.toLocaleString()}</div>
          <div className="l">Events analysed</div><div className="s">real govt data</div></div>
      </div>

      <div className="card">
        <h3>⏰ When disruptions happen (by hour)</h3>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={data.by_hour}>
            <XAxis dataKey="hour" {...axis} />
            <YAxis {...axis} />
            <Tooltip {...tip} cursor={{ fill: '#f1f3f6' }} />
            <Bar dataKey="events" radius={[4, 4, 0, 0]}>
              {data.by_hour.map((d, i) => (
                <Cell key={i} fill={`hsl(${Math.max(8, 45 - d.events / 14)}, 90%, 52%)`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>🛣️ Worst corridors</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.corridors} layout="vertical" margin={{ left: 20 }}>
            <XAxis type="number" {...axis} />
            <YAxis type="category" dataKey="corridor" width={95} {...axis} />
            <Tooltip {...tip} cursor={{ fill: '#f1f3f6' }} />
            <Bar dataKey="events" radius={[0, 4, 4, 0]} fill="#ea4335" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>💥 Highest-impact causes</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.causes} layout="vertical" margin={{ left: 20 }}>
            <XAxis type="number" {...axis} />
            <YAxis type="category" dataKey="cause" width={95} {...axis} />
            <Tooltip {...tip} cursor={{ fill: '#f1f3f6' }} />
            <Bar dataKey="avg_impact" radius={[0, 4, 4, 0]} fill="#f9ab00" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>🟢 Calmest zones (best diversion targets)</h3>
        <table className="calm">
          <thead><tr><th>Zone</th><th>Events</th><th>Avg impact</th><th>Stability</th></tr></thead>
          <tbody>
            {data.calm_zones.map((z, i) => (
              <tr key={i}><td>{z.zone}</td><td>{z.events}</td>
                <td>{z.avg_impact}</td><td>{z.instability}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
