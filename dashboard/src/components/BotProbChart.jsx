/**
 * BotProbChart — scrolling line chart of P(bot) and Trust over time.
 * Shows the last N action evaluations.
 *
 * Props:
 *   feed   {Array}  array of DecisionResponse objects (newest first)
 *   limit  {number} max data points to show (default 40)
 */

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";

const CHALLENGE_LINE = 50;
const BLOCK_LINE     = 20;

function formatTick(val) {
  return `${Math.round(val * 100)}%`;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "white", border: "1px solid #D3D1C7",
      borderRadius: 8, padding: "8px 12px", fontSize: 12,
    }}>
      <p style={{ marginBottom: 4, color: "#5F5E5A" }}>Action #{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color, margin: "2px 0" }}>
          {p.name}: <strong>
            {p.name === "P(bot)" ? `${(p.value * 100).toFixed(1)}%` : p.value.toFixed(1)}
          </strong>
        </p>
      ))}
    </div>
  );
}

export default function BotProbChart({ feed = [], limit = 40 }) {
  // Feed is newest-first — reverse and slice for chronological chart
  const data = [...feed]
    .slice(0, limit)
    .reverse()
    .map((item, i) => ({
      idx:       i + 1,
      "P(bot)":  item.bot_prob ?? 0,
      "Trust":   item.trust_score ?? 100,
      decision:  item.decision,
    }));

  if (!data.length) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 32 }}>
        <div className="card-title">P(bot) &amp; Trust Over Time</div>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
          No actions evaluated yet — submit an action to begin.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">P(bot) &amp; Trust Over Time</div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 16, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E3DC" />

          <XAxis
            dataKey="idx"
            tick={{ fontSize: 10, fill: "#888780" }}
            label={{ value: "Action #", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#888780" }}
          />

          <YAxis
            yAxisId="trust"
            domain={[0, 100]}
            tick={{ fontSize: 10, fill: "#888780" }}
            width={36}
          />
          <YAxis
            yAxisId="prob"
            orientation="right"
            domain={[0, 1]}
            tickFormatter={formatTick}
            tick={{ fontSize: 10, fill: "#888780" }}
            width={38}
          />

          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11 }} />

          {/* Trust threshold lines */}
          <ReferenceLine yAxisId="trust" y={CHALLENGE_LINE}
            stroke="#BA7517" strokeDasharray="4 2" strokeWidth={1} />
          <ReferenceLine yAxisId="trust" y={BLOCK_LINE}
            stroke="#A32D2D" strokeDasharray="4 2" strokeWidth={1} />

          <Line
            yAxisId="trust"
            type="monotone"
            dataKey="Trust"
            stroke="#0D1B2A"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
          <Line
            yAxisId="prob"
            type="monotone"
            dataKey="P(bot)"
            stroke="#1D9E75"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: 12, marginTop: 4, fontSize: 10, color: "#888780" }}>
        <span style={{ color: "#BA7517" }}>— Challenge threshold (50)</span>
        <span style={{ color: "#A32D2D" }}>— Block threshold (20)</span>
      </div>
    </div>
  );
}
