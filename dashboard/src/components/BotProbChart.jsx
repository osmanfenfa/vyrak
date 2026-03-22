import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const CHALLENGE_LINE = 50;
const BLOCK_LINE = 20;

function percentTick(value) {
  return `${Math.round(value * 100)}%`;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #d9dee5",
        borderRadius: 8,
        padding: "8px 10px",
        fontSize: 12,
      }}
    >
      <p className="muted mono" style={{ marginBottom: 4 }}>
        Action #{label}
      </p>
      {payload.map((entry) => (
        <p key={entry.name} style={{ color: entry.color, margin: "2px 0" }}>
          {entry.name}:{" "}
          <strong>
            {entry.name === "P(bot)"
              ? `${(entry.value * 100).toFixed(1)}%`
              : entry.value.toFixed(1)}
          </strong>
        </p>
      ))}
    </div>
  );
}

export default function BotProbChart({ feed = [], limit = 40 }) {
  const data = [...feed]
    .slice(0, limit)
    .reverse()
    .map((item, idx) => ({
      idx: idx + 1,
      "P(bot)": item.bot_prob ?? 0,
      Trust: item.trust_score ?? 100,
    }));

  if (!data.length) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 30 }}>
        <div className="card-title">Bot Probability and Trust</div>
        <p className="muted">No evaluated actions yet.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Bot Probability and Trust</div>

      <ResponsiveContainer width="100%" height={230}>
        <LineChart data={data} margin={{ top: 4, right: 14, left: -22, bottom: 0 }}>
          <CartesianGrid stroke="#e4e8ee" strokeDasharray="3 3" />
          <XAxis dataKey="idx" tick={{ fontSize: 10, fill: "#667085" }} />
          <YAxis
            yAxisId="trust"
            domain={[0, 100]}
            width={36}
            tick={{ fontSize: 10, fill: "#667085" }}
          />
          <YAxis
            yAxisId="prob"
            orientation="right"
            domain={[0, 1]}
            width={38}
            tickFormatter={percentTick}
            tick={{ fontSize: 10, fill: "#667085" }}
          />

          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11 }} />

          <ReferenceLine
            yAxisId="trust"
            y={CHALLENGE_LINE}
            stroke="#b66b16"
            strokeDasharray="4 2"
          />
          <ReferenceLine
            yAxisId="trust"
            y={BLOCK_LINE}
            stroke="#b42318"
            strokeDasharray="4 2"
          />

          <Line
            yAxisId="trust"
            type="monotone"
            dataKey="Trust"
            stroke="#102338"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
          <Line
            yAxisId="prob"
            type="monotone"
            dataKey="P(bot)"
            stroke="#0f8b6f"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="muted mono" style={{ marginTop: 4 }}>
        Challenge threshold: 50 | Block threshold: 20
      </div>
    </div>
  );
}
