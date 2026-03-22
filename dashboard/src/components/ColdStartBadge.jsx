/**
 * ColdStartBadge — shows a ramp-up indicator when a session is in cold start.
 *
 * Props:
 *   rampActive   {boolean}
 *   vectorCount  {number}   actions recorded so far
 *   rampActions  {number}   total actions needed (default 50)
 */

export default function ColdStartBadge({ rampActive, vectorCount = 0, rampActions = 50 }) {
  if (!rampActive) return null;

  const pct = Math.min(100, Math.round((vectorCount / rampActions) * 100));

  return (
    <div style={{
      background: "#EDE9FE",
      border: "1px solid #C4B5FD",
      borderRadius: "var(--radius-md)",
      padding: "6px 10px",
      fontSize: "12px",
      color: "#4C1D95",
      display: "flex",
      flexDirection: "column",
      gap: "4px",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 600 }}>
        <span>⚡ Cold Start Ramp-Up</span>
        <span>{pct}%</span>
      </div>

      {/* Progress bar */}
      <div style={{
        height: "4px",
        background: "#DDD6FE",
        borderRadius: "99px",
        overflow: "hidden",
      }}>
        <div style={{
          height: "100%",
          width: `${pct}%`,
          background: "#7C3AED",
          borderRadius: "99px",
          transition: "width 0.4s ease",
        }} />
      </div>

      <span style={{ color: "#6D28D9", fontSize: "11px" }}>
        {vectorCount} / {rampActions} actions — cosine check disabled, looser ML threshold
      </span>
    </div>
  );
}
