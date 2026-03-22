/**
 * SessionSummary — compact stats panel showing session metrics.
 *
 * Props:
 *   summary   {object}  from GET /session/{id}/summary
 *   onRestore {Function}
 *   onEnd     {Function}
 */

import RiskBadge from "./RiskBadge.jsx";

function Stat({ label, value, accent }) {
  return (
    <div style={{
      background: "#F8F7F2",
      borderRadius: "var(--radius-md)",
      padding: "10px 14px",
      minWidth: 72,
      textAlign: "center",
    }}>
      <div style={{
        fontSize: 22, fontWeight: 700,
        color: accent || "var(--navy)",
        lineHeight: 1.1,
      }}>
        {value}
      </div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
        {label}
      </div>
    </div>
  );
}

export default function SessionSummary({ summary, onRestore, onEnd }) {
  if (!summary) return null;

  const m = summary.metrics || {};

  return (
    <div className="card flex-col" style={{ gap: 12 }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div className="card-title" style={{ marginBottom: 0 }}>Session Summary</div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {summary.session_id?.slice(0, 16)}…
          </div>
        </div>
        <RiskBadge label={summary.risk_rating} />
      </div>

      {/* Stats grid */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Stat label="Trust"     value={summary.final_trust?.toFixed(1)} accent="var(--navy)" />
        <Stat label="Actions"   value={m.total_actions ?? 0} />
        <Stat label="ALLOW"     value={m.allowed        ?? 0} accent="#1D9E75" />
        <Stat label="CHALLENGE" value={m.challenged      ?? 0} accent="#BA7517" />
        <Stat label="BLOCK"     value={m.blocked         ?? 0} accent="#A32D2D" />
        <Stat label="Anomalies" value={m.anomaly_count   ?? 0} accent={m.anomaly_count > 0 ? "#BA7517" : undefined} />
      </div>

      {/* Extra info */}
      <div style={{ display: "flex", gap: 12, fontSize: 11, color: "var(--text-muted)" }}>
        <span>⏱ {summary.session_duration_s?.toFixed(0)}s</span>
        <span>🤖 ML {summary.ml_trained ? "active" : "untrained"}</span>
        <span>📦 {summary.ml_train_samples ?? 0} training samples</span>
        {summary.shadow_mode && (
          <span className="pill pill-shadow" style={{ fontSize: 10 }}>SHADOW</span>
        )}
      </div>

      {/* Actions */}
      <div style={{ display: "flex", gap: 8 }}>
        {onRestore && (
          <button className="btn-ghost btn-sm" onClick={onRestore}>
            ↑ Restore Trust
          </button>
        )}
        {onEnd && (
          <button className="btn-danger btn-sm" onClick={onEnd}>
            End Session
          </button>
        )}
      </div>
    </div>
  );
}
