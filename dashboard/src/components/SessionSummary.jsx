import RiskBadge from "./RiskBadge.jsx";

function Stat({ label, value, accent }) {
  return (
    <div
      style={{
        background: "#f8fafb",
        border: "1px solid #e3e7ec",
        borderRadius: "10px",
        padding: "10px 12px",
        minWidth: 72,
        textAlign: "center",
      }}
    >
      <div
        style={{
          fontSize: 20,
          fontWeight: 700,
          color: accent || "var(--navy)",
          lineHeight: 1.1,
        }}
      >
        {value}
      </div>
      <div className="muted" style={{ fontSize: 10, marginTop: 2 }}>
        {label}
      </div>
    </div>
  );
}

export default function SessionSummary({ summary, onRestore, onEnd }) {
  if (!summary) return null;

  const metrics = summary.metrics || {};

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div className="card-title" style={{ marginBottom: 0 }}>
            Session Summary
          </div>
          <div className="mono muted">{summary.session_id?.slice(0, 16)}...</div>
        </div>
        <RiskBadge label={summary.risk_rating} />
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Stat label="Trust" value={summary.final_trust?.toFixed(1)} accent="#102338" />
        <Stat label="Actions" value={metrics.total_actions ?? 0} />
        <Stat label="Allow" value={metrics.allowed ?? 0} accent="#0f8b6f" />
        <Stat label="Challenge" value={metrics.challenged ?? 0} accent="#b66b16" />
        <Stat label="Block" value={metrics.blocked ?? 0} accent="#b42318" />
        <Stat label="Anomalies" value={metrics.anomaly_count ?? 0} accent="#b66b16" />
      </div>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <span className="mono muted">
          duration: {summary.session_duration_s?.toFixed(0)}s
        </span>
        <span className="mono muted">
          ml: {summary.ml_trained ? "active" : "untrained"}
        </span>
        <span className="mono muted">
          samples: {summary.ml_train_samples ?? 0}
        </span>
        {summary.shadow_mode && <span className="pill pill-shadow">SHADOW</span>}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        {onRestore && (
          <button className="btn-ghost btn-sm" onClick={onRestore}>
            Restore Trust
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
