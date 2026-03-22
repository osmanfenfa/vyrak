import { useState } from "react";
import RiskBadge from "./RiskBadge.jsx";

const ACTION_TYPES = [
  "login",
  "read",
  "write",
  "delete",
  "search",
  "export",
  "config",
  "admin",
];

function decisionAccent(decision) {
  if (decision === "ALLOW") return "#0f8b6f";
  if (decision === "CHALLENGE") return "#b66b16";
  if (decision === "BLOCK") return "#b42318";
  return "#98a2b3";
}

function FeedRow({ item }) {
  const [expanded, setExpanded] = useState(false);
  const hasAnomalies = (item.anomalies || []).length > 0;
  const accent = decisionAccent(item.decision);

  return (
    <div
      style={{
        padding: "9px 10px",
        borderRadius: "8px",
        border: "1px solid #e3e7ec",
        background: "#fbfcfd",
        marginBottom: 6,
        cursor: hasAnomalies ? "pointer" : "default",
      }}
      onClick={() => hasAnomalies && setExpanded((v) => !v)}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            className="mono"
            style={{
              fontWeight: 700,
              fontSize: 11,
              color: accent,
              textTransform: "uppercase",
            }}
          >
            {item.action_type}
          </span>
          <span className="muted">
            trust <strong>{item.trust_score?.toFixed(1)}</strong>
          </span>
          {item.bot_prob > 0.01 && (
            <span className="mono muted">
              p(bot) {(item.bot_prob * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <RiskBadge label={item.decision} size="sm" />
          {hasAnomalies && (
            <span className="mono muted">{expanded ? "hide" : "view"}</span>
          )}
        </div>
      </div>

      {expanded && hasAnomalies && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 8 }}>
          {item.anomalies.map((anomaly, idx) => (
            <span
              key={`${anomaly}-${idx}`}
              className="mono"
              style={{
                background: "#fdecec",
                color: "#b42318",
                border: "1px solid #f8d1d1",
                borderRadius: 999,
                padding: "2px 7px",
                fontSize: 10,
              }}
            >
              {anomaly}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ActionFeed({ feed = [], onSubmit, loading = false }) {
  const [selected, setSelected] = useState("read");

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div className="card-title">Live Action Feed</div>

      <div style={{ display: "flex", gap: 8 }}>
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          {ACTION_TYPES.map((actionType) => (
            <option key={actionType} value={actionType}>
              {actionType}
            </option>
          ))}
        </select>
        <button
          className="btn-teal"
          onClick={() => onSubmit?.(selected)}
          disabled={loading || !onSubmit}
        >
          {loading ? <span className="spinner" /> : "Evaluate"}
        </button>
      </div>

      <div className="scroll-y" style={{ maxHeight: 330 }}>
        {feed.length === 0 ? (
          <p className="muted">No actions yet.</p>
        ) : (
          feed.map((item, idx) => <FeedRow key={idx} item={item} />)
        )}
      </div>

      <div className="muted mono">
        {feed.length} evaluation{feed.length === 1 ? "" : "s"}
      </div>
    </div>
  );
}
