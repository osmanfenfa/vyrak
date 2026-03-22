/**
 * ActionFeed — live scrolling list of evaluated actions with decision badges.
 *
 * Props:
 *   feed          {Array}    DecisionResponse objects, newest first
 *   onSubmit      {Function} (actionType) => void
 *   loading       {boolean}
 */

import { useState } from "react";
import RiskBadge from "./RiskBadge.jsx";

const ACTION_TYPES = ["login","read","write","delete","search","export","config","admin"];

function decisionStyle(decision) {
  if (decision === "ALLOW")     return { borderLeft: "3px solid #1D9E75" };
  if (decision === "CHALLENGE") return { borderLeft: "3px solid #BA7517" };
  if (decision === "BLOCK")     return { borderLeft: "3px solid #A32D2D" };
  return {};
}

function FeedRow({ item }) {
  const [expanded, setExpanded] = useState(false);
  const hasAnomalies = item.anomalies?.length > 0;

  return (
    <div
      style={{
        padding: "8px 10px",
        fontSize: "12px",
        borderRadius: "var(--radius-sm)",
        background: "#FAFAF8",
        marginBottom: 4,
        cursor: hasAnomalies ? "pointer" : "default",
        ...decisionStyle(item.decision),
      }}
      onClick={() => hasAnomalies && setExpanded(e => !e)}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        {/* Left: action type + trust */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontWeight: 600, textTransform: "uppercase", fontSize: 11 }}>
            {item.action_type}
          </span>
          <span style={{ color: "var(--text-muted)" }}>
            trust <strong>{item.trust_score?.toFixed(1)}</strong>
          </span>
          {item.bot_prob > 0.01 && (
            <span style={{ color: "#1D9E75", fontSize: 11 }}>
              P(bot) {(item.bot_prob * 100).toFixed(1)}%
            </span>
          )}
        </div>

        {/* Right: decision badge + expand hint */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <RiskBadge label={item.decision} size="sm" />
          {hasAnomalies && (
            <span style={{ color: "#BA7517", fontSize: 10 }}>
              {expanded ? "▲" : "▼"} {item.anomalies.length}
            </span>
          )}
        </div>
      </div>

      {/* Anomaly detail */}
      {expanded && hasAnomalies && (
        <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: 4 }}>
          {item.anomalies.map((a, i) => (
            <span key={i} style={{
              background: "#FCEBEB", color: "#A32D2D",
              fontSize: 10, padding: "1px 6px",
              borderRadius: 99, fontFamily: "var(--font-mono)",
            }}>
              {a}
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
    <div className="card flex-col" style={{ gap: 10 }}>
      <div className="card-title">Live Action Feed</div>

      {/* Action submission */}
      <div style={{ display: "flex", gap: 8 }}>
        <select
          value={selected}
          onChange={e => setSelected(e.target.value)}
          style={{ flex: 1 }}
        >
          {ACTION_TYPES.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <button
          className="btn-teal"
          onClick={() => onSubmit?.(selected)}
          disabled={loading || !onSubmit}
        >
          {loading ? <span className="spinner" style={{ width:14, height:14 }} /> : "Evaluate"}
        </button>
      </div>

      {/* Feed list */}
      <div className="scroll-y" style={{ maxHeight: 320 }}>
        {feed.length === 0 ? (
          <p style={{ color: "var(--text-muted)", fontSize: 12, padding: "8px 0" }}>
            No actions yet.
          </p>
        ) : (
          feed.map((item, i) => <FeedRow key={i} item={item} />)
        )}
      </div>

      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
        {feed.length} evaluation{feed.length !== 1 ? "s" : ""} · click flagged rows to see anomalies
      </div>
    </div>
  );
}
