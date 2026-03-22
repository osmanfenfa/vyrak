/**
 * RiskBadge — maps a risk_rating string or decision to a coloured pill.
 *
 * Props:
 *   label   {string}  "LOW RISK" | "MEDIUM RISK" | "HIGH RISK" | "CRITICAL RISK"
 *                     OR "ALLOW" | "CHALLENGE" | "BLOCK"
 *   size    {"sm"|"md"}  default "md"
 */

export default function RiskBadge({ label = "LOW RISK", size = "md" }) {
  const map = {
    "LOW RISK":      { cls: "pill-allow",     text: "LOW RISK" },
    "MEDIUM RISK":   { cls: "pill-challenge",  text: "MEDIUM RISK" },
    "HIGH RISK":     { cls: "pill-block",      text: "HIGH RISK" },
    "CRITICAL RISK": { cls: "pill-block",      text: "CRITICAL" },
    "ALLOW":         { cls: "pill-allow",      text: "ALLOW" },
    "CHALLENGE":     { cls: "pill-challenge",  text: "CHALLENGE" },
    "BLOCK":         { cls: "pill-block",      text: "BLOCK" },
    "healthy":       { cls: "pill-healthy",    text: "HEALTHY" },
    "degraded":      { cls: "pill-degraded",   text: "DEGRADED" },
    "down":          { cls: "pill-down",       text: "DOWN" },
  };

  const entry = map[label] || { cls: "pill-degraded", text: label };

  return (
    <span
      className={`pill ${entry.cls}`}
      style={size === "sm" ? { fontSize: "10px", padding: "1px 6px" } : undefined}
    >
      {entry.text}
    </span>
  );
}
