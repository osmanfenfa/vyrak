/**
 * TrustGauge — animated semicircle gauge showing current trust score (0–100).
 *
 * Props:
 *   score      {number}  0–100
 *   label      {string}  risk label
 *   rampActive {boolean}
 */

import { useEffect, useRef } from "react";

const SIZE    = 180;
const R       = 70;
const CX      = SIZE / 2;
const CY      = SIZE / 2 + 10;
const CIRCUM  = Math.PI * R;   // half-circle circumference

function scoreColor(score) {
  if (score >= 80) return "#1D9E75";
  if (score >= 50) return "#BA7517";
  if (score >= 20) return "#E85D24";
  return "#A32D2D";
}

export default function TrustGauge({ score = 100, label = "LOW RISK", rampActive = false }) {
  const arcRef = useRef(null);

  useEffect(() => {
    if (!arcRef.current) return;
    const fraction = Math.min(1, Math.max(0, score / 100));
    const dashLen  = CIRCUM * fraction;
    arcRef.current.style.strokeDashoffset = CIRCUM - dashLen;
  }, [score]);

  const color = scoreColor(score);

  return (
    <div style={{ textAlign: "center" }}>
      <svg
        width={SIZE}
        height={SIZE / 2 + 30}
        viewBox={`0 0 ${SIZE} ${SIZE / 2 + 30}`}
        style={{ overflow: "visible" }}
      >
        {/* Track */}
        <path
          d={`M ${CX - R},${CY} A ${R},${R} 0 0 1 ${CX + R},${CY}`}
          fill="none"
          stroke="#E5E3DC"
          strokeWidth="14"
          strokeLinecap="round"
        />

        {/* Filled arc */}
        <path
          ref={arcRef}
          d={`M ${CX - R},${CY} A ${R},${R} 0 0 1 ${CX + R},${CY}`}
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={`${CIRCUM} ${CIRCUM}`}
          strokeDashoffset={CIRCUM}
          style={{ transition: "stroke-dashoffset 0.6s ease, stroke 0.4s ease" }}
        />

        {/* Score text */}
        <text
          x={CX} y={CY - 8}
          textAnchor="middle"
          fontSize="32"
          fontWeight="700"
          fill={color}
          style={{ transition: "fill 0.4s ease" }}
        >
          {Math.round(score)}
        </text>

        {/* "Trust Score" label */}
        <text x={CX} y={CY + 12} textAnchor="middle" fontSize="11" fill="#888780">
          Trust Score
        </text>

        {/* Min / Max labels */}
        <text x={CX - R - 4} y={CY + 16} textAnchor="end" fontSize="10" fill="#B4B2A9">0</text>
        <text x={CX + R + 4} y={CY + 16} textAnchor="start" fontSize="10" fill="#B4B2A9">100</text>
      </svg>

      {/* Risk label + cold start note */}
      <div style={{ marginTop: 4, display: "flex", justifyContent: "center", gap: 6 }}>
        <span
          className="pill"
          style={{
            background: color + "22",
            color,
            fontSize: "12px",
            padding: "2px 10px",
            borderRadius: "99px",
            fontWeight: 600,
          }}
        >
          {label}
        </span>
        {rampActive && (
          <span className="pill" style={{
            background: "#EDE9FE", color: "#6D28D9",
            fontSize: "11px", padding: "2px 8px", borderRadius: "99px",
          }}>
            RAMP-UP
          </span>
        )}
      </div>
    </div>
  );
}
