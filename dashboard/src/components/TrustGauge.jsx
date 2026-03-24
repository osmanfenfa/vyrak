import { useEffect, useRef } from "react";

const SIZE = 180;
const RADIUS = 70;
const CENTER_X = SIZE / 2;
const CENTER_Y = SIZE / 2 + 10;
const HALF_CIRCUMFERENCE = Math.PI * RADIUS;

function scoreColor(score) {
  if (score >= 80) return "#0f8b6f";
  if (score >= 50) return "#b66b16";
  if (score >= 20) return "#e85d24";
  return "#b42318";
}

export default function TrustGauge({ score = 100, label = "LOW RISK", rampActive = false }) {
  const arcRef = useRef(null);
  const safeScore = Math.max(0, Math.min(100, Number(score) || 0));
  const color = scoreColor(safeScore);

  useEffect(() => {
    if (!arcRef.current) return;
    const fraction = safeScore / 100;
    const dashLength = HALF_CIRCUMFERENCE * fraction;
    arcRef.current.style.strokeDashoffset = HALF_CIRCUMFERENCE - dashLength;
  }, [safeScore]);

  return (
    <div className="trust-gauge">
      <svg
        width={SIZE}
        height={SIZE / 2 + 30}
        viewBox={`0 0 ${SIZE} ${SIZE / 2 + 30}`}
        className="trust-gauge-svg"
      >
        <path
          d={`M ${CENTER_X - RADIUS},${CENTER_Y} A ${RADIUS},${RADIUS} 0 0 1 ${CENTER_X + RADIUS},${CENTER_Y}`}
          fill="none"
          stroke="var(--line)"
          strokeWidth="14"
          strokeLinecap="round"
        />

        <path
          ref={arcRef}
          d={`M ${CENTER_X - RADIUS},${CENTER_Y} A ${RADIUS},${RADIUS} 0 0 1 ${CENTER_X + RADIUS},${CENTER_Y}`}
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={`${HALF_CIRCUMFERENCE} ${HALF_CIRCUMFERENCE}`}
          strokeDashoffset={HALF_CIRCUMFERENCE}
          style={{ transition: "stroke-dashoffset 0.6s ease, stroke 0.4s ease" }}
        />

        <text
          x={CENTER_X}
          y={CENTER_Y - 8}
          textAnchor="middle"
          fontSize="32"
          fontWeight="700"
          fill={color}
          style={{ transition: "fill 0.4s ease" }}
        >
          {Math.round(safeScore)}
        </text>

        <text
          x={CENTER_X}
          y={CENTER_Y + 12}
          textAnchor="middle"
          fontSize="11"
          fill="var(--text-muted)"
        >
          Trust Score
        </text>

        <text
          x={CENTER_X - RADIUS - 4}
          y={CENTER_Y + 16}
          textAnchor="end"
          fontSize="10"
          fill="var(--text-muted)"
        >
          0
        </text>
        <text
          x={CENTER_X + RADIUS + 4}
          y={CENTER_Y + 16}
          textAnchor="start"
          fontSize="10"
          fill="var(--text-muted)"
        >
          100
        </text>
      </svg>

      <div className="trust-gauge-pills">
        <span
          className="pill"
          style={{
            background: `${color}1f`,
            color,
            fontSize: 12,
            padding: "2px 10px",
          }}
        >
          {label}
        </span>
        {rampActive && <span className="pill pill-shadow">RAMP-UP</span>}
      </div>
    </div>
  );
}
