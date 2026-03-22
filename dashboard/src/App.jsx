/**
 * App.jsx — Zelvryn live monitoring dashboard
 *
 * Layout (single page):
 *   ┌─── Header ──────────────────────────────────────────┐
 *   │  Logo · system health · model status · shadow badge │
 *   └─────────────────────────────────────────────────────┘
 *   ┌─ Session panel ─────────┐  ┌─ Analytics ────────────┐
 *   │  Session start / switch │  │  BotProbChart          │
 *   │  TrustGauge             │  │  FeatureImportance     │
 *   │  ColdStartBadge         │  │  SessionSummary        │
 *   │  ActionFeed             │  └────────────────────────┘
 *   └─────────────────────────┘
 */

import { useState } from "react";

import { useSessionLifecycle, useSession } from "./hooks/useSession.js";
import { useProfile }      from "./hooks/useProfile.js";
import { useModelStatus }  from "./hooks/useModelStatus.js";

import TrustGauge        from "./components/TrustGauge.jsx";
import ActionFeed        from "./components/ActionFeed.jsx";
import BotProbChart      from "./components/BotProbChart.jsx";
import FeatureImportance from "./components/FeatureImportance.jsx";
import SessionSummary    from "./components/SessionSummary.jsx";
import ColdStartBadge    from "./components/ColdStartBadge.jsx";
import RiskBadge         from "./components/RiskBadge.jsx";

// ── Header ────────────────────────────────────────────────────────────────────

function Header({ modelStatus, isHealthy, shadowMode, uptime }) {
  const fmt = s => s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s/60)}m` : `${(s/3600).toFixed(1)}h`;
  return (
    <header style={{
      background: "var(--navy)",
      color: "white",
      padding: "10px 24px",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      boxShadow: "var(--shadow-md)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{
          width: 32, height: 32,
          background: "var(--teal)",
          borderRadius: 8,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontWeight: 800, fontSize: 18, color: "white",
        }}>Z</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.01em" }}>Zelvryn</div>
          <div style={{ fontSize: 10, color: "#9FE1CB" }}>ABSA Dashboard · Vyrak</div>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 12 }}>
        {/* System health */}
        <RiskBadge label={isHealthy ? "healthy" : "degraded"} size="sm" />

        {/* Shadow mode indicator */}
        {shadowMode && (
          <span className="pill pill-shadow">SHADOW MODE</span>
        )}

        {/* Model version */}
        {modelStatus?.active_version && (
          <span style={{ color: "#9FE1CB", fontFamily: "var(--font-mono)", fontSize: 11 }}>
            {modelStatus.active_version}
            {modelStatus.active_cv_f1 > 0 && ` F1=${(modelStatus.active_cv_f1*100).toFixed(0)}%`}
          </span>
        )}

        {/* Uptime */}
        {uptime > 0 && (
          <span style={{ color: "#5F9EA0", fontSize: 11 }}>⏱ {fmt(Math.round(uptime))}</span>
        )}
      </div>
    </header>
  );
}

// ── Session starter ────────────────────────────────────────────────────────────

function SessionStarter({ onStart, starting, error }) {
  const [uid, setUid] = useState("alice@example.com");
  return (
    <div className="card" style={{ maxWidth: 400, margin: "40px auto" }}>
      <div className="card-title">Start Monitoring Session</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <input
          value={uid}
          onChange={e => setUid(e.target.value)}
          placeholder="User ID or email"
        />
        <button className="btn-teal" onClick={() => onStart(uid)} disabled={starting || !uid.trim()}>
          {starting ? "Starting…" : "Start Session"}
        </button>
        {error && <div className="error-bar">{error}</div>}
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const { systemHealth, modelStatus, isHealthy, hasModel, shadowMode, uptime } = useModelStatus();
  const { sessionId, userId, starting, error: startError, start, end } = useSessionLifecycle();
  const { summary, feed, loading, error: actionError, submitAction, restoreTrust } = useSession(sessionId);
  const { profile } = useProfile(userId);

  const handleEnd = async () => {
    await end();
  };

  // Derive trust and risk from latest feed item or summary
  const latestFeed   = feed[0];
  const currentTrust = latestFeed?.trust_score ?? summary?.final_trust ?? 100;
  const riskRating   = summary?.risk_rating ?? "LOW RISK";
  const rampActive   = latestFeed?.ramp_active ?? false;

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Header
        modelStatus={modelStatus}
        isHealthy={isHealthy}
        shadowMode={shadowMode}
        uptime={uptime}
      />

      <main style={{ flex: 1, padding: "20px 24px", maxWidth: 1200, margin: "0 auto", width: "100%" }}>
        {!sessionId ? (
          <SessionStarter onStart={start} starting={starting} error={startError} />
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: 16 }}>

            {/* ── Left column ──────────────────────────────── */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

              {/* Session identity */}
              <div className="card" style={{ padding: "10px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{userId}</div>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {sessionId?.slice(0, 20)}…
                    </div>
                  </div>
                  <button className="btn-ghost btn-sm" onClick={handleEnd}>End</button>
                </div>
              </div>

              {/* Trust gauge */}
              <div className="card" style={{ textAlign: "center", padding: "16px 8px" }}>
                <TrustGauge
                  score={currentTrust}
                  label={riskRating}
                  rampActive={rampActive}
                />
              </div>

              {/* Cold start badge */}
              {rampActive && (
                <ColdStartBadge
                  rampActive={rampActive}
                  vectorCount={profile?.vector_count ?? 0}
                  rampActions={50}
                />
              )}

              {/* Error display */}
              {actionError && <div className="error-bar">{actionError}</div>}

              {/* Action feed */}
              <ActionFeed
                feed={feed}
                onSubmit={submitAction}
                loading={loading}
              />
            </div>

            {/* ── Right column ─────────────────────────────── */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

              {/* P(bot) chart */}
              <BotProbChart feed={feed} limit={40} />

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {/* Feature importances */}
                <FeatureImportance
                  features={summary?.top_features ?? []}
                  title="Top Discriminating Features"
                />

                {/* Session summary */}
                <SessionSummary
                  summary={summary}
                  onRestore={() => restoreTrust(20)}
                  onEnd={handleEnd}
                />
              </div>
            </div>
          </div>
        )}
      </main>

      <footer style={{
        textAlign: "center",
        padding: "10px",
        fontSize: 11,
        color: "var(--text-muted)",
        borderTop: "1px solid var(--gray-mid)",
      }}>
        Zelvryn v2.0 · Adaptive Behavioral Signature Authentication · Vyrak
      </footer>
    </div>
  );
}