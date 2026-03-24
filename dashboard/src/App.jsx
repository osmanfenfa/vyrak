import { useState } from "react";

import { useSessionLifecycle, useSession } from "./hooks/useSession.js";
import { useProfile } from "./hooks/useProfile.js";
import { useModelStatus } from "./hooks/useModelStatus.js";

import TrustGauge from "./components/TrustGauge.jsx";
import ActionFeed from "./components/ActionFeed.jsx";
import BotProbChart from "./components/BotProbChart.jsx";
import FeatureImportance from "./components/FeatureImportance.jsx";
import SessionSummary from "./components/SessionSummary.jsx";
import ColdStartBadge from "./components/ColdStartBadge.jsx";
import RiskBadge from "./components/RiskBadge.jsx";
import { collectDeviceMetadata } from "./utils/deviceMetadata.js";

function formatUptime(seconds) {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function Header({ modelStatus, isHealthy, shadowMode, uptime }) {
  return (
    <header className="app-header">
      <div className="app-header-inner">
        <div className="app-brand">
          <div className="brand-mark">Z</div>
          <div>
            <div className="brand-title">Zelvryn Dashboard</div>
            <div className="brand-subtitle">
              Adaptive Behavioral Signature Authentication
            </div>
          </div>
        </div>

        <div className="app-meta">
          <RiskBadge label={isHealthy ? "healthy" : "degraded"} size="sm" />
          {shadowMode && <span className="pill pill-shadow">SHADOW MODE</span>}
          {modelStatus?.active_version && (
            <span className="mono muted">
              {modelStatus.active_version}
              {modelStatus.active_cv_f1 > 0
                ? ` F1 ${(modelStatus.active_cv_f1 * 100).toFixed(0)}%`
                : ""}
            </span>
          )}
          {uptime > 0 && (
            <span className="mono muted">{formatUptime(Math.round(uptime))}</span>
          )}
        </div>
      </div>
    </header>
  );
}

function SessionStarter({ onStart, starting, error }) {
  const [uid, setUid] = useState("zelvryn@example.com");
  const [collecting, setCollecting] = useState(false);
  const [metadataError, setMetadataError] = useState(null);

  const handleStart = async () => {
    if (!uid.trim() || starting || collecting) return;

    setCollecting(true);
    setMetadataError(null);
    try {
      const metadata = await collectDeviceMetadata();
      await onStart(uid.trim(), metadata);
    } catch {
      // Continue even if metadata collection fails.
      setMetadataError("Device metadata could not be fully collected.");
      await onStart(uid.trim(), null);
    } finally {
      setCollecting(false);
    }
  };

  return (
    <div className="card session-starter">
      <div className="card-title">Start Session</div>
      <p>Enter a user id to begin live monitoring with real device signals.</p>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <input
          value={uid}
          onChange={(e) => setUid(e.target.value)}
          placeholder="User ID or email"
        />
        <button
          className="btn-teal"
          onClick={handleStart}
          disabled={starting || collecting || !uid.trim()}
        >
          {collecting ? "Collecting Device Data..." : starting ? "Starting..." : "Start Monitoring"}
        </button>
        <div className="starter-note">
          Browser may ask for location permission. If denied, IP-level location is used.
        </div>
        {metadataError && <div className="error-bar">{metadataError}</div>}
        {error && <div className="error-bar">{error}</div>}
      </div>
    </div>
  );
}

export default function App() {
  const [sessionMetadata, setSessionMetadata] = useState(null);
  const { modelStatus, isHealthy, shadowMode, uptime } = useModelStatus();
  const {
    sessionId,
    userId,
    starting,
    error: startError,
    start,
    end,
  } = useSessionLifecycle();
  const {
    summary,
    feed,
    loading,
    error: actionError,
    submitAction,
    restoreTrust,
  } = useSession(sessionId);
  const { profile } = useProfile(userId);

  const latestFeed = feed[0];
  const currentTrust = latestFeed?.trust_score ?? summary?.final_trust ?? 100;
  const riskRating = summary?.risk_rating ?? "LOW RISK";
  const rampActive = latestFeed?.ramp_active ?? false;

  const handleStart = async (uid, metadata = null) => {
    const data = await start(uid, metadata);
    if (data) {
      setSessionMetadata(metadata);
    }
    return data;
  };

  const handleEnd = async () => {
    await end();
    setSessionMetadata(null);
  };

  const handleSubmitAction = (actionType) =>
    submitAction(actionType, "resource", sessionMetadata);

  return (
    <div className="app-shell">
      <Header
        modelStatus={modelStatus}
        isHealthy={isHealthy}
        shadowMode={shadowMode}
        uptime={uptime}
      />

      <main className="app-main">
        {!sessionId ? (
          <SessionStarter onStart={handleStart} starting={starting} error={startError} />
        ) : (
          <div className="dashboard-grid">
            <section className="left-stack">
              <div className="card">
                <div className="session-meta">
                  <div>
                    <div style={{ fontWeight: 700 }}>{userId}</div>
                    <div className="mono muted">
                      {sessionId?.slice(0, 22)}...
                    </div>
                    <div className="session-signal-row mono muted">
                      {sessionMetadata?.ip ? `IP ${sessionMetadata.ip}` : "IP unavailable"}
                      {sessionMetadata?.geo_city ? ` • ${sessionMetadata.geo_city}` : ""}
                      {sessionMetadata?.geo_country ? `, ${sessionMetadata.geo_country}` : ""}
                    </div>
                  </div>
                  <button className="btn-ghost btn-sm" onClick={handleEnd}>
                    End
                  </button>
                </div>
              </div>

              <div className="card" style={{ textAlign: "center" }}>
                <TrustGauge
                  score={currentTrust}
                  label={riskRating}
                  rampActive={rampActive}
                />
              </div>

              {rampActive && (
                <ColdStartBadge
                  rampActive={rampActive}
                  vectorCount={profile?.vector_count ?? 0}
                  rampActions={50}
                />
              )}

              {actionError && <div className="error-bar">{actionError}</div>}

              <ActionFeed feed={feed} onSubmit={handleSubmitAction} loading={loading} />
            </section>

            <section className="right-stack">
              <BotProbChart feed={feed} limit={40} />

              <div className="split-grid">
                <FeatureImportance
                  features={summary?.top_features ?? []}
                  title="Top Features"
                />
                <SessionSummary
                  summary={summary}
                  onRestore={() => restoreTrust(20)}
                  onEnd={handleEnd}
                />
              </div>
            </section>
          </div>
        )}
      </main>

      <footer className="footer">
        Zelvryn v2.0 | Adaptive Behavioral Signature Authentication
      </footer>
    </div>
  );
}
