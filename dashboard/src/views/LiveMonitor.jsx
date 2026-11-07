import { useCallback, useState } from "react";

import { useAuth } from "../auth/useAuth.js";
import { useSessionLifecycle, useSession } from "../hooks/useSession.js";
import { useProfile } from "../hooks/useProfile.js";
import { collectDeviceMetadata } from "../utils/deviceMetadata.js";

import TrustGauge from "../components/TrustGauge.jsx";
import ActionFeed from "../components/ActionFeed.jsx";
import BotProbChart from "../components/BotProbChart.jsx";
import FeatureImportance from "../components/FeatureImportance.jsx";
import SessionSummary from "../components/SessionSummary.jsx";
import ColdStartBadge from "../components/ColdStartBadge.jsx";

const KINDS = [
  { value: "human", label: "Person" },
  { value: "agent", label: "AI agent" },
  { value: "service", label: "Service account" },
];

function SessionStarter({ onStart, starting, error }) {
  const { principal, can } = useAuth();
  const actsForOthers = can("sessions:act_any");
  const [uid, setUid] = useState(actsForOthers ? "support-agent-3" : principal.subject);
  const [kind, setKind] = useState(actsForOthers ? "agent" : "human");
  const [peerGroup, setPeerGroup] = useState("");
  const [collecting, setCollecting] = useState(false);
  const [metadataError, setMetadataError] = useState(null);

  const handleStart = async (e) => {
    e.preventDefault();
    const target = actsForOthers ? uid.trim() : principal.subject;
    if (!target || starting || collecting) return;
    const options = { identityKind: kind, peerGroup: peerGroup.trim() || null };

    setCollecting(true);
    setMetadataError(null);
    let metadata = null;
    try {
      // Device signals only make sense for a person at this browser
      if (kind === "human") metadata = await collectDeviceMetadata();
    } catch {
      setMetadataError("Device metadata could not be fully collected.");
    } finally {
      setCollecting(false);
    }
    await onStart(target, metadata, options);
  };

  return (
    <form className="card session-starter" onSubmit={handleStart}>
      <div className="card-title">Start a monitored session</div>
      <p className="muted" style={{ marginTop: 0 }}>
        {actsForOthers
          ? "As an integration you can evaluate any identity in your organisation, including AI agents and service accounts."
          : "You are monitoring your own session. Every action you take is scored against your normal behaviour."}
      </p>
      <div className="form-grid">
        <label className="field">
          <span>Identity</span>
          <input value={actsForOthers ? uid : principal.subject} onChange={(e) => setUid(e.target.value)}
            disabled={!actsForOthers} placeholder="User, agent or service id" />
        </label>
        {actsForOthers && (
          <div className="form-row">
            <label className="field">
              <span>Kind</span>
              <select value={kind} onChange={(e) => setKind(e.target.value)}>
                {KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Peer group (optional)</span>
              <input value={peerGroup} onChange={(e) => setPeerGroup(e.target.value)}
                placeholder="e.g. support_agent" maxLength={64} />
            </label>
          </div>
        )}
        <button className="btn-teal" type="submit" disabled={starting || collecting}>
          {collecting ? "Collecting device data…" : starting ? "Starting…" : "Start monitoring"}
        </button>
        {kind === "human" && (
          <div className="starter-note">
            Your browser may ask for location permission. If you decline, IP-level location is used.
          </div>
        )}
        {metadataError && <div className="error-bar">{metadataError}</div>}
        {error && <div className="error-bar">{error}</div>}
      </div>
    </form>
  );
}

export default function LiveMonitor() {
  const { principal, can } = useAuth();
  const [sessionMetadata, setSessionMetadata] = useState(null);
  const [sessionInfo, setSessionInfo] = useState(null);
  const { sessionId, userId, starting, error: startError, start, end } = useSessionLifecycle();

  const handleSessionMissing = useCallback(async () => {
    await end();
    setSessionMetadata(null);
    setSessionInfo(null);
  }, [end]);

  const { summary, feed, loading, error: actionError, submitAction, restoreTrust } =
    useSession(sessionId, { onSessionMissing: handleSessionMissing });

  // Only fetch the profile when this credential may read it (avoids 403 noise)
  const canReadProfile = userId && (userId === principal.subject || can("profiles:read_any"));
  const { profile } = useProfile(canReadProfile ? userId : null);

  const latestFeed = feed[0];
  const currentTrust = latestFeed?.trust_score ?? summary?.final_trust ?? sessionInfo?.trust_score ?? 100;
  const riskRating = summary?.risk_rating ?? "LOW RISK";
  const rampActive = latestFeed?.ramp_active ?? sessionInfo?.ramp_active ?? false;
  const shadow = sessionInfo?.shadow_mode;

  const handleStart = async (uid, metadata, options) => {
    const data = await start(uid, metadata, options);
    if (data) {
      setSessionMetadata(metadata);
      setSessionInfo(data);
    }
    return data;
  };

  const handleEnd = async () => {
    await end();
    setSessionMetadata(null);
    setSessionInfo(null);
  };

  const handleSubmitAction = (actionType, records) =>
    submitAction(actionType, "resource", {
      ...(sessionMetadata || {}),
      ...(records ? { records } : {}),
    });

  if (!sessionId) {
    return <SessionStarter onStart={handleStart} starting={starting} error={startError} />;
  }

  return (
    <div className="dashboard-grid">
      <section className="left-stack">
        <div className="card">
          <div className="session-meta">
            <div>
              <div style={{ fontWeight: 700 }}>
                {userId}{" "}
                <span className="pill pill-degraded" style={{ fontSize: 10 }}>
                  {sessionInfo?.identity_kind || "human"}
                </span>
              </div>
              <div className="mono muted">{sessionId?.slice(0, 22)}…</div>
              <div className="session-signal-row mono muted">
                {sessionMetadata?.ip ? `IP ${sessionMetadata.ip}` : "No device signals"}
                {sessionMetadata?.geo_city ? ` • ${sessionMetadata.geo_city}` : ""}
                {sessionMetadata?.geo_country ? `, ${sessionMetadata.geo_country}` : ""}
              </div>
            </div>
            <button className="btn-ghost btn-sm" onClick={handleEnd}>End</button>
          </div>
          {shadow && (
            <div className="notice" style={{ marginTop: 10 }}>
              Shadow mode: nothing is blocked. Expand an action to see what enforcement would have done.
            </div>
          )}
          {sessionInfo?.device_risk?.composite_score > 0 && (
            <div className="notice" style={{ marginTop: 10 }}>
              Device/network risk {Math.round(sessionInfo.device_risk.composite_score * 100)}% lowered the
              starting trust. {(sessionInfo.device_risk.triggered_signals || []).join(" · ")}
            </div>
          )}
        </div>

        <div className="card" style={{ textAlign: "center" }}>
          <TrustGauge score={currentTrust} label={riskRating} rampActive={rampActive} />
        </div>

        {rampActive && (
          <ColdStartBadge rampActive={rampActive} vectorCount={profile?.vector_count ?? 0} rampActions={50} />
        )}

        {actionError && <div className="error-bar">{actionError}</div>}

        <ActionFeed feed={feed} onSubmit={handleSubmitAction} loading={loading} />
      </section>

      <section className="right-stack">
        <BotProbChart feed={feed} limit={40} />
        <div className="split-grid">
          <FeatureImportance features={summary?.top_features ?? []} title="Top Features" />
          <SessionSummary summary={summary} onRestore={() => restoreTrust(20)} onEnd={handleEnd} />
        </div>
      </section>
    </div>
  );
}
