import { useCallback, useEffect, useState } from "react";
import { adminGate } from "./api/client.js";

const DEFAULT_ADMIN_USER = "admin@zelvryn.local";

const CONFIG_FIELDS = [
  { key: "trust_decay_mild", label: "Trust Decay Mild", step: "0.1" },
  { key: "trust_decay_moderate", label: "Trust Decay Moderate", step: "0.1" },
  { key: "trust_decay_severe", label: "Trust Decay Severe", step: "0.1" },
  { key: "trust_decay_ai_bot", label: "Trust Decay AI Bot", step: "0.1" },
  { key: "trust_challenge_threshold", label: "Challenge Threshold", step: "0.1" },
  { key: "trust_block_threshold", label: "Block Threshold", step: "0.1" },
  { key: "ml_bot_confidence_thresh", label: "ML Bot Threshold", step: "0.01" },
  { key: "cold_start_ramp_actions", label: "Cold Start Ramp Actions", step: "1" },
  { key: "ewma_decay_factor", label: "EWMA Decay Factor", step: "0.01" },
  { key: "ml_retrain_interval", label: "Retrain Interval", step: "1" },
  { key: "canary_traffic_percent", label: "Canary Traffic %", step: "1" },
];

function toDraft(config = {}) {
  const draft = {};
  CONFIG_FIELDS.forEach(({ key }) => {
    draft[key] = config[key] ?? "";
  });
  return draft;
}

function parseMaybeNumber(value) {
  if (value === "" || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename || "audit_export";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export default function AdminGate() {
  const [adminUser, setAdminUser] = useState(DEFAULT_ADMIN_USER);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  const [overview, setOverview] = useState(null);
  const [config, setConfig] = useState(null);
  const [models, setModels] = useState(null);
  const [audit, setAudit] = useState([]);
  const [draft, setDraft] = useState({});

  const [canaryVersion, setCanaryVersion] = useState("");
  const [promoteVersion, setPromoteVersion] = useState("");
  const [rollbackVersion, setRollbackVersion] = useState("");

  const refresh = useCallback(async () => {
    if (!adminUser.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const [ov, cfg, mdl, aud] = await Promise.all([
        adminGate.overview(adminUser),
        adminGate.getConfig(adminUser),
        adminGate.listModels(adminUser),
        adminGate.listAudit(adminUser, { limit: 25 }),
      ]);
      setOverview(ov);
      setConfig(cfg.config);
      setDraft(toDraft(cfg.config));
      setModels(mdl);
      setAudit(aud.items || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [adminUser]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleSaveConfig = async () => {
    if (!config) return;
    const patch = {};

    for (const field of CONFIG_FIELDS) {
      const key = field.key;
      const next = parseMaybeNumber(draft[key]);
      if (Number.isNaN(next)) {
        setError(`Invalid numeric value for ${key}`);
        return;
      }
      if (next === null) continue;
      if (Number(config[key]) !== next) {
        patch[key] = next;
      }
    }

    if (Object.keys(patch).length === 0) {
      setNotice("No config changes to save.");
      return;
    }

    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const result = await adminGate.patchConfig(adminUser, patch);
      setConfig(result.config);
      setDraft(toDraft(result.config));
      setNotice(`Updated: ${result.changed_keys.join(", ")}`);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleToggleShadow = async () => {
    const next = !overview?.shadow_mode;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await adminGate.setShadowMode(adminUser, next);
      setNotice(`Shadow mode set to ${next ? "ON" : "OFF"}.`);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleCanary = async () => {
    if (!canaryVersion.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const traffic = parseMaybeNumber(draft.canary_traffic_percent);
      await adminGate.setCanary(adminUser, canaryVersion.trim(), traffic);
      setNotice(`Canary promoted: ${canaryVersion.trim()}`);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const handlePromote = async () => {
    if (!promoteVersion.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await adminGate.promoteModel(adminUser, promoteVersion.trim(), "admin-gate promotion");
      setNotice(`Promoted active: ${promoteVersion.trim()}`);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleRollback = async () => {
    setBusy(true);
    setError(null);
    try {
      await adminGate.rollbackModel(
        adminUser,
        rollbackVersion.trim() || null,
        "admin-gate rollback",
      );
      setNotice("Rollback completed.");
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleExportAudit = async (format) => {
    setBusy(true);
    setError(null);
    try {
      const result = await adminGate.exportAudit(adminUser, { format, limit: 2000 });
      triggerDownload(result.blob, result.filename || `audit_export.${format}`);
      setNotice(`Audit exported (${format}).`);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <div className="app-brand">
            <div className="brand-mark">A</div>
            <div>
              <div className="brand-title">Zelvryn Admin Gate</div>
              <div className="brand-subtitle">Live operations control panel</div>
            </div>
          </div>
          <button className="btn-ghost btn-sm" onClick={refresh} disabled={loading || busy}>
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
      </header>

      <main className="app-main" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="card">
          <div className="card-title">Admin Identity</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8 }}>
            <input
              value={adminUser}
              onChange={(e) => setAdminUser(e.target.value)}
              placeholder="Admin user id"
            />
            <button className="btn-teal" onClick={refresh} disabled={loading || busy || !adminUser.trim()}>
              Connect
            </button>
          </div>
        </div>

        {(error || notice) && (
          <div className={error ? "error-bar" : "card"} style={!error ? { padding: "8px 10px" } : {}}>
            {error || notice}
          </div>
        )}

        <div className="dashboard-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <section className="left-stack">
            <div className="card">
              <div className="card-title">System Controls</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {CONFIG_FIELDS.map((field) => (
                  <label key={field.key} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span className="muted mono">{field.label}</span>
                    <input
                      type="number"
                      step={field.step}
                      value={draft[field.key] ?? ""}
                      onChange={(e) => setDraft((prev) => ({ ...prev, [field.key]: e.target.value }))}
                    />
                  </label>
                ))}
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                <button className="btn-teal" onClick={handleSaveConfig} disabled={busy || loading}>
                  Save Config
                </button>
                <button className="btn-ghost" onClick={handleToggleShadow} disabled={busy || loading}>
                  Shadow: {overview?.shadow_mode ? "ON" : "OFF"}
                </button>
              </div>
            </div>

            <div className="card">
              <div className="card-title">Model Management</div>
              <div style={{ display: "grid", gap: 8 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8 }}>
                  <input
                    value={canaryVersion}
                    onChange={(e) => setCanaryVersion(e.target.value)}
                    placeholder="Version for canary"
                  />
                  <button className="btn-ghost" onClick={handleCanary} disabled={busy}>
                    Canary
                  </button>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8 }}>
                  <input
                    value={promoteVersion}
                    onChange={(e) => setPromoteVersion(e.target.value)}
                    placeholder="Version to promote active"
                  />
                  <button className="btn-teal" onClick={handlePromote} disabled={busy}>
                    Promote
                  </button>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8 }}>
                  <input
                    value={rollbackVersion}
                    onChange={(e) => setRollbackVersion(e.target.value)}
                    placeholder="Rollback target (blank = latest retired)"
                  />
                  <button className="btn-danger" onClick={handleRollback} disabled={busy}>
                    Rollback
                  </button>
                </div>
              </div>
            </div>
          </section>

          <section className="right-stack">
            <div className="card">
              <div className="card-title">Registry Snapshot</div>
              <div className="mono muted" style={{ marginBottom: 8 }}>
                active {models?.summary?.active_version || "none"} | canary {models?.summary?.canary_version || "none"} | traffic {models?.summary?.canary_traffic_percent ?? 0}%
              </div>
              <div className="scroll-y" style={{ maxHeight: 240 }}>
                {(models?.versions || []).map((row) => (
                  <div
                    key={row.version}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1.2fr 0.8fr 0.5fr",
                      gap: 8,
                      borderBottom: "1px solid #eef1f4",
                      padding: "6px 0",
                    }}
                  >
                    <span className="mono">{row.version}</span>
                    <span className="muted">{row.state}</span>
                    <span className="mono">{(Number(row.cv_f1 || 0) * 100).toFixed(0)}%</span>
                  </div>
                ))}
                {(!models?.versions || models.versions.length === 0) && (
                  <p className="muted">No models available.</p>
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-title">Audit Log</div>
              <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                <button className="btn-ghost btn-sm" onClick={() => handleExportAudit("jsonl")} disabled={busy}>
                  Export JSONL
                </button>
                <button className="btn-ghost btn-sm" onClick={() => handleExportAudit("csv")} disabled={busy}>
                  Export CSV
                </button>
              </div>
              <div className="scroll-y" style={{ maxHeight: 300 }}>
                {audit.map((item, idx) => (
                  <div
                    key={`${item.request_id || "r"}-${idx}`}
                    style={{
                      border: "1px solid #e7ecf2",
                      borderRadius: 8,
                      padding: "7px 8px",
                      marginBottom: 6,
                      background: "#fbfcfd",
                    }}
                  >
                    <div className="mono" style={{ fontSize: 11 }}>
                      {item.event_type || "event"} | actor {item.actor_id || item.user_id || "unknown"}
                    </div>
                    <div className="muted mono" style={{ fontSize: 11 }}>
                      {new Date((item.logged_at || item.timestamp || 0) * 1000).toLocaleString()}
                    </div>
                  </div>
                ))}
                {audit.length === 0 && <p className="muted">No audit entries found.</p>}
              </div>
            </div>
          </section>
        </div>
      </main>

      <footer className="footer">
        /admin-gate | shadow {overview?.shadow_mode ? "on" : "off"} | sessions {overview?.active_sessions ?? 0}
      </footer>
    </div>
  );
}
