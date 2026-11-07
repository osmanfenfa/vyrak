import { useCallback, useEffect, useState } from "react";
import { apiKeys, events as eventsApi, tenant } from "../api/client.js";
import { useAuth } from "../auth/useAuth.js";
import RiskBadge from "../components/RiskBadge.jsx";

const KEY_ROLES = [
  { value: "integration", label: "Integration (send events)" },
  { value: "analyst", label: "Analyst (read & label alerts)" },
  { value: "viewer", label: "Viewer (read only)" },
];

const SAMPLE_EVENTS = JSON.stringify(
  [
    { identity_id: "support-agent-3", identity_kind: "agent", session_id: "run-42",
      action_type: "read", resource_id: "ticket_118", records: 3 },
    { identity_id: "support-agent-3", identity_kind: "agent", session_id: "run-42",
      action_type: "export", resource_id: "customers", records: 4000 },
  ],
  null,
  2,
);

function TenantSettingsCard() {
  const { principal } = useAuth();
  const [settings, setSettings] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    tenant.getSettings().then(setSettings).catch((e) => setError(e.message));
  }, []);

  const toggle = async () => {
    setBusy(true);
    setError(null);
    try {
      setSettings(await tenant.patchSettings({ shadow_mode: !settings.shadow_mode }));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">Enforcement for {principal.tenant_id}</div>
      {settings ? (
        <>
          <p style={{ marginTop: 0 }}>
            {settings.shadow_mode ? (
              <>
                <span className="pill pill-shadow">Shadow mode</span> Zelvryn observes and raises alerts
                but never challenges or blocks. Start here, review alerts for a few weeks, then enforce.
              </>
            ) : (
              <>
                <span className="pill pill-allow">Enforcing</span> Decisions are returned as CHALLENGE or
                BLOCK and your application should act on them.
              </>
            )}
          </p>
          <button className={settings.shadow_mode ? "btn-teal" : "btn-ghost"} onClick={toggle} disabled={busy}>
            {busy ? "Saving…" : settings.shadow_mode ? "Switch to enforcement" : "Switch back to shadow mode"}
          </button>
        </>
      ) : (
        !error && <p className="muted">Loading…</p>
      )}
      {error && <div className="error-bar" style={{ marginTop: 8 }}>{error}</div>}
    </div>
  );
}

function ApiKeysCard() {
  const [keys, setKeys] = useState([]);
  const [name, setName] = useState("backend");
  const [role, setRole] = useState("integration");
  const [created, setCreated] = useState(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      setKeys((await apiKeys.list()).items || []);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const create = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setCopied(false);
    try {
      setCreated(await apiKeys.create(name.trim(), [role]));
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const revoke = async (keyId) => {
    if (!window.confirm("Revoke this key? Anything using it stops working immediately.")) return;
    try {
      await apiKeys.revoke(keyId);
      await load();
    } catch (e) {
      setError(e.message);
    }
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(created.api_key);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">API keys</div>
      <form className="form-row" onSubmit={create}>
        <label className="field">
          <span>Name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={100} />
        </label>
        <label className="field">
          <span>Role</span>
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            {KEY_ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
          </select>
        </label>
        <button className="btn-teal" type="submit" disabled={busy} style={{ alignSelf: "end" }}>
          {busy ? "Creating…" : "Create key"}
        </button>
      </form>

      {created && (
        <div className="secret-box">
          <div style={{ fontWeight: 600 }}>Copy this key now. It won't be shown again.</div>
          <code className="mono">{created.api_key}</code>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn-ghost btn-sm" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
            <button className="btn-ghost btn-sm" onClick={() => setCreated(null)}>Done</button>
          </div>
        </div>
      )}

      {error && <div className="error-bar" style={{ marginTop: 8 }}>{error}</div>}

      <div className="table-wrap" style={{ marginTop: 12 }}>
        <table className="table">
          <thead><tr><th>Name</th><th>Role</th><th>Last used</th><th>Status</th><th /></tr></thead>
          <tbody>
            {keys.length === 0 && (
              <tr><td colSpan={5} className="muted">No keys yet.</td></tr>
            )}
            {keys.map((k) => (
              <tr key={k.key_id}>
                <td>
                  <div style={{ fontWeight: 600 }}>{k.name}</div>
                  <div className="mono muted" style={{ fontSize: 11 }}>zk_{k.key_id}_…</div>
                </td>
                <td>{(k.roles || []).join(", ")}</td>
                <td className="muted" style={{ fontSize: 12 }}>
                  {k.last_used_at ? new Date(k.last_used_at * 1000).toLocaleString() : "never"}
                </td>
                <td>{k.revoked ? <span className="pill pill-block">revoked</span> : <span className="pill pill-allow">active</span>}</td>
                <td style={{ textAlign: "right" }}>
                  {!k.revoked && <button className="btn-ghost btn-sm" onClick={() => revoke(k.key_id)}>Revoke</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EventTesterCard() {
  const [text, setText] = useState(SAMPLE_EVENTS);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const send = async () => {
    setError(null);
    let parsed;
    try {
      parsed = JSON.parse(text);
      if (!Array.isArray(parsed)) throw new Error("Expected a JSON array of events.");
    } catch (e) {
      setError(`Invalid JSON: ${e.message}`);
      return;
    }
    setBusy(true);
    try {
      setResult(await eventsApi.ingest(parsed));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">Send test events</div>
      <p className="muted" style={{ marginTop: 0 }}>
        The same request your backend makes with an integration key: <span className="mono">POST /events</span>.
        No SDK needed; forward your application's audit events.
      </p>
      <textarea className="mono" rows={10} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
        style={{ width: "100%", fontSize: 12 }} />
      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
        <button className="btn-teal" onClick={send} disabled={busy}>{busy ? "Sending…" : "Send events"}</button>
        <button className="btn-ghost" onClick={() => { setText(SAMPLE_EVENTS); setResult(null); }}>Reset sample</button>
      </div>
      {error && <div className="error-bar" style={{ marginTop: 8 }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 12 }}>
          <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
            {result.processed} processed in {result.duration_ms} ms · would be: ALLOW {result.would_be.ALLOW} ·
            CHALLENGE {result.would_be.CHALLENGE} · BLOCK {result.would_be.BLOCK}
            {result.alerts.length > 0 && ` · ${result.alerts.length} alert(s) opened, see the Alerts tab`}
          </div>
          {result.results.map((r, i) => (
            <div key={i} className="timeline-item">
              <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                <span className="mono">{r.identity_id} · {r.session_id}</span>
                <span style={{ display: "flex", gap: 6 }}>
                  <RiskBadge label={r.decision} size="sm" />
                  {r.would_be_decision !== r.decision && (
                    <span className="mono muted" style={{ fontSize: 11 }}>would be {r.would_be_decision}</span>
                  )}
                </span>
              </div>
              {r.reasons.length > 0 && <ul>{r.reasons.map((x, j) => <li key={j}>{x}</li>)}</ul>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function IntegrationsView() {
  const { can } = useAuth();
  return (
    <div className="integrations-layout">
      <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
        {can("apikeys:manage") && <TenantSettingsCard />}
        {can("apikeys:manage") && <ApiKeysCard />}
      </div>
      {can("events:ingest") && <EventTesterCard />}
    </div>
  );
}
