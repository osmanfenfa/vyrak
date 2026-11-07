import { useCallback, useEffect, useState } from "react";
import { alerts as alertsApi, downloadBlob } from "../api/client.js";
import { useAuth } from "../auth/useAuth.js";
import RiskBadge from "../components/RiskBadge.jsx";

const POLL_MS = 15_000;

const STATUS_LABELS = {
  open: "Open",
  confirmed_malicious: "Confirmed malicious",
  false_positive: "False positive",
  benign_expected: "Expected behaviour",
};

const KIND_LABELS = { human: "Person", agent: "AI agent", service: "Service" };

function timeAgo(ts) {
  if (!ts) return "–";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

function Severity({ value }) {
  const cls = { high: "pill-block", medium: "pill-challenge", low: "pill-degraded" }[value] || "pill-degraded";
  return <span className={`pill ${cls}`}>{value}</span>;
}

function AlertDetail({ alertId, canLabel, onLabelled, onClose }) {
  const [alert, setAlert] = useState(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setAlert(null);
    setError(null);
    alertsApi.get(alertId)
      .then((a) => { if (!cancelled) { setAlert(a); setNote(a.label?.note || ""); } })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [alertId]);

  const label = async (status) => {
    setBusy(true);
    setError(null);
    try {
      const updated = await alertsApi.label(alertId, status, note);
      setAlert(updated);
      onLabelled?.(updated);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  if (error && !alert) return <div className="card"><div className="error-bar">{error}</div></div>;
  if (!alert) return <div className="card muted">Loading alert…</div>;

  return (
    <div className="card alert-detail">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "flex-start" }}>
        <div>
          <div className="card-title" style={{ marginBottom: 4 }}>
            {alert.identity_id}{" "}
            <span className="muted" style={{ fontWeight: 400 }}>· {KIND_LABELS[alert.identity_kind] || alert.identity_kind}</span>
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
            <Severity value={alert.severity} />
            <RiskBadge label={alert.latest_decision} size="sm" />
            {!alert.enforced && <span className="pill pill-shadow">shadow</span>}
            <span className="muted" style={{ fontSize: 12 }}>
              trust {Number(alert.latest_trust ?? 0).toFixed(0)} · {STATUS_LABELS[alert.status]}
            </span>
          </div>
        </div>
        <button className="btn-ghost btn-sm" onClick={onClose}>Close</button>
      </div>

      <p style={{ margin: "12px 0" }}>{alert.summary}</p>

      <div className="muted mono" style={{ fontSize: 11, marginBottom: 6 }}>
        Signals: {(alert.anomaly_types || []).join(", ")}
      </div>

      <div className="timeline scroll-y">
        {[...alert.events].reverse().map((e, i) => (
          <div key={i} className="timeline-item">
            <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
              <span className="mono"><strong>{e.action_type}</strong> · {new Date(e.timestamp * 1000).toLocaleString()}</span>
              <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <span className="mono muted">trust {Number(e.trust_score).toFixed(0)}</span>
                <RiskBadge label={e.decision} size="sm" />
              </span>
            </div>
            <ul>
              {(e.reasons?.length ? e.reasons : e.anomalies).map((r, j) => <li key={j}>{r}</li>)}
            </ul>
          </div>
        ))}
      </div>

      {canLabel ? (
        <div className="label-box">
          <label className="field">
            <span>Analyst note (optional)</span>
            <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)} maxLength={2000} />
          </label>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="btn-danger btn-sm" disabled={busy} onClick={() => label("confirmed_malicious")}>Confirm malicious</button>
            <button className="btn-ghost btn-sm" disabled={busy} onClick={() => label("false_positive")}>False positive</button>
            <button className="btn-ghost btn-sm" disabled={busy} onClick={() => label("benign_expected")}>Expected behaviour</button>
            {alert.status !== "open" && (
              <button className="btn-ghost btn-sm" disabled={busy} onClick={() => label("open")}>Reopen</button>
            )}
          </div>
          {alert.label && (
            <div className="muted" style={{ fontSize: 12 }}>
              Labelled “{STATUS_LABELS[alert.label.status]}” by {alert.label.analyst} {timeAgo(alert.label.at)}
            </div>
          )}
          {error && <div className="error-bar">{error}</div>}
        </div>
      ) : (
        <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>Viewers can read alerts but not label them.</div>
      )}
    </div>
  );
}

export default function AlertsView() {
  const { can } = useAuth();
  const canLabel = can("alerts:label");
  const [filters, setFilters] = useState({ status: "open", severity: "", identity_id: "" });
  const [data, setData] = useState({ total: 0, items: [] });
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [exporting, setExporting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await alertsApi.list({ ...filters, limit: 100 }));
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    load();
    const t = setInterval(load, POLL_MS);
    return () => clearInterval(t);
  }, [load]);

  const exportTraining = async () => {
    setExporting(true);
    try {
      downloadBlob(await alertsApi.exportTraining(), "labelled_alerts.jsonl");
    } catch (e) {
      setError(e.message);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="alerts-layout">
      <section style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <div className="card-title" style={{ margin: 0 }}>Alerts <span className="muted">({data.total})</span></div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn-ghost btn-sm" onClick={load} disabled={loading}>{loading ? "Refreshing…" : "Refresh"}</button>
              {canLabel && (
                <button className="btn-ghost btn-sm" onClick={exportTraining} disabled={exporting}
                  title="Labelled alerts as training data for scripts/train_from_labels.py">
                  {exporting ? "Exporting…" : "Export training data"}
                </button>
              )}
            </div>
          </div>
          <div className="filter-row">
            <select value={filters.status} onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))}>
              <option value="">All statuses</option>
              {Object.entries(STATUS_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            <select value={filters.severity} onChange={(e) => setFilters((f) => ({ ...f, severity: e.target.value }))}>
              <option value="">All severities</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            <input placeholder="Identity" value={filters.identity_id}
              onChange={(e) => setFilters((f) => ({ ...f, identity_id: e.target.value.trim() }))} />
          </div>
        </div>

        {error && <div className="error-bar">{error}</div>}

        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          {data.items.length === 0 ? (
            <p className="muted" style={{ padding: 16, margin: 0 }}>
              {loading ? "Loading…" : "No alerts match these filters. Alerts appear when an identity does something unusual."}
            </p>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr><th>Severity</th><th>Identity</th><th>What happened</th><th>Decision</th><th>Status</th><th>Updated</th></tr>
                </thead>
                <tbody>
                  {data.items.map((a) => (
                    <tr key={a.alert_id} onClick={() => setSelected(a.alert_id)}
                      className={selected === a.alert_id ? "row-selected" : ""} tabIndex={0}
                      onKeyDown={(e) => { if (e.key === "Enter") setSelected(a.alert_id); }}>
                      <td><Severity value={a.severity} /></td>
                      <td>
                        <div style={{ fontWeight: 600 }}>{a.identity_id}</div>
                        <div className="muted" style={{ fontSize: 11 }}>{KIND_LABELS[a.identity_kind] || a.identity_kind}</div>
                      </td>
                      <td className="cell-summary">{a.summary}</td>
                      <td>
                        <RiskBadge label={a.latest_decision} size="sm" />
                        {!a.enforced && <div className="muted" style={{ fontSize: 10 }}>shadow</div>}
                      </td>
                      <td style={{ fontSize: 12 }}>{STATUS_LABELS[a.status]}</td>
                      <td className="muted" style={{ fontSize: 12, whiteSpace: "nowrap" }}>{timeAgo(a.updated_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <aside>
        {selected ? (
          <AlertDetail alertId={selected} canLabel={canLabel} onClose={() => setSelected(null)}
            onLabelled={() => load()} />
        ) : (
          <div className="card muted">Select an alert to see its timeline and the reasons behind each decision.</div>
        )}
      </aside>
    </div>
  );
}
