import { useCallback, useEffect, useState } from "react";
import { customers } from "../api/client.js";
import CopyLink from "../components/CopyLink.jsx";

function slugify(name) {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);
}

function OnboardForm({ onCreated }) {
  const [name, setName] = useState("");
  const [tenantId, setTenantId] = useState("");
  const [idEdited, setIdEdited] = useState(false);
  const [adminEmail, setAdminEmail] = useState("");
  const [adminName, setAdminName] = useState("");
  const [shadow, setShadow] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const data = await customers.create({
        name: name.trim(),
        tenant_id: (idEdited ? tenantId : slugify(name)) || null,
        admin_email: adminEmail.trim(),
        admin_name: adminName.trim(),
        shadow_mode: shadow,
      });
      setResult(data);
      setName(""); setTenantId(""); setIdEdited(false); setAdminEmail(""); setAdminName(""); setShadow(true);
      onCreated?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">Onboard a customer</div>
      <p className="muted" style={{ marginTop: 0 }}>
        Creates the organisation and an invite link for its first admin. They set their own password
        and invite the rest of their team.
      </p>
      <form className="form-grid" onSubmit={submit}>
        <div className="form-row">
          <label className="field">
            <span>Organisation name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} required minLength={2} maxLength={120}
              placeholder="Acme Corp" />
          </label>
          <label className="field">
            <span>Organisation id</span>
            <input className="mono" value={idEdited ? tenantId : slugify(name)}
              onChange={(e) => { setIdEdited(true); setTenantId(e.target.value.toLowerCase()); }}
              pattern="[a-z0-9][a-z0-9\-_]*" maxLength={40} title="Lowercase letters, numbers, - and _" placeholder="acme-corp" />
          </label>
        </div>
        <div className="form-row">
          <label className="field">
            <span>First admin's email</span>
            <input type="email" value={adminEmail} onChange={(e) => setAdminEmail(e.target.value)} required
              placeholder="security-lead@acme.com" />
          </label>
          <label className="field">
            <span>First admin's name (optional)</span>
            <input value={adminName} onChange={(e) => setAdminName(e.target.value)} maxLength={120} />
          </label>
        </div>
        <label className="check-row">
          <input type="checkbox" checked={shadow} onChange={(e) => setShadow(e.target.checked)} />
          <span>Start in shadow mode: observe and alert only, never block (recommended for new customers)</span>
        </label>
        {error && <div className="error-bar">{error}</div>}
        <button className="btn-teal" type="submit" disabled={busy}>{busy ? "Creating…" : "Create customer"}</button>
      </form>
      {result && (
        <div style={{ marginTop: 12 }}>
          <CopyLink
            title={`${result.customer.name} is ready. Send this invite to ${result.account.email}:`}
            link={result.invite_link}
            note={`The link works once and expires in ${result.expires_in_days} days.`}
            onDone={() => setResult(null)}
          />
        </div>
      )}
    </div>
  );
}

function CustomerDetail({ tenantId, onChanged, onClose }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [adminEmail, setAdminEmail] = useState("");
  const [link, setLink] = useState(null);

  const load = useCallback(async () => {
    try {
      setData(await customers.get(tenantId));
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, [tenantId]);

  useEffect(() => { setData(null); setLink(null); load(); }, [load]);

  const setStatus = async (status) => {
    const verb = status === "suspended" ? "Suspend" : "Reactivate";
    if (status === "suspended" && !window.confirm(
      `${verb} ${data.name}? Everyone in it, and all its API keys, lose access immediately.`)) return;
    setBusy(true);
    try {
      await customers.update(tenantId, { status });
      await load();
      onChanged?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const inviteAdmin = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await customers.inviteAdmin(tenantId, adminEmail.trim());
      setLink(res);
      setAdminEmail("");
      await load();
      onChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  if (!data) return <div className="card muted">{error || "Loading…"}</div>;

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <div>
          <div className="card-title" style={{ marginBottom: 2 }}>{data.name}</div>
          <div className="mono muted" style={{ fontSize: 12 }}>{data.tenant_id}</div>
        </div>
        <button className="btn-ghost btn-sm" onClick={onClose}>Close</button>
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", margin: "12px 0" }}>
        {data.status === "active"
          ? <span className="pill pill-allow">active</span>
          : <span className="pill pill-block">suspended</span>}
        <span className="muted" style={{ fontSize: 12 }}>
          {data.active_people} active · {data.pending_invites} invited · {data.open_alerts} open alerts
        </span>
      </div>
      {data.status === "active" ? (
        <button className="btn-danger btn-sm" disabled={busy} onClick={() => setStatus("suspended")}>Suspend access</button>
      ) : (
        <button className="btn-teal btn-sm" disabled={busy} onClick={() => setStatus("active")}>Reactivate</button>
      )}

      <div className="table-wrap" style={{ marginTop: 14 }}>
        <table className="table">
          <thead><tr><th>Person</th><th>Role</th><th>Status</th></tr></thead>
          <tbody>
            {data.accounts.map((a) => (
              <tr key={a.email}>
                <td><div style={{ fontWeight: 600 }}>{a.name || "—"}</div><div className="muted" style={{ fontSize: 12 }}>{a.email}</div></td>
                <td>{a.roles.join(", ")}</td>
                <td style={{ fontSize: 12 }}>{a.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <form className="form-row" style={{ marginTop: 14 }} onSubmit={inviteAdmin}>
        <label className="field">
          <span>Invite another admin (e.g. if theirs left)</span>
          <input type="email" value={adminEmail} onChange={(e) => setAdminEmail(e.target.value)} required />
        </label>
        <button className="btn-ghost" type="submit" disabled={busy} style={{ alignSelf: "end" }}>Invite admin</button>
      </form>
      {link && (
        <div style={{ marginTop: 10 }}>
          <CopyLink title={`Invite for ${link.account.email}`} link={link.invite_link}
            note={`Works once, expires in ${link.expires_in_days} days.`} onDone={() => setLink(null)} />
        </div>
      )}
      {error && <div className="error-bar" style={{ marginTop: 8 }}>{error}</div>}
    </div>
  );
}

export default function CustomersView() {
  const [items, setItems] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(
    () =>
      customers
        .list()
        .then((data) => { setItems(data.items || []); setError(null); })
        .catch((e) => setError(e.message)),
    [],
  );

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="alerts-layout">
      <section style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
        <OnboardForm onCreated={load} />
        {error && <div className="error-bar">{error}</div>}
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div className="card-title" style={{ padding: "14px 16px 0" }}>Customers ({items.length})</div>
          {items.length === 0 ? (
            <p className="muted" style={{ padding: 16, margin: 0 }}>No customers yet. Onboard your first one above.</p>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Organisation</th><th>Status</th><th>People</th><th>Open alerts</th><th>Created</th></tr></thead>
                <tbody>
                  {items.map((c) => (
                    <tr key={c.tenant_id} tabIndex={0} className={selected === c.tenant_id ? "row-selected" : ""}
                      onClick={() => setSelected(c.tenant_id)}
                      onKeyDown={(e) => { if (e.key === "Enter") setSelected(c.tenant_id); }}>
                      <td><div style={{ fontWeight: 600 }}>{c.name}</div><div className="mono muted" style={{ fontSize: 11 }}>{c.tenant_id}</div></td>
                      <td>{c.status === "active" ? <span className="pill pill-allow">active</span> : <span className="pill pill-block">suspended</span>}</td>
                      <td style={{ fontSize: 12 }}>{c.active_people} active{c.pending_invites ? `, ${c.pending_invites} invited` : ""}</td>
                      <td>{c.open_alerts}</td>
                      <td className="muted" style={{ fontSize: 12 }}>{new Date(c.created_at * 1000).toLocaleDateString()}</td>
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
          <CustomerDetail tenantId={selected} onChanged={load} onClose={() => setSelected(null)} />
        ) : (
          <div className="card muted">Select a customer to see its people, suspend it, or invite another admin.</div>
        )}
      </aside>
    </div>
  );
}
