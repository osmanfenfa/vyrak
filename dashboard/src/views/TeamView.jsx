import { useCallback, useEffect, useState } from "react";
import { team } from "../api/client.js";
import { useAuth } from "../auth/useAuth.js";
import CopyLink from "../components/CopyLink.jsx";

const ROLES = [
  { value: "admin", label: "Admin", hint: "manages people, API keys and settings" },
  { value: "analyst", label: "Analyst", hint: "reviews and labels alerts" },
  { value: "viewer", label: "Viewer", hint: "read only" },
];

function timeAgo(ts) {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

export default function TeamView() {
  const { principal } = useAuth();
  const [people, setPeople] = useState([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("analyst");
  const [link, setLink] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      setPeople((await team.list()).items || []);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const run = async (fn) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const invite = (e) => {
    e.preventDefault();
    run(async () => {
      const res = await team.invite(email.trim(), role);
      setLink({ title: `Invite for ${res.account.email}`, url: res.invite_link,
        note: `Works once and expires in ${res.expires_in_days} days.` });
      setEmail("");
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, maxWidth: 980 }}>
      <div className="card">
        <div className="card-title">Invite someone</div>
        <form className="form-row" onSubmit={invite}>
          <label className="field">
            <span>Email</span>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label className="field">
            <span>Role</span>
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              {ROLES.map((r) => <option key={r.value} value={r.value}>{r.label} ({r.hint})</option>)}
            </select>
          </label>
          <button className="btn-teal" type="submit" disabled={busy} style={{ alignSelf: "end" }}>Create invite</button>
        </form>
        <p className="muted" style={{ fontSize: 12, margin: "8px 0 0" }}>
          Email delivery isn't connected yet. Copy the link and send it yourself.
        </p>
        {link && (
          <div style={{ marginTop: 10 }}>
            <CopyLink title={link.title} link={link.url} note={link.note} onDone={() => setLink(null)} />
          </div>
        )}
      </div>

      {error && <div className="error-bar">{error}</div>}

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div className="card-title" style={{ padding: "14px 16px 0" }}>People in {principal.tenant_id}</div>
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Person</th><th>Role</th><th>Status</th><th>Last sign-in</th><th /></tr></thead>
            <tbody>
              {people.map((p) => {
                const self = p.email === principal.subject;
                return (
                  <tr key={p.email}>
                    <td>
                      <div style={{ fontWeight: 600 }}>{p.name || "—"}{self && <span className="muted"> (you)</span>}</div>
                      <div className="muted" style={{ fontSize: 12 }}>{p.email}</div>
                    </td>
                    <td>
                      <select value={p.roles[0]} disabled={self || busy} aria-label={`Role for ${p.email}`}
                        onChange={(e) => run(() => team.update(p.email, { role: e.target.value }))}
                        style={{ minWidth: 110 }}>
                        {ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
                      </select>
                    </td>
                    <td>
                      {p.status === "active" && <span className="pill pill-allow">active</span>}
                      {p.status === "invited" && <span className="pill pill-shadow">invited</span>}
                      {p.status === "disabled" && <span className="pill pill-block">disabled</span>}
                    </td>
                    <td className="muted" style={{ fontSize: 12 }}>{timeAgo(p.last_login_at)}</td>
                    <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                      {!self && p.status === "invited" && (
                        <button className="btn-ghost btn-sm" disabled={busy} onClick={() => run(async () => {
                          const r = await team.inviteLink(p.email);
                          setLink({ title: `New invite for ${p.email}`, url: r.invite_link,
                            note: `Works once and expires in ${r.expires_in_days} days.` });
                        })}>New invite link</button>
                      )}
                      {!self && p.status === "active" && (
                        <>
                          <button className="btn-ghost btn-sm" disabled={busy} onClick={() => run(async () => {
                            const r = await team.resetLink(p.email);
                            setLink({ title: `Password reset for ${p.email}`, url: r.reset_link,
                              note: `Works once and expires in ${r.expires_in_hours} hours.` });
                          })}>Reset password</button>{" "}
                          <button className="btn-ghost btn-sm" disabled={busy} onClick={() => {
                            if (window.confirm(`Disable ${p.email}? They are signed out immediately.`)) {
                              run(() => team.update(p.email, { status: "disabled" }));
                            }
                          }}>Disable</button>
                        </>
                      )}
                      {!self && p.status === "disabled" && (
                        <button className="btn-ghost btn-sm" disabled={busy}
                          onClick={() => run(() => team.update(p.email, { status: "active" }))}>Enable</button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
