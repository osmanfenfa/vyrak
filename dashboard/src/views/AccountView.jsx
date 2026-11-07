import { useEffect, useState } from "react";
import { auth } from "../api/client.js";
import { useAuth } from "../auth/useAuth.js";

const MIN_LENGTH = 10;

export default function AccountView() {
  const { principal, startSession } = useAuth();
  const [info, setInfo] = useState(null);
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    auth.account().then(setInfo).catch((e) => setError(e.message));
  }, []);

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    setDone(false);
    if (next.length < MIN_LENGTH) return setError(`Use at least ${MIN_LENGTH} characters.`);
    if (next !== confirm) return setError("The new passwords don't match.");
    setBusy(true);
    try {
      await startSession(await auth.changePassword(current, next));
      setCurrent(""); setNext(""); setConfirm("");
      setDone(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const account = info?.account;

  return (
    <div style={{ display: "grid", gap: 12, maxWidth: 560 }}>
      <div className="card">
        <div className="card-title">Your account</div>
        <dl className="kv">
          <dt>Name</dt><dd>{account?.name || "—"}</dd>
          <dt>Email</dt><dd>{principal.subject}</dd>
          <dt>Organisation</dt><dd>{info?.organisation || principal.tenant_id}</dd>
          <dt>Role</dt><dd>{principal.roles.join(", ")}</dd>
        </dl>
      </div>

      {account && (
        <form className="card form-grid" onSubmit={submit}>
          <div className="card-title" style={{ marginBottom: 0 }}>Change password</div>
          <input type="text" autoComplete="username" value={principal.subject} readOnly hidden />
          <label className="field">
            <span>Current password</span>
            <input type="password" autoComplete="current-password" value={current}
              onChange={(e) => setCurrent(e.target.value)} required />
          </label>
          <label className="field">
            <span>New password (at least {MIN_LENGTH} characters)</span>
            <input type="password" autoComplete="new-password" value={next}
              onChange={(e) => setNext(e.target.value)} required />
          </label>
          <label className="field">
            <span>Confirm new password</span>
            <input type="password" autoComplete="new-password" value={confirm}
              onChange={(e) => setConfirm(e.target.value)} required />
          </label>
          {error && <div className="error-bar">{error}</div>}
          {done && <div className="notice" style={{ margin: 0 }}>Password changed. Other devices have been signed out.</div>}
          <button className="btn-teal" type="submit" disabled={busy}>{busy ? "Saving…" : "Change password"}</button>
        </form>
      )}
    </div>
  );
}
