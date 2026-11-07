import { useEffect, useState } from "react";
import { auth } from "../api/client.js";
import { useAuth } from "../auth/useAuth.js";
import { goTo } from "../utils/route.js";
import { AuthCard } from "./LoginScreen.jsx";

const ROLE_LABELS = { admin: "Admin", analyst: "Analyst", viewer: "Viewer", platform_admin: "Platform operator" };
const MIN_LENGTH = 10;

export default function AcceptInvite({ token }) {
  const { startSession, signOut } = useAuth();
  const [invite, setInvite] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!token) {
      setLoadError("This invite link is incomplete. Ask your admin to send it again.");
      return;
    }
    auth.describeInvite(token)
      .then((data) => { setInvite(data); setName(data.name || ""); })
      .catch((e) => setLoadError(e.status === 404 ? "This invite link is invalid, already used or expired. Ask your admin for a new one." : e.message));
  }, [token]);

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    if (password.length < MIN_LENGTH) return setError(`Use at least ${MIN_LENGTH} characters.`);
    if (password !== confirm) return setError("The passwords don't match.");
    setBusy(true);
    try {
      signOut();                       // never mix with another signed-in session
      await startSession(await auth.acceptInvite(token, name.trim(), password));
      goTo("");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  if (loadError) {
    return (
      <AuthCard subtitle="Join your organisation">
        <div className="error-bar">{loadError}</div>
        <button className="btn-ghost" style={{ marginTop: 12, width: "100%" }} onClick={() => goTo("")}>Go to sign in</button>
      </AuthCard>
    );
  }
  if (!invite) return <AuthCard subtitle="Join your organisation"><p className="muted">Checking your invite…</p></AuthCard>;

  return (
    <AuthCard subtitle="Join your organisation">
      <form onSubmit={submit} className="form-grid">
        <div className="notice" style={{ margin: 0 }}>
          You've been invited to <strong>{invite.organisation}</strong> as{" "}
          <strong>{invite.roles.map((r) => ROLE_LABELS[r] || r).join(", ")}</strong>.
        </div>
        <label className="field">
          <span>Email</span>
          <input value={invite.email} disabled autoComplete="username" />
        </label>
        <label className="field">
          <span>Your name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={120} autoFocus />
        </label>
        <label className="field">
          <span>Choose a password (at least {MIN_LENGTH} characters)</span>
          <input type="password" autoComplete="new-password" value={password}
            onChange={(e) => setPassword(e.target.value)} required />
        </label>
        <label className="field">
          <span>Confirm password</span>
          <input type="password" autoComplete="new-password" value={confirm}
            onChange={(e) => setConfirm(e.target.value)} required />
        </label>
        {error && <div className="error-bar">{error}</div>}
        <button className="btn-teal" type="submit" disabled={busy}>{busy ? "Creating account…" : "Create account"}</button>
      </form>
    </AuthCard>
  );
}
