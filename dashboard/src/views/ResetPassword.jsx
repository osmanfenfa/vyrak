import { useState } from "react";
import { auth } from "../api/client.js";
import { useAuth } from "../auth/useAuth.js";
import { goTo } from "../utils/route.js";
import { AuthCard } from "./LoginScreen.jsx";

const MIN_LENGTH = 10;

export default function ResetPassword({ token }) {
  const { startSession, signOut } = useAuth();
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    if (password.length < MIN_LENGTH) return setError(`Use at least ${MIN_LENGTH} characters.`);
    if (password !== confirm) return setError("The passwords don't match.");
    setBusy(true);
    try {
      signOut();
      await startSession(await auth.resetPassword(token, password));
      goTo("");
    } catch (err) {
      setError(err.status === 404 ? "This reset link is invalid, already used or expired. Ask your admin for a new one." : err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthCard subtitle="Choose a new password">
      <form onSubmit={submit} className="form-grid">
        <label className="field">
          <span>New password (at least {MIN_LENGTH} characters)</span>
          <input type="password" autoComplete="new-password" value={password}
            onChange={(e) => setPassword(e.target.value)} required autoFocus />
        </label>
        <label className="field">
          <span>Confirm new password</span>
          <input type="password" autoComplete="new-password" value={confirm}
            onChange={(e) => setConfirm(e.target.value)} required />
        </label>
        <p className="muted" style={{ fontSize: 12, margin: 0 }}>This signs you out on every other device.</p>
        {error && <div className="error-bar">{error}</div>}
        <button className="btn-teal" type="submit" disabled={busy || !token}>{busy ? "Saving…" : "Set new password"}</button>
      </form>
    </AuthCard>
  );
}
