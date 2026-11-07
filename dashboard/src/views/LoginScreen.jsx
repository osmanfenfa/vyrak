import { useState } from "react";
import { useAuth } from "../auth/useAuth.js";

export function AuthCard({ children, subtitle }) {
  return (
    <div className="login-shell">
      <div className="card login-card">
        <div className="app-brand" style={{ marginBottom: 16 }}>
          <div className="brand-mark">Z</div>
          <div>
            <div className="brand-title" style={{ color: "var(--navy)" }}>Zelvryn</div>
            <div className="muted" style={{ fontSize: 13 }}>
              {subtitle || "Post-login protection for people, AI agents and service accounts"}
            </div>
          </div>
        </div>
        {children}
      </div>
    </div>
  );
}

export default function LoginScreen() {
  const { signInWithPassword, signInWithToken, notice } = useAuth();
  const [mode, setMode] = useState("password");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "password") await signInWithPassword(email.trim(), password);
      else await signInWithToken(token);
    } catch (err) {
      if (mode === "token" && err.status === 401) {
        setError("That token was rejected: it is invalid, expired or revoked.");
      } else {
        setError(err.message);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthCard>
      <form onSubmit={submit} className="form-grid">
        {notice && <div className="notice">{notice}</div>}

        {mode === "password" ? (
          <>
            <label className="field">
              <span>Email</span>
              <input type="email" autoComplete="username" value={email}
                onChange={(e) => setEmail(e.target.value)} required autoFocus />
            </label>
            <label className="field">
              <span>Password</span>
              <input type="password" autoComplete="current-password" value={password}
                onChange={(e) => setPassword(e.target.value)} required />
            </label>
          </>
        ) : (
          <label className="field">
            <span>API key or identity-provider token</span>
            <textarea rows={4} value={token} onChange={(e) => setToken(e.target.value)} required
              placeholder="zk_…  or  eyJhbGciOi…" className="mono" spellCheck={false} />
          </label>
        )}

        {error && <div className="error-bar">{error}</div>}

        <button className="btn-teal" type="submit" disabled={busy} style={{ width: "100%" }}>
          {busy ? "Signing in…" : "Sign in"}
        </button>

        <div className="muted" style={{ fontSize: 12, textAlign: "center" }}>
          {mode === "password" ? (
            <>
              Forgot your password? Ask your organisation's admin for a reset link.
              <br />
              <button type="button" className="link-btn" onClick={() => { setMode("token"); setError(null); }}>
                Sign in with an API key or SSO token instead
              </button>
            </>
          ) : (
            <button type="button" className="link-btn" onClick={() => { setMode("password"); setError(null); }}>
              Sign in with email and password
            </button>
          )}
        </div>
      </form>
    </AuthCard>
  );
}
