import { useCallback, useEffect, useMemo, useState } from "react";
import { auth, setAuthToken, setUnauthorizedHandler } from "../api/client.js";
import { AuthContext } from "./useAuth.js";
import { can } from "./permissions.js";

const STORAGE_KEY = "zelvryn.auth";

function readStored() {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const stored = JSON.parse(raw);
    if (stored.expiresAt && stored.expiresAt < Date.now()) return null;
    return stored;
  } catch {
    return null;
  }
}

function writeStored(value) {
  try {
    if (value) window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    else window.sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // Storage blocked (private mode): the sign-in still works for this page.
  }
}

/** Expiry from a JWT's exp claim; API keys don't expire client-side. */
function tokenExpiry(token) {
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  try {
    const payload = JSON.parse(atob(parts[1].replace(/-/g, "+").replace(/_/g, "/")));
    return payload.exp ? payload.exp * 1000 : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }) {
  const [state, setState] = useState(() => {
    const stored = readStored();
    if (stored) setAuthToken(stored.token);
    return stored;
  });
  const [notice, setNotice] = useState(null);

  const signOut = useCallback((message = null) => {
    setAuthToken(null);
    writeStored(null);
    setState(null);
    setNotice(message);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(() => signOut("Your sign-in expired or was revoked. Please sign in again."));
    return () => setUnauthorizedHandler(null);
  }, [signOut]);

  // Re-validate a restored credential (it may have been revoked).
  useEffect(() => {
    if (!state?.token) return;
    auth.me(state.token).catch((e) => {
      if (e.status === 401) signOut("Your sign-in expired or was revoked. Please sign in again.");
    });
    // Only on first load:
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sign out automatically when a JWT expires while the page is open.
  useEffect(() => {
    if (!state?.expiresAt) return;
    const ms = state.expiresAt - Date.now();
    const t = setTimeout(() => signOut("Your sign-in expired. Please sign in again."), Math.max(0, ms));
    return () => clearTimeout(t);
  }, [state?.expiresAt, signOut]);

  const signInWithToken = useCallback(async (rawToken) => {
    const token = rawToken.trim();
    const principal = await auth.me(token);
    const next = { token, principal, expiresAt: tokenExpiry(token) };
    setAuthToken(token);
    writeStored(next);
    setNotice(null);
    setState(next);
    return principal;
  }, []);

  /** Store a session returned by login / accept-invite / reset / change-password. */
  const startSession = useCallback(
    (sessionResponse) => signInWithToken(sessionResponse.access_token),
    [signInWithToken],
  );

  const signInWithPassword = useCallback(
    async (email, password) => startSession(await auth.login(email, password)),
    [startSession],
  );

  const value = useMemo(
    () => ({
      principal: state?.principal || null,
      token: state?.token || null,
      notice,
      signInWithToken,
      signInWithPassword,
      startSession,
      signOut,
      can: (permission) => can(state?.principal, permission),
    }),
    [state, notice, signInWithToken, signInWithPassword, startSession, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
