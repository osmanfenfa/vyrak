/**
 * Zelvryn API client.
 *
 * One credential per browser tab: a JWT (from email/password sign-in or your
 * identity provider) or an API key. It is attached to every
 * request automatically. A 401 clears it and notifies the app so the user is
 * sent back to the sign-in screen.
 */

const API_BASE =
  (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

let authToken = null;
let onUnauthorized = null;

export function setAuthToken(token) {
  authToken = token || null;
}

export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

export function apiBase() {
  return API_BASE;
}

function buildQueryString(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === null || value === undefined || value === "") return;
    query.set(key, String(value));
  });
  const text = query.toString();
  return text ? `?${text}` : "";
}

async function parsePayload(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function errorFrom(response, payload) {
  const detail = payload?.detail;
  const message =
    (typeof detail === "string" && detail) ||
    payload?.message ||
    (Array.isArray(payload?.errors) && payload.errors[0]?.msg) ||
    `Request failed with status ${response.status}`;
  const error = new Error(message);
  error.status = response.status;
  error.payload = payload;
  return error;
}

async function send(path, { method = "GET", body, token, auth = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  const bearer = token ?? (auth ? authToken : null);
  if (bearer) headers.Authorization = `Bearer ${bearer}`;

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    const error = new Error(`Cannot reach the API at ${API_BASE}. Is it running?`);
    error.status = 0;
    throw error;
  }

  if (response.status === 401 && auth && !token && authToken && onUnauthorized) {
    onUnauthorized();
  }
  return response;
}

async function request(path, options = {}) {
  const response = await send(path, options);
  const payload = await parsePayload(response);
  if (!response.ok) throw errorFrom(response, payload);
  return payload;
}

async function requestBlob(path, options = {}) {
  const response = await send(path, options);
  if (!response.ok) throw errorFrom(response, await parsePayload(response));
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") || "";
  const filenameMatch = disposition.match(/filename="?([^"]+)"?/i);
  return {
    blob,
    filename: filenameMatch?.[1] || null,
    contentType: response.headers.get("Content-Type") || "application/octet-stream",
  };
}

// ── Auth & accounts ────────────────────────────────────────────────────────
export const auth = {
  /** Who does this credential belong to? Used to validate a pasted token. */
  me(token) {
    return request("/auth/me", { token });
  },
  login(email, password) {
    return request("/auth/login", { method: "POST", auth: false, body: { email, password } });
  },
  describeInvite(token) {
    return request(`/auth/invites/${encodeURIComponent(token)}`, { auth: false });
  },
  acceptInvite(token, name, password) {
    return request("/auth/accept-invite", { method: "POST", auth: false, body: { token, name, password } });
  },
  resetPassword(token, password) {
    return request("/auth/reset-password", { method: "POST", auth: false, body: { token, password } });
  },
  account() {
    return request("/auth/account");
  },
  changePassword(currentPassword, newPassword) {
    return request("/auth/change-password", {
      method: "POST",
      body: { current_password: currentPassword, new_password: newPassword },
    });
  },
};

// ── Team (tenant admin) ────────────────────────────────────────────────────
export const team = {
  list() {
    return request("/team/users");
  },
  invite(email, role, name = "") {
    return request("/team/invites", { method: "POST", body: { email, role, name } });
  },
  update(email, patch) {
    return request(`/team/users/${encodeURIComponent(email)}`, { method: "PATCH", body: patch });
  },
  inviteLink(email) {
    return request(`/team/users/${encodeURIComponent(email)}/invite-link`, { method: "POST" });
  },
  resetLink(email) {
    return request(`/team/users/${encodeURIComponent(email)}/reset-link`, { method: "POST" });
  },
};

// ── Customers (platform operator) ──────────────────────────────────────────
export const customers = {
  list() {
    return request("/platform/customers");
  },
  create(payload) {
    return request("/platform/customers", { method: "POST", body: payload });
  },
  get(tenantId) {
    return request(`/platform/customers/${encodeURIComponent(tenantId)}`);
  },
  update(tenantId, patch) {
    return request(`/platform/customers/${encodeURIComponent(tenantId)}`, { method: "PATCH", body: patch });
  },
  inviteAdmin(tenantId, email, name = "") {
    return request(`/platform/customers/${encodeURIComponent(tenantId)}/admins`, {
      method: "POST",
      body: { email, name },
    });
  },
};

// ── Health ─────────────────────────────────────────────────────────────────
export const health = {
  get() {
    return request("/health", { auth: false });
  },
  model() {
    return request("/health/model", { auth: false });
  },
};

// ── Sessions ───────────────────────────────────────────────────────────────
export const sessions = {
  start(userId, metadata = null, options = {}) {
    return request("/session/start", {
      method: "POST",
      body: {
        user_id: userId,
        metadata,
        identity_kind: options.identityKind || "human",
        peer_group: options.peerGroup || null,
      },
    });
  },
  summary(sessionId) {
    return request(`/session/${sessionId}/summary`);
  },
  action(sessionId, actionType, resourceId = "resource", metadata = null) {
    return request(`/session/${sessionId}/action`, {
      method: "POST",
      body: { action_type: actionType, resource_id: resourceId, metadata },
    });
  },
  restoreTrust(sessionId, increment = 20, reason = "re_authentication") {
    return request(`/session/${sessionId}/restore-trust`, {
      method: "POST",
      body: { increment, reason },
    });
  },
  end(sessionId) {
    return request(`/session/${sessionId}`, { method: "DELETE" });
  },
};

// ── Profiles ───────────────────────────────────────────────────────────────
export const profiles = {
  get(userId) {
    return request(`/profile/${encodeURIComponent(userId)}`);
  },
  reset(userId) {
    return request(`/profile/${encodeURIComponent(userId)}/reset`, { method: "POST" });
  },
  delete(userId) {
    return request(`/profile/${encodeURIComponent(userId)}`, { method: "DELETE" });
  },
};

// ── Alerts ─────────────────────────────────────────────────────────────────
export const alerts = {
  list(params = {}) {
    return request(`/alerts${buildQueryString(params)}`);
  },
  get(alertId) {
    return request(`/alerts/${encodeURIComponent(alertId)}`);
  },
  label(alertId, status, note = "") {
    return request(`/alerts/${encodeURIComponent(alertId)}`, {
      method: "PATCH",
      body: { status, note },
    });
  },
  exportTraining() {
    return requestBlob("/alerts/training-export");
  },
};

// ── Tenant administration ──────────────────────────────────────────────────
export const apiKeys = {
  list() {
    return request("/auth/api-keys");
  },
  create(name, roles = ["integration"]) {
    return request("/auth/api-keys", { method: "POST", body: { name, roles } });
  },
  revoke(keyId) {
    return request(`/auth/api-keys/${encodeURIComponent(keyId)}`, { method: "DELETE" });
  },
};

export const tenant = {
  getSettings() {
    return request("/tenant/settings");
  },
  patchSettings(patch) {
    return request("/tenant/settings", { method: "PATCH", body: patch });
  },
};

export const events = {
  ingest(eventsList) {
    return request("/events", { method: "POST", body: { events: eventsList } });
  },
};

// ── Platform operations (platform_admin) ───────────────────────────────────
// The first argument is kept for compatibility with AdminGate.jsx; requests
// are authorised by the signed-in token.
export const adminGate = {
  overview() {
    return request("/admin-gate");
  },
  getConfig() {
    return request("/admin-gate/config");
  },
  patchConfig(_actor, patch) {
    return request("/admin-gate/config", { method: "PATCH", body: patch });
  },
  setShadowMode(_actor, enabled) {
    return request("/admin-gate/shadow-mode", {
      method: "PATCH",
      body: { enabled: Boolean(enabled) },
    });
  },
  listModels() {
    return request("/admin-gate/models");
  },
  setCanary(_actor, version, trafficPercent = null) {
    return request("/admin-gate/models/canary", {
      method: "POST",
      body: { version, traffic_percent: trafficPercent },
    });
  },
  promoteModel(_actor, version, notes = "") {
    return request("/admin-gate/models/promote", {
      method: "POST",
      body: { version, notes },
    });
  },
  rollbackModel(_actor, targetVersion = null, notes = "") {
    return request("/admin-gate/models/rollback", {
      method: "POST",
      body: { target_version: targetVersion, notes },
    });
  },
  listAudit(_actor, params = {}) {
    return request(`/admin-gate/audit${buildQueryString(params)}`);
  },
  exportAudit(_actor, params = {}) {
    return requestBlob(`/admin-gate/audit/export${buildQueryString(params)}`);
  },
};

export function downloadBlob({ blob, filename }, fallbackName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename || fallbackName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
