const API_BASE =
  (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

const USER_STORAGE_KEY = "zelvryn.user_id";

function getStoredUserId() {
  try {
    return window.localStorage.getItem(USER_STORAGE_KEY);
  } catch {
    return null;
  }
}

function setStoredUserId(userId) {
  try {
    window.localStorage.setItem(USER_STORAGE_KEY, userId);
  } catch {
    // Ignore storage failures in restricted browser contexts.
  }
}

function normalizeUserId(userId) {
  return typeof userId === "string" ? userId.trim() : "";
}

function buildHeaders({ authToken, extraHeaders } = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(extraHeaders || {}),
  };

  if (authToken) {
    headers.Authorization = `Bearer ${authToken}`;
  }

  return headers;
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

async function request(path, options = {}) {
  const {
    method = "GET",
    body,
    authToken,
    extraHeaders,
  } = options;

  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: buildHeaders({ authToken, extraHeaders }),
    body: body ? JSON.stringify(body) : undefined,
  });

  const payload = await parsePayload(response);

  if (!response.ok) {
    const message =
      payload?.detail ||
      payload?.message ||
      `Request failed with status ${response.status}`;

    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  return payload;
}

async function requestBlob(path, options = {}) {
  const {
    method = "GET",
    body,
    authToken,
    extraHeaders,
  } = options;

  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: buildHeaders({ authToken, extraHeaders }),
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const payload = await parsePayload(response);
    const message =
      payload?.detail ||
      payload?.message ||
      `Request failed with status ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") || "";
  const filenameMatch = disposition.match(/filename="?([^"]+)"?/i);
  return {
    blob,
    filename: filenameMatch?.[1] || null,
    contentType: response.headers.get("Content-Type") || "application/octet-stream",
  };
}

function authForUser(userId) {
  const explicitUser = normalizeUserId(userId);
  if (explicitUser) return explicitUser;

  const storedUser = normalizeUserId(getStoredUserId());
  if (storedUser) return storedUser;

  return null;
}

function requireAuthUser(userId, contextLabel) {
  const resolved = authForUser(userId);
  if (resolved) return resolved;

  const error = new Error(
    `No user available for ${contextLabel}. Start a session with a valid user ID first.`,
  );
  error.status = 400;
  throw error;
}

export const health = {
  get() {
    return request("/health");
  },
  model() {
    return request("/health/model");
  },
};

export const sessions = {
  async start(userId, metadata = null) {
    const cleanUserId = normalizeUserId(userId);
    if (!cleanUserId) {
      const error = new Error("User ID is required to start a session.");
      error.status = 400;
      throw error;
    }

    const token = requireAuthUser(cleanUserId, "session start");
    const data = await request("/session/start", {
      method: "POST",
      authToken: token,
      body: {
        user_id: cleanUserId,
        metadata,
      },
    });
    setStoredUserId(cleanUserId);
    return data;
  },

  summary(sessionId) {
    return request(`/session/${sessionId}/summary`, {
      authToken: requireAuthUser(null, "session summary"),
    });
  },

  action(sessionId, actionType, resourceId = "resource", metadata = null) {
    return request(`/session/${sessionId}/action`, {
      method: "POST",
      authToken: requireAuthUser(null, "session action"),
      body: {
        action_type: actionType,
        resource_id: resourceId,
        metadata,
      },
    });
  },

  restoreTrust(sessionId, increment = 20, reason = "re_authentication") {
    return request(`/session/${sessionId}/restore-trust`, {
      method: "POST",
      authToken: requireAuthUser(null, "restore trust"),
      body: {
        increment,
        reason,
      },
    });
  },

  end(sessionId) {
    return request(`/session/${sessionId}`, {
      method: "DELETE",
      authToken: requireAuthUser(null, "session end"),
    });
  },
};

export const profiles = {
  get(userId) {
    return request(`/profile/${encodeURIComponent(userId)}`, {
      authToken: requireAuthUser(userId, "profile read"),
    });
  },

  reset(userId) {
    return request(`/profile/${encodeURIComponent(userId)}/reset`, {
      method: "POST",
      authToken: requireAuthUser(userId, "profile reset"),
    });
  },

  delete(userId) {
    return request(`/profile/${encodeURIComponent(userId)}`, {
      method: "DELETE",
      authToken: requireAuthUser(userId, "profile delete"),
    });
  },
};

export const adminGate = {
  overview(userId) {
    return request("/admin-gate", {
      authToken: requireAuthUser(userId, "admin overview"),
    });
  },

  getConfig(userId) {
    return request("/admin-gate/config", {
      authToken: requireAuthUser(userId, "admin config read"),
    });
  },

  patchConfig(userId, patch) {
    return request("/admin-gate/config", {
      method: "PATCH",
      authToken: requireAuthUser(userId, "admin config patch"),
      body: patch,
    });
  },

  setShadowMode(userId, enabled) {
    return request("/admin-gate/shadow-mode", {
      method: "PATCH",
      authToken: requireAuthUser(userId, "admin shadow mode patch"),
      body: { enabled: Boolean(enabled) },
    });
  },

  listModels(userId) {
    return request("/admin-gate/models", {
      authToken: requireAuthUser(userId, "admin model list"),
    });
  },

  setCanary(userId, version, trafficPercent = null) {
    return request("/admin-gate/models/canary", {
      method: "POST",
      authToken: requireAuthUser(userId, "admin canary promote"),
      body: {
        version,
        traffic_percent: trafficPercent,
      },
    });
  },

  promoteModel(userId, version, notes = "") {
    return request("/admin-gate/models/promote", {
      method: "POST",
      authToken: requireAuthUser(userId, "admin model promote"),
      body: { version, notes },
    });
  },

  rollbackModel(userId, targetVersion = null, notes = "") {
    return request("/admin-gate/models/rollback", {
      method: "POST",
      authToken: requireAuthUser(userId, "admin model rollback"),
      body: {
        target_version: targetVersion,
        notes,
      },
    });
  },

  listAudit(userId, params = {}) {
    const qs = buildQueryString(params);
    return request(`/admin-gate/audit${qs}`, {
      authToken: requireAuthUser(userId, "admin audit query"),
    });
  },

  exportAudit(userId, params = {}) {
    const qs = buildQueryString(params);
    return requestBlob(`/admin-gate/audit/export${qs}`, {
      authToken: requireAuthUser(userId, "admin audit export"),
    });
  },
};
