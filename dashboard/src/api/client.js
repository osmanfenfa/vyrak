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

  const text = await response.text();
  let payload = null;

  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

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
