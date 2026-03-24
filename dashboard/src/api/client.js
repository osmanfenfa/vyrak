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
  return userId || getStoredUserId() || "alice@example.com";
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
    const token = authForUser(userId);
    const data = await request("/session/start", {
      method: "POST",
      authToken: token,
      body: {
        user_id: userId,
        metadata,
      },
    });
    setStoredUserId(userId);
    return data;
  },

  summary(sessionId) {
    return request(`/session/${sessionId}/summary`, {
      authToken: authForUser(),
    });
  },

  action(sessionId, actionType, resourceId = "resource", metadata = null) {
    return request(`/session/${sessionId}/action`, {
      method: "POST",
      authToken: authForUser(),
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
      authToken: authForUser(),
      body: {
        increment,
        reason,
      },
    });
  },

  end(sessionId) {
    return request(`/session/${sessionId}`, {
      method: "DELETE",
      authToken: authForUser(),
    });
  },
};

export const profiles = {
  get(userId) {
    return request(`/profile/${encodeURIComponent(userId)}`, {
      authToken: authForUser(userId),
    });
  },

  reset(userId) {
    return request(`/profile/${encodeURIComponent(userId)}/reset`, {
      method: "POST",
      authToken: authForUser(userId),
    });
  },

  delete(userId) {
    return request(`/profile/${encodeURIComponent(userId)}`, {
      method: "DELETE",
      authToken: authForUser(userId),
    });
  },
};
