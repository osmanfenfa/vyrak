import { useState, useEffect, useCallback, useRef } from "react";
import { sessions } from "../api/client.js";

const POLL_INTERVAL_MS = 3000;
const MAX_FEED_ITEMS = 100;

export function useSession(sessionId, options = {}) {
  const { onSessionMissing } = options;
  const [summary, setSummary] = useState(null);
  const [feed, setFeed] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);
  const missingHandledRef = useRef(false);

  const handleMissingSession = useCallback(() => {
    if (missingHandledRef.current) return;
    missingHandledRef.current = true;

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setSummary(null);
    setFeed([]);
    setLoading(false);
    setError("Session expired or missing. Please start a new session.");
    onSessionMissing?.();
  }, [onSessionMissing]);

  const fetchSummary = useCallback(async () => {
    if (!sessionId) return;

    try {
      const data = await sessions.summary(sessionId);
      setSummary(data);
      setError(null);
    } catch (e) {
      if (e.status === 404) {
        handleMissingSession();
        return;
      }
      setError(e.message);
    }
  }, [sessionId, handleMissingSession]);

  useEffect(() => {
    if (!sessionId) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      missingHandledRef.current = false;
      setSummary(null);
      setFeed([]);
      return;
    }

    missingHandledRef.current = false;
    fetchSummary();
    timerRef.current = setInterval(fetchSummary, POLL_INTERVAL_MS);
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [sessionId, fetchSummary]);

  const submitAction = useCallback(
    async (actionType, resourceId = "resource", metadata = null) => {
      if (!sessionId) return null;
      setLoading(true);

      try {
        const result = await sessions.action(sessionId, actionType, resourceId, metadata);
        setFeed((prev) => [result, ...prev].slice(0, MAX_FEED_ITEMS));
        setError(null);
        return result;
      } catch (e) {
        if (e.status === 404) {
          handleMissingSession();
          return null;
        }
        setError(e.message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [sessionId, handleMissingSession],
  );

  const restoreTrust = useCallback(
    async (increment = 20) => {
      if (!sessionId) return;

      try {
        const result = await sessions.restoreTrust(sessionId, increment);
        setFeed((prev) => [result, ...prev].slice(0, MAX_FEED_ITEMS));
        await fetchSummary();
      } catch (e) {
        if (e.status === 404) {
          handleMissingSession();
          return;
        }
        setError(e.message);
      }
    },
    [sessionId, fetchSummary, handleMissingSession],
  );

  return {
    summary,
    feed,
    loading,
    error,
    submitAction,
    restoreTrust,
    refresh: fetchSummary,
  };
}

export function useSessionLifecycle() {
  const [sessionId, setSessionId] = useState(null);
  const [userId, setUserId] = useState(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState(null);

  const start = useCallback(async (uid, metadata = null) => {
    setStarting(true);
    setError(null);

    try {
      const data = await sessions.start(uid, metadata);
      setSessionId(data.session_id);
      setUserId(uid);
      return data;
    } catch (e) {
      setError(e.message);
      return null;
    } finally {
      setStarting(false);
    }
  }, []);

  const end = useCallback(async () => {
    if (!sessionId) return;

    try {
      await sessions.end(sessionId);
    } catch {
      // Ignore close failures; local UI session state still resets.
    }

    setSessionId(null);
    setUserId(null);
  }, [sessionId]);

  return { sessionId, userId, starting, error, start, end };
}
