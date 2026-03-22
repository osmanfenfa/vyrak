/**
 * hooks/useSession.js
 * Polls a session summary and maintains a local action feed.
 *
 * Usage:
 *   const { summary, feed, loading, error, submitAction } = useSession(sessionId);
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { sessions } from "../api/client.js";

const POLL_INTERVAL_MS = 3000;
const MAX_FEED_ITEMS   = 100;

export function useSession(sessionId) {
  const [summary,  setSummary]  = useState(null);
  const [feed,     setFeed]     = useState([]);   // latest action results
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);
  const timerRef               = useRef(null);

  // ── Poll session summary ──────────────────────────────────────────────────
  const fetchSummary = useCallback(async () => {
    if (!sessionId) return;
    try {
      const data = await sessions.summary(sessionId);
      setSummary(data);
      setError(null);
    } catch (e) {
      if (e.status !== 404) setError(e.message);
    }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) { setSummary(null); setFeed([]); return; }

    fetchSummary();
    timerRef.current = setInterval(fetchSummary, POLL_INTERVAL_MS);
    return () => clearInterval(timerRef.current);
  }, [sessionId, fetchSummary]);

  // ── Submit an action and append result to feed ────────────────────────────
  const submitAction = useCallback(async (actionType, resourceId = "resource") => {
    if (!sessionId) return null;
    setLoading(true);
    try {
      const result = await sessions.action(sessionId, actionType, resourceId);
      setFeed(prev => [result, ...prev].slice(0, MAX_FEED_ITEMS));
      setError(null);
      return result;
    } catch (e) {
      setError(e.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  // ── Restore trust after challenge ─────────────────────────────────────────
  const restoreTrust = useCallback(async (increment = 20) => {
    if (!sessionId) return;
    try {
      const result = await sessions.restoreTrust(sessionId, increment);
      setFeed(prev => [result, ...prev].slice(0, MAX_FEED_ITEMS));
      await fetchSummary();
    } catch (e) {
      setError(e.message);
    }
  }, [sessionId, fetchSummary]);

  return { summary, feed, loading, error, submitAction, restoreTrust, refresh: fetchSummary };
}

// ── Session lifecycle hook ────────────────────────────────────────────────────

export function useSessionLifecycle() {
  const [sessionId, setSessionId] = useState(null);
  const [userId,    setUserId]    = useState(null);
  const [starting,  setStarting]  = useState(false);
  const [error,     setError]     = useState(null);

  const start = useCallback(async (uid) => {
    setStarting(true);
    setError(null);
    try {
      const data = await sessions.start(uid);
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
    try { await sessions.end(sessionId); } catch {}
    setSessionId(null);
    setUserId(null);
  }, [sessionId]);

  return { sessionId, userId, starting, error, start, end };
}
