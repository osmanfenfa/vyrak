/**
 * hooks/useModelStatus.js
 * Polls /health and /health/model every 10 seconds.
 */

import { useState, useEffect, useCallback } from "react";
import { health } from "../api/client.js";

const POLL_MS = 10_000;

export function useModelStatus() {
  const [systemHealth, setSystemHealth] = useState(null);
  const [modelStatus,  setModelStatus]  = useState(null);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState(null);

  const fetchAll = useCallback(async () => {
    try {
      const [sys, model] = await Promise.all([health.get(), health.model()]);
      setSystemHealth(sys);
      setModelStatus(model);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const t = setInterval(fetchAll, POLL_MS);
    return () => clearInterval(t);
  }, [fetchAll]);

  // Derived convenience booleans
  const isHealthy   = systemHealth?.status === "healthy";
  const hasModel    = Boolean(modelStatus?.active_version);
  const hasCanary   = Boolean(modelStatus?.canary_version);
  const shadowMode  = systemHealth?.shadow_mode ?? false;
  const uptime      = systemHealth?.uptime_s ?? 0;

  return {
    systemHealth, modelStatus,
    isHealthy, hasModel, hasCanary, shadowMode, uptime,
    loading, error, refresh: fetchAll,
  };
}
