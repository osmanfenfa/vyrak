/**
 * hooks/useProfile.js
 * Fetches and manages a user's behavioural profile.
 */

import { useState, useEffect, useCallback } from "react";
import { profiles } from "../api/client.js";

export function useProfile(userId) {
  const [profile,  setProfile]  = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);

  const fetchProfile = useCallback(async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const data = await profiles.get(userId);
      setProfile(data);
      setError(null);
    } catch (e) {
      setError(e.message);
      if (e.status === 404) setProfile(null);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  const resetProfile = useCallback(async () => {
    if (!userId) return;
    setLoading(true);
    try {
      await profiles.reset(userId);
      await fetchProfile();   // reload after reset
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [userId, fetchProfile]);

  const deleteProfile = useCallback(async () => {
    if (!userId) return;
    try {
      await profiles.delete(userId);
      setProfile(null);
    } catch (e) {
      setError(e.message);
    }
  }, [userId]);

  return { profile, loading, error, refresh: fetchProfile, resetProfile, deleteProfile };
}
