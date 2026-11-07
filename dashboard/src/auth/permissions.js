// Mirrors api/auth.py PERMISSIONS (platform_admin holds everything).
const PERMISSIONS = {
  "sessions:act_any": ["integration", "admin"],
  "sessions:read_any": ["integration", "admin", "analyst", "viewer"],
  "profiles:read_any": ["admin", "analyst", "viewer"],
  "profiles:reset": ["admin"],
  "alerts:read": ["admin", "analyst", "viewer"],
  "alerts:label": ["admin", "analyst"],
  "events:ingest": ["integration", "admin"],
  "audit:read": ["admin", "analyst"],
  "apikeys:manage": ["admin"],
  "team:manage": ["admin"],
};

export function can(principal, permission) {
  const roles = principal?.roles || [];
  if (roles.includes("platform_admin")) return true;
  return (PERMISSIONS[permission] || []).some((r) => roles.includes(r));
}
