import { useEffect, useMemo, useState } from "react";

import { AuthProvider } from "./auth/AuthContext.jsx";
import { useAuth } from "./auth/useAuth.js";
import { useModelStatus } from "./hooks/useModelStatus.js";
import RiskBadge from "./components/RiskBadge.jsx";

import LoginScreen from "./views/LoginScreen.jsx";
import LiveMonitor from "./views/LiveMonitor.jsx";
import AlertsView from "./views/AlertsView.jsx";
import IntegrationsView from "./views/IntegrationsView.jsx";
import AdminGate from "./AdminGate.jsx";
import CustomersView from "./views/CustomersView.jsx";
import TeamView from "./views/TeamView.jsx";
import AccountView from "./views/AccountView.jsx";
import AcceptInvite from "./views/AcceptInvite.jsx";
import ResetPassword from "./views/ResetPassword.jsx";
import { parseHash } from "./utils/route.js";

const ROLE_LABELS = {
  platform_admin: "Platform operator",
  admin: "Tenant admin",
  analyst: "Analyst",
  viewer: "Viewer",
  integration: "Integration",
  user: "User",
};

// Each tab is shown only to roles that can use it (the API enforces the same).
// The platform operator works across customers, so they get operator tabs only.
const isOperator = (roles) => roles.includes("platform_admin");

const TABS = [
  {
    id: "customers", label: "Customers", component: CustomersView,
    visible: (_can, roles) => isOperator(roles),
  },
  {
    id: "alerts", label: "Alerts", component: AlertsView,
    visible: (can, roles) => !isOperator(roles) && can("alerts:read"),
  },
  {
    id: "monitor", label: "Live session", component: LiveMonitor,
    visible: (can, roles) => !isOperator(roles) && (can("sessions:act_any") || roles.includes("user")),
  },
  {
    id: "team", label: "Team", component: TeamView,
    visible: (can, roles) => !isOperator(roles) && can("team:manage"),
  },
  {
    id: "integrations", label: "Integrations", component: IntegrationsView,
    visible: (can, roles) => !isOperator(roles) && (can("apikeys:manage") || can("events:ingest")),
  },
  {
    id: "platform", label: "Platform settings", component: AdminGate,
    visible: (_can, roles) => isOperator(roles),
  },
  {
    id: "account", label: "Account", component: AccountView,
    visible: () => true,
  },
];

function formatUptime(seconds) {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function initialTab() {
  const { path } = parseHash();
  if (path) return path;
  // Old deployments linked to /admin-gate
  if (window.location.pathname.replace(/\/+$/, "") === "/admin-gate") return "platform";
  return null;
}

function Header({ tabs, active, onSelect }) {
  const { principal, signOut } = useAuth();
  const { modelStatus, isHealthy, uptime, error } = useModelStatus();

  return (
    <header className="app-header">
      <div className="app-header-inner">
        <div className="app-brand">
          <div className="brand-mark">Z</div>
          <div>
            <div className="brand-title">Zelvryn</div>
            <div className="brand-subtitle">{principal.tenant_id}</div>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Sections">
          {tabs.map((t) => (
            <button key={t.id} className={`nav-tab ${active === t.id ? "nav-tab-active" : ""}`}
              aria-current={active === t.id ? "page" : undefined} onClick={() => onSelect(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>

        <div className="app-meta">
          <RiskBadge label={error ? "down" : isHealthy ? "healthy" : "degraded"} size="sm" />
          {modelStatus?.active_version && (
            <span className="mono header-muted hide-sm">{modelStatus.active_version}</span>
          )}
          {uptime > 0 && <span className="mono header-muted hide-sm">{formatUptime(Math.round(uptime))}</span>}
          <div className="user-chip" title={principal.roles.join(", ")}>
            <span className="user-name">{principal.subject}</span>
            <span className="user-role">{principal.roles.map((r) => ROLE_LABELS[r] || r).join(", ")}</span>
          </div>
          <button className="btn-ghost btn-sm header-btn" onClick={() => signOut()}>Sign out</button>
        </div>
      </div>
    </header>
  );
}

function Console() {
  const { principal, can } = useAuth();
  const tabs = useMemo(
    () => TABS.filter((t) => t.visible(can, principal.roles)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [principal],
  );
  const [active, setActive] = useState(() => {
    const wanted = initialTab();
    return tabs.some((t) => t.id === wanted) ? wanted : tabs[0]?.id;
  });

  useEffect(() => {
    if (!tabs.some((t) => t.id === active)) setActive(tabs[0]?.id);
  }, [tabs, active]);

  const select = (id) => {
    setActive(id);
    window.history.replaceState(null, "", `/#/${id}`);
  };

  const Current = tabs.find((t) => t.id === active)?.component;

  return (
    <div className="app-shell">
      <Header tabs={tabs} active={active} onSelect={select} />
      <main className="app-main">
        {Current ? (
          <Current principal={principal} />
        ) : (
          <div className="card">
            This credential has no dashboard sections. API keys with the integration role are meant for
            your backend, not this console.
          </div>
        )}
      </main>
      <footer className="footer">Zelvryn · post-login behavioural protection</footer>
    </div>
  );
}

function useHashRoute() {
  const [route, setRoute] = useState(parseHash);
  useEffect(() => {
    const onChange = () => setRoute(parseHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

function Root() {
  const { principal } = useAuth();
  const { path, params } = useHashRoute();
  // Public pages reached from one-time links (work signed in or out)
  if (path === "accept-invite") return <AcceptInvite token={params.get("token")} />;
  if (path === "reset-password") return <ResetPassword token={params.get("token")} />;
  return principal ? <Console /> : <LoginScreen />;
}

export default function App() {
  return (
    <AuthProvider>
      <Root />
    </AuthProvider>
  );
}
