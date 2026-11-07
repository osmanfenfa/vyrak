/** Hash routes: "#/alerts", "#/accept-invite?token=…". */
export function parseHash() {
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?");
  return { path, params: new URLSearchParams(query) };
}

export function goTo(path) {
  window.history.replaceState(null, "", `/#/${path}`);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}
