const IP_GEO_ENDPOINT = "https://ipapi.co/json/";
const GEO_TIMEOUT_MS = 9000;

function buildDeviceDescriptor() {
  const platform = navigator.platform || "unknown-platform";
  const cores = navigator.hardwareConcurrency || "na";
  const memory = navigator.deviceMemory || "na";
  const language = navigator.language || "na";

  return `${platform}|cores:${cores}|mem:${memory}|lang:${language}`;
}

async function fetchIpAndCoarseGeo() {
  try {
    const response = await fetch(IP_GEO_ENDPOINT);
    if (!response.ok) return null;

    const payload = await response.json();
    return {
      ip: payload?.ip || "",
      geo_country: payload?.country_code || payload?.country_name || "",
      geo_city: payload?.city || "",
      latitude:
        typeof payload?.latitude === "number" ? payload.latitude : null,
      longitude:
        typeof payload?.longitude === "number" ? payload.longitude : null,
    };
  } catch {
    return null;
  }
}

async function fetchPreciseGeoLocation() {
  if (!("geolocation" in navigator)) return null;

  try {
    const position = await new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, {
        enableHighAccuracy: true,
        timeout: GEO_TIMEOUT_MS,
      });
    });

    return {
      latitude: position.coords.latitude,
      longitude: position.coords.longitude,
    };
  } catch {
    return null;
  }
}

export async function collectDeviceMetadata() {
  const [ipGeo, preciseGeo] = await Promise.all([
    fetchIpAndCoarseGeo(),
    fetchPreciseGeoLocation(),
  ]);

  const latitude = preciseGeo?.latitude ?? ipGeo?.latitude ?? null;
  const longitude = preciseGeo?.longitude ?? ipGeo?.longitude ?? null;

  return {
    ip: ipGeo?.ip || "",
    device: buildDeviceDescriptor(),
    user_agent: navigator.userAgent || "",
    geo_country: ipGeo?.geo_country || "",
    geo_city: ipGeo?.geo_city || "",
    latitude,
    longitude,
    screen_resolution: `${window.screen?.width || 0}x${window.screen?.height || 0}`,
    timezone_offset: new Date().getTimezoneOffset(),
  };
}
