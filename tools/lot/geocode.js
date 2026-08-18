// Address -> lat/long. Census first (authoritative for US addresses, free, no
// key); Nominatim as a fallback for anything Census cannot match.

const USER_AGENT = 'home-lot-boundary/0.1 (https://github.com/biodoc/home)';

async function getJson(url, { timeoutMs = 20000, headers = {} } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { 'User-Agent': USER_AGENT, Accept: 'application/json', ...headers },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} from ${new URL(url).host}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

export async function geocodeCensus(address) {
  const url =
    'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress' +
    `?address=${encodeURIComponent(address)}&benchmark=Public_AR_Current&format=json`;
  const data = await getJson(url);
  const match = data?.result?.addressMatches?.[0];
  if (!match) return null;
  return {
    lon: Number(match.coordinates.x),
    lat: Number(match.coordinates.y),
    matchedAddress: match.matchedAddress,
    source: 'census',
  };
}

export async function geocodeNominatim(address) {
  const url =
    'https://nominatim.openstreetmap.org/search' +
    `?q=${encodeURIComponent(address)}&format=jsonv2&limit=1&addressdetails=0`;
  const data = await getJson(url);
  const hit = Array.isArray(data) ? data[0] : null;
  if (!hit) return null;
  return {
    lon: Number(hit.lon),
    lat: Number(hit.lat),
    matchedAddress: hit.display_name,
    source: 'nominatim',
  };
}

/**
 * Geocode an address, trying each provider in turn.
 * Returns `{ lat, lon, matchedAddress, source }` or throws if nothing matches.
 */
export async function geocode(address, { providers = [geocodeCensus, geocodeNominatim] } = {}) {
  const failures = [];
  for (const provider of providers) {
    try {
      const hit = await provider(address);
      if (hit) return hit;
      failures.push(`${provider.name}: no match`);
    } catch (err) {
      failures.push(`${provider.name}: ${err.message}`);
    }
  }
  throw new Error(`Could not geocode "${address}".\n  ${failures.join('\n  ')}`);
}
