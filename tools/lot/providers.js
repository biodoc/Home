// Parcel-geometry providers: point -> lot polygon.
//
// Zillow is deliberately absent. It has no public parcel API (the old one was
// retired in 2021), its lot-line overlay is licensed vendor data served as
// vector tiles, and scraping it violates its terms of use. Parcel geometry
// comes from the record holders instead: county/state assessor GIS layers, or
// a licensed national aggregator.

import { readFile } from 'node:fs/promises';
import { esriPolygonToGeoJSON, pointInGeometry, polygonsOf } from './geometry.js';

const USER_AGENT = 'home-lot-boundary/0.1 (https://github.com/biodoc/home)';

async function getJson(url, { timeoutMs = 30000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { 'User-Agent': USER_AGENT, Accept: 'application/json' },
    });
    const text = await res.text();
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} from ${new URL(url).host}: ${text.slice(0, 200)}`);
    }
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(`Non-JSON response from ${new URL(url).host}: ${text.slice(0, 200)}`);
    }
    // ArcGIS reports errors with HTTP 200 and an `error` member.
    if (data?.error) {
      const e = data.error;
      throw new Error(`ArcGIS error ${e.code}: ${e.message}${e.details?.length ? ` (${e.details.join('; ')})` : ''}`);
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

/** Strip a trailing /query so callers can paste either form of a layer URL. */
export function normalizeLayerUrl(url) {
  return url.replace(/\/+$/, '').replace(/\/query$/i, '');
}

/** Fetch a layer's metadata — used to validate a URL before relying on it. */
export async function describeArcgisLayer(layerUrl) {
  const meta = await getJson(`${normalizeLayerUrl(layerUrl)}?f=json`);
  return {
    name: meta.name,
    type: meta.type,
    geometryType: meta.geometryType,
    fields: (meta.fields || []).map((f) => f.name),
    maxRecordCount: meta.maxRecordCount,
  };
}

/**
 * Query an ArcGIS FeatureServer/MapServer layer for the parcel containing a
 * point. Works against any public county or state parcel service.
 */
export async function fetchParcelFromArcgis({ lat, lon, layerUrl }) {
  const base = normalizeLayerUrl(layerUrl);
  const params = new URLSearchParams({
    f: 'json',
    geometry: JSON.stringify({ x: lon, y: lat, spatialReference: { wkid: 4326 } }),
    geometryType: 'esriGeometryPoint',
    inSR: '4326',
    outSR: '4326', // KML requires WGS84; make the server reproject.
    spatialRel: 'esriSpatialRelIntersects',
    outFields: '*',
    returnGeometry: 'true',
    where: '1=1',
  });

  const data = await getJson(`${base}/query?${params}`);
  const features = data?.features || [];
  if (features.length === 0) return null;

  // A point on a shared boundary can intersect several parcels; prefer one that
  // strictly contains the point, then the smallest.
  const candidates = features
    .map((f) => ({ geometry: esriPolygonToGeoJSON(f.geometry), properties: f.attributes || {} }))
    .filter((c) => c.geometry);
  if (candidates.length === 0) return null;

  const containing = candidates.filter((c) => pointInGeometry([lon, lat], c.geometry));
  const pool = containing.length ? containing : candidates;
  pool.sort((a, b) => vertexCount(a.geometry) - vertexCount(b.geometry));

  return { ...pool[0], source: `arcgis:${new URL(base).host}`, alternatives: candidates.length - 1 };
}

function vertexCount(geometry) {
  return polygonsOf(geometry).reduce(
    (n, rings) => n + rings.reduce((m, ring) => m + ring.length, 0),
    0,
  );
}

/**
 * Regrid's national parcel API. Requires a paid token (REGRID_TOKEN), but it is
 * the only single endpoint with nationwide coverage.
 */
export async function fetchParcelFromRegrid({ lat, lon, token }) {
  if (!token) throw new Error('Regrid requires a token (set REGRID_TOKEN or pass --token)');
  const url =
    'https://app.regrid.com/api/v2/parcels/point' +
    `?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius=1&limit=1&token=${encodeURIComponent(token)}`;
  const data = await getJson(url);
  const feature = data?.parcels?.features?.[0] || data?.features?.[0];
  if (!feature?.geometry) return null;
  const fields = feature.properties?.fields || feature.properties || {};
  return { geometry: feature.geometry, properties: fields, source: 'regrid' };
}

/**
 * Read a parcel from a local GeoJSON file. Useful when your county publishes a
 * download rather than a live service, and it makes the pipeline usable offline.
 */
export async function fetchParcelFromFile({ lat, lon, path }) {
  const data = JSON.parse(await readFile(path, 'utf8'));
  const features =
    data.type === 'FeatureCollection' ? data.features
    : data.type === 'Feature' ? [data]
    : [{ type: 'Feature', geometry: data, properties: {} }];

  const polygons = features.filter((f) => polygonsOf(f.geometry).length > 0);
  if (polygons.length === 0) throw new Error(`No polygon features in ${path}`);

  const hit =
    Number.isFinite(lat) && Number.isFinite(lon)
      ? polygons.find((f) => pointInGeometry([lon, lat], f.geometry))
      : null;
  const chosen = hit || (polygons.length === 1 ? polygons[0] : null);
  if (!chosen) {
    throw new Error(
      `${path} holds ${polygons.length} polygons and none contains the point — ` +
        'narrow the file down or pass --lat/--lon inside the target lot.',
    );
  }
  return { geometry: chosen.geometry, properties: chosen.properties || {}, source: `file:${path}` };
}
