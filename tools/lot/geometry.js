// Geometry helpers: Esri <-> GeoJSON conversion, geodesic area, centroid, bbox.

const EARTH_RADIUS_M = 6378137;
const toRad = (deg) => (deg * Math.PI) / 180;

/** Signed planar area of a ring in degree-space. Positive = counter-clockwise. */
export function signedRingArea(ring) {
  let sum = 0;
  for (let i = 0, n = ring.length; i < n; i++) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[(i + 1) % n];
    sum += x1 * y2 - x2 * y1;
  }
  return sum / 2;
}

export function isClockwise(ring) {
  return signedRingArea(ring) < 0;
}

/** Ensure first and last vertex are identical, as GeoJSON and KML both require. */
export function closeRing(ring) {
  if (ring.length < 3) return ring.slice();
  const [fx, fy] = ring[0];
  const [lx, ly] = ring[ring.length - 1];
  return fx === lx && fy === ly ? ring.slice() : [...ring, [fx, fy]];
}

/**
 * Convert an Esri polygon geometry (`{rings: [...]}`) to a GeoJSON Polygon or
 * MultiPolygon.
 *
 * Esri winds outer rings clockwise and holes counter-clockwise, and packs every
 * ring of every part into one flat list. GeoJSON (RFC 7946) wants the opposite
 * winding and explicit nesting, so we regroup on the winding order: each
 * clockwise ring opens a new polygon, each counter-clockwise ring is a hole in
 * the polygon most recently opened.
 */
export function esriPolygonToGeoJSON(esri) {
  const rings = (esri?.rings || []).filter((r) => Array.isArray(r) && r.length >= 4);
  if (rings.length === 0) return null;

  const polygons = [];
  for (const raw of rings) {
    const ring = closeRing(raw.map(([x, y]) => [x, y]));
    if (isClockwise(ring) || polygons.length === 0) {
      // Outer ring: re-wind counter-clockwise for GeoJSON.
      polygons.push([isClockwise(ring) ? ring.slice().reverse() : ring]);
    } else {
      // Hole: re-wind clockwise for GeoJSON.
      polygons[polygons.length - 1].push(ring.slice().reverse());
    }
  }

  return polygons.length === 1
    ? { type: 'Polygon', coordinates: polygons[0] }
    : { type: 'MultiPolygon', coordinates: polygons };
}

/** Every polygon (array of rings) in a Polygon or MultiPolygon geometry. */
export function polygonsOf(geometry) {
  if (!geometry) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates;
  return [];
}

// WGS84 ellipsoid.
const WGS84_A = 6378137;
const WGS84_E2 = 0.00669437999014;

/** Metres per radian of latitude (meridional) and of longitude (normal) at `lat`. */
function metresPerRadian(lat) {
  const phi = toRad(lat);
  const s = Math.sin(phi);
  const w = 1 - WGS84_E2 * s * s;
  return {
    north: (WGS84_A * (1 - WGS84_E2)) / Math.pow(w, 1.5),
    east: (WGS84_A / Math.sqrt(w)) * Math.cos(phi),
  };
}

/**
 * Area of a ring in square metres.
 *
 * Projects the ring onto a local tangent plane using the WGS84 radii of
 * curvature at its mean latitude, then takes the planar shoelace area. At
 * parcel scale this tracks true ellipsoidal area closely; a spherical
 * excess formula would read ~0.7% high near the equator, which is enough to
 * move a reported acreage. It is only meant for lot-sized polygons — over
 * many degrees of latitude the single tangent plane stops being a good fit.
 */
export function ringAreaM2(ring) {
  const r = closeRing(ring);
  if (r.length < 4) return 0;

  const lats = r.map(([, lat]) => lat);
  const lat0 = (Math.min(...lats) + Math.max(...lats)) / 2;
  const lon0 = r[0][0];
  const { north, east } = metresPerRadian(lat0);

  const pts = r.map(([lon, lat]) => [toRad(lon - lon0) * east, toRad(lat - lat0) * north]);

  let sum = 0;
  for (let i = 0, n = pts.length - 1; i < n; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[i + 1];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum / 2);
}

/** Net area of a geometry: outer rings minus their holes. */
export function areaM2(geometry) {
  let total = 0;
  for (const rings of polygonsOf(geometry)) {
    rings.forEach((ring, i) => {
      total += i === 0 ? ringAreaM2(ring) : -ringAreaM2(ring);
    });
  }
  return total;
}

export const M2_PER_ACRE = 4046.8564224;
export const M2_PER_SQFT = 0.09290304;

/**
 * Area-weighted centroid of the outer rings, computed in a local
 * equirectangular projection so the shoelace weighting is not distorted by
 * longitude convergence.
 */
export function centroid(geometry) {
  const outers = polygonsOf(geometry).map((rings) => rings[0]).filter(Boolean);
  if (outers.length === 0) return null;

  const origin = outers[0][0];
  const scale = Math.cos(toRad(origin[1]));
  let cx = 0;
  let cy = 0;
  let weight = 0;

  for (const ring of outers) {
    const pts = closeRing(ring).map(([lon, lat]) => [(lon - origin[0]) * scale, lat - origin[1]]);
    let a = 0;
    let sx = 0;
    let sy = 0;
    for (let i = 0, n = pts.length - 1; i < n; i++) {
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[i + 1];
      const cross = x1 * y2 - x2 * y1;
      a += cross;
      sx += (x1 + x2) * cross;
      sy += (y1 + y2) * cross;
    }
    a /= 2;
    if (a === 0) continue;
    cx += sx / (6 * a) * Math.abs(a);
    cy += sy / (6 * a) * Math.abs(a);
    weight += Math.abs(a);
  }

  if (weight === 0) {
    // Degenerate ring (zero area) — fall back to the mean vertex.
    const pts = outers.flat();
    return [
      pts.reduce((s, p) => s + p[0], 0) / pts.length,
      pts.reduce((s, p) => s + p[1], 0) / pts.length,
    ];
  }

  return [origin[0] + cx / weight / scale, origin[1] + cy / weight];
}

export function bbox(geometry) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const rings of polygonsOf(geometry)) {
    for (const ring of rings) {
      for (const [x, y] of ring) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  return Number.isFinite(minX) ? [minX, minY, maxX, maxY] : null;
}

/** True when the point falls inside the geometry (outer rings minus holes). */
export function pointInGeometry([lon, lat], geometry) {
  let inside = false;
  for (const rings of polygonsOf(geometry)) {
    if (!pointInRing([lon, lat], rings[0])) continue;
    const inHole = rings.slice(1).some((hole) => pointInRing([lon, lat], hole));
    if (!inHole) inside = true;
  }
  return inside;
}

function pointInRing([x, y], ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
