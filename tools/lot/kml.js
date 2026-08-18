// GeoJSON parcel feature -> KML document -> KMZ archive.

import { zipSync } from './zip.js';
import { areaM2, bbox, centroid, polygonsOf, M2_PER_ACRE, M2_PER_SQFT } from './geometry.js';

const esc = (value) =>
  String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');

/** KML wants 6 decimals of lon/lat (~0.1 m) — more is false precision. */
const coord = ([lon, lat]) => `${round(lon, 6)},${round(lat, 6)},0`;

function round(n, places) {
  const f = 10 ** places;
  return Math.round(n * f) / f;
}

function ringXml(ring, tag) {
  const coords = ring.map(coord).join(' ');
  return `        <${tag}><LinearRing><coordinates>${coords}</coordinates></LinearRing></${tag}>`;
}

function polygonXml(rings) {
  const parts = [
    '      <Polygon>',
    '        <tessellate>1</tessellate>',
    '        <altitudeMode>clampToGround</altitudeMode>',
    ringXml(rings[0], 'outerBoundaryIs'),
    ...rings.slice(1).map((hole) => ringXml(hole, 'innerBoundaryIs')),
    '      </Polygon>',
  ];
  return parts.join('\n');
}

function extendedData(props) {
  const rows = Object.entries(props)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `      <Data name="${esc(k)}"><value>${esc(v)}</value></Data>`);
  return rows.length ? `    <ExtendedData>\n${rows.join('\n')}\n    </ExtendedData>` : '';
}

function descriptionHtml(props) {
  const rows = Object.entries(props)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `<tr><th align="left">${esc(k)}</th><td>${esc(v)}</td></tr>`)
    .join('');
  return rows ? `<![CDATA[<table>${rows}</table>]]>` : '';
}

/** Human-readable lot size summary, appended to the parcel's attributes. */
export function areaSummary(geometry) {
  const m2 = areaM2(geometry);
  return {
    'Computed area (acres)': (m2 / M2_PER_ACRE).toFixed(4),
    'Computed area (sq ft)': Math.round(m2 / M2_PER_SQFT).toLocaleString('en-US'),
    'Computed area (sq m)': Math.round(m2).toLocaleString('en-US'),
  };
}

/**
 * Build the KML document for a parcel.
 *
 * @param {object} opts
 * @param {object} opts.geometry   GeoJSON Polygon or MultiPolygon (WGS84).
 * @param {object} opts.properties Parcel attributes to attach.
 * @param {string} opts.name       Placemark / document name.
 * @param {[number,number]} [opts.point] Geocoded address point to pin.
 * @param {string} [opts.pointLabel]
 */
export function buildKml({ geometry, properties = {}, name, point, pointLabel = 'Geocoded address' }) {
  const polys = polygonsOf(geometry);
  if (polys.length === 0) throw new Error('geometry has no polygon rings');

  const props = { ...properties, ...areaSummary(geometry) };
  const geometryXml =
    polys.length === 1
      ? polygonXml(polys[0])
      : ['      <MultiGeometry>', ...polys.map(polygonXml), '      </MultiGeometry>'].join('\n');

  const center = centroid(geometry);
  const box = bbox(geometry);
  const lookAt = center
    ? [
        '  <LookAt>',
        `    <longitude>${round(center[0], 6)}</longitude>`,
        `    <latitude>${round(center[1], 6)}</latitude>`,
        `    <range>${Math.max(150, Math.round(spanMeters(box) * 2.5))}</range>`,
        '    <tilt>0</tilt>',
        '  </LookAt>',
      ].join('\n')
    : '';

  const placemarks = [
    '  <Placemark>',
    `    <name>${esc(name)}</name>`,
    '    <styleUrl>#lot-boundary</styleUrl>',
    descriptionHtml(props) && `    <description>${descriptionHtml(props)}</description>`,
    extendedData(props),
    geometryXml,
    '  </Placemark>',
  ];

  if (point) {
    placemarks.push(
      '  <Placemark>',
      `    <name>${esc(pointLabel)}</name>`,
      '    <styleUrl>#address-pin</styleUrl>',
      `    <Point><coordinates>${coord(point)}</coordinates></Point>`,
      '  </Placemark>',
    );
  }

  if (center) {
    placemarks.push(
      '  <Placemark>',
      '    <name>Lot centroid</name>',
      '    <styleUrl>#centroid-pin</styleUrl>',
      `    <Point><coordinates>${coord(center)}</coordinates></Point>`,
      '  </Placemark>',
    );
  }

  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<kml xmlns="http://www.opengis.net/kml/2.2">',
    '<Document>',
    `  <name>${esc(name)}</name>`,
    '  <Style id="lot-boundary">',
    '    <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>',
    '    <PolyStyle><color>400000ff</color><fill>1</fill><outline>1</outline></PolyStyle>',
    '  </Style>',
    '  <Style id="address-pin">',
    '    <IconStyle><color>ff00aaff</color><scale>1.1</scale>',
    '      <Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href></Icon>',
    '    </IconStyle>',
    '  </Style>',
    '  <Style id="centroid-pin">',
    '    <IconStyle><color>ffffffff</color><scale>0.8</scale>',
    '      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>',
    '    </IconStyle>',
    '  </Style>',
    lookAt,
    ...placemarks,
    '</Document>',
    '</kml>',
    '',
  ]
    .filter((line) => line !== '' && line !== undefined && line !== false)
    .join('\n');
}

function spanMeters(box) {
  if (!box) return 100;
  const [minX, minY, maxX, maxY] = box;
  const midLat = ((minY + maxY) / 2) * (Math.PI / 180);
  const dx = (maxX - minX) * 111320 * Math.cos(midLat);
  const dy = (maxY - minY) * 110540;
  return Math.max(Math.hypot(dx, dy), 30);
}

/** Wrap a KML string as a KMZ archive (a ZIP whose root entry is doc.kml). */
export function kmlToKmz(kml, extraFiles = []) {
  return zipSync([{ name: 'doc.kml', data: kml }, ...extraFiles]);
}

/** Flat CSV of every boundary vertex: the raw lat/long list. */
export function verticesCsv(geometry, { name = 'parcel' } = {}) {
  const rows = ['polygon,ring,ring_type,vertex,latitude,longitude'];
  polygonsOf(geometry).forEach((rings, p) => {
    rings.forEach((ring, r) => {
      const type = r === 0 ? 'outer' : 'hole';
      ring.forEach(([lon, lat], i) => {
        rows.push(`${p + 1},${r + 1},${type},${i + 1},${round(lat, 7)},${round(lon, 7)}`);
      });
    });
  });
  return `# ${name}\n${rows.join('\n')}\n`;
}
