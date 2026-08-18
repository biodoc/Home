#!/usr/bin/env node
// Offline tests. Everything except the network providers is exercised here.
// Run: node tools/lot/test.js

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inflateRawSync } from 'node:zlib';

import {
  areaM2, bbox, centroid, closeRing, esriPolygonToGeoJSON, isClockwise,
  pointInGeometry, polygonsOf, ringAreaM2, M2_PER_ACRE,
} from './geometry.js';
import { crc32, zipSync } from './zip.js';
import { buildKml, kmlToKmz, verticesCsv } from './kml.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const tests = [];
const test = (name, fn) => tests.push([name, fn]);

// --- geometry -------------------------------------------------------------

test('closeRing appends the first vertex when the ring is open', () => {
  assert.deepEqual(closeRing([[0, 0], [1, 0], [1, 1]]), [[0, 0], [1, 0], [1, 1], [0, 0]]);
  const closed = [[0, 0], [1, 0], [1, 1], [0, 0]];
  assert.deepEqual(closeRing(closed), closed);
});

test('isClockwise matches the shoelace sign convention', () => {
  assert.equal(isClockwise([[0, 0], [1, 0], [1, 1], [0, 1]]), false); // CCW
  assert.equal(isClockwise([[0, 0], [0, 1], [1, 1], [1, 0]]), true); // CW
});

test('esriPolygonToGeoJSON re-winds a simple ring to GeoJSON order', () => {
  // Esri outer rings are clockwise; GeoJSON wants counter-clockwise.
  const geo = esriPolygonToGeoJSON({ rings: [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]] });
  assert.equal(geo.type, 'Polygon');
  assert.equal(geo.coordinates.length, 1);
  assert.equal(isClockwise(geo.coordinates[0]), false, 'outer ring must be CCW');
});

test('esriPolygonToGeoJSON nests holes inside their outer ring', () => {
  const outer = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]; // CW (Esri outer)
  const hole = [[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]; // CCW (Esri hole)
  const geo = esriPolygonToGeoJSON({ rings: [outer, hole] });
  assert.equal(geo.type, 'Polygon');
  assert.equal(geo.coordinates.length, 2, 'one outer + one hole');
  assert.equal(isClockwise(geo.coordinates[0]), false, 'outer CCW');
  assert.equal(isClockwise(geo.coordinates[1]), true, 'hole CW');
});

test('esriPolygonToGeoJSON splits multiple parts into a MultiPolygon', () => {
  const partA = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]];
  const partB = [[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]];
  const geo = esriPolygonToGeoJSON({ rings: [partA, partB] });
  assert.equal(geo.type, 'MultiPolygon');
  assert.equal(geo.coordinates.length, 2);
});

test('esriPolygonToGeoJSON rejects empty and degenerate input', () => {
  assert.equal(esriPolygonToGeoJSON({ rings: [] }), null);
  assert.equal(esriPolygonToGeoJSON({ rings: [[[0, 0], [1, 1]]] }), null);
  assert.equal(esriPolygonToGeoJSON(null), null);
});

test('ringAreaM2 matches exact ellipsoidal reference areas', () => {
  // References are the closed-form ellipsoidal areas of each lat/lon
  // quadrilateral: dlon * a^2 * (1 - e^2) / 2 * [q(lat2) - q(lat1)].
  const d = 0.001;
  const cases = [
    [0, 12309.07],  // 111.32 m x 110.57 m at the equator
    [40, 9481.61],  //  85.39 m x 111.03 m
    [60, 6216.71],  //  55.80 m x 111.41 m
  ];
  for (const [lat, expected] of cases) {
    const area = ringAreaM2([[0, lat], [d, lat], [d, lat + d], [0, lat + d], [0, lat]]);
    assert.ok(
      Math.abs(area - expected) / expected < 0.00002,
      `lat ${lat}: got ${area.toFixed(1)}, expected ~${expected}`,
    );
  }
});

test('areaM2 subtracts holes from the outer ring', () => {
  const outer = [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]];
  const hole = [[0.0002, 0.0002], [0.0002, 0.0008], [0.0008, 0.0008], [0.0008, 0.0002], [0.0002, 0.0002]];
  const solid = areaM2({ type: 'Polygon', coordinates: [outer] });
  const withHole = areaM2({ type: 'Polygon', coordinates: [outer, hole] });
  assert.ok(withHole < solid);
  assert.ok(Math.abs(withHole - solid * (1 - 0.36)) / solid < 0.01, 'hole covers 36% of the square');
});

test('a quarter-acre lot reports as roughly a quarter acre', () => {
  // ~32 m x ~32 m => ~1012 m^2 => ~0.25 acres.
  const lat = 40;
  const dLat = 32 / 110540;
  const dLon = 32 / (111320 * Math.cos((lat * Math.PI) / 180));
  const ring = [
    [-74, lat], [-74 + dLon, lat], [-74 + dLon, lat + dLat], [-74, lat + dLat], [-74, lat],
  ];
  const acres = areaM2({ type: 'Polygon', coordinates: [ring] }) / M2_PER_ACRE;
  assert.ok(Math.abs(acres - 0.25) < 0.01, `got ${acres} acres`);
});

test('centroid of a square is its middle', () => {
  const ring = [[-74, 40], [-73.999, 40], [-73.999, 40.001], [-74, 40.001], [-74, 40]];
  const [lon, lat] = centroid({ type: 'Polygon', coordinates: [ring] });
  assert.ok(Math.abs(lon - -73.9995) < 1e-6, `lon ${lon}`);
  assert.ok(Math.abs(lat - 40.0005) < 1e-6, `lat ${lat}`);
});

test('centroid falls back to the mean vertex for a zero-area ring', () => {
  const ring = [[0, 0], [1, 1], [2, 2], [0, 0]];
  const [lon, lat] = centroid({ type: 'Polygon', coordinates: [ring] });
  assert.ok(Number.isFinite(lon) && Number.isFinite(lat));
});

test('bbox spans every part of a MultiPolygon', () => {
  const geo = esriPolygonToGeoJSON({
    rings: [
      [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]],
      [[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]],
    ],
  });
  assert.deepEqual(bbox(geo), [0, 0, 6, 6]);
});

test('pointInGeometry honours holes', () => {
  const outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const hole = [[2, 2], [2, 8], [8, 8], [8, 2], [2, 2]];
  const geo = { type: 'Polygon', coordinates: [outer, hole] };
  assert.equal(pointInGeometry([1, 1], geo), true, 'inside outer, outside hole');
  assert.equal(pointInGeometry([5, 5], geo), false, 'inside the hole');
  assert.equal(pointInGeometry([20, 20], geo), false, 'outside entirely');
});

// --- zip / kmz ------------------------------------------------------------

test('crc32 matches the known check value for "123456789"', () => {
  assert.equal(crc32(Buffer.from('123456789')), 0xcbf43926);
});

test('zipSync produces an archive the platform unzip accepts', () => {
  const dir = mkdtempSync(join(tmpdir(), 'lotzip-'));
  const file = join(dir, 'a.zip');
  writeFileSync(file, zipSync([{ name: 'doc.kml', data: 'hello '.repeat(200) }]));
  const listing = execFileSync('unzip', ['-t', file], { encoding: 'utf8' });
  assert.match(listing, /No errors detected/);
  const body = execFileSync('unzip', ['-p', file, 'doc.kml'], { encoding: 'utf8' });
  assert.equal(body, 'hello '.repeat(200));
});

test('zipSync stores incompressible data rather than growing it', () => {
  const random = Buffer.from(Array.from({ length: 64 }, (_, i) => (i * 97 + 13) % 256));
  const buf = zipSync([{ name: 'r.bin', data: random }]);
  assert.equal(buf.readUInt16LE(8), 0, 'compression method should be 0 (stored)');
  assert.equal(buf.readUInt32LE(18), buf.readUInt32LE(22), 'stored sizes must match');
});

test('kmlToKmz round-trips through raw inflate', () => {
  const kml = '<?xml version="1.0"?><kml>' + '<x/>'.repeat(100) + '</kml>';
  const kmz = kmlToKmz(kml);
  assert.equal(kmz.readUInt32LE(0), 0x04034b50, 'local file header signature');
  const nameLen = kmz.readUInt16LE(26);
  assert.equal(kmz.subarray(30, 30 + nameLen).toString(), 'doc.kml');
  const compressed = kmz.subarray(30 + nameLen, 30 + nameLen + kmz.readUInt32LE(18));
  assert.equal(inflateRawSync(compressed).toString(), kml);
});

// --- kml document ---------------------------------------------------------

const SQUARE = {
  type: 'Polygon',
  coordinates: [[[-74, 40], [-73.999, 40], [-73.999, 40.001], [-74, 40.001], [-74, 40]]],
};

test('buildKml emits a well-formed document with the parcel ring', () => {
  const kml = buildKml({ geometry: SQUARE, name: 'Test Lot', properties: { APN: '123-45' } });
  assert.match(kml, /^<\?xml version="1\.0" encoding="UTF-8"\?>/);
  assert.match(kml, /<kml xmlns="http:\/\/www\.opengis\.net\/kml\/2\.2">/);
  assert.match(kml, /<name>Test Lot<\/name>/);
  assert.match(kml, /<outerBoundaryIs><LinearRing><coordinates>-74,40,0 /);
  assert.match(kml, /<Data name="APN"><value>123-45<\/value><\/Data>/);
  assert.match(kml, /<altitudeMode>clampToGround<\/altitudeMode>/);
  assert.equal((kml.match(/<Placemark>/g) || []).length, (kml.match(/<\/Placemark>/g) || []).length);
});

test('buildKml escapes XML metacharacters in names and attributes', () => {
  const kml = buildKml({
    geometry: SQUARE,
    name: 'Smith & Sons <Trust>',
    properties: { Owner: 'A "B" & C' },
  });
  assert.match(kml, /<name>Smith &amp; Sons &lt;Trust&gt;<\/name>/);
  assert.match(kml, /<value>A &quot;B&quot; &amp; C<\/value>/);
  assert.ok(!/<name>Smith & Sons/.test(kml), 'raw ampersand must not survive');
});

test('buildKml adds the address pin and centroid placemarks', () => {
  const kml = buildKml({ geometry: SQUARE, name: 'Lot', point: [-73.9995, 40.0005] });
  assert.equal((kml.match(/<Placemark>/g) || []).length, 3);
  assert.match(kml, /<Point><coordinates>-73\.9995,40\.0005,0<\/coordinates><\/Point>/);
  assert.match(kml, /Lot centroid/);
});

test('buildKml writes holes as innerBoundaryIs', () => {
  const withHole = {
    type: 'Polygon',
    coordinates: [
      SQUARE.coordinates[0],
      [[-73.9998, 40.0002], [-73.9998, 40.0008], [-73.9992, 40.0008], [-73.9992, 40.0002], [-73.9998, 40.0002]],
    ],
  };
  const kml = buildKml({ geometry: withHole, name: 'Lot' });
  assert.equal((kml.match(/<innerBoundaryIs>/g) || []).length, 1);
});

test('buildKml wraps multi-part parcels in a MultiGeometry', () => {
  const multi = esriPolygonToGeoJSON({
    rings: [
      [[0, 0], [0, 0.001], [0.001, 0.001], [0.001, 0], [0, 0]],
      [[1, 1], [1, 1.001], [1.001, 1.001], [1.001, 1], [1, 1]],
    ],
  });
  const kml = buildKml({ geometry: multi, name: 'Split lot' });
  assert.match(kml, /<MultiGeometry>/);
  assert.equal((kml.match(/<Polygon>/g) || []).length, 2);
});

test('buildKml refuses geometry with no rings', () => {
  assert.throws(() => buildKml({ geometry: { type: 'Point', coordinates: [0, 0] }, name: 'x' }), /no polygon rings/);
});

test('verticesCsv lists every vertex with a header', () => {
  const csv = verticesCsv(SQUARE, { name: 'Test Lot' });
  const lines = csv.trim().split('\n');
  assert.equal(lines[0], '# Test Lot');
  assert.equal(lines[1], 'polygon,ring,ring_type,vertex,latitude,longitude');
  assert.equal(lines.length - 2, SQUARE.coordinates[0].length);
  assert.equal(lines[2], '1,1,outer,1,40,-74');
});

// --- end to end (offline path) -------------------------------------------

test('CLI turns a local GeoJSON parcel into a valid KMZ', () => {
  const dir = mkdtempSync(join(tmpdir(), 'lotcli-'));
  const parcels = join(dir, 'parcels.geojson');
  writeFileSync(
    parcels,
    JSON.stringify({
      type: 'FeatureCollection',
      features: [
        { type: 'Feature', properties: { APN: '001-002-003', Owner: 'Doe & Co' }, geometry: SQUARE },
        {
          type: 'Feature',
          properties: { APN: 'neighbour' },
          geometry: { type: 'Polygon', coordinates: [[[-80, 30], [-79.999, 30], [-79.999, 30.001], [-80, 30.001], [-80, 30]]] },
        },
      ],
    }),
  );

  const prefix = join(dir, 'out');
  const stdout = execFileSync(
    process.execPath,
    [join(HERE, 'cli.js'), '--lat', '40.0005', '--lon', '-73.9995',
     '--geojson', parcels, '-o', prefix, '--formats', 'kmz,kml,geojson,csv', '--json'],
    { encoding: 'utf8' },
  );

  const summary = JSON.parse(stdout);
  assert.equal(summary.vertices, 5);
  assert.equal(summary.parts, 1);
  assert.ok(Math.abs(summary.areaAcres - 2.343) < 0.005, `acres ${summary.areaAcres}`);
  assert.equal(summary.files.length, 4);

  // The KMZ must be readable by a standard unzip, and carry the right parcel.
  assert.match(execFileSync('unzip', ['-t', `${prefix}.kmz`], { encoding: 'utf8' }), /No errors detected/);
  const kml = execFileSync('unzip', ['-p', `${prefix}.kmz`, 'doc.kml'], { encoding: 'utf8' });
  assert.match(kml, /001-002-003/);
  assert.match(kml, /Doe &amp; Co/);
  assert.ok(!kml.includes('neighbour'), 'must pick the parcel containing the point');

  const geojson = JSON.parse(readFileSync(`${prefix}.geojson`, 'utf8'));
  assert.equal(geojson.type, 'Feature');
  assert.equal(geojson.properties.APN, '001-002-003');
  assert.equal(polygonsOf(geojson.geometry).length, 1);

  const csv = readFileSync(`${prefix}.csv`, 'utf8').trim().split('\n');
  assert.equal(csv.length, 7); // comment + header + 5 vertices
});

test('CLI errors clearly when no parcel source is given', () => {
  assert.throws(
    () => execFileSync(process.execPath, [join(HERE, 'cli.js'), '--lat', '40', '--lon', '-74'], { encoding: 'utf8', stdio: 'pipe' }),
    (err) => /No parcel source given/.test(err.stderr),
  );
});

test('CLI reports which parcels are ambiguous instead of guessing', () => {
  const dir = mkdtempSync(join(tmpdir(), 'lotamb-'));
  const parcels = join(dir, 'p.geojson');
  writeFileSync(
    parcels,
    JSON.stringify({
      type: 'FeatureCollection',
      features: [
        { type: 'Feature', properties: {}, geometry: SQUARE },
        { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [[[-80, 30], [-79.999, 30], [-79.999, 30.001], [-80, 30.001], [-80, 30]]] } },
      ],
    }),
  );
  assert.throws(
    () => execFileSync(process.execPath, [join(HERE, 'cli.js'), '--lat', '10', '--lon', '10', '--geojson', parcels, '-o', join(dir, 'o')], { encoding: 'utf8', stdio: 'pipe' }),
    (err) => /none contains the point/.test(err.stderr),
  );
});

test('layers registry is valid JSON with usable entries', () => {
  const { layers } = JSON.parse(readFileSync(join(HERE, 'layers.json'), 'utf8'));
  for (const [slug, entry] of Object.entries(layers)) {
    assert.ok(entry.url?.startsWith('https://'), `${slug} needs an https url`);
    assert.match(entry.url, /\/(FeatureServer|MapServer)\/\d+$/, `${slug} url must end in a layer index`);
    assert.ok(entry.label, `${slug} needs a label`);
  }
});

// --- runner ---------------------------------------------------------------

let failed = 0;
for (const [name, fn] of tests) {
  try {
    fn();
    console.log(`  ok   ${name}`);
  } catch (err) {
    failed++;
    console.log(`  FAIL ${name}\n       ${err.message.split('\n')[0]}`);
  }
}
console.log(`\n${tests.length - failed}/${tests.length} passed`);
process.exit(failed ? 1 : 0);
