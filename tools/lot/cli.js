#!/usr/bin/env node
// Address -> parcel boundary -> KMZ / GeoJSON / CSV.
//
//   npm run lot -- "123 Main St, Springfield, IL" --layer <arcgis-layer-url>
//
// See tools/lot/README.md for where to find a layer URL for your county.

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { geocode } from './geocode.js';
import {
  describeArcgisLayer,
  fetchParcelFromArcgis,
  fetchParcelFromFile,
  fetchParcelFromRegrid,
} from './providers.js';
import { areaSummary, buildKml, kmlToKmz, verticesCsv } from './kml.js';
import { areaM2, bbox, centroid, polygonsOf, M2_PER_ACRE, M2_PER_SQFT } from './geometry.js';

const HERE = dirname(fileURLToPath(import.meta.url));

const USAGE = `
Look up a property's lot boundary by address and export it as KMZ.

Usage
  npm run lot -- "<address>" [options]
  npm run lot -- --lat <lat> --lon <lon> [options]
  npm run lot -- layers [--check]

Parcel source (pick one)
  --layer <url>       ArcGIS FeatureServer/MapServer parcel layer to query.
  --county <slug>     Use a layer from tools/lot/layers.json (see: layers).
  --regrid            Use the Regrid API. Needs --token or REGRID_TOKEN.
  --geojson <file>    Read the parcel from a local GeoJSON file instead.

Options
  -o, --out <prefix>  Output path prefix. Default: ./lot-output/<slugged-address>
  --name <text>       Placemark name. Default: the matched address.
  --token <token>     Regrid API token.
  --formats <list>    Comma-separated: kmz,kml,geojson,csv. Default: kmz,geojson,csv
  --json              Print the result summary as JSON.
  -h, --help          Show this message.

Examples
  npm run lot -- "1600 Pennsylvania Ave NW, Washington, DC" --county dc
  npm run lot -- "12 Oak St, Austin, TX" --layer https://services.arcgis.com/.../FeatureServer/0
  npm run lot -- --lat 30.2672 --lon -97.7431 --geojson ./my-parcels.geojson -o ./oak-st
`.trim();

function parseArgs(argv) {
  const opts = { formats: 'kmz,geojson,csv', positional: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      const value = argv[++i];
      if (value === undefined) throw new Error(`${arg} needs a value`);
      return value;
    };
    switch (arg) {
      case '-h': case '--help': opts.help = true; break;
      case '--check': opts.check = true; break;
      case '--json': opts.json = true; break;
      case '--regrid': opts.regrid = true; break;
      case '--layer': opts.layer = next(); break;
      case '--county': opts.county = next(); break;
      case '--geojson': opts.geojsonFile = next(); break;
      case '--token': opts.token = next(); break;
      case '--name': opts.name = next(); break;
      case '--formats': opts.formats = next(); break;
      case '--lat': opts.lat = Number(next()); break;
      case '--lon': case '--lng': opts.lon = Number(next()); break;
      case '-o': case '--out': opts.out = next(); break;
      default:
        if (arg.startsWith('-')) throw new Error(`Unknown option: ${arg}`);
        opts.positional.push(arg);
    }
  }
  return opts;
}

async function loadLayers() {
  return JSON.parse(await readFile(resolve(HERE, 'layers.json'), 'utf8'));
}

async function runLayers(opts) {
  const { layers } = await loadLayers();
  const slugs = Object.keys(layers);
  if (slugs.length === 0) {
    console.log('No layers registered. Add one to tools/lot/layers.json.');
    return 0;
  }

  if (!opts.check) {
    console.log('Registered parcel layers (use with --county <slug>):\n');
    for (const slug of slugs) {
      const entry = layers[slug];
      console.log(`  ${slug.padEnd(12)} ${entry.label}`);
      console.log(`  ${' '.repeat(12)} ${entry.url}`);
      console.log(`  ${' '.repeat(12)} verified: ${entry.verified ? 'yes' : 'no — run with --check'}\n`);
    }
    return 0;
  }

  let failures = 0;
  for (const slug of slugs) {
    const entry = layers[slug];
    process.stdout.write(`  ${slug.padEnd(12)} `);
    try {
      const meta = await describeArcgisLayer(entry.url);
      const polygonal = /polygon/i.test(meta.geometryType || '');
      console.log(`OK — "${meta.name}" (${meta.geometryType}, ${meta.fields.length} fields)`);
      if (!polygonal) {
        console.log(`  ${' '.repeat(12)} warning: not a polygon layer, boundaries will not export`);
        failures++;
      }
    } catch (err) {
      console.log(`FAILED — ${err.message}`);
      failures++;
    }
  }
  console.log(failures === 0 ? '\nAll layers reachable.' : `\n${failures} layer(s) need attention.`);
  return failures === 0 ? 0 : 1;
}

function slugify(text) {
  return (
    String(text)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 60) || 'lot'
  );
}

async function resolveLocation(opts) {
  const address = opts.positional.join(' ').trim();
  if (Number.isFinite(opts.lat) && Number.isFinite(opts.lon)) {
    return { lat: opts.lat, lon: opts.lon, matchedAddress: address || null, source: 'given' };
  }
  if (!address) throw new Error('Give an address, or --lat and --lon.\n\n' + USAGE);
  return geocode(address);
}

async function resolveParcel(opts, location) {
  const { lat, lon } = location;
  if (opts.geojsonFile) return fetchParcelFromFile({ lat, lon, path: opts.geojsonFile });
  if (opts.regrid) return fetchParcelFromRegrid({ lat, lon, token: opts.token || process.env.REGRID_TOKEN });

  let layerUrl = opts.layer;
  if (!layerUrl && opts.county) {
    const { layers } = await loadLayers();
    const entry = layers[opts.county];
    if (!entry) {
      throw new Error(
        `No layer registered for "${opts.county}". Known: ${Object.keys(layers).join(', ')}\n` +
          'Run `npm run lot -- layers` to see them, or pass --layer <url>.',
      );
    }
    layerUrl = entry.url;
  }

  if (!layerUrl) {
    throw new Error(
      'No parcel source given. Pass one of --layer, --county, --regrid, or --geojson.\n\n' + USAGE,
    );
  }
  return fetchParcelFromArcgis({ lat, lon, layerUrl });
}

async function main(argv) {
  const opts = parseArgs(argv);
  if (opts.help) {
    console.log(USAGE);
    return 0;
  }
  if (opts.positional[0] === 'layers') {
    opts.positional.shift();
    return runLayers(opts);
  }

  const location = await resolveLocation(opts);
  if (!opts.json) {
    console.log(`Location: ${location.lat.toFixed(6)}, ${location.lon.toFixed(6)} (${location.source})`);
    if (location.matchedAddress) console.log(`Matched:  ${location.matchedAddress}`);
  }

  const parcel = await resolveParcel(opts, location);
  if (!parcel) {
    console.error(
      'No parcel found at that point.\n' +
        '  - The geocoded point may sit in a right-of-way; nudge it with --lat/--lon.\n' +
        '  - The layer may not cover this address. Check with: npm run lot -- layers --check',
    );
    return 2;
  }

  const geometry = parcel.geometry;
  const rings = polygonsOf(geometry);
  const vertices = rings.reduce((n, r) => n + r.reduce((m, ring) => m + ring.length, 0), 0);
  const m2 = areaM2(geometry);
  const center = centroid(geometry);

  const name = opts.name || location.matchedAddress || `Lot ${center[1].toFixed(5)}, ${center[0].toFixed(5)}`;
  const properties = { ...parcel.properties, 'Source': parcel.source };
  if (location.matchedAddress) properties['Geocoded address'] = location.matchedAddress;

  const kml = buildKml({
    geometry,
    properties,
    name,
    point: [location.lon, location.lat],
    pointLabel: location.matchedAddress || 'Geocoded point',
  });

  const prefix = opts.out || `./lot-output/${slugify(location.matchedAddress || name)}`;
  await mkdir(dirname(resolve(prefix)), { recursive: true });

  const wanted = new Set(opts.formats.split(',').map((f) => f.trim().toLowerCase()).filter(Boolean));
  const written = [];
  if (wanted.has('kmz')) {
    await writeFile(`${prefix}.kmz`, kmlToKmz(kml));
    written.push(`${prefix}.kmz`);
  }
  if (wanted.has('kml')) {
    await writeFile(`${prefix}.kml`, kml, 'utf8');
    written.push(`${prefix}.kml`);
  }
  if (wanted.has('geojson')) {
    const feature = {
      type: 'Feature',
      geometry,
      properties: { ...properties, ...areaSummary(geometry), name },
    };
    await writeFile(`${prefix}.geojson`, JSON.stringify(feature, null, 2), 'utf8');
    written.push(`${prefix}.geojson`);
  }
  if (wanted.has('csv')) {
    await writeFile(`${prefix}.csv`, verticesCsv(geometry, { name }), 'utf8');
    written.push(`${prefix}.csv`);
  }

  const summary = {
    address: location.matchedAddress,
    geocodeSource: location.source,
    point: { lat: location.lat, lon: location.lon },
    parcelSource: parcel.source,
    centroid: { lat: center[1], lon: center[0] },
    bbox: bbox(geometry),
    vertices,
    parts: rings.length,
    areaAcres: Number((m2 / M2_PER_ACRE).toFixed(4)),
    areaSqFt: Math.round(m2 / M2_PER_SQFT),
    files: written,
  };

  if (opts.json) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    console.log(`Parcel:   ${parcel.source}`);
    console.log(`Boundary: ${vertices} vertices across ${rings.length} part(s)`);
    console.log(`Area:     ${summary.areaAcres} acres (${summary.areaSqFt.toLocaleString('en-US')} sq ft)`);
    console.log(`Centroid: ${center[1].toFixed(6)}, ${center[0].toFixed(6)}`);
    console.log(`Wrote:\n${written.map((f) => `  ${f}`).join('\n')}`);
  }
  return 0;
}

main(process.argv.slice(2))
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  });
