# Lot boundary → KMZ

Look up a property by address, pull its **parcel polygon** from an authoritative
GIS source, and export the boundary as **KMZ** (plus GeoJSON and a plain lat/long
CSV).

```bash
npm run lot -- "1600 Pennsylvania Ave NW, Washington, DC" --county dc
```

The shape of the output (this run used a local parcel file, so the numbers are
real; a live `--county` lookup prints the same fields):

```
$ npm run lot -- --lat 30.26730 --lon -97.74300 --geojson ./parcels.geojson -o ./evergreen

Location: 30.267300, -97.743000 (given)
Parcel:   file:./parcels.geojson
Boundary: 7 vertices across 1 part(s)
Area:     0.4218 acres (18,372 sq ft)
Centroid: 30.267412, -97.742925
Wrote:
  ./evergreen.kmz
  ./evergreen.geojson
  ./evergreen.csv
```

Open the `.kmz` in Google Earth, drop it on Google My Maps, or load it in QGIS.

## Why not Zillow?

Zillow can't serve this, for three separate reasons:

1. **No API.** Zillow's public API (`GetDeepSearchResults` and friends) was
   retired in 2021. What remains — Bridge Interactive / Zillow Group data feeds —
   is for licensed MLS participants and carries listing data, not parcel geometry.
2. **The lot lines aren't Zillow's data.** The boundary overlay on their map is
   licensed third-party parcel data rendered as vector map tiles. Even if you
   pulled the tiles, you'd be redistributing a vendor's licensed product.
3. **Scraping is off the table.** Zillow's terms of use prohibit automated
   access, and they enforce it. The site's own "lot size" figure is a single
   number anyway — an attribute, not a polygon.

The polygon exists because a county assessor drew it. That's where this tool
goes to get it, and the result is the authoritative version rather than a
redrawn copy.

## Parcel sources

Pick one:

| Flag | Source | Notes |
| --- | --- | --- |
| `--layer <url>` | Any ArcGIS parcel service | The main path. Free, no key. |
| `--county <slug>` | A layer from `layers.json` | Shorthand for a saved `--layer`. |
| `--regrid` | [Regrid](https://regrid.com) parcel API | Nationwide, paid. Needs `REGRID_TOKEN`. |
| `--geojson <file>` | A local GeoJSON file | For counties that publish downloads, not services. Works offline. |

### Finding your county's layer URL

Most US counties publish parcels as a public ArcGIS REST service. To find yours:

1. Search for `<county> <state> parcels ArcGIS REST`, or open the county / state
   GIS open-data portal and look for the parcel or "cadastral" dataset.
2. On the dataset page, find the **service URL**. You want the one ending in a
   layer index — `.../FeatureServer/0` or `.../MapServer/12`, not the bare
   service root.
3. Confirm it works before you trust it:

   ```bash
   npm run lot -- layers --check     # checks everything in layers.json
   ```

   Or check an ad-hoc URL by just running a lookup against it — a bad URL fails
   with the server's own error message.

4. Save it in `layers.json` so `--county <slug>` picks it up:

   ```json
   "travis-tx": {
     "label": "Travis County, TX — parcels",
     "url": "https://services.arcgis.com/.../FeatureServer/0",
     "verified": true
   }
   ```

The entries shipped in `layers.json` are a starting point and are marked
`"verified": false` — they were written without network access to test them.
Run `--check` before relying on any of them.

## Geocoding

Addresses resolve through the **US Census geocoder** first (free, no key,
authoritative for US street addresses), falling back to **Nominatim**
(OpenStreetMap) when Census can't match. Skip geocoding entirely with
`--lat` / `--lon`.

If the geocoded point lands in the street right-of-way rather than on the lot —
which happens with some rural and newly-built addresses — no parcel will match.
Nudge the point into the lot with explicit `--lat`/`--lon`.

## Output

`--formats` controls what gets written (default `kmz,geojson,csv`):

- **`.kmz`** — zipped KML. One polygon placemark styled with a red outline and
  translucent fill, a pin at the geocoded address, a pin at the lot centroid,
  and a `LookAt` framed on the lot. Every assessor attribute is carried through
  as `ExtendedData` and rendered in the balloon.
- **`.kml`** — same document, uncompressed.
- **`.geojson`** — an RFC 7946 Feature, for QGIS / PostGIS / Turf.
- **`.csv`** — one row per boundary vertex: `polygon, ring, ring_type, vertex,
  latitude, longitude`. This is the raw lat/long list.

Holes (a parcel with an interior exclusion) and multi-part parcels (a lot split
by a road) both survive the round trip — holes become `innerBoundaryIs`,
parts become a `MultiGeometry`.

## Accuracy notes

- Everything is emitted in **WGS84 (EPSG:4326)**, which is what KML requires.
  ArcGIS queries pass `outSR=4326` so the server does the reprojection from
  whatever the county stores natively.
- **Reported acreage is computed from the polygon**, not copied from the
  assessor's record. The two are usually within a rounding step of each other;
  when they diverge meaningfully, the assessor's own figure is still in the
  `ExtendedData` for comparison. Area uses a local ellipsoidal projection and
  matches the closed-form WGS84 quadrilateral area to within 0.002%.
- **A parcel boundary is not a survey.** County parcel layers are maintained for
  assessment, and their absolute positional accuracy is typically a few feet —
  sometimes considerably worse in rural areas. Don't site a fence off this.

## Tests

```bash
npm run test:lot
```

Runs offline: geometry conversions, area against closed-form ellipsoidal
references, ZIP/KMZ integrity (verified against the platform `unzip`), KML
structure and XML escaping, and an end-to-end CLI run through the `--geojson`
path. The network providers (Census, ArcGIS, Regrid) are not covered — they need
live endpoints.
