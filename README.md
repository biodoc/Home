# Home Asset Recognizer

In-browser app that detects items in room photos and builds a property inventory for insurance purposes. No backend, no API keys — object detection runs locally with TensorFlow.js + COCO-SSD.

## Run

```bash
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`).

## Use

1. **Capture room** — drop a photo (or take one on mobile) and label the room. The model detects up to ~80 common object classes (TV, couch, laptop, bottle, chair, etc.).
2. **Review** — add detected items to the inventory individually or all at once.
3. **Edit** — fill in estimated value, brand/model, serial #. Inventory is saved to your browser's localStorage automatically.
4. **Export** — download CSV or PDF for your insurer.

## Lot boundary export (`tools/lot`)

A separate Node CLI in this repo: give it a street address, and it fetches the
property's parcel polygon from the county GIS record and writes a **KMZ** you can
open in Google Earth, plus GeoJSON and a lat/long vertex CSV.

```bash
npm run lot -- "1600 Pennsylvania Ave NW, Washington, DC" --county dc
npm run lot -- layers          # registered parcel layers
npm run test:lot               # offline test suite
```

Parcel geometry comes from county/state ArcGIS parcel services (free), Regrid
(paid, nationwide), or a local GeoJSON file — not from Zillow, which has no
parcel API and licenses its lot-line overlay from a third party. See
[`tools/lot/README.md`](tools/lot/README.md).

## Notes

- The COCO-SSD model only knows ~80 categories. For anything it misses (jewelry, art, appliances, tools), use **+ Add row** to enter items manually.
- Everything runs locally in your browser. Photos are never uploaded.
- "Clear all" wipes localStorage; export first if you want a backup.
