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

## Notes

- The COCO-SSD model only knows ~80 categories. For anything it misses (jewelry, art, appliances, tools), use **+ Add row** to enter items manually.
- Everything runs locally in your browser. Photos are never uploaded.
- "Clear all" wipes localStorage; export first if you want a backup.
