# iPhone Test App (PWA)

Minimal installable web app: no build step, no backend, and no API keys. Covers:
- Tap counter saved on the device (localStorage)
- Device info and online/offline status
- Accelerometer readout (iOS asks for motion permission)
- GPS location
- Offline support through a service worker

## Install on iPhone

1. Host the folder over HTTPS. The included workflow deploys it to GitHub Pages.
   One-time setup: in the repo, go to **Settings → Pages → Source: GitHub Actions**.
2. On the iPhone, open the Pages URL in **Safari**.
3. Tap **Share → Add to Home Screen**.
4. Launch it from the icon. It opens full-screen and the header pill reads "installed".

## Local test (same Wi-Fi)

```bash
cd iphone-test-app && python3 -m http.server 8000
```
Open `http://<pc-ip>:8000` on the phone. The UI works over plain HTTP, but the service worker, motion and GPS need HTTPS.

## Updating

After you change files, bump `CACHE` in `sw.js`. Otherwise installed copies keep serving the old cached version.
