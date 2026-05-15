const KEY = 'asset-recognizer.inventory.v1';

export function loadInventory() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveInventory(items) {
  localStorage.setItem(KEY, JSON.stringify(items));
}

export function clearInventory() {
  localStorage.removeItem(KEY);
}
