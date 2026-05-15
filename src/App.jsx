import { useEffect, useMemo, useState } from 'react';
import PhotoUploader from './components/PhotoUploader.jsx';
import InventoryTable from './components/InventoryTable.jsx';
import { loadInventory, saveInventory, clearInventory } from './utils/storage.js';
import { exportCsv, exportPdf } from './utils/exporters.js';

function makeId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

export default function App() {
  const [items, setItems] = useState(() => loadInventory());

  useEffect(() => {
    saveInventory(items);
  }, [items]);

  const total = useMemo(
    () => items.reduce((sum, it) => sum + (Number(it.estimatedValue) || 0), 0),
    [items]
  );

  const addItems = (newItems) => {
    const stamp = new Date().toISOString();
    setItems((prev) => [
      ...prev,
      ...newItems.map((it) => ({
        id: makeId(),
        name: it.name,
        room: it.room || '',
        estimatedValue: '',
        notes: '',
        confidence: it.confidence ?? null,
        dateAdded: stamp,
      })),
    ]);
  };

  const addBlankRow = () => addItems([{ name: '' }]);

  const updateItem = (id, patch) => {
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)));
  };

  const removeItem = (id) => {
    setItems((prev) => prev.filter((it) => it.id !== id));
  };

  const clearAll = () => {
    if (items.length === 0) return;
    if (confirm(`Remove all ${items.length} items? This cannot be undone.`)) {
      setItems([]);
      clearInventory();
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Home Asset Recognizer</h1>
          <p>Photograph each room, review detected items, and export your inventory for insurance.</p>
        </div>
        <div className="totals">
          <div>
            <span className="num">{items.length}</span>
            <span className="label">items</span>
          </div>
          <div>
            <span className="num">${total.toFixed(2)}</span>
            <span className="label">est. total</span>
          </div>
        </div>
      </header>

      <div className="layout">
        <PhotoUploader onAdd={addItems} />

        <div className="panel">
          <h2>2. Inventory</h2>
          <div className="export-bar">
            <button onClick={addBlankRow}>+ Add row</button>
            <div className="spacer" />
            <button onClick={() => exportCsv(items)} disabled={items.length === 0}>Export CSV</button>
            <button onClick={() => exportPdf(items)} disabled={items.length === 0}>Export PDF</button>
            <button className="danger" onClick={clearAll} disabled={items.length === 0}>Clear all</button>
          </div>
          <InventoryTable items={items} onUpdate={updateItem} onRemove={removeItem} />
        </div>
      </div>
    </div>
  );
}
