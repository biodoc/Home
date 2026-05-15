export default function InventoryTable({ items, onUpdate, onRemove }) {
  if (items.length === 0) {
    return <div className="empty">No items yet. Upload a room photo to get started, or use the Add row button.</div>;
  }

  return (
    <table className="inventory">
      <thead>
        <tr>
          <th className="col-name">Item</th>
          <th className="col-room">Room</th>
          <th className="col-value">Value (USD)</th>
          <th className="col-notes">Notes</th>
          <th className="col-conf">Confidence</th>
          <th className="col-action"></th>
        </tr>
      </thead>
      <tbody>
        {items.map((it) => (
          <tr key={it.id}>
            <td className="col-name">
              <input
                value={it.name}
                onChange={(e) => onUpdate(it.id, { name: e.target.value })}
                placeholder="Item name"
              />
            </td>
            <td className="col-room">
              <input
                value={it.room || ''}
                onChange={(e) => onUpdate(it.id, { room: e.target.value })}
                placeholder="Room"
              />
            </td>
            <td className="col-value">
              <input
                type="number"
                step="0.01"
                min="0"
                value={it.estimatedValue ?? ''}
                onChange={(e) => onUpdate(it.id, { estimatedValue: e.target.value })}
                placeholder="0.00"
              />
            </td>
            <td className="col-notes">
              <input
                value={it.notes || ''}
                onChange={(e) => onUpdate(it.id, { notes: e.target.value })}
                placeholder="Brand, model, serial #…"
              />
            </td>
            <td className="col-conf">
              {it.confidence != null ? (
                <div title={`${(it.confidence * 100).toFixed(1)}%`}>
                  <div className="confidence-bar">
                    <div className="fill" style={{ width: `${Math.round(it.confidence * 100)}%` }} />
                  </div>
                </div>
              ) : (
                <span className="tag">manual</span>
              )}
            </td>
            <td className="col-action">
              <button className="danger" onClick={() => onRemove(it.id)}>Remove</button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
