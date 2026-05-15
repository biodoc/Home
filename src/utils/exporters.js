import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

function csvEscape(value) {
  const s = value == null ? '' : String(value);
  if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

export function exportCsv(items) {
  const headers = ['Item', 'Room', 'Estimated Value', 'Notes', 'Confidence', 'Date Added'];
  const rows = items.map((it) => [
    it.name,
    it.room || '',
    it.estimatedValue || '',
    it.notes || '',
    it.confidence != null ? Math.round(it.confidence * 100) + '%' : '',
    it.dateAdded || '',
  ]);
  const csv = [headers, ...rows].map((r) => r.map(csvEscape).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  triggerDownload(blob, `home-inventory-${todayStamp()}.csv`);
}

export function exportPdf(items) {
  const doc = new jsPDF({ unit: 'pt', format: 'letter' });
  const total = items.reduce((sum, it) => sum + (Number(it.estimatedValue) || 0), 0);

  doc.setFontSize(18);
  doc.text('Home Inventory', 40, 50);
  doc.setFontSize(10);
  doc.setTextColor(120);
  doc.text(`Generated ${new Date().toLocaleString()}`, 40, 68);
  doc.text(`Items: ${items.length}    Estimated total: $${total.toFixed(2)}`, 40, 82);
  doc.setTextColor(0);

  autoTable(doc, {
    startY: 100,
    head: [['Item', 'Room', 'Value (USD)', 'Confidence', 'Notes']],
    body: items.map((it) => [
      it.name,
      it.room || '',
      it.estimatedValue ? `$${Number(it.estimatedValue).toFixed(2)}` : '',
      it.confidence != null ? Math.round(it.confidence * 100) + '%' : '',
      it.notes || '',
    ]),
    styles: { fontSize: 9, cellPadding: 5 },
    headStyles: { fillColor: [79, 140, 255] },
    columnStyles: {
      0: { cellWidth: 110 },
      1: { cellWidth: 80 },
      2: { cellWidth: 70 },
      3: { cellWidth: 60 },
      4: { cellWidth: 'auto' },
    },
  });

  doc.save(`home-inventory-${todayStamp()}.pdf`);
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function todayStamp() {
  const d = new Date();
  return [d.getFullYear(), String(d.getMonth() + 1).padStart(2, '0'), String(d.getDate()).padStart(2, '0')].join('-');
}
