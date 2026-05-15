import { useRef, useState, useEffect } from 'react';
import { detectInImage, drawDetections, loadModel } from '../utils/detection.js';

export default function PhotoUploader({ onAdd }) {
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [predictions, setPredictions] = useState([]);
  const [status, setStatus] = useState('Loading detection model…');
  const [statusKind, setStatusKind] = useState('');
  const [working, setWorking] = useState(false);
  const [drag, setDrag] = useState(false);
  const [room, setRoom] = useState('');
  const [imageReady, setImageReady] = useState(false);

  useEffect(() => {
    loadModel()
      .then(() => {
        setStatus('Model ready. Drop a photo or click to upload.');
        setStatusKind('good');
      })
      .catch((err) => {
        console.error(err);
        setStatus('Failed to load model: ' + err.message);
        setStatusKind('error');
      });
  }, []);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setStatus('Please choose an image file.');
      setStatusKind('error');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => runDetection(e.target.result);
    reader.readAsDataURL(file);
  };

  const runDetection = async (dataUrl) => {
    setWorking(true);
    setPredictions([]);
    setStatus('Running detection…');
    setStatusKind('');
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = async () => {
        imageRef.current = img;
        setImageReady(true);
        const preds = await detectInImage(img);
        setPredictions(preds);
        if (canvasRef.current) drawDetections(canvasRef.current, img, preds);
        if (preds.length === 0) {
          setStatus('No items detected. Try a clearer photo, or add items manually.');
          setStatusKind('');
        } else {
          setStatus(`Detected ${preds.length} item${preds.length === 1 ? '' : 's'}.`);
          setStatusKind('good');
        }
        setWorking(false);
      };
      img.onerror = () => {
        setStatus('Could not load that image.');
        setStatusKind('error');
        setWorking(false);
      };
      img.src = dataUrl;
    } catch (err) {
      console.error(err);
      setStatus('Detection failed: ' + err.message);
      setStatusKind('error');
      setWorking(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const addAll = () => {
    if (predictions.length === 0) return;
    onAdd(predictions.map((p) => ({ name: p.class, confidence: p.score, room })));
    setStatus(`Added ${predictions.length} item${predictions.length === 1 ? '' : 's'} to inventory.`);
    setStatusKind('good');
    setPredictions([]);
  };

  const addOne = (idx) => {
    const p = predictions[idx];
    onAdd([{ name: p.class, confidence: p.score, room }]);
    const next = predictions.filter((_, i) => i !== idx);
    setPredictions(next);
    if (canvasRef.current && imageRef.current) {
      drawDetections(canvasRef.current, imageRef.current, next);
    }
  };

  return (
    <div className="panel">
      <h2>1. Capture room</h2>

      <div
        className={`dropzone${drag ? ' drag' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
      >
        <strong>Drop a room photo here</strong>
        <span>or click to choose / take a photo</span>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      <div style={{ marginTop: 12, display: 'flex', gap: 8, alignItems: 'center' }}>
        <label style={{ fontSize: 12, color: 'var(--muted)' }}>Room label:</label>
        <input
          type="text"
          value={room}
          onChange={(e) => setRoom(e.target.value)}
          placeholder="e.g. Living room"
          style={{
            flex: 1,
            background: 'var(--panel-2)',
            border: '1px solid var(--border)',
            color: 'var(--text)',
            padding: '6px 8px',
            borderRadius: 4,
            fontSize: 13,
          }}
        />
      </div>

      <div className={`status ${statusKind}`}>{working ? 'Working…' : status}</div>

      {imageReady && (
        <div className="preview-wrap">
          <canvas ref={canvasRef} />
        </div>
      )}

      {predictions.length > 0 && (
        <>
          <div style={{ marginTop: 14, marginBottom: 8 }} className="button-row">
            <button className="primary" onClick={addAll}>Add all {predictions.length} to inventory</button>
          </div>
          <div>
            {predictions.map((p, i) => (
              <div key={i} className="detected-row">
                <span style={{ flex: 1 }}>{p.class}</span>
                <span className="badge">{Math.round(p.score * 100)}%</span>
                <button onClick={() => addOne(i)}>Add</button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
