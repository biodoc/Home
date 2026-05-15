import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';

let modelPromise = null;

export function loadModel() {
  if (!modelPromise) {
    modelPromise = cocoSsd.load({ base: 'mobilenet_v2' });
  }
  return modelPromise;
}

export async function detectInImage(image) {
  const model = await loadModel();
  return model.detect(image);
}

export function drawDetections(canvas, image, predictions) {
  const ctx = canvas.getContext('2d');
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  const fontSize = Math.max(14, Math.round(canvas.width / 60));
  ctx.font = `${fontSize}px -apple-system, sans-serif`;
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 400));

  predictions.forEach((p, i) => {
    const [x, y, w, h] = p.bbox;
    const color = COLORS[i % COLORS.length];
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(x, y, w, h);

    const label = `${p.class} ${(p.score * 100).toFixed(0)}%`;
    const padding = 4;
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(x, y - fontSize - padding * 2, textWidth + padding * 2, fontSize + padding * 2);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + padding, y - padding);
  });
}

const COLORS = [
  '#4f8cff', '#43c47b', '#e9a23b', '#e5484d', '#9b6bff',
  '#1fbeb6', '#f06292', '#ffd54f', '#7ed957', '#ff8a65',
];
