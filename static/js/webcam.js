/**
 * webcam.js – shared webcam utilities (placeholder; inline JS handles camera
 * start/capture in each template for simplicity).
 */

// Exported helper: capture a frame from a <video> element as a JPEG data URL
function captureFrame(videoEl, quality = 0.92) {
  const canvas = document.createElement("canvas");
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  canvas.getContext("2d").drawImage(videoEl, 0, 0);
  return canvas.toDataURL("image/jpeg", quality);
}
