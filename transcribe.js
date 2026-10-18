import { pipeline } from
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const $ = id => document.getElementById(id);

const videoInput = $("videoFile");
const video = $("previewVideo");
const overlay = $("captionOverlay");
const status = $("status");
const transcript = $("transcript");
const loadButton = $("loadEngine");
const transcribeButton = $("transcribeAudio");
const exportButton = $("exportVideo");
const downloadLink = $("downloadLink");

let model = null;
let videoURL = "";
let captionChunks = [];
let originalTranscript = "";
let busy = false;

function setStatus(message) {
  status.textContent = message;
}

function updateStyle() {
  overlay.className =
    "pos-" + $("captionPosition").value +
    " style-" + $("captionStyle").value;

  const sizes = { small: "20px", medium: "30px", large: "40px" };
  overlay.style.fontSize = sizes[$("fontSize").value];
}

function getCurrentCaption(time) {
  const found = captionChunks.find(chunk =>
    time >= chunk.start && time < chunk.end
  );
  return found ? found.text : "";
}

function refreshPreviewCaption() {
  overlay.textContent = getCurrentCaption(video.currentTime);
  updateStyle();
}

function normalizeChunks(result, duration) {
  const chunks = (result.chunks || [])
    .map(item => ({
      text: (item.text || "").trim(),
      start: Array.isArray(item.timestamp) ? item.timestamp[0] : null,
      end: Array.isArray(item.timestamp) ? item.timestamp[1] : null
    }))
    .filter(item =>
      item.text &&
      Number.isFinite(item.start) &&
      Number.isFinite(item.end) &&
      item.end > item.start
    );

  if (chunks.length) return chunks;

  const words = (result.text || "").trim().split(/\s+/).filter(Boolean);
  const resultChunks = [];
  const groupSize = 6;
  const count = Math.ceil(words.length / groupSize);

  for (let i = 0; i < count; i++) {
    resultChunks.push({
      text: words.slice(i * groupSize, (i + 1) * groupSize).join(" "),
      start: duration * i / count,
      end: duration * (i + 1) / count
    });
  }

  return resultChunks;
}

function chunksFromEditedTranscript(text, duration) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const groups = [];

  for (let i = 0; i < words.length; i += 6) {
    groups.push(words.slice(i, i + 6).join(" "));
  }

  return groups.map((text, i) => ({
    text,
    start: duration * i / Math.max(groups.length, 1),
    end: duration * (i + 1) / Math.max(groups.length, 1)
  }));
}

videoInput.addEventListener("change", () => {
  const file = videoInput.files[0];
  if (!file) return;

  if (videoURL) URL.revokeObjectURL(videoURL);
  videoURL = URL.createObjectURL(file);
  video.src = videoURL;
  video.load();

  captionChunks = [];
  originalTranscript = "";
  transcript.value = "";
  overlay.textContent = "";
  downloadLink.hidden = true;
  setStatus("Video selected. Load the AI model when ready.");
});

video.addEventListener("timeupdate", refreshPreviewCaption);
video.addEventListener("seeked", refreshPreviewCaption);

$("captionStyle").addEventListener("change", updateStyle);
$("captionPosition").addEventListener("change", updateStyle);
$("fontSize").addEventListener("change", updateStyle);

loadButton.addEventListener("click", async () => {
  if (busy) return;

  try {
    busy = true;
    loadButton.disabled = true;
    setStatus("Loading AI speech model. The first download may take time...");

    model = await pipeline(
      "automatic-speech-recognition",
      "onnx-community/whisper-tiny",
      { dtype: "q8" }
    );

    setStatus("AI model loaded. Select a video and generate captions.");
  } catch (error) {
    setStatus("Model loading failed: " + error.message);
  } finally {
    busy = false;
    loadButton.disabled = false;
  }
});

transcribeButton.addEventListener("click", async () => {
  const file = videoInput.files[0];

  if (!file) {
    setStatus("Choose a video first.");
    return;
  }
  if (!model) {
    setStatus("Tap Load AI speech model first.");
    return;
  }
  if (busy) return;

  let audioContext;
  let audioVideo;
  let source;
  let processor;
  let silentGain;
  let finished = false;
  const samples = [];

  try {
    busy = true;
    transcribeButton.disabled = true;
    setStatus("Preparing video audio...");

    audioVideo = document.createElement("video");
    audioVideo.src = URL.createObjectURL(file);
    audioVideo.preload = "auto";
    audioVideo.playsInline = true;

    await new Promise((resolve, reject) => {
      audioVideo.onloadedmetadata = resolve;
      audioVideo.onerror = () => reject(new Error("Could not read this video."));
    });

    audioContext = new AudioContext();
    await audioContext.resume();

    source = audioContext.createMediaElementSource(audioVideo);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    silentGain = audioContext.createGain();
    silentGain.gain.value = 0;

    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(audioContext.destination);

    processor.onaudioprocess = event => {
      if (!finished) {
        samples.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      }
    };

    setStatus("Extracting speech audio in real time. Keep this page open...");
    await audioVideo.play();

    await new Promise((resolve, reject) => {
      audioVideo.onended = resolve;
      audioVideo.onerror = () => reject(new Error("Video playback failed."));
    });

    finished = true;
    processor.disconnect();
    source.disconnect();

    const total = samples.reduce((sum, part) => sum + part.length, 0);
    if (!total) throw new Error("No audio was captured from the video.");

    const combined = new Float32Array(total);
    let offset = 0;
    for (const part of samples) {
      combined.set(part, offset);
      offset += part.length;
    }

    setStatus("Preparing speech for AI recognition...");

    const inputRate = audioContext.sampleRate;
    const rate = 16000;
    const output = new Float32Array(Math.floor(combined.length * rate / inputRate));

    for (let i = 0; i < output.length; i++) {
      output[i] = combined[Math.floor(i * inputRate / rate)];
    }

    const language = $("language").value;
    const options = {
      sampling_rate: rate,
      return_timestamps: true,
      chunk_length_s: 20,
      stride_length_s: 3
    };

    if (language !== "auto") options.language = language;

    setStatus("AI is recognizing the speech. This may take a while...");
    const result = await model(output, options);

    transcript.value = result.text || "";
    originalTranscript = transcript.value;
    captionChunks = normalizeChunks(result, audioVideo.duration || video.duration || 1);

    setStatus(
      captionChunks.length
        ? "Transcript ready. Review the words and preview the captions."
        : "Transcript created, but no timed captions were returned. Try another video."
    );

    refreshPreviewCaption();
  } catch (error) {
    setStatus("Transcription failed: " + error.message);
  } finally {
    finished = true;
    if (processor) processor.disconnect();
    if (source) source.disconnect();

    if (audioContext && audioContext.state !== "closed") {
      await audioContext.close();
    }

    if (audioVideo) {
      const oldURL = audioVideo.src;
      audioVideo.pause();
      audioVideo.removeAttribute("src");
      audioVideo.load();
      if (oldURL.startsWith("blob:")) URL.revokeObjectURL(oldURL);
    }

    busy = false;
    transcribeButton.disabled = false;
  }
});

function drawCaption(ctx, text, width, height) {
  if (!text) return;

  const style = $("captionStyle").value;
  const position = $("captionPosition").value;
  const sizeChoice = $("fontSize").value;
  const fontSize = Math.max(
    18,
    Math.round(width * (sizeChoice === "small" ? 0.035 :
      sizeChoice === "large" ? 0.065 : 0.05))
  );

  const maxWidth = width * 0.86;
  const words = text.split(/\s+/);
  const lines = [];
  let line = "";

  ctx.font = `900 ${fontSize}px Arial`;

  for (const word of words) {
    const test = line ? line + " " + word : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);

  const lineHeight = fontSize * 1.2;
  const blockHeight = lines.length * lineHeight;
  let y;

  if (position === "top") y = height * 0.1 + fontSize;
  else if (position === "middle") y = (height - blockHeight) / 2 + fontSize;
  else y = height * 0.88 - blockHeight + fontSize;

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.font = `900 ${fontSize}px Arial`;

  lines.forEach((lineText, index) => {
    const x = width / 2;
    const lineY = y + index * lineHeight;

    if (style === "boxed" || style === "clean") {
      const measured = ctx.measureText(lineText).width;
      const padding = fontSize * 0.25;
      ctx.fillStyle = style === "boxed" ? "#ed2456" : "rgba(0,0,0,0.65)";
      ctx.fillRect(
        x - measured / 2 - padding,
        lineY - lineHeight / 2,
        measured + padding * 2,
        lineHeight
      );
    }

    ctx.lineWidth = Math.max(2, fontSize * 0.09);
    ctx.strokeStyle = style === "neon" ? "#003e68" : "#000000";
    ctx.fillStyle =
      style === "karaoke" || style === "gradient" ? "#ffe45e" :
      style === "neon" ? "#62fff3" : "#ffffff";

    if (style === "neon") {
      ctx.shadowColor = "#00cfff";
      ctx.shadowBlur = fontSize * 0.4;
    } else {
      ctx.shadowBlur = 0;
    }

    if (style !== "clean" && style !== "boxed") {
      ctx.strokeText(lineText, x, lineY);
    }
    ctx.fillText(lineText, x, lineY);
    ctx.shadowBlur = 0;
  });
}

function getExportChunks(duration) {
  const edited = transcript.value.trim();

  if (!edited) return [];

  if (edited !== originalTranscript) {
    return chunksFromEditedTranscript(edited, duration);
  }

  return captionChunks;
}

exportButton.addEventListener("click", async () => {
  const file = videoInput.files[0];

  if (!file) {
    setStatus("Choose a video first.");
    return;
  }
  if (!transcript.value.trim()) {
    setStatus("Generate or enter a transcript before exporting.");
    return;
  }
  if (busy) return;

  const supportedType = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm"
  ].find(type => MediaRecorder.isTypeSupported(type));

  if (!supportedType) {
    setStatus("This browser does not support WebM video recording. Try Chrome.");
    return;
  }

  let exportVideo;
  let audioContext;
  let source;
  let canvasStream;
  let combinedStream;
  let recorder;
  let animationId;
  let finished = false;
  const parts = [];

  try {
    busy = true;
    exportButton.disabled = true;
    downloadLink.hidden = true;
    setStatus("Preparing video export...");

    exportVideo = document.createElement("video");
    exportVideo.src = URL.createObjectURL(file);
    exportVideo.playsInline = true;
    exportVideo.preload = "auto";

    await new Promise((resolve, reject) => {
      exportVideo.onloadedmetadata = resolve;
      exportVideo.onerror = () => reject(new Error("Could not open video for export."));
    });

    const width = exportVideo.videoWidth || 1280;
    const height = exportVideo.videoHeight || 720;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas is unavailable in this browser.");

    const chunks = getExportChunks(exportVideo.duration);

    audioContext = new AudioContext();
    await audioContext.resume();

    source = audioContext.createMediaElementSource(exportVideo);
    const audioDestination = audioContext.createMediaStreamDestination();
    source.connect(audioDestination);

    canvasStream = canvas.captureStream(30);
    combinedStream = new MediaStream([
      ...canvasStream.getVideoTracks(),
      ...audioDestination.stream.getAudioTracks()
    ]);

    recorder = new MediaRecorder(combinedStream, { mimeType: supportedType });

    recorder.ondataavailable = event => {
      if (event.data && event.data.size) parts.push(event.data);
    };

    const recorderDone = new Promise((resolve, reject) => {
      recorder.onstop = resolve;
      recorder.onerror = () => reject(new Error("Recording failed."));
    });

    const drawFrame = () => {
      if (finished) return;

      ctx.drawImage(exportVideo, 0, 0, width, height);
      const current = chunks.find(item =>
        exportVideo.currentTime >= item.start &&
        exportVideo.currentTime < item.end
      );

      drawCaption(ctx, current ? current.text : "", width, height);
      animationId = requestAnimationFrame(drawFrame);
    };

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, width, height);
    recorder.start(1000);
    drawFrame();

    setStatus("Exporting in real time. Keep this page open until it finishes...");
    await exportVideo.play();

    await new Promise((resolve, reject) => {
      exportVideo.onended = resolve;
      exportVideo.onerror = () => reject(new Error("Video playback stopped unexpectedly."));
    });

    finished = true;
    cancelAnimationFrame(animationId);
    recorder.stop();
    await recorderDone;

    const blob = new Blob(parts, { type: supportedType });
    if (!blob.size) throw new Error("The exported video is empty.");

    const downloadURL = URL.createObjectURL(blob);
    downloadLink.href = downloadURL;
    downloadLink.download = "captioned-video.webm";
    downloadLink.textContent = "Download captioned video";
    downloadLink.hidden = false;

    setStatus("Export complete! Tap the download link to save your captioned video.");
  } catch (error) {
    finished = true;
    if (animationId) cancelAnimationFrame(animationId);
    if (recorder && recorder.state !== "inactive") {
      try { recorder.stop(); } catch {}
    }
    setStatus("Export failed: " + error.message);
  } finally {
    finished = true;

    if (exportVideo) {
      const oldURL = exportVideo.src;
      exportVideo.pause();
      exportVideo.removeAttribute("src");
      exportVideo.load();
      if (oldURL.startsWith("blob:")) URL.revokeObjectURL(oldURL);
    }

    if (audioContext && audioContext.state !== "closed") {
      await audioContext.close();
    }

    if (combinedStream) combinedStream.getTracks().forEach(track => track.stop());
    if (canvasStream) canvasStream.getTracks().forEach(track => track.stop());

    busy = false;
    exportButton.disabled = false;
  }
});

updateStyle();
