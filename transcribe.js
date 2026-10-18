import { pipeline } from
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const status = document.getElementById("status");
const transcript = document.getElementById("transcript");

let transcriber;

async function loadCaptionEngine() {
  status.textContent =
    "Loading the speech model. The first download may be large.";

  try {
    transcriber = await pipeline(
      "automatic-speech-recognition",
      "onnx-community/whisper-tiny",
      { dtype: "q8" }
    );

    status.textContent =
      "Speech model loaded. Audio transcription can now be tested.";
  } catch (error) {
    status.textContent =
      "Could not load the model. Check your internet connection, " +
      "available memory, and model compatibility. " + error.message;
  }
}

async function transcribeAudioFile(file) {
  if (!transcriber) {
    status.textContent = "Load the speech model first.";
    return;
  }

  try {
    status.textContent = "Reading audio on your device...";

    const context = new AudioContext();
    const buffer = await context.decodeAudioData(
      await file.arrayBuffer()
    );

    const source = buffer.getChannelData(0);
    const ratio = buffer.sampleRate / 16000;
    const length = Math.floor(source.length / ratio);
    const audio = new Float32Array(length);

    for (let i = 0; i < length; i++) {
      audio[i] = source[Math.floor(i * ratio)];
    }

    status.textContent = "Transcribing audio... Keep this page open.";

    const result = await transcriber(audio, {
      sampling_rate: 16000,
      task: "transcribe"
    });

    transcript.value = result.text || "";
    status.textContent =
      "Transcription finished. Review the text for mistakes.";

    await context.close();
  } catch (error) {
    status.textContent =
      "Transcription failed. Try a supported audio file. " +
      error.message;
  }
}

const loadButton = document.getElementById("loadEngine");
const audioInput = document.getElementById("audioFile");
const transcribeButton = document.getElementById("transcribeAudio");

loadButton?.addEventListener("click", loadCaptionEngine);

transcribeButton?.addEventListener("click", () => {
  const file = audioInput?.files?.[0];

  if (!file) {
    status.textContent = "Choose an audio file first.";
    return;
  }

  transcribeAudioFile(file);
});
