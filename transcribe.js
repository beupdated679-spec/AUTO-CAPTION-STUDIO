import { pipeline } from
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const status = document.getElementById("status");
const transcript = document.getElementById("transcript");
const audioInput = document.getElementById("audioFile");
const loadButton = document.getElementById("loadEngine");
const transcribeButton = document.getElementById("transcribeAudio");

let transcriber = null;

async function loadCaptionEngine() {
  try {
    loadButton.disabled = true;
    status.textContent = "Loading AI speech model. Please wait...";

    transcriber = await pipeline(
      "automatic-speech-recognition",
      "onnx-community/whisper-tiny",
      { dtype: "q8" }
    );

    status.textContent =
      "AI model loaded. Choose your video and generate captions.";
  } catch (error) {
    status.textContent = "Model loading failed: " + error.message;
  } finally {
    loadButton.disabled = false;
  }
}

async function transcribeVideo(file) {
  if (!transcriber) {
    status.textContent = "First tap Load speech model.";
    return;
  }

  if (!file) {
    status.textContent = "Please select a video first.";
    return;
  }

  let audioContext;
  let video;
  let source;
  let processor;
  let silentGain;
  let samples = [];
  let stream;
  let finished = false;

  try {
    status.textContent = "Opening video and preparing audio...";

    video = document.createElement("video");
    video.src = URL.createObjectURL(file);
    video.playsInline = true;
    video.preload = "auto";

    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve;
      video.onerror = () => reject(new Error("Could not open this video."));
    });

    audioContext = new AudioContext();
    await audioContext.resume();

    source = audioContext.createMediaElementSource(video);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    silentGain = audioContext.createGain();
    silentGain.gain.value = 0;

    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(audioContext.destination);

    processor.onaudioprocess = event => {
      if (!finished) {
        samples.push(
          new Float32Array(event.inputBuffer.getChannelData(0))
        );
      }
    };

    status.textContent =
      "Extracting audio in real time. Keep this page open...";

    await video.play();

    await new Promise((resolve, reject) => {
      video.onended = resolve;
      video.onerror = () => reject(new Error("Video playback failed."));
    });

    finished = true;
    processor.disconnect();
    source.disconnect();

    const totalLength = samples.reduce((sum, chunk) => sum + chunk.length, 0);

    if (!totalLength) {
      throw new Error("No audio was captured from this video.");
    }

    const combined = new Float32Array(totalLength);
    let offset = 0;

    for (const chunk of samples) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    status.textContent = "Converting audio for speech recognition...";

    const inputRate = audioContext.sampleRate;
    const outputRate = 16000;
    const outputLength = Math.floor(
      combined.length * outputRate / inputRate
    );
    const audio = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      audio[i] = combined[Math.floor(i * inputRate / outputRate)];
    }

    status.textContent = "AI is generating your transcript...";

    const result = await transcriber(audio, {
      sampling_rate: outputRate,
      task: "transcribe",
      chunk_length_s: 20,
      stride_length_s: 3
    });

    transcript.value = result.text || "";
    status.textContent = "Transcript ready! Review and edit the text.";
  } catch (error) {
    status.textContent = "Could not generate captions: " + error.message;
  } finally {
    finished = true;

    if (processor) processor.disconnect();
    if (source) source.disconnect();

    if (audioContext && audioContext.state !== "closed") {
      await audioContext.close();
    }

    if (video) {
      const oldUrl = video.src;
      video.pause();
      video.removeAttribute("src");
      video.load();

      if (oldUrl.startsWith("blob:")) URL.revokeObjectURL(oldUrl);
    }
  }
}

loadButton.addEventListener("click", loadCaptionEngine);

transcribeButton.addEventListener("click", () => {
  transcribeVideo(audioInput.files[0]);
});
