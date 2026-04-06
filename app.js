import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  TextStreamer,
  load_image,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";

import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4/build/pdf.min.mjs";
pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4/build/pdf.worker.min.mjs";

const statusEl   = document.getElementById("status");
const dropZone   = document.getElementById("drop-zone");
const fileInput  = document.getElementById("file-input");
const preview    = document.getElementById("preview");
const fileInfo   = document.getElementById("file-info");
const runBtn     = document.getElementById("run-btn");
const output     = document.getElementById("output");
const textInput  = document.getElementById("text-input");
const fileTab    = document.getElementById("file-tab");
const textTab    = document.getElementById("text-tab");
const textPreview    = document.getElementById("text-preview");
const renderContainer = document.getElementById("render-container");

const progressWrap   = document.getElementById("progress-wrap");
const progressBar    = document.getElementById("progress-bar");
const progressDetail = document.getElementById("progress-detail");

let processor, model;

// inputData: { type: "images", images: [Blob,...] } | { type: "text", text: "..." } | null
let inputData = null;
let activeTab = "file";

// ---- Privacy modal ----
const privacyLink    = document.getElementById("privacy-link");
const privacyOverlay = document.getElementById("privacy-overlay");
privacyLink.addEventListener("click", (e) => { e.preventDefault(); privacyOverlay.classList.add("open"); });
document.getElementById("privacy-close-btn").addEventListener("click", () => privacyOverlay.classList.remove("open"));
privacyOverlay.addEventListener("click", (e) => { if (e.target === privacyOverlay) privacyOverlay.classList.remove("open"); });

// ---- Tabs ----
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeTab = btn.dataset.tab;
    fileTab.classList.toggle("hidden", activeTab !== "file");
    textTab.classList.toggle("hidden", activeTab !== "text");
    updateRunBtn();
  });
});

// ---- Model loading ----
const MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(0) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function makeProgressHandler(label) {
  const fileProgress = {};
  return (info) => {
    console.log("[progress]", label, info.status, info.file, info);
    if (info.status === "progress" && info.total) {
      fileProgress[info.file] = { loaded: info.loaded, total: info.total };
      const totalLoaded = Object.values(fileProgress).reduce((s, f) => s + f.loaded, 0);
      const totalSize   = Object.values(fileProgress).reduce((s, f) => s + f.total, 0);
      const pct = Math.min(100, (totalLoaded / totalSize) * 100).toFixed(0);
      progressBar.style.width = pct + "%";
      statusEl.textContent = `${label}: ${pct}%`;
      progressDetail.textContent = `${formatBytes(totalLoaded)} / ${formatBytes(totalSize)}`;
      progressWrap.style.display = "block";
    } else if (info.status === "initiate") {
      statusEl.textContent = `${label}: loading ${info.file ?? ""}…`;
      progressWrap.style.display = "block";
    } else if (info.status === "ready") {
      progressDetail.textContent = `${info.file ?? ""} ready`;
    }
  };
}

async function loadModel() {
  try {
    if (!navigator.gpu) {
      statusEl.textContent = "Your browser does not support WebGPU. Please use a recent version of Chrome, Edge, or Opera.";
      statusEl.className = "error";
      return;
    }

    statusEl.textContent = "Downloading processor…";
    progressWrap.style.display = "block";
    progressBar.style.width = "0%";
    processor = await AutoProcessor.from_pretrained(MODEL_ID, {
      progress_callback: makeProgressHandler("Processor"),
    });

    statusEl.textContent = "Downloading model weights — this may take a while on first visit…";
    progressBar.style.width = "0%";

    model = await Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: "q4f16",
      device: "webgpu",
      progress_callback: makeProgressHandler("Model"),
    });

    progressWrap.style.display = "none";
    progressDetail.textContent = "";
    statusEl.textContent = "Model ready — upload a file or enter text to begin.";
    statusEl.className = "ready";
    updateRunBtn();
  } catch (e) {
    statusEl.textContent = `Failed to load model: ${e.message}`;
    statusEl.className = "error";
    console.error(e);
  }
}

loadModel();

// ---- File handling ----
function getFileExt(name) {
  return (name || "").split(".").pop().toLowerCase();
}

async function handleFile(file) {
  if (!file) return;
  const ext = getFileExt(file.name);
  const type = file.type;

  preview.hidden = true;
  textPreview.hidden = true;
  textPreview.textContent = "";
  fileInfo.hidden = true;
  fileInfo.innerHTML = "";
  inputData = null;

  try {
    if (type.startsWith("image/")) {
      await handleImage(file);
    } else if (type === "application/pdf" || ext === "pdf") {
      await handlePDF(file);
    } else if (type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" || ext === "docx" || ext === "doc") {
      await handleDocx(file);
    } else if (["txt", "md", "markdown"].includes(ext) || type.startsWith("text/")) {
      await handleTextFile(file);
    } else {
      statusEl.textContent = `Unsupported file type: ${ext}`;
      statusEl.className = "error";
    }
  } catch (e) {
    statusEl.textContent = `Error processing file: ${e.message}`;
    statusEl.className = "error";
    console.error(e);
  }

  updateRunBtn();
}

async function handleImage(file) {
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  inputData = { type: "images", images: [file] };
  fileInfo.innerHTML = `<strong>${file.name}</strong>`;
  fileInfo.hidden = false;
}

async function handlePDF(file) {
  statusEl.textContent = "Rendering PDF pages…";
  statusEl.className = "";
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pageCount = pdf.numPages;
  const images = [];

  for (let i = 1; i <= pageCount; i++) {
    statusEl.textContent = `Rendering PDF page ${i} / ${pageCount}…`;
    const page = await pdf.getPage(i);
    const scale = 2;
    const viewport = page.getViewport({ scale });
    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    const ctx = canvas.getContext("2d");
    await page.render({ canvasContext: ctx, viewport }).promise;
    const blob = await new Promise(r => canvas.toBlob(r, "image/png"));
    images.push(blob);
  }

  // Show first page as preview
  preview.src = URL.createObjectURL(images[0]);
  preview.hidden = false;
  fileInfo.innerHTML = `<strong>${file.name}</strong> <span class="badge">${pageCount} page${pageCount > 1 ? "s" : ""}</span>`;
  fileInfo.hidden = false;
  inputData = { type: "images", images };
  statusEl.textContent = "PDF ready.";
  statusEl.className = "ready";
}

async function handleDocx(file) {
  statusEl.textContent = "Converting Word document…";
  statusEl.className = "";
  const arrayBuffer = await file.arrayBuffer();
  const result = await mammoth.extractRawText({ arrayBuffer });
  const text = result.value.replace(/^\s*\n/, "");

  textPreview.textContent = text.slice(0, 500) + (text.length > 500 ? "\n…" : "");
  textPreview.hidden = false;
  fileInfo.innerHTML = `<strong>${file.name}</strong> <span class="badge">Word</span>`;
  fileInfo.hidden = false;
  inputData = { type: "text", text };
  statusEl.textContent = "Document ready.";
  statusEl.className = "ready";
}

async function handleTextFile(file) {
  const text = (await file.text()).replace(/^\s*\n/, "");

  textPreview.textContent = text.slice(0, 500) + (text.length > 500 ? "\n…" : "");
  textPreview.hidden = false;
  fileInfo.innerHTML = `<strong>${file.name}</strong> <span class="badge">Text</span>`;
  fileInfo.hidden = false;
  inputData = { type: "text", text };
}

function updateRunBtn() {
  if (activeTab === "text") {
    runBtn.disabled = !(model && textInput.value.trim());
  } else {
    runBtn.disabled = !(model && inputData);
  }
}

textInput.addEventListener("input", updateRunBtn);

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  handleFile(e.dataTransfer.files[0]);
});

// ---- Prompts ----
const promptSelect  = document.getElementById("prompt-select");
const editBtn       = document.getElementById("edit-prompt-btn");
const modalOverlay  = document.getElementById("modal-overlay");
const modalName     = document.getElementById("modal-name");
const modalText     = document.getElementById("modal-text");
const modalSaveBtn  = document.getElementById("modal-save-btn");
const modalResetBtn = document.getElementById("modal-reset-btn");
const modalCancelBtn = document.getElementById("modal-cancel-btn");

const STORAGE_KEY = "gemma4-prompts";

const DEFAULT_PROMPTS = [
  { name: "Opsummering", text: `Læs det vedhæftede dokument grundigt. Lav en struktureret opsummering af indholdet.

Krav til format:

Brug faste overskrifter til at adskille emnerne.

Brug bulletpoints til at beskrive de specifikke trin og regler.

Sørg for at få de vigtige detaljer med (deadlines, ansvarlige, specifikke krav), men hold sproget letforståeligt og direkte.

Start med et ultrakort resumé på 2-3 linjer af dokumentets overordnede formål.` },
  { name: "Prompt 2", text: "" },
  { name: "Prompt 3", text: "" },
  { name: "Prompt 4", text: "" },
  { name: "Prompt 5", text: "" },
];

function loadPrompts() {
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored ? JSON.parse(stored) : DEFAULT_PROMPTS.map(p => ({ ...p }));
}

function savePrompts(prompts) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(prompts));
}

let prompts = loadPrompts();

function renderPromptSelect(keepIndex) {
  const idx = keepIndex ?? 0;
  promptSelect.innerHTML = "";
  prompts.forEach((p, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = p.name;
    promptSelect.appendChild(opt);
  });
  promptSelect.value = idx;
}

renderPromptSelect();

function openModal() {
  const idx = Number(promptSelect.value);
  modalName.value = prompts[idx].name;
  modalText.value = prompts[idx].text;
  modalOverlay.classList.add("open");
  modalName.focus();
}

function closeModal() {
  modalOverlay.classList.remove("open");
}

editBtn.addEventListener("click", openModal);

modalCancelBtn.addEventListener("click", closeModal);

modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) closeModal();
});

modalSaveBtn.addEventListener("click", () => {
  const idx = Number(promptSelect.value);
  prompts[idx].name = modalName.value;
  prompts[idx].text = modalText.value;
  savePrompts(prompts);
  renderPromptSelect(idx);
  closeModal();
});

modalResetBtn.addEventListener("click", () => {
  const idx = Number(promptSelect.value);
  modalName.value = DEFAULT_PROMPTS[idx].name;
  modalText.value = DEFAULT_PROMPTS[idx].text;
});

// ---- Inference ----
let rawOutput = "";

function renderMarkdown() {
  output.innerHTML = marked.parse(rawOutput);
}

async function runInference(image, promptStr) {
  const messages = [{ role: "user", content: [] }];
  if (image) messages[0].content.push({ type: "image" });
  messages[0].content.push({ type: "text", text: promptStr });

  const chatPrompt = processor.apply_chat_template(messages, {
    enable_thinking: false,
    add_generation_prompt: true,
  });

  const inputs = image
    ? await processor(chatPrompt, image, null, { add_special_tokens: false })
    : await processor(chatPrompt, null, null, { add_special_tokens: false });

  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => { rawOutput += text; renderMarkdown(); },
  });

  await model.generate({
    ...inputs,
    max_new_tokens: 1024,
    do_sample: false,
    streamer,
  });
}

runBtn.addEventListener("click", async () => {
  // Resolve current input
  let currentInput = inputData;
  if (activeTab === "text") {
    currentInput = { type: "text", text: textInput.value.trim() };
  }
  if (!model || !currentInput) return;

  runBtn.disabled = true;
  rawOutput = "";
  output.innerHTML = "";
  statusEl.textContent = "Running inference…";
  statusEl.className = "";

  const promptStr = prompts[Number(promptSelect.value)].text;

  try {
    if (currentInput.type === "images") {
      for (let i = 0; i < currentInput.images.length; i++) {
        if (currentInput.images.length > 1) {
          statusEl.textContent = `Processing page ${i + 1} / ${currentInput.images.length}…`;
          if (i > 0) rawOutput += "\n\n---\n\n## Page " + (i + 1) + "\n\n";
        }
        const image = await load_image(URL.createObjectURL(currentInput.images[i]));
        await runInference(image, promptStr);
      }
    } else {
      // Text-only: prepend the user's text to the prompt
      await runInference(null, currentInput.text + "\n\n" + promptStr);
    }

    statusEl.textContent = "Done.";
    statusEl.className = "ready";
  } catch (e) {
    statusEl.textContent = `Inference error: ${e.message}`;
    statusEl.className = "error";
    console.error(e);
  } finally {
    updateRunBtn();
  }
});
