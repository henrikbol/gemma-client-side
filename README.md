# Gemma 4 — Browser-Based Document Analyzer

A fully client-side document analyzer powered by [Google Gemma 4](https://ai.google.dev/gemma) running directly in the browser via WebGPU. No server, no API keys, no data ever leaving your machine.

## Demo

Upload a PDF, Word document, image, or paste text — and get a structured analysis in seconds, all running locally on your GPU.

## Why this project

This is a demo case exploring **in-browser LLM inference** — running a multimodal language model entirely on the client using WebGPU and ONNX, with zero backend infrastructure. It demonstrates that meaningful AI-powered document workflows are possible without sending any data to the cloud.

## How it works

1. The app downloads a quantized (Q4F16) ONNX build of Gemma 4 from HuggingFace on first visit and caches it in the browser.
2. Inference runs locally via [Transformers.js](https://huggingface.co/docs/transformers.js) and WebGPU.
3. Documents are converted to images or text in-browser before being passed to the model.
4. Results stream back token-by-token and are rendered as Markdown.

No server is involved after the initial page load. See [privacy.md](privacy.md) for full details.

## Supported input formats

| Format | How it's processed |
|---|---|
| Images (PNG, JPG, etc.) | Passed directly to the multimodal model |
| PDF | Rendered page-by-page to images via pdf.js |
| Word (.docx) | Converted to HTML (Mammoth), then rendered to an image (html2canvas) |
| Text / Markdown | Passed as text input to the model |

## Tech stack

- **[Transformers.js](https://huggingface.co/docs/transformers.js)** — model loading and inference via ONNX + WebGPU
- **[Gemma 4 ONNX](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX)** — quantized multimodal model
- **[pdf.js](https://mozilla.github.io/pdf.js/)** — PDF rendering
- **[Mammoth](https://github.com/mwilliamson/mammoth.js)** — Word-to-HTML conversion
- **[html2canvas](https://html2canvas.hertzen.com/)** — DOM-to-image rendering
- **[Marked](https://marked.js.org/)** — Markdown rendering

## Getting started

No build step required. Serve the project with any static file server:

```bash
uv run python -m http.server 8000
```

Then open `http://localhost:8000` in a browser with WebGPU support (Chrome 113+, Edge 113+).

The first load will download ~2 GB of model weights, which are cached by the browser for subsequent visits.

## Requirements

- A browser with **WebGPU** support
- A GPU with enough VRAM to run the quantized model (~4 GB recommended)

## Privacy

All document processing and inference happens locally in your browser. No data is sent to any server. See [privacy.md](privacy.md) for details.
