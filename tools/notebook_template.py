"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "phi4_multimodal_pipeline",
    "repo_name": "phi4-multimodal-pipeline",
    "stem": "phi4_multimodal",
    "notebook_name": "phi4_multimodal_colab.ipynb",
    "profile": "MULTI-CAPABILITY",
    "pipeline_class": "Phi4MultimodalPipeline",
    "weights_key": "phi4-multimodal-instruct",
    "model_load": "Phi4MultimodalPipeline.from_pretrained(allow_remote_code=True, weights_dir=WEIGHTS_DIR)",
    "runtime_imports": ["torch", "transformers"],
    "title": "Phi-4 Multimodal — DIMER multi-capability tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/phi4-multimodal-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/phi4-multimodal-pipeline/blob/main/tutorials/phi4_multimodal_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2FPhi--4--multimodal--instruct-ffcc4d?style=flat",
            "https://huggingface.co/microsoft/Phi-4-multimodal-instruct",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2503.01743-b31b1b.svg", "https://arxiv.org/abs/2503.01743"),
    ],
    "capability": "text-, image-, and audio-conditioned text generation using one pinned `microsoft/Phi-4-multimodal-instruct` checkpoint",
    "intro": (
        "At inference the 5.6B language model consumes a chat prompt in which `<|image_k|>` and `<|audio_k|>` "
        "placeholders are replaced by SigLIP vision embeddings and conformer speech embeddings (each modality routed "
        "through its own LoRA adapter inside the checkpoint), then generates text autoregressively; the pipeline "
        "builds that prompt, runs greedy decoding by default and returns the generated text with provenance. "
        "**No adaptation occurs:** no training, fine-tuning, in-context conditioning beyond the single instruction, or "
        "preprocessing fitting happens in this notebook — the upstream checkpoint supplies the weights, the processor "
        "configuration **and the model code**, and the carried pipeline module adds snapshot verification, the "
        "explicit remote-code opt-in, the input contract, a fixed output contract and the `validate_inputs` and "
        "`evaluation_report` helpers. **Trust boundary (MOD9/MOD10):** this release ships its model, configuration "
        "and processor as Python files that `transformers` must execute (`trust_remote_code=True` inside the carried "
        "loader, gated behind `allow_remote_code=True`); the standalone path pins each of those files by SHA-256 in "
        "the inline manifest, fetches them at the immutable revision and re-hashes them before they are imported, so "
        "the code that runs is exactly the reviewed revision's — no other remote code is executed. The default samples "
        "are a text instruction, a synthetic traffic-sign image drawn in code, and the checkpoint's own example speech "
        "clip; their outputs are demonstration (plumbing) evidence, not a benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream revision including its remote model code, acknowledge the custom-code trust boundary "
        "explicitly, validate each capability's input into an input manifest, run text-only, image + text and audio + "
        "text generation through one public API, exercise optional BYOD image and audio inputs, produce an evaluation "
        "report that is honestly `not-measurable` for open-ended generation, and export machine-readable outputs plus "
        "provenance."
    ),
    "exclusions": (
        "fine-tuning (the checkpoint's `sample_finetune_*.py` scripts are pinned but never run), FlashAttention 2 "
        "(available only by explicit caller choice with a compatible `flash-attn` build), video, tool use, "
        "multi-turn chat state, calibrated answer confidence, or any claim that upstream benchmark results were "
        "reproduced. Generated descriptions and transcripts can be wrong and the pipeline does not detect it."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12) with a **CUDA GPU** and roughly 16 GB of GPU memory for the ~11.2 GB snapshot loaded in its stored bfloat16 precision; `from_pretrained` refuses `device='cuda'` without a CUDA device. Attention is `eager` for portability. The pinned `torch==2.6.0` install and the three SafeTensors shards are the largest downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a chat prompt, greedy decoding and `trust_remote_code` mean.",
        "- **Data:** the default image is a synthetic red octagon with a white border drawn in code (no download, no ground truth); the default audio is `examples/what_is_the_traffic_sign_in_the_image.wav`, a short speech clip that ships inside the pinned checkpoint snapshot itself and is therefore fetched from the Hugging Face Hub at the immutable revision and digest-verified with the weights. Optional BYOD uploads are gated off by default so the sample path can run top-to-bottom without interaction; expected BYOD inputs are one image file decodable by Pillow and/or one audio file decodable by `soundfile`. Do not upload confidential or restricted media to a hosted notebook environment unless you are authorized to do so. Inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Prepare the samples or optional BYOD\n\n"
                "Three inputs, one per capability. **Text:** a fixed instruction. **Image:** a synthetic 256×256 "
                "traffic-sign-like image drawn in code — a red octagon with a white border on a light background, "
                "deterministic, digest printed; it is not a photograph, so the model's description is a sanity check "
                "of the vision path, not a correctness measurement. **Audio:** the checkpoint's own "
                "`examples/what_is_the_traffic_sign_in_the_image.wav` (already staged and digest-verified in Section 3 "
                "because it is a manifest entry), decoded with the pinned `soundfile` dependency into a float32 "
                "waveform plus sampling rate and passed as the `(array, sampling_rate)` tuple the upstream processor "
                "expects. BYOD is optional and disabled by default; `USE_BYOD` enables upload dialogs for an image "
                "and/or an audio file — skip either dialog by cancelling it. Look for a dictionary naming each "
                "sample's kind and digest."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "import soundfile as sf\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "TEXT_INSTRUCTION = 'In two sentences, explain what automatic speech recognition does.'\n"
                "IMAGE_INSTRUCTION = 'Describe the most prominent traffic sign in the image.'\n"
                "AUDIO_INSTRUCTION = 'Transcribe the attached speech.'\n"
                "SAMPLE_AUDIO = WEIGHTS_DIR / 'examples' / 'what_is_the_traffic_sign_in_the_image.wav'\n\n"
                "def _synthetic_sign(side=256):\n"
                "    # Deterministic red octagon with a white border on a light grey background: no randomness, stable digest.\n"
                "    canvas = Image.new('RGB', (side, side), (235, 235, 235))\n"
                "    draw = ImageDraw.Draw(canvas)\n"
                "    centre, radius = side / 2, side * 0.42\n"
                "    octagon = [(centre + radius * np.cos(np.pi / 8 + k * np.pi / 4), centre + radius * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)]\n"
                "    draw.polygon(octagon, fill=(200, 16, 24), outline=(255, 255, 255), width=max(2, side // 40))\n"
                "    return canvas\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    print('Upload an image (or cancel to keep the synthetic sign).')\n"
                "    uploaded = files.upload()\n"
                "    if uploaded:\n"
                "        image_name, image_bytes = next(iter(uploaded.items()))\n"
                "        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')\n"
                "        image_kind = 'BYOD'\n"
                "    else:\n"
                "        image, image_name, image_kind = _synthetic_sign(), 'synthetic_octagon_256.png', 'synthetic'\n"
                "    print('Upload an audio file (or cancel to keep the checkpoint example clip).')\n"
                "    uploaded = files.upload()\n"
                "    if uploaded:\n"
                "        audio_name, audio_bytes = next(iter(uploaded.items()))\n"
                "        waveform, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')\n"
                "        audio_kind = 'BYOD'\n"
                "    else:\n"
                "        waveform, sampling_rate = sf.read(SAMPLE_AUDIO, dtype='float32')\n"
                "        audio_name, audio_kind = SAMPLE_AUDIO.name, 'checkpoint-example'\n"
                "else:\n"
                "    image, image_name, image_kind = _synthetic_sign(), 'synthetic_octagon_256.png', 'synthetic'\n"
                "    waveform, sampling_rate = sf.read(SAMPLE_AUDIO, dtype='float32')\n"
                "    audio_name, audio_kind = SAMPLE_AUDIO.name, 'checkpoint-example'\n"
                "if waveform.ndim > 1:\n"
                "    waveform = waveform.mean(axis=1)\n"
                "audio = (waveform, sampling_rate)\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "audio_sha256 = hashlib.sha256(np.ascontiguousarray(waveform, dtype=np.float32).tobytes()).hexdigest()\n"
                "sample_kind = 'BYOD' if USE_BYOD else 'synthetic'\n"
                "print({{'image': {{'kind': image_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256}}, 'audio': {{'kind': audio_kind, 'name': audio_name, 'seconds': round(len(waveform) / sampling_rate, 2), 'sampling_rate': sampling_rate, 'waveform_sha256': audio_sha256}}}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `generate` "
                "applies — a non-empty instruction, at most `MAX_IMAGES` images and `MAX_AUDIOS` audio clips, "
                "`max_new_tokens` 1..`MAX_NEW_TOKENS`, `temperature` 0..`MAX_TEMPERATURE` — and returns an **input "
                "manifest** naming the schema and ceilings, each media input's observed properties (image mode and "
                "size; audio samples, sampling rate, duration), the request (prompt length, decoding settings) and the "
                "verdict. One manifest is produced per capability and the three are written together to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a "
                "request with five images and records the pipeline's own error message as a finding. Inside the "
                "upstream processor images are tiled for the SigLIP encoder and audio is converted to speech features; "
                "nothing else is dropped or altered."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MAX_IMAGES': MAX_IMAGES, 'MAX_AUDIOS': MAX_AUDIOS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'MAX_TEMPERATURE': MAX_TEMPERATURE}}}})\n"
                "input_manifest = {{\n"
                "    'text': validate_inputs(TEXT_INSTRUCTION, max_new_tokens=96),\n"
                "    'image': validate_inputs(IMAGE_INSTRUCTION, images=[image], max_new_tokens=96, names=[image_name]),\n"
                "    'audio': validate_inputs(AUDIO_INSTRUCTION, audios=[audio], max_new_tokens=128, names=[audio_name]),\n"
                "}}\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(IMAGE_INSTRUCTION, images=[image] * (MAX_IMAGES + 1))\n"
                "except ValueError as exc:\n"
                "    input_manifest['image']['findings'].append({{'input': 'too-many-images-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Capability A — text-only generation\n\n"
                "**Input contract:** one instruction string, no media. **Output contract:** `text` is the decoded "
                "continuation after the `<|assistant|>` tag, greedy (`temperature=0.0` → `do_sample=False`) and capped "
                "at `max_new_tokens`; the result also echoes the media counts, decoding settings, attention backend, "
                "device and weight source. The text is generated language, not a calibrated statement — a fluent "
                "answer is not evidence of correctness."
            ),
            "code": (
                "text_result = pipe.generate(TEXT_INSTRUCTION, max_new_tokens=96, temperature=0.0)\n"
                "print({{'image_count': text_result['image_count'], 'audio_count': text_result['audio_count'], 'temperature': text_result['temperature'], 'attention_implementation': text_result['attention_implementation'], 'device': text_result['device'], 'source': text_result['source']}})\n"
                "print(text_result['text'])"
            ),
        },
        {
            "md": (
                "## 7. Capability B — image + text\n\n"
                "**Input contract:** one instruction plus up to `MAX_IMAGES` PIL images; the pipeline inserts one "
                "`<|image_k|>` placeholder per image and the upstream processor tiles each image for the SigLIP "
                "encoder. **Output contract:** generated text describing or answering about the images. On the "
                "synthetic octagon the description is a sanity check that the vision path executes — the image has no "
                "ground-truth caption, and a plausible \"stop sign\" answer must not be read as recognition accuracy."
            ),
            "code": (
                "image_result = pipe.generate(IMAGE_INSTRUCTION, images=[image], max_new_tokens=96, temperature=0.0)\n"
                "print({{'image_count': image_result['image_count'], 'audio_count': image_result['audio_count']}})\n"
                "print(image_result['text'])"
            ),
        },
        {
            "md": (
                "## 8. Capability C — audio + text\n\n"
                "**Input contract:** one instruction plus up to `MAX_AUDIOS` `(waveform, sampling_rate)` tuples; the "
                "pipeline inserts one `<|audio_k|>` placeholder per clip and the upstream processor computes speech "
                "features. **Output contract:** generated text — here a transcription request. The checkpoint's example "
                "clip is a short spoken question; its file name suggests the expected words, but that is not a "
                "reference transcript and no accuracy is inferred from this one clip."
            ),
            "code": (
                "audio_result = pipe.generate(AUDIO_INSTRUCTION, audios=[audio], max_new_tokens=128, temperature=0.0)\n"
                "print({{'image_count': audio_result['image_count'], 'audio_count': audio_result['audio_count']}})\n"
                "print(audio_result['text'])"
            ),
        },
        {
            "md": (
                "## 9. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. The "
                "repository ships **no metric helper** for open-ended generation, so the verdict is `not-measurable` "
                "by construction: the report records, per capability, whether the output was non-empty (plumbing "
                "evidence only) and what labelled data would make it measurable — reference answers with a task metric "
                "for text, image-question pairs with reference answers (VQA accuracy) or captions (CIDEr) for images, "
                "and reference transcripts scored with a word error rate for audio. The report is written to "
                "`outputs/{stem}_evaluation_report.json`. A `sample-sanity` verdict is deliberately not available here "
                "because no metric exists in the repository to back it."
            ),
            "code": (
                "results = {{'text': text_result, 'image': image_result, 'audio': audio_result}}\n"
                "report = evaluation_report(results, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "print('No reference answers, captions or transcripts exist for these samples; the outputs above are sanity evidence only.')"
            ),
        },
        {
            "md": (
                "## 10. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves each capability's full result, the evaluation report, the input "
                "manifests, the sample identities and digests, the notebook's source (repository, revision, embedded "
                "module digest, generator), the model identifier, the immutable model revision (which pins the remote "
                "code as well as the weights), the model licence, the acknowledged trust boundary, and the runtime "
                "identity (Python, `torch`, `transformers`, device, attention backend). The three generated texts are "
                "also written as one plain-text file. No credentials are recorded."
            ),
            "code": (
                "payload = {{\n"
                "    'capabilities': results,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'samples': {{'text': {{'instruction': TEXT_INSTRUCTION}}, 'image': {{'kind': image_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256}}, 'audio': {{'kind': audio_kind, 'name': audio_name, 'sampling_rate': sampling_rate, 'waveform_sha256': audio_sha256}}}},\n"
                "    'trust_boundary': {{'remote_code_executed': True, 'remote_code_files': list(REMOTE_CODE_FILES), 'pinned_by': 'inline manifest SHA-256 at MODEL_REVISION, verified by verify_snapshot before import', 'opt_in': 'allow_remote_code=True'}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'attention_implementation': pipe.attention_implementation,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_generations.txt', 'w', encoding='utf-8') as handle:\n"
                "    for label, result in results.items():\n"
                "        handle.write(f'[{{label}}]\\n{{result[\"text\"]}}\\n\\n')\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The three outputs share one generative model but have different evidence sources and failure modes. "
        "Non-empty text only establishes that the inference path executed; the evaluation report is `not-measurable` "
        "because the repository ships no metric and the samples carry no references. Image and audio interpretations "
        "can be wrong, the synthetic octagon is not a photograph of any sign, the example clip has no reference "
        "transcript, and no calibrated answer confidence is returned. The model executes upstream Python code: that "
        "code is pinned by digest at the immutable revision and re-hashed before import, which fixes *which* code runs "
        "but does not make it audited — review the pinned files before any deployment. The tutorial deliberately does "
        "not expose fine-tuning or claim that upstream benchmark results were reproduced.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model **and its remote code**, execute the explicitly acknowledged "
        "custom-code boundary, validate the demonstrated inputs, run the demonstrated public pipeline capabilities, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It "
        "does **not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, "
        "or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a photograph of a real traffic sign and a recording you have "
        "transcribed yourself; pass both media in one call (`images=[image], audios=[audio]`) with the checkpoint's "
        "other example clip `what_is_shown_in_this_image.wav` to reproduce the upstream image-question demo; raise "
        "`temperature` above 0 and observe non-deterministic sampling; compare `attention_implementation='flash_attention_2'` "
        "only after installing a compatible `flash-attn` build.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/phi4-multimodal-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/phi4-multimodal-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/phi4-multimodal-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Pinned upstream sample code: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/{MODEL_REVISION}/sample_inference_phi4mm.py\n"
        "- Technical report: https://arxiv.org/abs/2503.01743"
    ),
}
