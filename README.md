# MediGuide: EdgeAI-Assistive-Medicine-Detector
Real-time embedded medicine recognition system on MaixPy Edge-AI featuring PaddleOCR, 4-stage glyph/phonetic error correction, temporal voting, and UART telemetry.
<div align="center">

#  Edge-AI Assistive Medicine Recognition System
### Real-Time Embedded OCR, Hierarchical Error Correction & Temporal Voting for the Visually Impaired

[![Platform: MaixPy Edge AI](https://img.shields.io/badge/Platform-MaixPy%20Edge%20AI-FF6F00.svg)]()
[![Model: PP-OCR](https://img.shields.io/badge/Vision%20Model-PaddleOCR%20(PP--OCR)-blue.svg)]()
[![Language: Python](https://img.shields.io/badge/Language-Python%203-3776AB.svg?logo=python)](https://www.python.org/)
[![Interface: UART](https://img.shields.io/badge/Hardware-UART%20Serial%20(9600%20baud)-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Author:** Jingkun Li (Jarred)  
*Assistive Computer Vision & Embedded Edge-AI Project*

</div>

---

##  Motivation & Overview

Medication errors pose severe health risks for visually impaired and elderly individuals, as pharmaceutical packaging often features small fonts, glossy curved surfaces, and visually identical boxes. While cloud-based vision APIs suffer from network latency and privacy concerns, naive on-device OCR frequently misrecognizes characters under non-ideal lighting or motion blur—which is unacceptable in safety-critical medical scenarios.

This project implements a standalone, real-time **Assistive Medicine Box Recognition System** deployed on a **MaixPy Edge-AI vision processor**. By combining an embedded **PP-OCR deep neural network** with a custom **4-stage hierarchical post-OCR correction pipeline**, **phonetic fuzzy matching**, and **temporal multi-frame consensus voting**, the system reliably identifies medications and transmits structured alerts via **UART (`9600 baud`)** to downstream voice synthesis / smart pillbox actuators.

---

##   Key Engineering & Algorithmic Innovations

### 1. Center-Cropped ROI Acceleration (`85%` Field of View)
To eliminate peripheral background clutter and accelerate neural network inference on edge hardware, the camera capture pipeline dynamically crops a centered **Region of Interest (ROI)** covering $85\% \times 85\%$ of the frame (`ocr_roi`), guiding users to hold medicine packages within the optimal focal zone.

### 2. 4-Stage Hierarchical Post-OCR Correction & Confidence Scoring
Curved blister packs and specular reflections often cause OCR character confusion between visually similar glyph structures (homoglyphs, e.g., `囊` $\leftrightarrow$ `襄`, `颗` $\leftrightarrow$ `顺`) or homophones (`瘟` $\leftrightarrow$ `温`, `芬` $\leftrightarrow$ `分`, `林` $\leftrightarrow$ `淋`). To prevent misidentification without retraining heavy neural weights, a lightweight 4-stage NLP post-processing pipeline (`smart_correct_with_confidence`) is implemented[cite: 8]:

| Stage | Correction Strategy | Algorithmic Mechanism | Assigned Confidence ($C$) |
| :---: | :--- | :--- | :---: |
| **Stage 1** | **Exact Lexicon Match** | Regex non-CJK filtering (`[^\u4e00-\u9fff]`) + direct lookup against `STANDARD_MEDICINES` database[cite: 8] | **`0.95` (95%)**[cite: 8] |
| **Stage 2** | **Character-Level Homoglyph Fixing** | Replaces common OCR visual/phonetic confusions via `CHAR_CORRECTION` dictionary mapping[cite: 8] | **`0.88` (88%)**[cite: 8] |
| **Stage 3** | **Phrase-Level Dictionary Mapping** | Resolves multi-character corrupted stems via `CORRECTION_MAP`[cite: 8] | **`0.85` (85%)**[cite: 8] |
| **Stage 4** | **Phonetic Sequence Fuzzy Matching** | Converts characters to phonetic Pinyin representations (`rough_pinyin`) and computes Gestalt pattern matching ratio via `difflib.SequenceMatcher` (threshold $\ge 0.65$)[cite: 8] | **`0.65 – 0.80`** $\min(0.65 + 0.15 \cdot S, 0.80)$[cite: 8] |

> *Note: Any detection failing domain keyword validation (`MEDICINE_KEYWORDS`) or scoring below the $0.65$ confidence cutoff is immediately discarded[cite: 8].*

### 3. Temporal Multi-Frame Consensus Voting (`VOTE_THRESHOLD = 3`)
In assistive healthcare applications, a single-frame false positive could lead to dangerous medication mistakes. The system enforces a temporal state buffer (`vote_confirm`)[cite: 8]:
* A medicine identity is **only confirmed after being consecutively validated across $\ge 3$ frames** with confidence $\ge 0.65$[cite: 8].
* Once confirmed, the voting accumulator clears and enforces a **2-second non-blocking transmission cooldown (`SEND_INTERVAL = 2 s`)** to prevent auditory/serial flooding[cite: 8].

### 4. Embedded UART Hardware Interfacing (`UART2`)
Confirmed medication labels and confidence percentages are formatted and encoded into **`GBK` byte streams** (`f"{text},{int(conf*100)}%\n"`) for compatibility with embedded Chinese TTS (Text-to-Speech) voice synthesis chips and external microcontrollers[cite: 8]:
* **Pin Multiplexing:** `Pin A29` $\rightarrow$ `UART2_RX` | `Pin A28` $\rightarrow$ `UART2_TX`[cite: 8]
* **Serial Protocol:** `/dev/ttyS2` at `9600 baud`[cite: 8]

---

##  System Pipeline Architecture

```text
[Camera Frame] ──► [85% Center ROI Crop] ──► [Embedded PP-OCR Inference]
                                                      │
                                                      ▼
[Discard Frame] ◄── (Conf < 0.65) ── [4-Stage Glyph & Phonetic Correction]
                                                      │ (Conf >= 0.65)
                                                      ▼
[On-Screen HUD] ◄─────────────────── [Multi-Frame Consensus Voter (N >= 3)]
                                                      │ (Confirmed & Cooldown > 2s)
                                                      ▼
                                     [UART2 Serial TX (GBK, 9600 baud) ──► Voice Module]
