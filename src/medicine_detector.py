Project: Edge-AI Assistive Medicine Recognition System
Author: Jingkun Li (Jarred)
Platform: MaixPy Edge-AI Vision Processor (PP-OCR + UART2)
Description: Real-time embedded OCR medicine box detector for the visually impaired,
             featuring a 4-stage glyph/phonetic error-correction pipeline,
             multi-frame temporal consensus voting, and UART serial output.
"""

from maix import camera, display, image, nn, app, uart, time, pinmap
import re
import difflib

# ================== 1. Initialization ==================
model = "/root/models/pp_ocr.mud"
ocr = nn.PP_OCR(model)
cam = camera.Camera(ocr.input_width(), ocr.input_height(), ocr.input_format())
disp = display.Display()

# UART Serial Configuration (9600 baud)
my_uart = None
try:
    pinmap.set_pin_function("A29", "UART2_RX")
    pinmap.set_pin_function("A28", "UART2_TX")
    my_uart = uart.UART("/dev/ttyS2", 9600)
except Exception as e:
    print("UART Error:", e)

# Font Configuration
image.load_font("ppocr", "/maixapp/share/font/ppocr_keys_v1.ttf", size=28)
image.set_default_font("ppocr")

MEDICINE_KEYWORDS = ["药", "片", "胶囊", "颗粒", "清瘟", "布洛", "阿司"]

# ================== 2. Standard Medicine Lexicon ==================
STANDARD_MEDICINES = [
    "感冒胶囊",
    "连花清瘟胶囊",
    "布洛芬缓释胶囊",
    "阿司匹林",
    "板蓝根颗粒"
]

# ================== 3. Error Correction Dictionaries ==================
CHAR_CORRECTION = {
    "襄": "囊", "温": "瘟", "淋": "林",
    "分": "芬", "兰": "蓝", "顺": "颗"
}

CORRECTION_MAP = {
    "连花清温": "连花清瘟",
    "阿司匹淋": "阿司匹林",
    "布洛分": "布洛芬",
    "板兰根": "板蓝根",
}

# ================== 4. Utility & Phonetic Functions ==================
def clean_text(text):
    return re.sub(r"[^\u4e00-\u9fff]", "", text)

def is_medicine(text):
    return any(k in text for k in MEDICINE_KEYWORDS)

def char_level_correct(text):
    return "".join(CHAR_CORRECTION.get(c, c) for c in text)

# Lightweight phonetic mapping for fuzzy matching
def rough_pinyin(word):
    table = {
        "清": "qing", "温": "wen", "瘟": "wen",
        "芬": "fen", "分": "fen",
        "林": "lin", "淋": "lin",
        "囊": "nang", "胶": "jiao",
        "颗": "ke", "粒": "li"
    }
    return "".join(table.get(c, c) for c in word)

# ================== 5. 4-Stage Hierarchical Correction & Confidence ==================
def smart_correct_with_confidence(text):
    """
    return: (corrected_text, confidence[0~1])
    """
    if not text or not is_medicine(text):
        return text, 0.0

    # Stage 1: Exact lexicon hit
    if text in STANDARD_MEDICINES:
        return text, 0.95

    # Stage 2: Character-level homoglyph correction
    char_fixed = char_level_correct(text)
    if char_fixed in STANDARD_MEDICINES:
        return char_fixed, 0.88

    # Stage 3: Phrase-level mapping table correction
    if char_fixed in CORRECTION_MAP:
        fixed = CORRECTION_MAP[char_fixed]
        return fixed, 0.85

    # Stage 4: Phonetic Pinyin + difflib sequence similarity
    py = rough_pinyin(char_fixed)
    best = (0, char_fixed)

    for med in STANDARD_MEDICINES:
        score = difflib.SequenceMatcher(
            None, py, rough_pinyin(med)
        ).ratio()
        if score > best[0]:
            best = (score, med)

    if best[0] >= 0.65:
        # Map similarity score to confidence interval [0.65, 0.80]
        confidence = 0.65 + best[0] * 0.15
        return best[1], min(confidence, 0.80)

    return char_fixed, 0.0

# ================== 6. Temporal Multi-Frame Consensus Voting ==================
VOTE_THRESHOLD = 3
vote_buffer = {}

def vote_confirm(text):
    vote_buffer[text] = vote_buffer.get(text, 0) + 1
    if vote_buffer[text] >= VOTE_THRESHOLD:
        vote_buffer.clear()
        return True
    return False

# ================== 7. Center ROI Configuration ==================
roi_w, roi_h = int(cam.width() * 0.85), int(cam.height() * 0.85)
roi_x, roi_y = (cam.width() - roi_w) // 2, (cam.height() - roi_h) // 2
ocr_roi = (roi_x, roi_y, roi_w, roi_h)

# ================== 8. Main Real-Time Inference Loop ==================
last_send_time = 0
SEND_INTERVAL = 2

while not app.need_exit():
    img = cam.read()
    img_roi = img.crop(*ocr_roi)
    objs = ocr.detect(img_roi)

    for obj in objs:
        raw = obj.char_str()
        text = clean_text(raw)

        text, conf = smart_correct_with_confidence(text)

        # Discard low-confidence detections
        if conf < 0.65:
            continue

        # Multi-frame consensus confirmation & UART transmission
        if vote_confirm(text) and my_uart:
            now = time.time()
            if now - last_send_time > SEND_INTERVAL:
                my_uart.write(
                    f"{text},{int(conf*100)}%\n".encode("gbk", errors="ignore")
                )
                last_send_time = now

        # On-screen HUD rendering
        box = obj.box
        x = box.x4 + roi_x
        y = box.y4 + roi_y
        img.draw_string(
            x, y,
            f"{text} {int(conf*100)}%",
            image.COLOR_RED
        )

    img.draw_rect(*ocr_roi, image.COLOR_GREEN, 2)
    disp.show(img)
