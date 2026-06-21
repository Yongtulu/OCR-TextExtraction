#!/usr/bin/env python3
"""
OCR Text Capture & Translation
Preprocessing: OpenCV pipeline (grayscale → denoise → binarize → perspective)
OCR:           EasyOCR (offline deep learning model)
Translation:   argostranslate (offline local model)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import functools
import numpy as np
import cv2


# ── OpenCV Preprocessing Pipeline ────────────────────────────────────────────

def preprocess(image_path: str, params: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    steps = {}

    img = cv2.imread(image_path)
    steps["① Original"] = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["② Grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h = params["denoise_h"]
    denoised = cv2.fastNlMeansDenoising(gray, h=h, templateWindowSize=7, searchWindowSize=21)
    steps["③ Denoised"] = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    block = max(3, params["thresh_block"] | 1)
    C     = params["thresh_c"]
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, C
    )
    steps["④ Binarized"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    corrected = _perspective_correct(img, params["canny_lo"], params["canny_hi"])
    steps["⑤ Perspective"] = corrected

    return corrected, steps


def _perspective_correct(img: np.ndarray, canny_lo: int, canny_hi: int) -> np.ndarray:
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, canny_lo, canny_hi)
    edges   = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_pts = None
    for cnt in contours[:5]:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            doc_pts = approx.reshape(4, 2).astype(np.float32)
            break

    if doc_pts is None:
        return img

    pts = _order_points(doc_pts)
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (w, h))


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect    = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ── EasyOCR ───────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def get_ocr_engine():
    import easyocr
    return easyocr.Reader(["ch_sim", "en"])


def run_ocr(bgr: np.ndarray, conf_thresh: float = 0.5, iou_thresh: float = 0.5):
    engine = get_ocr_engine()
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw    = engine.readtext(rgb, text_threshold=conf_thresh)

    boxes   = [box for box, text, conf in raw]
    results = [(text, float(conf)) for box, text, conf in raw]

    if len(boxes) > 1:
        boxes, results = _nms(boxes, results, iou_thresh)

    return results, boxes


def _box_to_xyxy(box):
    pts = np.array(box)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = _box_to_xyxy(a)
    bx1, by1, bx2, by2 = _box_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _nms(boxes, results, iou_thresh):
    order   = sorted(range(len(results)), key=lambda i: results[i][1], reverse=True)
    keep, dropped = [], set()
    for i in order:
        if i in dropped:
            continue
        keep.append(i)
        for j in order:
            if j != i and j not in dropped:
                if _iou(boxes[i], boxes[j]) > iou_thresh:
                    dropped.add(j)
    return [boxes[k] for k in keep], [results[k] for k in keep]


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_boxes(bgr: np.ndarray, boxes: list, results: list) -> np.ndarray:
    out = bgr.copy()
    for box, (text, conf) in zip(boxes, results):
        pts     = np.array(box, dtype=np.int32)
        color   = _conf_color(conf)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        x, y  = pts[0]
        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def _conf_color(conf: float):
    if conf >= 0.9: return (60, 200, 50)
    if conf >= 0.7: return (0, 180, 230)
    return (60, 60, 220)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ── Translation ───────────────────────────────────────────────────────────────

LANG_MAP = {
    "zh-cn": "zh", "zh-tw": "zh", "zh": "zh",
    "en": "en", "ja": "ja", "ko": "ko",
    "fr": "fr", "de": "de", "es": "es",
    "it": "it", "pt": "pt", "ru": "ru", "ar": "ar",
}


def translate_to_chinese(text: str) -> str:
    from argostranslate import translate as argo
    from langdetect import detect
    try:
        src_code = LANG_MAP.get(detect(text), "en")
    except Exception:
        src_code = "en"
    if src_code == "zh":
        return text
    installed = {lang.code: lang for lang in argo.get_installed_languages()}
    zh = installed.get("zh")
    if not zh:
        raise RuntimeError("Chinese language pack not installed. Run setup_models.py first.")
    src = installed.get(src_code)
    if src:
        t = src.get_translation(zh)
        if t:
            return t.translate(text)
    en = installed.get("en")
    if src and en:
        t1, t2 = src.get_translation(en), en.get_translation(zh)
        if t1 and t2:
            return t2.translate(t1.translate(text))
    return f"[Unsupported language: {src_code}]\n{text}"


# ── GUI Constants ─────────────────────────────────────────────────────────────

STEP_NAMES = ["① Original", "② Grayscale", "③ Denoised",
              "④ Binarized", "⑤ Perspective", "⑥ Detections"]

# (label, key, min, max, default, affected steps)
PARAM_DEFS = [
    ("Denoise h",       "denoise_h",    1,  30,  10, "→ ③④⑤"),
    ("Binarize block",  "thresh_block", 3,  99,  31, "→ ④"),
    ("Binarize C",      "thresh_c",     1,  30,  10, "→ ④"),
    ("Canny low",       "canny_lo",    10, 200,  50, "→ ⑤"),
    ("Canny high",      "canny_hi",    50, 300, 150, "→ ⑤"),
]

PARAM_AFFECTS = {
    "denoise_h":    ["③ Denoised", "④ Binarized", "⑤ Perspective"],
    "thresh_block": ["④ Binarized"],
    "thresh_c":     ["④ Binarized"],
    "canny_lo":     ["⑤ Perspective"],
    "canny_hi":     ["⑤ Perspective"],
}


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpenCV Preprocessing + EasyOCR")
        self.geometry("1340x820")
        self.minsize(960, 620)

        self._image_path: str | None = None
        self._steps: dict[str, Image.Image] = {}
        self._debounce_id = None

        self._build_ui()

    def _build_ui(self):
        # Top bar
        bar = ttk.Frame(self, padding=(10, 8, 10, 4))
        bar.pack(fill=tk.X)

        ttk.Label(bar, text="Image path:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.path_var, width=54)
        entry.pack(side=tk.LEFT, padx=(4, 6))
        entry.bind("<Return>", lambda _: self._load_from_entry())

        ttk.Button(bar, text="Browse…", command=self._browse).pack(side=tk.LEFT)
        self.run_btn = ttk.Button(bar, text="  Run OCR & Translate  ", command=self._start)
        self.run_btn.pack(side=tk.LEFT, padx=(12, 0))

        # conf / IOU sliders
        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=(18, 12), pady=2)

        self._conf_var = tk.DoubleVar(value=0.5)
        self._iou_var  = tk.DoubleVar(value=0.5)

        for label, var in [("conf", self._conf_var), ("IOU", self._iou_var)]:
            ttk.Label(bar, text=f"{label}:").pack(side=tk.LEFT)
            ttk.Scale(bar, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                      variable=var, length=110).pack(side=tk.LEFT, padx=(2, 2))
            val_lbl = ttk.Label(bar, width=4)
            val_lbl.pack(side=tk.LEFT, padx=(0, 2))
            var.trace_add("write", lambda *_, v=var, l=val_lbl:
                          l.config(text=f"{v.get():.2f}"))
            tk.Label(bar, text="→ ⑥ Detections", fg="#4caf50",
                     font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 12))

        # Progress bar
        self._progress = ttk.Progressbar(self, mode="indeterminate")

        # Main pane
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        # Left: step tabs + preprocessing sliders
        left = ttk.Frame(pane)
        pane.add(left, weight=5)

        self._notebook = ttk.Notebook(left)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        self._canvases: dict[str, tk.Canvas] = {}
        for name in STEP_NAMES:
            frm = ttk.Frame(self._notebook)
            self._notebook.add(frm, text=name)
            c = tk.Canvas(frm, bg="#2b2b2b", highlightthickness=0)
            c.pack(fill=tk.BOTH, expand=True)
            c.bind("<Configure>", lambda _, n=name: self._redraw_step(n))
            self._canvases[name] = c

        ctrl = ttk.LabelFrame(left, text="Preprocessing Parameters", padding=(10, 6))
        ctrl.pack(fill=tk.X, pady=(6, 0))
        self._build_sliders(ctrl)

        # Right: text results
        right = ttk.Frame(pane, padding=(8, 0, 0, 0))
        pane.add(right, weight=4)

        ttk.Label(right, text="Recognized text (original · confidence):").pack(anchor=tk.W)
        self.orig_box = self._make_textbox(right)
        self.orig_box.pack(fill=tk.BOTH, expand=True)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(right, text="Chinese translation:").pack(anchor=tk.W)
        self.trans_box = self._make_textbox(right)
        self.trans_box.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — enter an image path or click Browse")
        ttk.Label(self, textvariable=self.status_var, anchor=tk.W,
                  foreground="#888").pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 4))

    def _build_sliders(self, parent):
        self._param_vars: dict[str, tk.IntVar] = {}

        for i, (label, key, lo, hi, default, affects) in enumerate(PARAM_DEFS):
            col = (i % 2) * 4
            row = i // 2

            var = tk.IntVar(value=default)
            self._param_vars[key] = var

            ttk.Label(parent, text=label, width=16, anchor=tk.E).grid(
                row=row, column=col, padx=(0, 4), pady=3, sticky=tk.E)
            ttk.Scale(parent, from_=lo, to=hi, orient=tk.HORIZONTAL,
                      variable=var, length=160,
                      command=lambda _, k=key: self._on_param_change(k)).grid(
                row=row, column=col+1, padx=(0, 4), pady=3)
            ttk.Label(parent, textvariable=var, width=4).grid(
                row=row, column=col+2, padx=(0, 6), pady=3, sticky=tk.W)
            tk.Label(parent, text=affects, fg="#4caf50",
                     font=("Helvetica", 10, "bold")).grid(
                row=row, column=col+3, padx=(0, 20), pady=3, sticky=tk.W)

        ttk.Button(parent, text="Reset defaults", command=self._reset_params).grid(
            row=3, column=0, columnspan=8, pady=(6, 2))

    def _reset_params(self):
        for _, key, _, _, default, _ in PARAM_DEFS:
            self._param_vars[key].set(default)
        if self._image_path:
            self._run_preprocess()

    # ── Param change → debounced reprocess ───

    def _on_param_change(self, key: str):
        if self._image_path is None:
            return
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(
            300, lambda: self._run_preprocess_partial(key))

    def _run_preprocess_partial(self, changed_key: str):
        affected = PARAM_AFFECTS.get(changed_key, [])
        if not affected:
            return
        threading.Thread(target=self._preprocess_thread,
                         args=(self._image_path, affected), daemon=True).start()

    def _run_preprocess(self):
        threading.Thread(target=self._preprocess_thread,
                         args=(self._image_path, None), daemon=True).start()

    def _preprocess_thread(self, path: str, only_steps):
        try:
            params  = {k: v.get() for k, v in self._param_vars.items()}
            _, steps = preprocess(path, params)
            for name, bgr in steps.items():
                if only_steps is None or name in only_steps:
                    pil = bgr_to_pil(bgr)
                    self._steps[name] = pil
                    self._ui(lambda n=name: self._redraw_step(n))
        except Exception as e:
            self._ui(lambda: self.status_var.set(f"Preprocessing error: {e}"))

    # ── Image loading ───

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                       ("All files", "*.*")]
        )
        if path:
            self.path_var.set(path)
            self._load_image(path)

    def _load_from_entry(self):
        path = self.path_var.get().strip()
        if os.path.isfile(path):
            self._load_image(path)
        else:
            self.status_var.set("File not found — check the path")

    def _load_image(self, path: str):
        self._image_path = path
        self.status_var.set(f"Loaded: {os.path.basename(path)}")
        self._run_preprocess()
        self._notebook.select(0)

    def _redraw_step(self, name: str):
        canvas = self._canvases.get(name)
        pil    = self._steps.get(name)
        if canvas is None or pil is None:
            return
        w = canvas.winfo_width()  or 500
        h = canvas.winfo_height() or 500
        img   = pil.copy()
        img.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        canvas._photo = photo
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, anchor=tk.CENTER, image=photo)

    # ── OCR worker ───

    def _start(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("Warning", "Please select or enter an image path first.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("Error", f"File not found:\n{path}")
            return

        self.run_btn.config(state=tk.DISABLED)
        for box in (self.orig_box, self.trans_box):
            box.delete("1.0", tk.END)
        self._progress.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=2)
        self._progress.start(12)
        threading.Thread(target=self._worker, args=(path,), daemon=True).start()

    def _worker(self, path: str):
        try:
            self._ui(lambda: self.status_var.set("OpenCV preprocessing…"))
            params = {k: v.get() for k, v in self._param_vars.items()}
            final_bgr, steps = preprocess(path, params)

            for name, bgr in steps.items():
                pil = bgr_to_pil(bgr)
                self._steps[name] = pil
                self._ui(lambda n=name: self._redraw_step(n))

            self._ui(lambda: self.status_var.set("EasyOCR inference…"))
            conf = self._conf_var.get()
            iou  = self._iou_var.get()
            results, boxes = run_ocr(final_bgr, conf_thresh=conf, iou_thresh=iou)

            if not results:
                self._ui(lambda: self.status_var.set("No text detected."))
                return

            annotated = draw_boxes(final_bgr, boxes, results)
            self._steps["⑥ Detections"] = bgr_to_pil(annotated)
            self._ui(lambda: self._redraw_step("⑥ Detections"))
            self._ui(lambda: self._notebook.select(STEP_NAMES.index("⑥ Detections")))

            lines = "\n".join(f"{t}  [{c:.2%}]" for t, c in results)
            plain = "\n".join(t for t, _ in results)
            self._ui(lambda: self.orig_box.insert(tk.END, lines))

            self._ui(lambda: self.status_var.set("Translating…"))
            zh = translate_to_chinese(plain)
            self._ui(lambda: self.trans_box.insert(tk.END, zh))
            self._ui(lambda: self.status_var.set(
                f"Done — {len(results)} region(s) detected"))

        except Exception as e:
            msg = str(e)
            self._ui(lambda: self.status_var.set(f"Error: {msg}"))
            self._ui(lambda: messagebox.showerror("Error", msg))
        finally:
            self._ui(self._finish)

    def _finish(self):
        self.run_btn.config(state=tk.NORMAL)
        self._progress.stop()
        self._progress.pack_forget()

    @staticmethod
    def _make_textbox(parent) -> tk.Text:
        f   = ttk.Frame(parent)
        f.pack(fill=tk.BOTH, expand=True)
        box = tk.Text(f, wrap=tk.WORD, font=("Helvetica", 13),
                      relief=tk.FLAT, padx=6, pady=4)
        sb  = ttk.Scrollbar(f, orient=tk.VERTICAL, command=box.yview)
        box.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return box

    def _ui(self, fn):
        self.after(0, fn)


if __name__ == "__main__":
    app = App()
    app.mainloop()
