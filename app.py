"""
Forensica - Neo-Brutalist Gradio single-file app.

Loud, colourful, high-contrast neo-brutalism UI:
  • Chunky black borders, hard drop-shadows, tile-based layout
  • Rotated stickers, ALL-CAPS mono labels, primary-colour blocks
  • Six-model dual-domain analysis (4 Synthetic + 2 Photographic)
  • Heatmaps (saliency + overlay) for Custom CNN and ResNet50V2
  • Instruments (per-model gauges), Ranking, and full Report dossier
  • Responsive layout for phone / tablet / desktop (CSS breakpoints)

Run:
    pip install -r requirements.txt gradio
    python app.py
"""

from __future__ import annotations
import os, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import gradio as gr

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization, Dense


# ────────────────────────────────────────────────────────────
# 1. Model loading (with compatibility shims for older .h5)
# ────────────────────────────────────────────────────────────
class CompatibleBatchNorm(BatchNormalization):
    def __init__(self, **kw):
        for k in ("renorm", "renorm_clipping", "renorm_momentum", "fused"):
            kw.pop(k, None)
        super().__init__(**kw)


class CompatibleDense(Dense):
    def __init__(self, **kw):
        kw.pop("quantization_config", None)
        super().__init__(**kw)


CO = {"BatchNormalization": CompatibleBatchNorm, "Dense": CompatibleDense}
MODELS_DIR = os.environ.get("MODELS_DIR", "TrainedModels")


def _load(path, co=None):
    if co:
        return load_model(path, custom_objects=co, compile=False)
    return load_model(path, compile=False)


def _load_all():
    print(f"Loading models from {MODELS_DIR} …")

    print("  [1/6] Custom CNN …")
    m1 = _load(f"{MODELS_DIR}/custom_cnn.h5", CO);           gc.collect()
    print("  [2/6] ResNet50 …")
    m2 = _load(f"{MODELS_DIR}/resnet50.h5",   CO);           gc.collect()
    print("  [3/6] VGG16 …")
    m3 = _load(f"{MODELS_DIR}/vgg16.h5",      CO);           gc.collect()
    print("  [4/6] MobileNetV2 Synthetic …")
    m4 = _load(f"{MODELS_DIR}/mobilenet.h5",  CO);           gc.collect()
    print("  [5/6] MobileNetV2 Photographic …")
    m5 = _load(f"{MODELS_DIR}/realworld_mobilenet.h5", CO);  gc.collect()
    print("  [6/6] ResNet50V2 Photographic …")
    m6 = _load(f"{MODELS_DIR}/realworld_resnet.keras");      gc.collect()

    print("✅ All 6 models loaded")
    return m1, m2, m3, m4, m5, m6


custom_cnn, resnet_model, vgg_model, mobilenet_model, rw_mobile_model, rw_resnet_model \
    = _load_all()


MODEL_META = [
    ("cnn",       "Custom CNN",   "Synthetic",    0.9562, "cifake"),
    ("resnet_c",  "ResNet50",     "Synthetic",    0.8167, "cifake"),
    ("vgg",       "VGG16",        "Synthetic",    0.8776, "cifake"),
    ("mobile_c",  "MobileNetV2",  "Synthetic",    0.8707, "cifake"),
    ("resnet_rw", "ResNet50V2",   "Photographic", 0.9617, "realworld"),
    ("mobile_rw", "MobileNetV2",  "Photographic", 0.8940, "realworld"),
]
MODELS = {
    "cnn":       custom_cnn,
    "resnet_c":  resnet_model,
    "vgg":       vgg_model,
    "mobile_c":  mobilenet_model,
    "resnet_rw": rw_resnet_model,
    "mobile_rw": rw_mobile_model,
}

# Neo-brutalist tile colours cycled through the gauge grid
TILE_COLORS = ["#FFDE59", "#FF5C8A", "#4CC9F0", "#B8FF5C", "#FF7A29", "#C77DFF"]


# ────────────────────────────────────────────────────────────
# 2. Preprocessing + saliency helpers
# ────────────────────────────────────────────────────────────
def preprocess_cifake(img: Image.Image):
    img = img.convert("RGB")
    small = img.resize((32, 32), Image.LANCZOS).resize((224, 224), Image.LANCZOS)
    arr = np.array(small, dtype=np.float32) / 255.0
    return arr, np.expand_dims(arr, 0)


def preprocess_realworld(img: Image.Image):
    img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, np.expand_dims(arr, 0)


def saliency(model, batch):
    t = tf.Variable(tf.cast(batch, tf.float32), trainable=True)
    with tf.GradientTape() as tape:
        tape.watch(t)
        loss = model(t, training=False)[:, 0]
    g = tape.gradient(loss, t).numpy()[0]
    h = np.max(np.abs(g), axis=-1)
    if h.max() > 0:
        h = h / h.max()
    return h


def heatmap_rgb(hm):
    hm = cv2.resize(hm, (224, 224))
    hm_u8 = np.uint8(255 * np.clip(hm, 0, 1))
    color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def overlay(arr, hm):
    hm = cv2.resize(hm, (224, 224))
    color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    orig = np.uint8(255 * arr)
    return cv2.addWeighted(orig, 0.55, color, 0.45, 0)


# ────────────────────────────────────────────────────────────
# 3. Core inference
# ────────────────────────────────────────────────────────────
IDLE_VERDICT = """
<div class="nb-card nb-card--verdict nb-tilt-l">
  <span class="nb-tag nb-tag--black">PLATE II · VERDICT</span>
  <h2 class="nb-verdict-headline">Awaiting <em>specimen</em>.</h2>
  <p class="nb-verdict-sub">Drop an image on the left, then hit <b>BEGIN ANALYSIS</b>.</p>
</div>
"""


def run_analysis(image: Image.Image):
    if image is None:
        return (None, None, None, None, None,
                IDLE_VERDICT, "", "", "", "")

    arr_c, batch_c = preprocess_cifake(image)
    arr_r, batch_r = preprocess_realworld(image)

    preds = {}
    for mid, _, _, _, pre in MODEL_META:
        batch = batch_c if pre == "cifake" else batch_r
        preds[mid] = float(MODELS[mid].predict(batch, verbose=0)[0][0])

    ranked = []
    for mid, name, dom, acc, _ in MODEL_META:
        p = preds[mid]
        is_real = p > 0.5
        conf = (p if is_real else 1 - p) * 100
        strength = abs(p - 0.5) * 2
        rank_score = acc * strength
        ranked.append({
            "id": mid, "name": name, "domain": dom, "acc": acc,
            "pred": p, "is_real": is_real, "conf": conf,
            "strength": strength, "score": rank_score,
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[0]

    hm_cnn = saliency(custom_cnn, batch_c)
    hm_rn  = saliency(rw_resnet_model, batch_r)
    original_img   = np.uint8(255 * arr_c)
    cnn_activation = heatmap_rgb(hm_cnn)
    rn_activation  = heatmap_rgb(hm_rn)
    cnn_overlay_   = overlay(arr_c, hm_cnn)
    rn_overlay_    = overlay(arr_r, hm_rn)

    # ── Verdict ──────────────────────────────────────────────
    verdict_label = "REAL" if top["is_real"] else "AI-GENERATED"
    verdict_bg = "#B8FF5C" if top["is_real"] else "#FF5C8A"
    verdict_md = f"""
<div class="nb-card nb-card--verdict nb-tilt-l" style="background:{verdict_bg};">
  <span class="nb-tag nb-tag--black">PLATE II · VERDICT</span>
  <h2 class="nb-verdict-headline">
    Likely <em>{verdict_label.lower()}</em>.
  </h2>
  <div class="nb-verdict-meta">
    <span class="nb-chip nb-chip--white">{top['name']}</span>
    <span class="nb-chip nb-chip--yellow">{top['domain'].upper()}</span>
    <span class="nb-chip nb-chip--cyan">{top['conf']:.1f}% CONF</span>
    <span class="nb-chip nb-chip--white">ACC {top['acc']*100:.2f}%</span>
  </div>
</div>
"""

    # ── Gauges ───────────────────────────────────────────────
    gauges = ['<div class="nb-grid nb-grid--gauges">']
    for i, m in enumerate(ranked):
        tile = TILE_COLORS[i % len(TILE_COLORS)]
        verdict_color = "#B8FF5C" if m["is_real"] else "#FF5C8A"
        vtxt = "REAL" if m["is_real"] else "FAKE"
        gauges.append(f"""
        <div class="nb-gauge" style="background:{tile};">
          <div class="nb-gauge-head">
            <span class="nb-gauge-name">{m['name']}</span>
            <span class="nb-gauge-domain">{m['domain'].upper()}</span>
          </div>
          <div class="nb-verdict-pill" style="background:{verdict_color};">{vtxt}</div>
          <div class="nb-bar"><span style="width:{m['conf']:.1f}%; background:#000;"></span></div>
          <div class="nb-gauge-meta">
            <span>CONF {m['conf']:.1f}%</span>
            <span>ACC {m['acc']*100:.1f}%</span>
          </div>
        </div>""")
    gauges.append("</div>")
    gauges_md = "\n".join(gauges)

    # ── Ranking ──────────────────────────────────────────────
    rows = ""
    for i, m in enumerate(ranked, 1):
        row_bg = "background:#FFDE59;" if i == 1 else ""
        vtxt = "REAL" if m["is_real"] else "FAKE"
        vcolor = "#B8FF5C" if m["is_real"] else "#FF5C8A"
        rows += f"""
        <tr style="{row_bg}">
          <td class="nb-rank">#{i:02d}</td>
          <td><b>{m['name']}</b><div class="nb-dim">{m['domain']}</div></td>
          <td><span class="nb-verdict-pill nb-verdict-pill--sm" style="background:{vcolor};">{vtxt}</span></td>
          <td class="nb-mono">{m['conf']:.1f}%</td>
          <td class="nb-mono">{m['acc']*100:.2f}%</td>
          <td class="nb-mono"><b>{m['score']*100:.2f}</b></td>
        </tr>"""
    ranking_md = f"""
<div class="nb-card nb-card--rank nb-tilt-r">
  <span class="nb-tag nb-tag--pink">PLATE V · RANKING · ACC × CONFIDENCE STRENGTH</span>
  <div class="nb-table-wrap">
    <table class="nb-table">
      <thead>
        <tr><th>#</th><th>MODEL</th><th>VERDICT</th><th>CONFIDENCE</th><th>ACCURACY</th><th>SCORE</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>
"""

    # ── Signals ──────────────────────────────────────────────
    real_votes = sum(1 for m in ranked if m["is_real"])
    fake_votes = len(ranked) - real_votes
    avg_conf   = sum(m["conf"] for m in ranked) / len(ranked)
    signals_md = f"""
<div class="nb-card nb-card--signals">
  <span class="nb-tag nb-tag--cyan">PLATE VI · SIGNALS</span>
  <div class="nb-stat-grid">
    <div class="nb-stat" style="background:#B8FF5C;"><div class="nb-stat-num">{real_votes}/6</div><div class="nb-stat-lbl">REAL VOTES</div></div>
    <div class="nb-stat" style="background:#FF5C8A;"><div class="nb-stat-num">{fake_votes}/6</div><div class="nb-stat-lbl">FAKE VOTES</div></div>
    <div class="nb-stat" style="background:#4CC9F0;"><div class="nb-stat-num">{avg_conf:.1f}%</div><div class="nb-stat-lbl">MEAN CONF</div></div>
    <div class="nb-stat" style="background:#FFDE59;"><div class="nb-stat-num">{top['conf']:.0f}%</div><div class="nb-stat-lbl">TOP SIGNAL</div></div>
  </div>
  <ul class="nb-signals">
    <li>▸ <b>{real_votes} of 6</b> models judge REAL; <b>{fake_votes}</b> judge AI-generated.</li>
    <li>▸ Mean confidence across the panel: <b>{avg_conf:.1f}%</b>.</li>
    <li>▸ Strongest signal - <b>{top['name']}</b> ({top['domain']}) at <b>{top['conf']:.1f}%</b>.</li>
    <li>▸ Weakest signal - <b>{ranked[-1]['name']}</b> at <b>{ranked[-1]['conf']:.1f}%</b>.</li>
  </ul>
</div>
"""

    # ── Raw ──────────────────────────────────────────────────
    raw_rows = "".join(
        f"<tr><td><b>{m['name']}</b></td>"
        f"<td class='nb-dim'>{m['domain']}</td>"
        f"<td class='nb-mono'>{m['pred']:.6f}</td>"
        f"<td class='nb-mono'>{m['pred']*100:.2f}% real / {(1-m['pred'])*100:.2f}% fake</td></tr>"
        for m in ranked
    )
    raw_md = f"""
<div class="nb-card nb-card--raw">
  <span class="nb-tag nb-tag--yellow">RAW SCORES · SIGMOID OUTPUT (&gt;0.5 ⇒ REAL)</span>
  <div class="nb-table-wrap">
    <table class="nb-table">
      <thead><tr><th>MODEL</th><th>DOMAIN</th><th>P(REAL)</th><th>SPLIT</th></tr></thead>
      <tbody>{raw_rows}</tbody>
    </table>
  </div>
</div>
"""

    # ── Technical ────────────────────────────────────────────
    tech_md = """
<div class="nb-card nb-card--tech">
  <span class="nb-tag nb-tag--black">TECHNICAL &amp; DISCLAIMER</span>
  <ul class="nb-signals">
    <li>▸ <b>Synthetic domain</b> - CIFAKE-style preprocessing: 32×32 downscale then 224×224 upscale. Trained to catch low-resolution generative artefacts.</li>
    <li>▸ <b>Photographic domain</b> - direct 224×224 resize. Trained on real-world photos vs. modern high-resolution generative imagery (GRAVEX-200K).</li>
    <li>▸ <b>Heatmaps</b> - gradient saliency from two independent models: Custom CNN (Synthetic) and ResNet50V2 (Photographic). Bright/yellow zones = highest influence.</li>
    <li>▸ <b>Ranking</b> - score = accuracy × |pred − 0.5| × 2. Measures reliability; never issues a single verdict.</li>
    <li>▸ <b>Disclaimer</b> - evidence, not judgment. Research use only, not for legal use. Heavily filtered or re-photographed images may produce unreliable results.</li>
  </ul>
  <div class="nb-table-wrap">
    <table class="nb-table">
      <thead><tr><th>FIELD</th><th>VALUE</th></tr></thead>
      <tbody>
        <tr><td>Total models</td><td class="nb-mono">6 (4 Synthetic + 2 Photographic)</td></tr>
        <tr><td>Synthetic dataset</td><td class="nb-mono">CIFAKE - 120,000 images</td></tr>
        <tr><td>Photographic dataset</td><td class="nb-mono">GRAVEX-200K - 200,000 images</td></tr>
        <tr><td>Total training images</td><td class="nb-mono">320,000</td></tr>
        <tr><td>Best accuracy</td><td class="nb-mono">ResNet50V2 - 96.17%</td></tr>
        <tr><td>Best AUC-ROC</td><td class="nb-mono">ResNet50V2 - 99.48%</td></tr>
        <tr><td>Synthetic preprocessing</td><td class="nb-mono">32×32 → 224×224 upscale</td></tr>
        <tr><td>Photographic preprocessing</td><td class="nb-mono">Direct 224×224 resize</td></tr>
        <tr><td>Ranking method</td><td class="nb-mono">Accuracy × Confidence Strength</td></tr>
        <tr><td>Heatmap sources</td><td class="nb-mono">Custom CNN · ResNet50V2</td></tr>
        <tr><td>Input resolution</td><td class="nb-mono">224×224 px</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""

    return (
        Image.fromarray(original_img),
        Image.fromarray(cnn_activation),
        Image.fromarray(rn_activation),
        Image.fromarray(cnn_overlay_),
        Image.fromarray(rn_overlay_),
        verdict_md, gauges_md, ranking_md, signals_md + raw_md, tech_md,
    )


# ────────────────────────────────────────────────────────────
# 4. Neo-brutalist CSS (responsive)
# ────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
  --ink: #0a0a0a;
  --paper: #FFF8E7;
  --line: #0a0a0a;
  --yellow: #FFDE59;
  --pink:   #FF5C8A;
  --cyan:   #4CC9F0;
  --lime:   #B8FF5C;
  --orange: #FF7A29;
  --violet: #C77DFF;
  --dark:   #2A2A3C;   /* deep slate for white-text plates */
  --shadow: 6px 6px 0 0 var(--ink);
  --shadow-lg: 10px 10px 0 0 var(--ink);
  --display: 'Archivo Black', 'Space Grotesk', sans-serif;
  --body:    'Space Grotesk', system-ui, sans-serif;
  --mono:    'JetBrains Mono', ui-monospace, monospace;
}

body, .gradio-container {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--body) !important;
  background-image:
    linear-gradient(var(--ink) 1px, transparent 1px),
    linear-gradient(90deg, var(--ink) 1px, transparent 1px);
  background-size: 40px 40px;
  background-position: -1px -1px;
  background-blend-mode: normal;
}
.gradio-container {
  max-width: 1400px !important; margin: 0 auto !important;
  padding: 24px 16px !important;
}

/* ─── Masthead ─── */
.nb-masthead {
  background: var(--ink); color: var(--paper);
  border: 3px solid var(--ink); box-shadow: var(--shadow);
  padding: 12px 18px; display: flex; align-items: center;
  justify-content: space-between; flex-wrap: wrap; gap: 8px;
  font-family: var(--mono); font-size: 11px; letter-spacing: .18em; text-transform: uppercase;
}
.nb-masthead .brand { color: var(--yellow); font-family: var(--display); letter-spacing: .05em; font-size: 16px; }
.nb-masthead .dot { display:inline-block; width:8px; height:8px; background:var(--lime); border-radius:50%; margin-right:6px; }

/* ─── Hero ─── */
.nb-hero {
  margin: 28px 0 32px; padding: 32px 24px;
  background: var(--yellow);
  border: 4px solid var(--ink); box-shadow: var(--shadow-lg);
  position: relative;
}
.nb-hero .nb-sticker {
  position: absolute; top: -18px; right: 24px;
  background: var(--pink); color: var(--ink);
  border: 3px solid var(--ink); padding: 6px 12px;
  font-family: var(--mono); font-size: 11px; font-weight: 700;
  letter-spacing: .18em; text-transform: uppercase;
  transform: rotate(4deg); box-shadow: 4px 4px 0 var(--ink);
}
.nb-hero .nb-eyebrow {
  font-family: var(--mono); font-size: 11px; letter-spacing: .3em;
  text-transform: uppercase; margin-bottom: 14px; font-weight: 700;
}
.nb-hero h1 {
  font-family: var(--display); font-weight: 900;
  font-size: clamp(44px, 9vw, 110px); line-height: 0.92;
  letter-spacing: -0.02em; margin: 0; color: var(--ink);
  text-transform: uppercase;
}
.nb-hero h1 em {
  font-style: normal; background: var(--ink); color: var(--yellow);
  padding: 0 12px; display: inline-block; transform: rotate(-1deg);
}
.nb-hero p { max-width: 40rem; margin-top: 20px; line-height: 1.55; font-size: 16px; font-weight: 500; }

.nb-hero-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 22px; }
.nb-chip {
  border: 2px solid var(--ink); padding: 4px 10px;
  font-family: var(--mono); font-size: 11px; letter-spacing: .12em;
  text-transform: uppercase; font-weight: 700; box-shadow: 3px 3px 0 var(--ink);
}
.nb-chip--white  { background: var(--dark); color: var(--paper); }
.nb-chip--yellow { background: var(--yellow); }
.nb-chip--cyan   { background: var(--cyan); }
.nb-chip--pink   { background: var(--pink); }
.nb-chip--lime   { background: var(--lime); }

/* ─── Section tags ─── */
.nb-tag {
  display: inline-block; padding: 6px 12px; margin-bottom: 14px;
  border: 3px solid var(--ink); box-shadow: 4px 4px 0 var(--ink);
  font-family: var(--mono); font-size: 11px; font-weight: 700;
  letter-spacing: .18em; text-transform: uppercase; color: var(--ink);
}
.nb-tag--black  { background: var(--ink); color: var(--yellow); }
.nb-tag--yellow { background: var(--yellow); }
.nb-tag--pink   { background: var(--pink); color: #fff; }
.nb-tag--cyan   { background: var(--cyan); }
.nb-tag--lime   { background: var(--lime); }

/* ─── Cards ─── */
.nb-card {
  background: #fff; border: 4px solid var(--ink); box-shadow: var(--shadow-lg);
  padding: 22px; margin: 18px 0;
}
.nb-card--verdict { background: var(--lime); }
.nb-card--rank    { background: #fff; }
.nb-card--signals { background: var(--cyan); }
.nb-card--raw     { background: #fff; }
.nb-card--tech    { background: var(--yellow); }
.nb-tilt-l { transform: rotate(-0.5deg); }
.nb-tilt-r { transform: rotate(0.4deg); }

.nb-verdict-headline {
  font-family: var(--display); font-size: clamp(36px, 7vw, 78px);
  line-height: 1.02; margin: 14px 0 18px; color: var(--ink);
  text-transform: uppercase; letter-spacing: -0.01em;
}
.nb-verdict-headline em {
  font-style: normal; background: var(--ink); color: var(--yellow);
  padding: 0 10px; display: inline-block; transform: rotate(-1deg);
}
.nb-verdict-sub { font-family: var(--mono); font-size: 13px; letter-spacing: .1em; text-transform: uppercase; }
.nb-verdict-meta { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }

/* ─── Gauges ─── */
.nb-grid { display: grid; gap: 16px; }
.nb-grid--gauges {
  grid-template-columns: repeat(1, 1fr);
}
@media (min-width: 560px)  { .nb-grid--gauges { grid-template-columns: repeat(2, 1fr); } }
@media (min-width: 900px)  { .nb-grid--gauges { grid-template-columns: repeat(3, 1fr); } }
@media (min-width: 1200px) { .nb-grid--gauges { grid-template-columns: repeat(3, 1fr); } }

.nb-gauge {
  border: 3px solid var(--ink); box-shadow: var(--shadow);
  padding: 14px; transition: transform .15s ease, box-shadow .15s ease;
}
.nb-gauge:hover { transform: translate(-2px, -2px); box-shadow: 8px 8px 0 var(--ink); }
.nb-gauge:nth-child(2n) { transform: rotate(-0.6deg); }
.nb-gauge:nth-child(3n) { transform: rotate(0.5deg); }
.nb-gauge:hover { transform: translate(-2px, -2px) rotate(0); }
.nb-gauge-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.nb-gauge-name { font-family: var(--display); font-size: 18px; text-transform: uppercase; }
.nb-gauge-domain, .nb-gauge-meta {
  font-family: var(--mono); font-size: 10px; letter-spacing: .18em;
  text-transform: uppercase; font-weight: 700;
}
.nb-verdict-pill {
  display: inline-block; margin: 10px 0 10px;
  padding: 6px 14px; border: 3px solid var(--ink);
  font-family: var(--mono); font-size: 12px; font-weight: 700;
  letter-spacing: .2em; box-shadow: 3px 3px 0 var(--ink);
}
.nb-verdict-pill--sm { padding: 2px 8px; font-size: 10px; box-shadow: 2px 2px 0 var(--ink); }
.nb-bar {
  height: 10px; background: #fff; border: 2px solid var(--ink);
  position: relative; overflow: hidden;
}
.nb-bar span { position: absolute; inset: 0; display: block; height: 100%; }
.nb-gauge-meta { display: flex; justify-content: space-between; margin-top: 10px; }

/* ─── Tables ─── */
.nb-table-wrap { overflow-x: auto; border: 3px solid var(--ink); background:#fff !important; }
.nb-table { width: 100%; border-collapse: collapse; font-family: var(--body); background:#fff !important; }
.nb-table th, .nb-table td {
  padding: 10px 12px; border-bottom: 2px solid var(--ink); text-align: left;
  font-size: 14px; color: var(--ink) !important; background: #fff !important;
}
.nb-table td * { color: var(--ink) !important; background: transparent !important; }
.nb-table th {
  background: var(--ink); color: var(--paper);
  font-family: var(--mono); font-size: 11px; letter-spacing: .18em;
  text-transform: uppercase; font-weight: 700;
}
.nb-table tr:last-child td { border-bottom: 0; }
.nb-rank { font-family: var(--display); font-size: 16px; }
.nb-mono { font-family: var(--mono); font-size: 12px; font-weight: 700; }
.nb-dim  { font-family: var(--mono); font-size: 10px; letter-spacing: .15em;
           text-transform: uppercase; color: #555; }

/* ─── Signals ─── */
.nb-stat-grid {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
  margin: 12px 0 16px;
}
@media (min-width: 720px) { .nb-stat-grid { grid-template-columns: repeat(4, 1fr); } }
.nb-stat {
  border: 3px solid var(--ink); box-shadow: 4px 4px 0 var(--ink);
  padding: 12px; text-align: center;
}
.nb-stat-num { font-family: var(--display); font-size: clamp(22px, 4vw, 32px); }
.nb-stat-lbl { font-family: var(--mono); font-size: 10px; letter-spacing: .18em;
               text-transform: uppercase; font-weight: 700; margin-top: 2px; }

.nb-signals { list-style: none; padding: 0; margin: 8px 0 0; }
.nb-signals li {
  padding: 10px 12px; margin: 8px 0;
  background: var(--ink) !important; color: var(--yellow) !important;
  border: 2px solid var(--ink); box-shadow: 3px 3px 0 var(--ink);
  font-size: 15px; font-weight: 600;
}
.nb-signals li * { color: var(--yellow) !important; background: transparent !important; }
.nb-signals li b, .nb-signals li strong { color: #fff !important; }

/* ─── Specimen panel ─── */
.nb-specimen-head {
  background: var(--cyan); border: 3px solid var(--ink); box-shadow: var(--shadow);
  padding: 14px 16px; margin-bottom: 14px;
}
.nb-specimen-head h3 { font-family: var(--display); font-size: 22px; margin: 4px 0 0; text-transform: uppercase; }
.nb-caption { font-family: var(--mono); font-size: 10px; letter-spacing: .2em;
              text-transform: uppercase; margin-top: 10px; color: #333; }

/* ─── Heatmap section header ─── */
.nb-plate-head {
  background: var(--violet); border: 3px solid var(--ink); box-shadow: var(--shadow);
  padding: 14px 16px; margin: 24px 0 14px; color: var(--ink);
}
.nb-plate-head h3 { font-family: var(--display); font-size: 22px; margin: 4px 0 0; text-transform: uppercase; }
.nb-plate-head--orange { background: var(--orange); }
.nb-plate-head--pink   { background: var(--pink); color: #fff; }
.nb-plate-head--lime   { background: var(--lime); }

/* ─── Footer ─── */
.nb-footer {
  margin-top: 40px; padding: 20px; background: var(--ink); color: var(--paper);
  border: 3px solid var(--ink); box-shadow: var(--shadow-lg);
  font-family: var(--mono); font-size: 11px; letter-spacing: .15em; text-transform: uppercase;
  display: grid; gap: 10px;
}
.nb-footer .brand { color: var(--yellow); font-family: var(--display); font-size: 16px; letter-spacing: .05em; }
.nb-footer a { color: var(--cyan); }

/* ─── Gradio component overrides ─── */
.gr-button {
  background: var(--ink) !important; color: var(--yellow) !important;
  border: 3px solid var(--ink) !important; border-radius: 0 !important;
  font-family: var(--mono) !important; font-size: 12px !important; font-weight: 700 !important;
  letter-spacing: .22em !important; text-transform: uppercase !important;
  padding: 14px 20px !important; box-shadow: 5px 5px 0 var(--ink) !important;
  transition: transform .12s ease, box-shadow .12s ease !important;
}
.gr-button:hover {
  transform: translate(-2px, -2px) !important;
  box-shadow: 8px 8px 0 var(--ink) !important;
  background: var(--pink) !important; color: #fff !important;
}
.gr-button.secondary, .gr-button[variant="secondary"] {
  background: #fff !important; color: var(--ink) !important;
}
.gr-button.secondary:hover { background: var(--cyan) !important; }

.gr-image, .gr-box, .gr-panel, .block, .gr-form {
  background: #fff !important;
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
  box-shadow: 5px 5px 0 var(--ink) !important;
}
.gr-image { padding: 6px !important; }

.tabitem, .tab-nav, .gr-tab-item { background: transparent !important; }
.tab-nav button, .gr-tab-item button, button[role="tab"] {
  background: #fff !important; color: var(--ink) !important;
  border: 3px solid var(--ink) !important; border-radius: 0 !important;
  font-family: var(--mono) !important; font-size: 11px !important; font-weight: 700 !important;
  letter-spacing: .18em !important; text-transform: uppercase !important;
  padding: 10px 16px !important; margin-right: 6px !important;
  box-shadow: 3px 3px 0 var(--ink) !important;
}
button[role="tab"][aria-selected="true"], button.selected {
  background: var(--pink) !important; color: #fff !important;
}
/* Force labels: black plate + yellow text so nothing camouflages on dark bars */
label, .label-wrap, .label-wrap span,
.gr-box > .label-wrap, .gr-form > .label-wrap,
span[data-testid="block-label"], .block > .label-wrap {
  background: var(--ink) !important;
  color: var(--yellow) !important;
  font-family: var(--mono) !important; font-size: 11px !important;
  letter-spacing: .18em !important; text-transform: uppercase !important; font-weight: 700 !important;
  padding: 6px 10px !important;
  border: 0 !important;
  border-radius: 0 !important;
}
label *, .label-wrap *, span[data-testid="block-label"] * {
  color: var(--yellow) !important;
  background: transparent !important;
  fill: var(--yellow) !important;
}
/* Button inner spans must inherit color, no inner dark chip */
.gr-button, .gr-button * {
  background-image: none !important;
  text-shadow: none !important;
}
.gr-button span, .gr-button div, .gr-button p {
  background: transparent !important;
  color: inherit !important;
}
.gr-button.secondary span, .gr-button[variant="secondary"] span,
.gr-button.secondary div, .gr-button[variant="secondary"] div {
  color: var(--ink) !important;
}

footer.svelte-1rjryqp, footer.svelte-gpiuxh, .footer,
.gradio-container > footer { display: none !important; }

.gradio-container p, .gradio-container li, .gradio-container td,
.gradio-container th, .gradio-container b, .gradio-container strong,
.gradio-container code { color: var(--ink); }

/* ─── Responsive niceties ─── */

/* Force Gradio rows to wrap on tablet + phone (default is nowrap flex) */
@media (max-width: 900px) {
  .gradio-container .gr-row,
  .gradio-container div[class*="row"] { flex-wrap: wrap !important; }
  .gradio-container .gr-column,
  .gradio-container div[class*="column"] { min-width: 100% !important; }
}

/* Tablet */
@media (max-width: 1024px) {
  .gradio-container { padding: 18px 14px !important; }
  :root { --shadow: 5px 5px 0 0 var(--ink); --shadow-lg: 7px 7px 0 0 var(--ink); }
  body, .gradio-container { background-size: 32px 32px; }
}

/* Phone */
@media (max-width: 720px) {
  .gradio-container { padding: 14px 10px !important; overflow-x: hidden !important; }
  body, .gradio-container { background-size: 24px 24px; }
  :root { --shadow: 4px 4px 0 0 var(--ink); --shadow-lg: 5px 5px 0 0 var(--ink); }

  .nb-masthead { font-size: 9px; padding: 10px 12px; gap: 6px; }
  .nb-masthead .brand { font-size: 13px; }

  .nb-hero { padding: 22px 16px; margin: 20px 0 22px; }
  .nb-hero .nb-sticker { right: 12px; top: -14px; font-size: 9px; padding: 5px 9px; }
  .nb-hero h1 { font-size: clamp(36px, 12vw, 56px); }
  .nb-hero p { font-size: 14px; margin-top: 14px; }
  .nb-hero-chips { gap: 6px; margin-top: 16px; }
  .nb-chip { font-size: 9px; padding: 3px 7px; box-shadow: 2px 2px 0 var(--ink); }

  .nb-card { padding: 14px; margin: 14px 0; border-width: 3px; }
  .nb-tilt-l, .nb-tilt-r { transform: none; }
  .nb-verdict-headline { font-size: clamp(28px, 8vw, 44px); }

  .nb-specimen-head h3, .nb-plate-head h3 { font-size: 18px; }
  .nb-specimen-head, .nb-plate-head { padding: 12px; }

  .nb-gauge { padding: 12px; }
  .nb-gauge:nth-child(2n), .nb-gauge:nth-child(3n) { transform: none; }
  .nb-gauge-name { font-size: 16px; }

  .nb-table th, .nb-table td { padding: 8px 8px; font-size: 12px; }
  .nb-stat { padding: 10px 8px; }

  .gr-button {
    letter-spacing: .12em !important; padding: 12px 14px !important;
    font-size: 11px !important; box-shadow: 4px 4px 0 var(--ink) !important;
    width: 100% !important;
  }
  .gr-image, .gr-box, .gr-panel, .block, .gr-form {
    box-shadow: 4px 4px 0 var(--ink) !important;
  }

  .nb-footer { padding: 14px; font-size: 9px; letter-spacing: .1em; }
  .nb-footer .brand { font-size: 13px; }

  .tab-nav button, button[role="tab"] {
    font-size: 10px !important; padding: 8px 10px !important; margin-right: 4px !important;
  }
}

/* Very small phones */
@media (max-width: 380px) {
  .nb-hero h1 { font-size: clamp(30px, 13vw, 44px); }
  .nb-hero .nb-sticker { display: none; }
}
"""


# ────────────────────────────────────────────────────────────
# 5. Gradio UI
# ────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Forensica - AI Image Forensics") as demo:

    gr.HTML("""
    <div class="nb-masthead">
      <div><span class="dot"></span>FORENSICA · SYSTEM ONLINE</div>
      <div class="brand">FORENSICA</div>
      <div>VOL. I / ISSUE 06 · GRADIO ED.</div>
    </div>

    <section class="nb-hero">
      <span class="nb-sticker">NEW! 6 MODELS</span>
      <div class="nb-eyebrow">№ 001 - FORENSIC ANALYSIS SYSTEM</div>
      <h1>The image,<br/><em>interrogated.</em></h1>
      <p>Forensica detects the <b>probability of an image being AI-generated or Real</b>.
         Six deep-learning models across two domains examine any image side by side,
         publishing not a verdict but the <b>evidence</b>, ranked by reliability.
         Interpretation is left to you, the reader.</p>
      <div class="nb-hero-chips">
        <span class="nb-chip nb-chip--yellow">4× SYNTHETIC</span>
        <span class="nb-chip nb-chip--cyan">2× PHOTOGRAPHIC</span>
        <span class="nb-chip nb-chip--lime">96.17% BEST ACC</span>
        <span class="nb-chip nb-chip--pink">320K TRAINING IMGS</span>
        <span class="nb-chip nb-chip--white">HEATMAPS · GRAD-SAL</span>
      </div>
    </section>
    """)

    with gr.Row(equal_height=False):
        # LEFT - Specimen
        with gr.Column(scale=4, min_width=280):
            gr.HTML("""
            <div class="nb-specimen-head">
              <span class="nb-tag nb-tag--black">PLATE I</span>
              <h3>Specimen</h3>
            </div>
            """)
            img_in = gr.Image(type="pil", label="INPUT IMAGE", height=380,
                              sources=["upload", "clipboard"])
            with gr.Row():
                run_btn   = gr.Button("▶ BEGIN ANALYSIS", elem_id="run-btn")
                clear_btn = gr.Button("✕ CLEAR", variant="secondary")
            gr.HTML('<p class="nb-caption">ACCEPTED · JPG · PNG · WEBP · BMP · TIFF</p>')

        # RIGHT - Report
        with gr.Column(scale=8, min_width=320):
            verdict = gr.HTML(IDLE_VERDICT)

            gr.HTML("""
            <div class="nb-plate-head nb-plate-head--orange">
              <span class="nb-tag nb-tag--black">PLATE III</span>
              <h3>Heatmaps</h3>
              <p class="nb-caption">SALIENCY & OVERLAY · CNN [SYNTHETIC] · RESNET50V2 [PHOTOGRAPHIC]</p>
            </div>
            """)
            with gr.Row():
                original_out = gr.Image(label="ORIGINAL", height=200, interactive=False)
                cnn_act_out  = gr.Image(label="CNN SALIENCY", height=200, interactive=False)
                rn_act_out   = gr.Image(label="RESNET50V2 SALIENCY", height=200, interactive=False)
            with gr.Row():
                cnn_over_out = gr.Image(label="CNN OVERLAY", height=200, interactive=False)
                rn_over_out  = gr.Image(label="RESNET50V2 OVERLAY", height=200, interactive=False)

            gr.HTML("""
            <div class="nb-plate-head nb-plate-head--lime">
              <span class="nb-tag nb-tag--black">PLATE IV</span>
              <h3>Instruments</h3>
              <p class="nb-caption">INDIVIDUAL MODEL GAUGES</p>
            </div>
            """)
            gauges_out = gr.HTML()

            ranking_out = gr.HTML()

            gr.HTML("""
            <div class="nb-plate-head nb-plate-head--pink">
              <span class="nb-tag nb-tag--yellow">PLATE VI</span>
              <h3>Report</h3>
              <p class="nb-caption" style="color:#fff;">FULL FORENSIC DOSSIER</p>
            </div>
            """)
            with gr.Tabs():
                with gr.TabItem("SIGNALS & RAW"):
                    signals_out = gr.HTML()
                with gr.TabItem("TECHNICAL & DISCLAIMER"):
                    tech_out = gr.HTML()

    gr.HTML("""
    <div class="nb-footer">
      <div><span class="brand">FORENSICA</span> · BUILT BY TEAM N:U:N</div>
      <div>SYNTHETIC - CNN 95.62% · RESNET50 81.67% · VGG16 87.76% · MOBILENETV2 87.07%</div>
      <div>PHOTOGRAPHIC - RESNET50V2 96.17% · MOBILENETV2 89.40%</div>
      <div>320K TRAINING IMAGES · 2 DOMAINS · EVIDENCE, NOT JUDGMENT.</div>
    </div>
    """)

    outputs = [original_out, cnn_act_out, rn_act_out, cnn_over_out, rn_over_out,
               verdict, gauges_out, ranking_out, signals_out, tech_out]

    run_btn.click(run_analysis, inputs=img_in, outputs=outputs)
    clear_btn.click(
        lambda: (None,) * 5 + (IDLE_VERDICT, "", "", "", ""),
        outputs=outputs,
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
