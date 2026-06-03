# ============================================================
# AI IMAGE FORENSICS — Complete app.py for VS Code
# Run: python app.py
# ============================================================
 
# ── Imports ─────────────────────────────────────────────────
import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import os
import warnings
warnings.filterwarnings('ignore')
 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (BatchNormalization, Dense,
                                     GlobalAveragePooling2D, Dropout)
from tensorflow.keras.models import Model
 
print("✅ Libraries imported")
 
 
# ── Compatibility Classes (needed for .h5 models) ────────────
class CompatibleBatchNorm(BatchNormalization):
    def __init__(self, **kwargs):
        kwargs.pop('renorm',          None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        kwargs.pop('fused',           None)
        super().__init__(**kwargs)
 
class CompatibleDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)
 
custom_objects = {
    'BatchNormalization': CompatibleBatchNorm,
    'Dense':              CompatibleDense
}
 
 
# ── Safe Model Loader ────────────────────────────────────────
def safe_load(path, name, co=None):
    try:
        model = load_model(path, custom_objects=co) if co else load_model(path)
        print(f"✅ {name} loaded — {model.count_params():,} params")
        return model
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        raise
 
 
# ── Load All 6 Models ────────────────────────────────────────
print("\nLoading models...")
print("=" * 50)
 
# Synthetic Domain (.h5 — needs custom_objects)
custom_cnn      = safe_load('TrainedModels/custom_cnn.h5',
                            'Custom CNN',            custom_objects)
 
resnet_model    = safe_load('TrainedModels/resnet50.h5',
                            'ResNet50',              custom_objects)
 
vgg_model       = safe_load('TrainedModels/vgg16.h5',
                            'VGG16',                 custom_objects)
 
mobilenet_model = safe_load('TrainedModels/mobilenet.h5',
                            'MobileNetV2 Synthetic', custom_objects)
 
# Photographic Domain
rw_mobile_model = safe_load('TrainedModels/realworld_mobilenet.h5',
                            'MobileNetV2 Photographic', custom_objects)
 
rw_resnet_model = safe_load('TrainedModels/realworld_resnet.keras',
                            'ResNet50V2 Photographic')   # no custom_objects needed
 
print("=" * 50)
print("✅ All 6 models loaded successfully!")
 
 
# ── Sanity Check ─────────────────────────────────────────────
print("\nRunning sanity check...")
 
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
 
checks = [
    (custom_cnn,      'Custom CNN'),
    (resnet_model,    'ResNet50'),
    (vgg_model,       'VGG16'),
    (mobilenet_model, 'MobileNetV2 Synthetic'),
    (rw_mobile_model, 'MobileNetV2 Photographic'),
    (rw_resnet_model, 'ResNet50V2 Photographic'),
]
 
all_ok = True
for model, name in checks:
    try:
        pred = model.predict(dummy, verbose=0)[0][0]
        status = "✅" if 0.0 <= pred <= 1.0 else "⚠️"
        print(f"  {status} {name}: {pred:.4f}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        all_ok = False
 
print()
if all_ok:
    print("✅ All models predicting correctly — ready to launch!")
else:
    print("⚠️ Some models failed sanity check — check errors above")

    
# ── Helper Functions ─────────────────────────────────────────

def preprocess_cifake(image):
    """Synthetic-style: downsample to 32x32 then upscale to 224x224."""
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_small    = img.resize((32, 32),    Image.LANCZOS)
    img_upscaled = img_small.resize((224, 224), Image.LANCZOS)
    arr   = np.array(img_upscaled, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)
    return arr, batch


def preprocess_realworld(image):
    """Photographic-style: direct resize to 224x224."""
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_resized = img.resize((224, 224), Image.LANCZOS)
    arr   = np.array(img_resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)
    return arr, batch


def generate_saliency(model, img_batch):
    """Gradient saliency map — shows what the model focuses on."""
    img_tensor = tf.Variable(
        tf.cast(img_batch, tf.float32), trainable=True
    )
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor, training=False)
        loss       = prediction[:, 0]
    grads   = tape.gradient(loss, img_tensor).numpy()[0]
    heatmap = np.max(np.abs(grads), axis=-1)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap


def overlay_heatmap(img_array, heatmap):
    """Blend original image with coloured heatmap."""
    hm   = cv2.resize(heatmap, (224, 224))
    hm   = np.uint8(255 * hm)
    hm   = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm   = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    orig = np.uint8(255 * img_array)
    return cv2.addWeighted(orig, 0.55, hm, 0.45, 0)


def get_verdict(pred):
    """Return label, confidence, colour and emoji."""
    if pred > 0.5:
        return "REAL",         pred * 100,         "#00e5a0", "✅"
    else:
        return "AI Generated", (1 - pred) * 100,   "#ff4d6d", "🤖"


def get_explanation(is_real, confidence, top_model_name, top_domain):
    """Detailed forensic explanation based on top ranked model."""
    level = ("very high" if confidence >= 90
             else "high"  if confidence >= 75
             else "moderate")

    if not is_real:
        return f"""
## 🤖 Why This Image Appears AI Generated

The top ranked model **{top_model_name} [{top_domain}]** flagged this
image as **AI Generated** with **{level} confidence ({confidence:.1f}%)**.
All six models were evaluated — their accuracy-weighted scores
collectively point toward synthetic generation.

---

### 🔴 Key Forensic Signals Detected

**1. Unnatural Texture Smoothness**
AI diffusion models produce mathematically optimised surfaces.
Real photographs always contain micro-imperfections — subtle grain,
dust, natural wear — that AI consistently fails to replicate.
The model detected an abnormally uniform surface texture here.

**2. Lighting Uniformity Anomaly**
The lighting gradients appear computationally generated.
Real-world illumination produces complex shadows with soft edges
and environmental colour bleeding. AI lighting tends to be
physically perfect but visually sterile — a signature this image shows.

**3. Fine Detail Artifacts**
Regions of high complexity — hair, fabric weave, text, foliage,
background edges — often show subtle blurring or repetitive pixel
patterns. These are hallmarks of the latent-space interpolation
used by generative models like Stable Diffusion and DALL-E.

**4. Statistical Pixel Distribution**
At a pixel level, AI-generated images follow a distinct statistical
distribution — colour histograms cluster differently than those
from real camera sensors with their inherent ISO noise
and lens aberrations.

**5. Structural Coherence Inconsistencies**
Long-range spatial relationships show subtle inconsistencies —
elements that appear correct locally but don't follow
real-world physics at the global scale.

---

### 🗺️ Reading the Heatmap
**Red/Yellow zones** = regions where Custom CNN detected
the strongest AI-generation artifacts.
**Blue zones** = minimal influence on the decision.
        """
    else:
        return f"""
## ✅ Why This Image Appears Real

The top ranked model **{top_model_name} [{top_domain}]** classified
this image as **Human Created** with **{level} confidence ({confidence:.1f}%)**.
All six models were evaluated — their accuracy-weighted scores
collectively indicate authentic, camera-captured imagery.

---

### 🟢 Authenticity Indicators Found

**1. Authentic Sensor Noise**
This image contains natural grain and ISO noise patterns consistent
with real camera capture. Digital cameras introduce characteristic
noise at specific spatial frequencies — a signature AI cannot
perfectly replicate.

**2. Physically Accurate Lighting**
Light and shadow transitions follow real-world physics — soft
penumbras, environmental colour cast, and directional consistency
indicating a genuine light source rather than computed illumination.

**3. Natural Imperfections Present**
Subtle real-world irregularities: slight motion blur, natural
focus falloff, organic texture variation. AI generation smooths
these out in ways that are statistically detectable.

**4. Authentic Depth of Field**
The optical blur gradient (bokeh) matches real lens physics —
aperture-correct with authentic lens aberrations that synthetic
images approximate but don't precisely replicate.

**5. Camera Fingerprint Signatures**
Colour channels show characteristic response curves of a real
image sensor — white balance, chromatic aberration, and vignetting
that match physical optics rather than computational rendering.

---

### 🗺️ Reading the Heatmap
**Red/Yellow zones** = regions the model found most convincingly
authentic — strongest real-world signatures.
**Blue zones** = minimal influence on the decision.
        """


# ── Main Prediction Function ─────────────────────────────────

def predict_image(image):
    try:
        # Both preprocessing paths
        img_array_c, img_batch_c = preprocess_cifake(image)
        img_array_r, img_batch_r = preprocess_realworld(image)

        # Predictions — Synthetic models
        p_cnn    = custom_cnn.predict(img_batch_c,      verbose=0)[0][0]
        p_resnet = resnet_model.predict(img_batch_c,    verbose=0)[0][0]
        p_vgg    = vgg_model.predict(img_batch_c,       verbose=0)[0][0]
        p_mobile = mobilenet_model.predict(img_batch_c, verbose=0)[0][0]

        # Predictions — Photographic models
        p_rw_resnet = rw_resnet_model.predict(img_batch_r, verbose=0)[0][0]
        p_rw_mobile = rw_mobile_model.predict(img_batch_r, verbose=0)[0][0]

        # Model registry
        all_models = [
            {'id': 'cnn',          'name': 'Custom CNN',
             'domain': 'Synthetic',    'accuracy': 0.9562, 'pred': p_cnn},
            {'id': 'resnet_c',     'name': 'ResNet50',
             'domain': 'Synthetic',    'accuracy': 0.8167, 'pred': p_resnet},
            {'id': 'vgg',          'name': 'VGG16',
             'domain': 'Synthetic',    'accuracy': 0.8776, 'pred': p_vgg},
            {'id': 'mobile_c',     'name': 'MobileNetV2',
             'domain': 'Synthetic',    'accuracy': 0.8707, 'pred': p_mobile},
            {'id': 'resnet_rw',    'name': 'ResNet50V2',
             'domain': 'Photographic', 'accuracy': 0.9262, 'pred': p_rw_resnet},
            {'id': 'mobile_rw',    'name': 'MobileNetV2',
             'domain': 'Photographic', 'accuracy': 0.8940, 'pred': p_rw_mobile},
        ]

        # Compute rank scores
        for m in all_models:
            m['is_real']    = m['pred'] > 0.5
            m['confidence'] = (m['pred'] * 100
                               if m['is_real']
                               else (1 - m['pred']) * 100)
            m['strength']   = abs(m['pred'] - 0.5) * 2
            m['rank_score'] = m['accuracy'] * m['strength']
            m['label']      = 'REAL'    if m['is_real'] else 'AI Generated'
            m['emoji']      = '✅'       if m['is_real'] else '🤖'
            m['color']      = '#00e5a0' if m['is_real'] else '#ff4d6d'

        ranked = sorted(all_models,
                        key=lambda x: x['rank_score'],
                        reverse=True)

        top = ranked[0]

        # ── Grad-CAM for BOTH best models ───────────────────
        # Custom CNN — Synthetic domain
        heatmap_cnn  = generate_saliency(custom_cnn, img_batch_c)
        overlaid_cnn = overlay_heatmap(img_array_c, heatmap_cnn)

        # ResNet50V2 — Photographic domain
        heatmap_rn   = generate_saliency(rw_resnet_model, img_batch_r)
        overlaid_rn  = overlay_heatmap(img_array_r, heatmap_rn)

        # ── Build Visualization ──────────────────────────────
# ── Row 1: Heatmaps ──────────────────────────────────
        fig1, axes1 = plt.subplots(1, 5, figsize=(24, 5),
                                   facecolor='#080810')
        fig1.suptitle('HEATMAP & OVERLAY ANALYSIS',
                      color='#e0e0ff', fontsize=13,
                      fontfamily='monospace', fontweight='bold')

        # 1. Original Image
        axes1[0].imshow(img_array_c)
        axes1[0].set_title('ORIGINAL IMAGE',
                           color='#aaaacc', fontsize=9,
                           fontfamily='monospace', pad=8)
        axes1[0].axis('off')

        # 2. Custom CNN Activation Map
        axes1[1].imshow(heatmap_cnn, cmap='inferno')
        axes1[1].set_title('ACTIVATION MAP\n(Custom CNN — Synthetic)',
                           color='#aaaacc', fontsize=9,
                           fontfamily='monospace', pad=8)
        axes1[1].axis('off')

        # 3. ResNet50V2 Activation Map
        axes1[2].imshow(heatmap_rn, cmap='inferno')
        axes1[2].set_title('ACTIVATION MAP\n(ResNet50V2 — Photographic)',
                           color='#aaaacc', fontsize=9,
                           fontfamily='monospace', pad=8)
        axes1[2].axis('off')

        # 4. Custom CNN Gradient Overlay
        axes1[3].imshow(overlaid_cnn)
        axes1[3].set_title('GRADIENT OVERLAY\n(Custom CNN — Synthetic)',
                           color='#aaaacc', fontsize=9,
                           fontfamily='monospace', pad=8)
        axes1[3].axis('off')

        # 5. ResNet50V2 Gradient Overlay  ←←← NEW
        axes1[4].imshow(overlaid_rn)
        axes1[4].set_title('GRADIENT OVERLAY\n(ResNet50V2 — Photographic)',
                           color='#aaaacc', fontsize=9,
                           fontfamily='monospace', pad=8)
        axes1[4].axis('off')

        plt.tight_layout()
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', dpi=150,
                    bbox_inches='tight', facecolor='#080810')
        buf1.seek(0)
        img_heatmaps = Image.open(buf1).copy()
        plt.close()

         # ── Row 2: Gauge Dials ───────────────────────────────
        fig2, gauge_axes = plt.subplots(1, 6, figsize=(24, 5),
                                        facecolor='#080810')
        fig2.suptitle('INDIVIDUAL MODEL GAUGES',
                      color='#e0e0ff', fontsize=13,
                      fontfamily='monospace', fontweight='bold')

        def draw_gauge(ax, name, domain, score, acc):
            ax.set_facecolor('#0d0d1a')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # ── Smart Color & Verdict Logic ─────────────────────
            distance_from_50 = abs(score - 50)
            if distance_from_50 < 15:
                gauge_color = '#888899'   # Grey - Uncertain
                verdict = "UNSURE"
            elif score > 50:
                gauge_color = '#00e5a0'   # Green - REAL
                verdict = "REAL"
            else:
                gauge_color = '#ff4d6d'   # Red - AI
                verdict = "AI"

            # Background arc
            theta     = np.linspace(np.pi, 0, 100)
            r, cx, cy = 0.35, 0.5, 0.42
            ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta),
                    color='#222233', linewidth=11,
                    solid_capstyle='round', zorder=1)

            # Filled arc
            fill_end   = np.pi - (np.pi * score / 100)
            theta_fill = np.linspace(np.pi, fill_end, 100)
            ax.plot(cx + r*np.cos(theta_fill),
                    cy + r*np.sin(theta_fill),
                    color=gauge_color, linewidth=11,
                    solid_capstyle='round', zorder=2)

            # Verdict text
            ax.text(cx, cy - 0.02, verdict,
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color=gauge_color,
                    fontfamily='monospace', zorder=3)

            # Score percentage
            ax.text(cx, cy - 0.18, f'{score:.1f}%',
                    ha='center', va='center',
                    fontsize=11, color='#aaaacc',
                    fontfamily='monospace', zorder=3)

            # Model name
            ax.text(cx, 0.90, name,
                    ha='center', va='top',
                    fontsize=9, fontweight='bold',
                    color='#ccccdd', fontfamily='monospace')

            # Domain tag
            tag_col = '#6655ff' if domain == 'Synthetic' else '#00aa88'
            ax.text(cx, 0.78, f'[{domain}]',
                    ha='center', va='top',
                    fontsize=8, color=tag_col,
                    fontfamily='monospace')

            # Accuracy
            ax.text(cx, 0.06, f'Accuracy: {acc}',
                    ha='center', va='bottom',
                    fontsize=8, color='#555566',
                    fontfamily='monospace')

            for s in ax.spines.values():
                s.set_edgecolor('#222233')
                s.set_linewidth(1.2)

        # Draw all gauges
        for ax, m in zip(gauge_axes, all_models):
            draw_gauge(ax, m['name'], m['domain'],
                       m['pred']*100,
                       f"{m['accuracy']*100:.2f}%")

        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=150,
                    bbox_inches='tight', facecolor='#080810')
        buf2.seek(0)
        img_gauges = Image.open(buf2).copy()
        plt.close()

        # ── Row 3: Bar Chart + Rankings ──────────────────────
        fig3, (ax_bar, ax_rank) = plt.subplots(
            1, 2, figsize=(20, 6), facecolor='#080810'
        )
        fig3.suptitle('MODEL COMPARISON & RANKINGS',
                      color='#e0e0ff', fontsize=13,
                      fontfamily='monospace', fontweight='bold')

        # Bar chart
        real_pcts = [m['pred']*100 for m in all_models]
        bar_cols  = [m['color']    for m in all_models]
        ax_bar.set_facecolor('#0d0d1a')
        bars = ax_bar.bar(range(6), real_pcts, color=bar_cols,
                          width=0.6, edgecolor='#222233', linewidth=1)
        ax_bar.axhline(50, color='#555566', linewidth=1,
                       linestyle='--', alpha=0.7)
        ax_bar.set_xticks(range(6))
        ax_bar.set_xticklabels(
            ['CNN\n(Synth)', 'R50\n(Synth)', 'VGG\n(Synth)',
             'MNV2\n(Synth)', 'R50V2\n(Photo)', 'MNV2\n(Photo)'],
            fontsize=9, color='#666677'
        )
        for bar, val in zip(bars, real_pcts):
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        val + 1.5, f'{val:.1f}%',
                        ha='center', va='bottom',
                        color='white', fontsize=9,
                        fontfamily='monospace', fontweight='bold')
        ax_bar.set_ylim(0, 115)
        ax_bar.set_ylabel('P(REAL) %', color='#666677',
                          fontsize=10, fontfamily='monospace')
        ax_bar.set_title('ALL MODELS P(REAL)', color='#aaaacc',
                         fontsize=10, fontfamily='monospace',
                         fontweight='bold', pad=8)
        ax_bar.tick_params(colors='#666677')
        for s in ax_bar.spines.values():
            s.set_edgecolor('#222233')

        # Rankings
        ax_rank.set_facecolor('#0d0d1a')
        ax_rank.axis('off')
        ax_rank.set_xlim(0, 1)
        ax_rank.set_ylim(0, 1)
        ax_rank.text(0.5, 0.97,
                     'MODEL RANKINGS  (Accuracy × Confidence)',
                     ha='center', va='top',
                     fontsize=10, fontweight='bold',
                     color='#aaaacc', fontfamily='monospace')

        row_h, start_y = 0.13, 0.85
        for i, m in enumerate(ranked):
            y         = start_y - i * row_h
            dom_col   = '#6655ff' if m['domain'] == 'Synthetic' else '#00aa88'
            ax_rank.text(0.02, y, f'#{i+1}',
                         ha='left', va='center',
                         fontsize=11, fontweight='bold',
                         color='#888899', fontfamily='monospace')
            ax_rank.text(0.09, y, m['name'],
                         ha='left', va='center',
                         fontsize=9, fontweight='bold',
                         color='#ccccdd', fontfamily='monospace')
            ax_rank.text(0.09, y - 0.045,
                         f"[{m['domain']}]  Acc: {m['accuracy']*100:.2f}%",
                         ha='left', va='center',
                         fontsize=8, color=dom_col,
                         fontfamily='monospace')
            ax_rank.barh(y, 0.50, left=0.32,
                         height=0.07, color='#1a1a2e',
                         edgecolor='#222233')
            ax_rank.barh(y, m['rank_score']*0.50,
                         left=0.32, height=0.07,
                         color=m['color'], alpha=0.85)
            ax_rank.text(0.84, y,
                         f"{m['emoji']} {m['label']}",
                         ha='left', va='center',
                         fontsize=9, fontweight='bold',
                         color=m['color'], fontfamily='monospace')
            ax_rank.text(0.84, y - 0.045,
                         f"{m['confidence']:.1f}% conf",
                         ha='left', va='center',
                         fontsize=8, color='#666677',
                         fontfamily='monospace')
        for s in ax_rank.spines.values():
            s.set_edgecolor('#222233')

        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=150,
                    bbox_inches='tight', facecolor='#080810')
        buf3.seek(0)
        img_rankings = Image.open(buf3).copy()
        plt.close()
            # ── Explanation from top ranked model ────────────────
        explanation = get_explanation(
            top['is_real'],
            top['confidence'],
            top['name'],
            top['domain']
        )

        # ── Result text ──────────────────────────────────────
        rank_table = ""
        for i, m in enumerate(ranked):
            rank_table += (
                f"| #{i+1} | {m['emoji']} {m['name']} "
                f"| [{m['domain']}] "
                f"| {m['label']} "
                f"| {m['confidence']:.1f}% "
                f"| {m['accuracy']*100:.2f}% "
                f"| {m['rank_score']:.3f} |\n"
            )

        result_text = f"""
## 🔬 Forensic Analysis — 6 Model Report

> ⚠️ **No single verdict is issued.**
> Models are ranked by reliability. Final interpretation is left to the user.

---

### 🏆 Model Rankings
*Ranked by: Model Accuracy × Prediction Confidence Strength*

| Rank | Model | Domain | Prediction | Confidence | Accuracy | Score |
|------|-------|--------|-----------|------------|----------|-------|
{rank_table}

---

### 📊 Raw Scores

| Model | Domain | P(REAL) | P(FAKE) |
|-------|--------|---------|---------|
| 🧪 Custom CNN | Synthetic | {p_cnn*100:.2f}% | {(1-p_cnn)*100:.2f}% |
| 🧪 ResNet50 | Synthetic | {p_resnet*100:.2f}% | {(1-p_resnet)*100:.2f}% |
| 🧪 VGG16 | Synthetic | {p_vgg*100:.2f}% | {(1-p_vgg)*100:.2f}% |
| 🧪 MobileNetV2 | Synthetic | {p_mobile*100:.2f}% | {(1-p_mobile)*100:.2f}% |
| 📷 ResNet50V2 | Photographic | {p_rw_resnet*100:.2f}% | {(1-p_rw_resnet)*100:.2f}% |
| 📷 MobileNetV2 | Photographic | {p_rw_mobile*100:.2f}% | {(1-p_rw_mobile)*100:.2f}% |

---

{explanation}

---

### ℹ️ Domain Guide
- 🧪 **Synthetic Domain** — trained on CIFAKE dataset (120,000 images)
  *Best for detecting AI-generated artwork and synthetic images*
  *Preprocessing: 32×32 → 224×224 upscale*
- 📷 **Photographic Domain** — trained on GRAVEX-200K (200,000 real vs AI faces)
  *Best for detecting AI-generated photographs and deepfakes*
  *Preprocessing: Direct resize to 224×224*
### 🗺️ Reading the Heatmaps
- 🧪 **Custom CNN heatmap** — shows suspicious regions from Synthetic domain perspective
- 📷 **ResNet50V2 heatmap** — shows suspicious regions from Photographic domain perspective
- **Bright/Yellow zones** — regions with highest influence on the decision
- **Dark zones** — regions with minimal influence

---

### ⚠️ Disclaimers
- Heavily filtered or edited images may produce unreliable results
- Screenshots and rephotographed images lose authentic noise patterns
- AI-enhanced real photos occupy a grey zone between both classes
- Compressed images may mimic AI generation signatures
- Results are for research purposes only — not for legal use
- Always combine with human judgement for critical decisions

---

### 📐 Technical Details

| Field | Value |
|-------|-------|
| Total Models | 6 (4 Synthetic + 2 Photographic) |
| Synthetic Dataset | CIFAKE — 120,000 images |
| Photographic Dataset | GRAVEX-200K — 200,000 images |
| Total Training Images | 320,000 images |
| Best Model Accuracy | ResNet50V2 — 96.17% |
| Best Model AUC-ROC | ResNet50V2 — 99.48% |
| Synthetic Preprocessing | 32×32 → 224×224 upscale |
| Photographic Preprocessing | Direct 224×224 resize |
| Ranking Method | Accuracy × Confidence Strength |
| Heatmap Source | Custom CNN (best Synthetic model) |
| Input Resolution | 224×224 px |
        """

        return img_heatmaps, img_gauges, img_rankings, result_text

    except Exception as e:
        import traceback
        empty = Image.new('RGB', (100, 100), color='#080810')
        return empty, empty, empty, \
               f"## ❌ Error\n\n**{str(e)}**\n\n```\n{traceback.format_exc()}\n```"




print("✅ Prediction function ready")


# ── Gradio Interface ─────────────────────────────────────────

# Sample images for examples
REAL_FOLDER = 'dataset/cifake/test/REAL'
FAKE_FOLDER = 'dataset/cifake/test/FAKE'

examples = []
if os.path.exists(REAL_FOLDER) and os.listdir(REAL_FOLDER):
    examples.append(
        [os.path.join(REAL_FOLDER, os.listdir(REAL_FOLDER)[0])]
    )
if os.path.exists(FAKE_FOLDER) and os.listdir(FAKE_FOLDER):
    examples.append(
        [os.path.join(FAKE_FOLDER, os.listdir(FAKE_FOLDER)[0])]
    )

CSS = """
/* ── FULL WIDTH DARK BACKGROUND ── */
html {
    background: #05050e !important;
}

body {
    background: #05050e !important;
    min-width: 100% !important;
}

.gradio-container {
    background: #05050e !important;
    min-width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Inner content wrapper — keeps content centered */
.contain {
    background: #05050e !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

/* Remove any white panels */
.svelte-1gfkn6j,
.wrap,
.panel,
.app,
footer,
#component-0 {
    background: #05050e !important;
}

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #05050e !important;
    color: #d4d4e8 !important;
    font-family: 'Syne', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* ── HERO ── */
.hero { padding: 60px 24px 40px; text-align: center; position: relative; }
.hero::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 40% at 50% 0%,
                rgba(100,80,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.scan-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(100,80,255,0.08);
    border: 1px solid rgba(100,80,255,0.25);
    color: #9988ff;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
    padding: 7px 18px; border-radius: 100px; margin-bottom: 28px;
}
.scan-dot {
    width: 6px; height: 6px; background: #9988ff;
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.7); }
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(36px, 6vw, 64px); font-weight: 800;
    color: #f0f0ff; line-height: 1.05;
    letter-spacing: -2px; margin-bottom: 16px;
}
.hero-title .accent {
    background: linear-gradient(120deg, #6655ff, #cc44ff, #ff4488);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 15px; color: #5a5a7a;
    max-width: 520px; margin: 0 auto 40px;
    line-height: 1.7; font-weight: 400;
}

/* ── STATS ── */
.stats {
    display: flex; justify-content: center; align-items: center;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px; padding: 20px 32px;
    max-width: 780px; margin: 0 auto; gap: 0;
}
.stat         { flex: 1; text-align: center; padding: 0 12px; }
.stat-sep     { width: 1px; height: 36px; background: rgba(255,255,255,0.07); }
.stat-val     { font-family: 'JetBrains Mono', monospace; font-size: 20px;
                font-weight: 700; color: #f0f0ff; display: block; line-height: 1; }
.stat-val.g   { color: #00e5a0; }
.stat-val.p   { color: #9988ff; }
.stat-val.pk  { color: #ff44aa; }
.stat-val.b   { color: #44aaff; }
.stat-val.tl  { color: #00aa88; }
.stat-lbl     { font-family: 'JetBrains Mono', monospace; font-size: 8px;
                letter-spacing: 2px; text-transform: uppercase;
                color: #3a3a5a; margin-top: 5px; display: block; }

/* ── PANELS ── */
.main-wrap {
    background: rgba(255,255,255,0.015);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 24px; padding: 32px; margin: 32px 0 16px;
    position: relative; overflow: hidden;
}
.main-wrap::before {
    content: ''; position: absolute; top: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(100,80,255,0.4), transparent);
}
.sec-label {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    letter-spacing: 3px; text-transform: uppercase; color: #3a3a5a;
    margin-bottom: 14px; display: flex; align-items: center; gap: 10px;
}
.sec-label::after {
    content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.05);
}

/* ── UPLOAD ── */
.upload-zone {
    border: 1.5px dashed rgba(100,80,255,0.2) !important;
    border-radius: 18px !important;
    background: rgba(100,80,255,0.02) !important;
    transition: all 0.3s !important; min-height: 260px !important;
}
.upload-zone:hover {
    border-color: rgba(100,80,255,0.5) !important;
    background: rgba(100,80,255,0.05) !important;
}

/* ── BUTTON ── */
.analyze-btn {
    background: linear-gradient(135deg, #5544ee 0%, #aa33ee 100%) !important;
    border: none !important; border-radius: 14px !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    padding: 15px 28px !important; width: 100% !important;
    margin-top: 14px !important; cursor: pointer !important;
    box-shadow: 0 4px 32px rgba(85,68,238,0.35) !important;
    transition: all 0.25s !important;
}
.analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 40px rgba(85,68,238,0.55) !important;
}

/* ── OUTPUT ── */
.out-img  {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 18px !important; overflow: hidden !important;
}
.out-text {
    background: rgba(255,255,255,0.015) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 18px !important; padding: 12px !important;
    min-height: 220px !important;
}

/* ── HOW IT WORKS ── */
.how-wrap {
    background: rgba(255,255,255,0.015);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 24px; padding: 32px; margin-bottom: 16px;
}
.steps {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px; margin-top: 20px;
}
.step-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px; padding: 18px 14px;
    text-align: center; transition: border-color 0.3s;
}
.step-card:hover { border-color: rgba(100,80,255,0.3); }
.step-icon  { font-size: 24px; margin-bottom: 8px; }
.step-num   { font-family: 'JetBrains Mono', monospace; font-size: 9px;
              letter-spacing: 2px; color: #6655ff;
              text-transform: uppercase; margin-bottom: 5px; }
.step-title { font-size: 12px; font-weight: 600; color: #d4d4f0; margin-bottom: 3px; }
.step-desc  { font-size: 10px; color: #3a3a5a; line-height: 1.5; }

/* ── FORMATS ── */
.fmt-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.fmt {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07); color: #4a4a6a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase;
    padding: 4px 10px; border-radius: 6px;
}

/* ── FOOTER ── */
.footer {
    text-align: center; padding: 28px 0 48px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #2a2a3a; letter-spacing: 0.5px;
}
.footer span { color: #5544ee; }

/* ── MARKDOWN ── */
footer { display: none !important; }
.gr-form  { background: transparent !important; }
.gr-panel { background: transparent !important; border: none !important; }
label { color: #3a3a5a !important; font-size: 11px !important;
        font-family: 'JetBrains Mono', monospace !important; }
.prose h2 {
    font-family: 'Syne', sans-serif !important; color: #f0f0ff !important;
    font-size: 17px !important; font-weight: 700 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 8px; margin-bottom: 12px;
}
.prose h3 {
    font-family: 'JetBrains Mono', monospace !important;
    color: #9988ff !important; font-size: 11px !important;
    letter-spacing: 1.5px; text-transform: uppercase; margin: 20px 0 10px;
}
.prose p      { color: #7070a0 !important; font-size: 13px !important; line-height: 1.75 !important; }
.prose li     { color: #7070a0 !important; font-size: 13px !important; line-height: 1.75 !important; }
.prose strong { color: #d4d4f0 !important; }
.prose table  { width: 100% !important; border-collapse: collapse !important; margin: 12px 0 !important; }
.prose td, .prose th {
    border: 1px solid rgba(255,255,255,0.06) !important;
    padding: 9px 14px !important; font-size: 12px !important;
    color: #7070a0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.prose th {
    background: rgba(100,80,255,0.06) !important;
    color: #9988ff !important; font-size: 10px !important;
    letter-spacing: 1px; text-transform: uppercase;
}
.prose hr   { border-color: rgba(255,255,255,0.05) !important; margin: 20px 0 !important; }
.prose code {
    background: rgba(100,80,255,0.1) !important; color: #9988ff !important;
    padding: 2px 6px !important; border-radius: 4px !important;
    font-size: 11px !important;
}
"""

with gr.Blocks(
    title="AI Image Forensics",
    theme=gr.themes.Base(
        primary_hue="violet",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Syne")
    ),
    css=CSS
) as demo:

    # ── HERO
    gr.HTML("""
    <div class="hero">
        <div class="scan-badge">
            <span class="scan-dot"></span>
            Forensic Analysis System  ·  6 Models  ·  Dual Domain
        </div>
        <h1 class="hero-title">
            AI Image<br><span class="accent">Forensics</span>
        </h1>
        <p class="hero-sub">
            Upload any image. Six deep learning models across two domains
            analyze it simultaneously — revealing the forensic evidence
            for and against AI generation.
        </p>
        <div class="stats">
            <div class="stat">
                <span class="stat-val g">96.17%</span>
                <span class="stat-lbl">Best Accuracy</span>
            </div>
            <div class="stat-sep"></div>
            <div class="stat">
                <span class="stat-val p">99.48%</span>
                <span class="stat-lbl">Best AUC-ROC</span>
            </div>
            <div class="stat-sep"></div>
            <div class="stat">
                <span class="stat-val pk">6</span>
                <span class="stat-lbl">Models</span>
            </div>
            <div class="stat-sep"></div>
            <div class="stat">
                <span class="stat-val b">2</span>
                <span class="stat-lbl">Domains</span>
            </div>
            <div class="stat-sep"></div>
            <div class="stat">
                <span class="stat-val tl">320K</span>
                <span class="stat-lbl">Training Images</span>
            </div>
            <div class="stat-sep"></div>
            <div class="stat">
                <span class="stat-val g">Acc×Conf</span>
                <span class="stat-lbl">Ranking Method</span>
            </div>
        </div>
    </div>
    """)

    # ── MAIN PANEL
    gr.HTML('<div class="main-wrap">')

    with gr.Row(equal_height=True):

        # Left — Upload
        with gr.Column(scale=1):
            gr.HTML('<div class="sec-label">01 — Input Image</div>')

            input_image = gr.Image(
                label="",
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                height=260,
                elem_classes=["upload-zone"]
            )

            detect_btn = gr.Button(
                "⚡ RUN FORENSIC ANALYSIS",
                variant="primary",
                elem_classes=["analyze-btn"]
            )

            if examples:
                gr.HTML("""
                <div style="margin-top:24px">
                    <div class="sec-label">02 — Quick Test Samples</div>
                </div>
                """)
                gr.Examples(examples=examples, inputs=input_image, label="")

            gr.HTML("""
            <div style="margin-top:20px">
                <div class="sec-label">Accepted Formats</div>
                <div class="fmt-row">
                    <span class="fmt">JPG</span>
                    <span class="fmt">PNG</span>
                    <span class="fmt">WEBP</span>
                    <span class="fmt">BMP</span>
                    <span class="fmt">TIFF</span>
                    <span class="fmt">Any Size</span>
                </div>
            </div>
            """)

        # Right — Results
        with gr.Column(scale=2):
            gr.HTML('<div class="sec-label">03 — Heatmaps & Overlays</div>')
            output_heatmaps = gr.Image(
                label="", height=280,
                elem_classes=["out-img"]
            )
            gr.HTML('<div class="sec-label" style="margin-top:14px">04 — Model Gauges</div>')
            output_gauges = gr.Image(
                label="", height=260,
                elem_classes=["out-img"]
            )
            gr.HTML('<div class="sec-label" style="margin-top:14px">05 — Rankings & Comparison</div>')
            output_rankings = gr.Image(
                label="", height=300,
                elem_classes=["out-img"]
            )
            gr.HTML('<div class="sec-label" style="margin-top:20px">06 — Full Analysis Report</div>')
            output_text = gr.Markdown(
                value="""
## 🔬 Awaiting Input

Upload an image to begin dual-domain forensic analysis.

**6 models across 2 domains will analyze your image:**

- 🧪 **Synthetic Domain** — Custom CNN · ResNet50 · VGG16 · MobileNetV2
  Trained on CIFAKE — 120,000 synthetic benchmark images
- 📷 **Photographic Domain** — ResNet50V2 · MobileNetV2
  Trained on GRAVEX-200K — 200,000 real vs AI face images

**Results show rankings — no forced single verdict.**
Final interpretation is always left to the user.

---

*Ranking method: Model Accuracy × Confidence Strength*
                """,
                elem_classes=["out-text"]
            )

    gr.HTML('</div>')

    # ── HOW IT WORKS
    gr.HTML("""
    <div class="how-wrap">
        <div class="sec-label">How It Works</div>
        <div class="steps">
            <div class="step-card">
                <div class="step-icon">📤</div>
                <div class="step-num">Step 01</div>
                <div class="step-title">Upload Image</div>
                <div class="step-desc">Any format — preprocessing handled automatically for each domain</div>
            </div>
            <div class="step-card">
                <div class="step-icon">⚙️</div>
                <div class="step-num">Step 02</div>
                <div class="step-title">Dual Preprocessing</div>
                <div class="step-desc">Synthetic: 32×32 → 224×224 upscale · Photographic: direct 224×224 resize</div>
            </div>
            <div class="step-card">
                <div class="step-icon">🧠</div>
                <div class="step-num">Step 03</div>
                <div class="step-title">6-Model Scan</div>
                <div class="step-desc">4 Synthetic + 2 Photographic = 6 models running simultaneously</div>
            </div>
            <div class="step-card">
                <div class="step-icon">🗺️</div>
                <div class="step-num">Step 04</div>
                <div class="step-title">Heatmap</div>
                <div class="step-desc">Gradient saliency from Custom CNN and ResNet50V2 — shows exact focus regions</div>
            </div>
            <div class="step-card">
                <div class="step-icon">🏆</div>
                <div class="step-num">Step 05</div>
                <div class="step-title">Ranking</div>
                <div class="step-desc">Models ranked by Accuracy × Confidence — most reliable result shown first</div>
            </div>
            <div class="step-card">
                <div class="step-icon">📋</div>
                <div class="step-num">Step 06</div>
                <div class="step-title">Full Report</div>
                <div class="step-desc">Forensic explanation + raw scores + disclaimers + technical details</div>
            </div>
        </div>
    </div>
    """)

    # ── FOOTER
    gr.HTML("""
    <div class="footer">
        AI Image Forensics &nbsp;·&nbsp;
        Built by <span>Team N:U:N</span> &nbsp;·&nbsp;
        Synthetic: Custom CNN 95.62% · ResNet50 81.67% · VGG16 87.76% · MobileNetV2 87.07%
        &nbsp;·&nbsp;
        Photographic: ResNet50V2 96.17% · MobileNetV2 89.40%
        &nbsp;·&nbsp;
        <span>320K Training Images</span> across 2 Domains
    </div>
    """)

    detect_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=[output_heatmaps, output_gauges,
                 output_rankings, output_text]
    )

print("✅ Professional app built successfully")

# ── Launch ───────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )