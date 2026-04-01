"""Generate a publication-quality PPEDCRF architecture/pipeline diagram.

Two-row U-shaped layout:
  Row 1 (L->R): Input Frames -> Unary Predictor -> DCRF Refinement
  Turn:         down (vertical connector on right side)
  Row 2 (R->L): NCP Scaling -> Gaussian Noise -> Released Frame
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ===================== GLOBAL LAYOUT =====================
FIG_W, FIG_H = 10.4, 7.8
BOX_W  = 2.30
BOX_H  = 1.25
H_GAP  = 0.55
V_GAP  = 1.70

Y_R1 = FIG_H - 2.20
Y_R2 = Y_R1 - V_GAP - BOX_H

X_LEFT  = 1.45
X_MID   = X_LEFT + BOX_W + H_GAP
X_RIGHT = X_MID  + BOX_W + H_GAP

# ===================== COLOURS =====================
C = dict(
    input="#E8D5B7", unary="#B8D4E3", dcrf="#A8D5BA",
    ncp="#E3C4D4", noise="#F5E6A3", output="#D5C8E8",
    arrow="#444444", border="#555555",
)

# ===================== FIGURE =====================
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

# ===================== HELPERS =====================
def box(cx, cy, w, h, fc, title, lines, eq=None):
    bx = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.10", facecolor=fc,
        edgecolor=C["border"], linewidth=1.5, zorder=2)
    ax.add_patch(bx)
    ax.text(cx, cy + h*0.23, title,
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", zorder=3)
    ax.text(cx, cy - h*0.18, "\n".join(lines),
            ha="center", va="center", fontsize=8.2,
            linespacing=1.2, color="#333", zorder=3)
    if eq:
        ax.text(cx, cy - h/2 - 0.18, eq,
                ha="center", va="top", fontsize=8.8,
                style="italic", color="#222", zorder=3)

def h_arr(x1, y, x2, **kw):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle="Simple,tail_width=0.5,head_width=5,head_length=3.5",
        color=kw.get("c", C["arrow"]), linewidth=1.5, zorder=1))

def v_arr(x, y1, y2, **kw):
    ax.add_patch(FancyArrowPatch(
        (x, y1), (x, y2),
        arrowstyle="Simple,tail_width=0.5,head_width=5,head_length=3.5",
        color=kw.get("c", C["arrow"]), linewidth=1.5, zorder=1))

# ===================== BOXES =====================
# Row 1
box(X_LEFT,  Y_R1, BOX_W, BOX_H, C["input"],
    "Input Frames",
    ["Video sequence", r"$\mathcal{X} = \{I_t\}$"],
    eq=r"$I_t$")

box(X_MID,   Y_R1, BOX_W, BOX_H, C["unary"],
    "Unary Predictor",
    ["Per-pixel sensitivity", "logit estimation"],
    eq=r"$u_t = g_\theta(I_t)$")

box(X_RIGHT, Y_R1, BOX_W, BOX_H, C["dcrf"],
    "DCRF Refinement",
    ["Spatial smoothing +", "temporal consistency"],
    eq=r"$p_t$  (continuous mask)")

# Row 2
box(X_RIGHT, Y_R2, BOX_W, BOX_H, C["ncp"],
    "NCP Scaling",
    ["Normalized control", "penalty modulation"],
    eq=r"$\alpha_t = \alpha \cdot p_t / \max(p_t)$")

box(X_MID,   Y_R2, BOX_W, BOX_H, C["noise"],
    "Gaussian Noise",
    ["DP-style calibrated", r"$\eta_t \sim \mathcal{N}(0,\,\sigma_0^2)$"],
    eq=r"$\alpha_t \odot p_t \odot \eta_t$")

box(X_LEFT,  Y_R2, BOX_W, BOX_H, C["output"],
    "Released Frame",
    ["Sanitized output", "for downstream use"],
    eq=r"$I_t'$")

# ===================== FLOW ARROWS =====================
h_arr(X_LEFT  + BOX_W/2 + 0.04, Y_R1,  X_MID   - BOX_W/2 - 0.04)
h_arr(X_MID   + BOX_W/2 + 0.04, Y_R1,  X_RIGHT - BOX_W/2 - 0.04)

# Turn: DCRF down to NCP
v_arr(X_RIGHT, Y_R1 - BOX_H/2 - 0.04, Y_R2 + BOX_H/2 + 0.04)

# Row 2 R->L
h_arr(X_RIGHT - BOX_W/2 - 0.04, Y_R2,  X_MID   + BOX_W/2 + 0.04)
h_arr(X_MID   - BOX_W/2 - 0.04, Y_R2,  X_LEFT  + BOX_W/2 + 0.04)

# ===================== TEMPORAL FEEDBACK LOOP =====================
dcrf_top = Y_R1 + BOX_H / 2
fb_y = dcrf_top + 0.55
lx = X_RIGHT - BOX_W * 0.25
rx = X_RIGHT + BOX_W * 0.25
ax.plot([rx, rx], [dcrf_top + 0.02, fb_y], color="#388E3C", lw=1.8, zorder=1)
ax.plot([lx, rx], [fb_y, fb_y],            color="#388E3C", lw=1.8, zorder=1)
ax.add_patch(FancyArrowPatch(
    (lx, fb_y), (lx, dcrf_top + 0.02),
    arrowstyle="Simple,tail_width=0.5,head_width=5,head_length=3.5",
    color="#388E3C", linewidth=1.8, zorder=1))
ax.text(X_RIGHT, fb_y + 0.15,
        r"$\hat{p}_{t-1}$  (temporal feedback)",
        ha="center", va="bottom", fontsize=9,
        color="#2E7D32", fontweight="bold", zorder=3)

# ===================== PRIVACY BUDGET =====================
# Place between the two rows, pointing into Gaussian Noise from above
mid_gap_y = (Y_R1 - BOX_H/2 + Y_R2 + BOX_H/2) / 2
v_arr(X_MID, mid_gap_y + 0.10, Y_R2 + BOX_H/2 + 0.04, c="#E65100")
ax.text(X_MID, mid_gap_y + 0.22,
        r"Privacy budget  $\sigma_0(\varepsilon,\delta)$",
        ha="center", va="bottom", fontsize=9,
        color="#BF360C", fontweight="bold", zorder=3)

# ===================== ANNOTATIONS =====================
ann_y = Y_R2 - BOX_H/2 - 0.65

ax.text(X_LEFT, ann_y,
        r"Attacker observes only $I_t'$" "\n"
        r"and computes $f(I_t')$ for retrieval",
        ha="center", va="top", fontsize=8,
        color="#B71C1C", style="italic",
        bbox=dict(boxstyle="round,pad=0.12", fc="#FFEBEE",
                  ec="#E57373", lw=0.8), zorder=3)

ax.text(X_RIGHT, ann_y,
        "Downstream detector /\nsegmenter uses sanitized frame",
        ha="center", va="top", fontsize=8,
        color="#1565C0", style="italic",
        bbox=dict(boxstyle="round,pad=0.12", fc="#E3F2FD",
                  ec="#90CAF9", lw=0.8), zorder=3)

# ===================== TITLE =====================
ax.text(FIG_W / 2, FIG_H - 0.25,
        "PPEDCRF Pipeline: Release-Side Selective Perturbation",
        ha="center", va="top", fontsize=13, fontweight="bold",
        color="#222", zorder=3)

# ===================== SAVE =====================
out = r"c:\source\PPEDCRF\paper\figs\architecture_of_solution.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved to {out}")
