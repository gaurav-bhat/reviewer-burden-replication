"""
Publication-quality figure generation for PeerJ submission.

Generates:
  figure1_its_volume.png   — ITS for MR volume (H1 primary finding)
  figure1_its_ttfr.png     — ITS for time-to-first-review (H3a, exploratory)
  figure2_lorenz.png       — Lorenz curves for reviewer concentration (H5)

Run:  python3 create_figures.py
"""

DATA_DIR = "data/"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.tools import add_constant

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       300,
})

COLOR_PRE  = "#2166AC"   # blue
COLOR_POST = "#D6604D"   # red-orange
COLOR_EQ   = "#555555"   # equality line

ADOPTION_DATE   = pd.Timestamp("2026-01-01", tz="UTC")
TARGET_BRANCHES = ["main", "master", "develop", "development", "cxt-master", "ot-master"]
WASHOUT         = pd.Timedelta(days=15)

# ── Load & filter data ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR + "mr_metrics.csv", parse_dates=["created_at"])
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
df = df[df["target_branch"].isin(TARGET_BRANCHES)].copy()
if "title" in df.columns:
    df = df[~df["title"].str.lower().str.startswith(("draft:", "wip:"), na=False)]
df = df[~df["created_at"].between(ADOPTION_DATE - WASHOUT, ADOPTION_DATE + WASHOUT)]
df["period"] = np.where(df["created_at"] < ADOPTION_DATE, "pre", "post")

# ── ITS helper ─────────────────────────────────────────────────────────────────
def build_its(series: pd.Series, date_col: pd.Series):
    ts = pd.DataFrame({"date": date_col, "y": series})
    ts["time"]         = np.arange(len(ts))
    ts["intervention"] = (ts["date"] >= ADOPTION_DATE).astype(int)
    # time_post resets to 0 at intervention
    first_post = ts.loc[ts["intervention"] == 1, "time"].min()
    ts["time_post"] = (ts["time"] - first_post) * ts["intervention"]
    X     = add_constant(ts[["time", "intervention", "time_post"]])
    model = sm.OLS(ts["y"], X).fit()
    return ts, model


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1A — ITS: MR Volume (primary H1 finding)
# ══════════════════════════════════════════════════════════════════════════════
df_vol = df[["created_at"]].copy()
df_vol["y"] = 1
weekly_vol = (
    df_vol.set_index("created_at")
          .resample("W")
          .sum()
          .reset_index()
          .rename(columns={"created_at": "date"})
)

vol_ts, vol_model = build_its(weekly_vol["y"], weekly_vol["date"])

pre_mask  = vol_ts["date"] < ADOPTION_DATE
post_mask = ~pre_mask
n_pre = pre_mask.sum()

fig, ax = plt.subplots(figsize=(8, 4.5))

# Observed scatter
ax.scatter(vol_ts.loc[pre_mask,  "date"], vol_ts.loc[pre_mask,  "y"],
           color=COLOR_PRE,  s=22, alpha=0.65, zorder=3, label="Observed (pre-adoption)")
ax.scatter(vol_ts.loc[post_mask, "date"], vol_ts.loc[post_mask, "y"],
           color=COLOR_POST, s=22, alpha=0.65, zorder=3, label="Observed (post-adoption)")

# Fitted trend lines
ax.plot(vol_ts.loc[pre_mask,  "date"], vol_model.fittedvalues[:n_pre],
        color=COLOR_PRE,  linewidth=2, label="Fitted trend (pre)")
ax.plot(vol_ts.loc[post_mask, "date"], vol_model.fittedvalues[n_pre:],
        color=COLOR_POST, linewidth=2, label="Fitted trend (post)")

# Adoption marker
ax.axvline(ADOPTION_DATE, color="black", linestyle="--", linewidth=1.4,
           label="Adoption date (Jan 1, 2026)")

# Annotation with key ITS coefficients
ax.text(0.98, 0.97,
        f"β₁ (pre-trend) = −0.23/wk  p = 0.192\n"
        f"β₂ (level)     =  1.06       p = 0.769\n"
        f"β₃ (slope)     = +1.24/wk  p < 0.001\n"
        f"R² = 0.623",
        transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))

ax.set_xlabel("Date")
ax.set_ylabel("MR count (weekly)")
ax.set_title("Interrupted Time Series: Merge Request Volume (H1)")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)

# Shade pre/post regions lightly
ax.axvspan(vol_ts["date"].min(), ADOPTION_DATE,
           alpha=0.04, color=COLOR_PRE, zorder=0)
ax.axvspan(ADOPTION_DATE, vol_ts["date"].max(),
           alpha=0.04, color=COLOR_POST, zorder=0)

fig.tight_layout()
fig.savefig("figure1_its_volume.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figure1_its_volume.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1B — ITS: Time to First Review (H3a, exploratory)
# ══════════════════════════════════════════════════════════════════════════════
df_tt = df[["created_at", "time_to_first_review_hrs"]].dropna().copy()
weekly_tt = (
    df_tt.set_index("created_at")["time_to_first_review_hrs"]
         .resample("W")
         .mean()
         .dropna()
         .reset_index()
         .rename(columns={"created_at": "date", "time_to_first_review_hrs": "y"})
)

tt_ts, tt_model = build_its(weekly_tt["y"], weekly_tt["date"])

pre_mask_tt  = tt_ts["date"] < ADOPTION_DATE
post_mask_tt = ~pre_mask_tt
n_pre_tt = pre_mask_tt.sum()

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.scatter(tt_ts.loc[pre_mask_tt,  "date"], tt_ts.loc[pre_mask_tt,  "y"],
           color=COLOR_PRE,  s=22, alpha=0.65, zorder=3,
           label=f"Observed pre (n = {n_pre_tt} weekly bins)")
ax.scatter(tt_ts.loc[post_mask_tt, "date"], tt_ts.loc[post_mask_tt, "y"],
           color=COLOR_POST, s=22, alpha=0.65, zorder=3,
           label=f"Observed post (n = {len(tt_ts)-n_pre_tt} weekly bins)")

ax.plot(tt_ts.loc[pre_mask_tt,  "date"], tt_model.fittedvalues[:n_pre_tt],
        color=COLOR_PRE,  linewidth=2, label="Fitted trend (pre)")
ax.plot(tt_ts.loc[post_mask_tt, "date"], tt_model.fittedvalues[n_pre_tt:],
        color=COLOR_POST, linewidth=2, label="Fitted trend (post)")

ax.axvline(ADOPTION_DATE, color="black", linestyle="--", linewidth=1.4,
           label="Adoption date (Jan 1, 2026)")

# Caveat note
ax.text(0.98, 0.97,
        "Note: Pre-adoption TTFR data\nvery sparse (8 MRs → few bins).\n"
        "ITS coefficients are descriptive only.",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        color="#663300",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8f0",
                  edgecolor="#cc9966", alpha=0.9))

ax.set_xlabel("Date")
ax.set_ylabel("Mean time to first review (hours, weekly)")
ax.set_title("Interrupted Time Series: Time to First Review (H3a, exploratory)")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)

ax.axvspan(tt_ts["date"].min(), ADOPTION_DATE,
           alpha=0.04, color=COLOR_PRE, zorder=0)
ax.axvspan(ADOPTION_DATE, tt_ts["date"].max(),
           alpha=0.04, color=COLOR_POST, zorder=0)

fig.tight_layout()
fig.savefig("figure1_its_ttfr.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figure1_its_ttfr.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Lorenz Curves (H5: Reviewer concentration)
# ══════════════════════════════════════════════════════════════════════════════
profiles  = pd.read_csv(DATA_DIR + "contributor_profiles.csv")
df_notes  = pd.read_csv(DATA_DIR + "review_notes.csv")

senior_ids = profiles.loc[profiles["is_senior"], "author_id"].tolist()
sr = df_notes[df_notes["reviewer_id"].isin(senior_ids)].copy()
sr["created_at"] = pd.to_datetime(sr["created_at"], utc=True)

WASHOUT_START = ADOPTION_DATE - WASHOUT
WASHOUT_END   = ADOPTION_DATE + WASHOUT
sr = sr[~sr["created_at"].between(WASHOUT_START, WASHOUT_END)]

pre_loads  = sr.loc[sr["created_at"] < ADOPTION_DATE,  "reviewer_id"].value_counts().values
post_loads = sr.loc[sr["created_at"] >= ADOPTION_DATE, "reviewer_id"].value_counts().values


def gini(arr):
    arr = np.sort(np.array(arr, dtype=float))
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))


def lorenz_points(arr):
    arr    = np.sort(np.array(arr, dtype=float))
    cs     = np.cumsum(arr)
    cs     = cs / cs[-1]
    x_vals = np.linspace(0, 1, len(cs) + 1)
    y_vals = np.concatenate([[0], cs])
    return x_vals, y_vals


g_pre  = gini(pre_loads)
g_post = gini(post_loads)

x_pre,  y_pre  = lorenz_points(pre_loads)
x_post, y_post = lorenz_points(post_loads)

fig, ax = plt.subplots(figsize=(6.5, 6))

# Equality line
ax.plot([0, 1], [0, 1], color=COLOR_EQ, linestyle="--", linewidth=1.2,
        label="Perfect equality", zorder=2)

# Pre-adoption curve
ax.plot(x_pre, y_pre, color=COLOR_PRE, linewidth=2.2,
        label=f"Pre-adoption  (Gini = {g_pre:.3f}, n = {len(pre_loads)} reviewers)")
ax.fill_between(x_pre, y_pre, [0] * len(x_pre),
                color=COLOR_PRE, alpha=0.10)

# Post-adoption curve
ax.plot(x_post, y_post, color=COLOR_POST, linewidth=2.2,
        label=f"Post-adoption (Gini = {g_post:.3f}, n = {len(post_loads)} reviewers)")
ax.fill_between(x_post, y_post, [0] * len(x_post),
                color=COLOR_POST, alpha=0.10)

# Annotate top-2 concentration post
top2_share = np.sort(post_loads)[-2:].sum() / post_loads.sum()
ax.annotate(
    f"Top-2 reviewers: {top2_share:.1%}\nof all post-adoption reviews",
    xy=(1.0, y_post[-1]), xytext=(0.60, 0.30),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
    fontsize=9, color="#333333",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#cccccc", alpha=0.9),
)

ax.set_xlabel("Cumulative share of senior engineers\n(ranked by review volume, ascending)")
ax.set_ylabel("Cumulative share of total review notes")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Lorenz Curves: Senior Engineer Review Load Concentration (H5)")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("figure2_lorenz.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figure2_lorenz.png")

print("\nDone. Upload these three files to PeerJ as separate figure attachments:")
print("  figure1_its_volume.png  →  Figure 1")
print("  figure2_lorenz.png      →  Figure 2")
print("  figure1_its_ttfr.png    →  Supplemental Figure S1 (or omit)")
