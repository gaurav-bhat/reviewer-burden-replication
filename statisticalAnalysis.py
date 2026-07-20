"""
Statistical Analysis Script
"The Reviewer's Burden: Code Review Load on Senior Engineers After AI Tool Adoption"
Published in PeerJ Computer Science.

Replication package: https://github.com/gaurav-bhat/reviewer-burden-replication

Input CSVs (in data/ directory):
    - data/mr_metrics.csv
    - data/review_notes.csv
    - data/approvals.csv
    - data/contributor_profiles.csv

Output (written to current directory):
    - results_summary.csv
    - full_results.txt
"""

DATA_DIR = "data/"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
import sys

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

ADOPTION_DATE   = pd.Timestamp("2026-01-01", tz="UTC")
WASHOUT_DAYS    = 15                            # days either side of boundary
ALPHA           = 0.05
# H2 (PR size) and H6 (churn) require lines_added/files_changed which GitLab
# did not return — active hypotheses are H1, H3, H4, H5 only
N_HYPOTHESES    = 4
ALPHA_ADJUSTED  = ALPHA / N_HYPOTHESES          # Bonferroni = 0.0125

BOT_KEYWORDS    = ['bot', 'ci', 'automation', 'pipeline', 'dependabot']
# Include each project's trunk branch (cxt-master=hawkeye, ot-master=mktpui)
TARGET_BRANCHES = ['main', 'master', 'develop', 'development', 'cxt-master', 'ot-master']

SPRINT_WEEKS    = 2                             # sprint length for H1

OUTPUT_FILE     = "full_results.txt"

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════

print("Loading data...")
df_mr       = pd.read_csv(DATA_DIR + "mr_metrics.csv",          parse_dates=['created_at','merged_at','first_review_at'])
df_notes    = pd.read_csv(DATA_DIR + "review_notes.csv",         parse_dates=['created_at'])
df_approvals= pd.read_csv(DATA_DIR + "approvals.csv")
df_profiles = pd.read_csv(DATA_DIR + "contributor_profiles.csv")

# ── Clean MRs ─────────────────────────────────────────────────────────
df_mr = df_mr[df_mr['target_branch'].isin(TARGET_BRANCHES)].copy()
df_mr = df_mr[~df_mr['title'].str.lower().str.startswith(('draft:', 'wip:'), na=False)]

# ── Remove bot authors ────────────────────────────────────────────────
bot_ids = df_profiles[
    df_profiles['author_username'].str.lower().str.contains('|'.join(BOT_KEYWORDS), na=False)
]['author_id'].tolist()
df_mr    = df_mr[~df_mr['author_id'].isin(bot_ids)]
df_notes = df_notes[~df_notes['reviewer_id'].isin(bot_ids)]

# ── Period classification with washout window ─────────────────────────
washout_start = ADOPTION_DATE - pd.Timedelta(days=WASHOUT_DAYS)
washout_end   = ADOPTION_DATE + pd.Timedelta(days=WASHOUT_DAYS)

df_mr = df_mr[
    ~df_mr['created_at'].between(washout_start, washout_end)
].copy()

df_mr['period'] = np.where(df_mr['created_at'] < ADOPTION_DATE, 'pre', 'post')

pre  = df_mr[df_mr['period'] == 'pre']
post = df_mr[df_mr['period'] == 'post']

# ── Senior engineers ──────────────────────────────────────────────────
senior_ids = df_profiles[df_profiles['is_senior'] == True]['author_id'].tolist()

print(f"  Total MRs     : {len(df_mr):,}  (pre={len(pre):,}, post={len(post):,})")
print(f"  Senior devs   : {len(senior_ids)}")
print(f"  Bonferroni α  : {ALPHA_ADJUSTED:.4f}\n")

# ══════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def normality_test(series, label=""):
    """Shapiro-Wilk test. Returns (is_normal, W, p)."""
    series = series.dropna()
    # Shapiro-Wilk is reliable up to n=5000; use KS for larger samples
    if len(series) <= 5000:
        stat, p = stats.shapiro(series)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.kstest(series, 'norm',
                               args=(series.mean(), series.std()))
        test_name = "Kolmogorov-Smirnov"
    is_normal = p > 0.05
    return is_normal, stat, p, test_name


def cohens_d(a, b):
    """Cohen's d for independent samples."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.std(a, ddof=1)**2 +
         (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0


def rank_biserial(U, n1, n2):
    """Rank-biserial correlation from Mann-Whitney U statistic."""
    return 1 - (2 * U) / (n1 * n2)


def effect_label(r):
    """Interpret effect size magnitude (Cohen 1988)."""
    r = abs(r)
    if r < 0.10: return "negligible"
    if r < 0.30: return "small"
    if r < 0.50: return "medium"
    return "large"


def test_hypothesis(pre_vals, post_vals, h_label, metric_label,
                    expected_direction="increase"):
    """
    Run full hypothesis test pipeline:
    1. Normality (Shapiro-Wilk / KS)
    2. Parametric (t-test) or non-parametric (Mann-Whitney U)
    3. Effect size
    4. Decision at Bonferroni-adjusted alpha
    Returns a result dict.
    """
    pre_vals  = pd.Series(pre_vals).dropna()
    post_vals = pd.Series(post_vals).dropna()

    norm_pre,  w_pre,  p_pre,  test_pre  = normality_test(pre_vals)
    norm_post, w_post, p_post, test_post = normality_test(post_vals)
    both_normal = norm_pre and norm_post

    if both_normal:
        stat, p_val = stats.ttest_ind(pre_vals, post_vals)
        test_used   = "Independent t-test"
        d           = cohens_d(post_vals.values, pre_vals.values)
        effect_size = d
        effect_type = "Cohen's d"
    else:
        stat, p_val = stats.mannwhitneyu(
            post_vals, pre_vals, alternative='two-sided')
        test_used   = "Mann-Whitney U"
        n1, n2      = len(post_vals), len(pre_vals)
        rb          = rank_biserial(stat, n1, n2)
        effect_size = rb
        effect_type = "Rank-biserial r"

    significant  = p_val < ALPHA_ADJUSTED
    direction_ok = (np.median(post_vals) > np.median(pre_vals)
                    if expected_direction == "increase"
                    else np.median(post_vals) < np.median(pre_vals))
    supported    = significant and direction_ok

    pct_change = ((np.mean(post_vals) - np.mean(pre_vals)) /
                  np.mean(pre_vals) * 100) if np.mean(pre_vals) != 0 else np.nan

    result = {
        'Hypothesis'     : h_label,
        'Metric'         : metric_label,
        'Pre Mean (SD)'  : f"{np.mean(pre_vals):.2f} ({np.std(pre_vals):.2f})",
        'Post Mean (SD)' : f"{np.mean(post_vals):.2f} ({np.std(post_vals):.2f})",
        'Δ%'             : f"{pct_change:+.1f}%",
        'Pre Median'     : round(np.median(pre_vals), 2),
        'Post Median'    : round(np.median(post_vals), 2),
        'Normality (pre)': f"{'Normal' if norm_pre else 'Non-normal'} (p={p_pre:.4f})",
        'Test Used'      : test_used,
        'Statistic'      : round(stat, 3),
        'p-value'        : round(p_val, 6),
        'Significant'    : "Yes" if significant else "No",
        f'{effect_type}' : round(effect_size, 3),
        'Effect Size'    : effect_label(effect_size),
        'Outcome'        : "SUPPORTED" if supported else "NOT SUPPORTED",
    }
    return result


def gini_coefficient(values):
    """Gini coefficient for an array of non-negative values."""
    values = np.sort(np.array(values, dtype=float))
    n      = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index  = np.arange(1, n + 1)
    return (2 * np.sum(index * values) -
            (n + 1) * np.sum(values)) / (n * np.sum(values))

# ══════════════════════════════════════════════════════════════════════
# 3. HYPOTHESIS TESTS
# ══════════════════════════════════════════════════════════════════════

results = []
log     = []

def log_print(msg=""):
    print(msg)
    log.append(msg)

log_print("=" * 65)
log_print("HYPOTHESIS TEST RESULTS")
log_print(f"Bonferroni-adjusted significance threshold: α = {ALPHA_ADJUSTED:.4f}")
log_print("=" * 65)

# ── H1: PR Volume ─────────────────────────────────────────────────────
log_print("\n── H1: Pull Request Volume ──────────────────────────────────")

def sprint_volumes(df, sprint_weeks=2):
    df = df.copy()
    df['sprint'] = ((df['created_at'] - df['created_at'].min()) /
                    pd.Timedelta(weeks=sprint_weeks)).astype(int)
    return df.groupby('sprint').size()

pre_sprints  = sprint_volumes(pre,  SPRINT_WEEKS)
post_sprints = sprint_volumes(post, SPRINT_WEEKS)

r1 = test_hypothesis(pre_sprints, post_sprints, "H1", "MR volume per sprint")
results.append(r1)
log_print(f"  Pre  sprints  : {len(pre_sprints)} sprints, mean={pre_sprints.mean():.1f} MRs")
log_print(f"  Post sprints  : {len(post_sprints)} sprints, mean={post_sprints.mean():.1f} MRs")
log_print(f"  Δ%            : {r1['Δ%']}")
log_print(f"  Test          : {r1['Test Used']}, stat={r1['Statistic']}, p={r1['p-value']}")
log_print(f"  Effect size   : {list(r1.items())[-3][1]} ({r1['Effect Size']})")
log_print(f"  Outcome       : {r1['Outcome']}")

# ── H2: Review Surface Area ───────────────────────────────────────────
log_print("\n── H2: Pull Request Size ────────────────────────────────────")
log_print("  SKIPPED — lines_added / files_changed not available in GitLab export")
log_print("  (GitLab REST API requires per-MR diff calls; data not collected)")

# ── H3: Reviewer Turnaround Time ──────────────────────────────────────
log_print("\n── H3: Reviewer Turnaround Time ─────────────────────────────")

for metric, label in [
    ('time_to_first_review_hrs', 'Time to first review (hrs)'),
    ('time_to_merge_hrs',        'Time to merge (hrs)'),
]:
    valid = df_mr[df_mr[metric] > 0]
    pre_v  = valid[valid['period'] == 'pre'][metric]
    post_v = valid[valid['period'] == 'post'][metric]
    r = test_hypothesis(pre_v, post_v, "H3", label)
    results.append(r)
    log_print(f"  [{label}]")
    log_print(f"    Pre median={pre_v.median():.1f} hrs  "
              f"Post median={post_v.median():.1f} hrs  "
              f"Δ%={r['Δ%']}  p={r['p-value']}  {r['Outcome']}")

# ── H4: Review Cycle Depth ────────────────────────────────────────────
log_print("\n── H4: Review Cycle Depth ───────────────────────────────────")

# Use the review_round_trips column from MR data (only non-null = MRs that
# received at least one formal review; nulls = merged without review)
reviewed_pre  = df_mr[(df_mr['period'] == 'pre')  & df_mr['review_round_trips'].notna()]
reviewed_post = df_mr[(df_mr['period'] == 'post') & df_mr['review_round_trips'].notna()]

pre_rt  = reviewed_pre['review_round_trips']
post_rt = reviewed_post['review_round_trips']

# Review coverage rate (% of MRs receiving any formal review)
coverage_pre  = df_mr[df_mr['period'] == 'pre']['review_round_trips'].notna().mean()  * 100
coverage_post = df_mr[df_mr['period'] == 'post']['review_round_trips'].notna().mean() * 100

log_print(f"  Review coverage (MRs with any formal review): "
          f"pre={coverage_pre:.1f}%  post={coverage_post:.1f}%  "
          f"Δ={coverage_post-coverage_pre:+.1f}pp")
log_print(f"  NOTE: H4 statistical test limited to reviewed MRs only "
          f"(pre n={len(pre_rt)}, post n={len(post_rt)})")

if len(pre_rt) >= 5:
    r4 = test_hypothesis(pre_rt, post_rt, "H4", "Review round trips (reviewed MRs only)")
    results.append(r4)
    high_friction_pre  = (pre_rt  >= 3).mean() * 100
    high_friction_post = (post_rt >= 3).mean() * 100
    log_print(f"  Pre median={pre_rt.median():.1f}  Post median={post_rt.median():.1f}  "
              f"Δ%={r4['Δ%']}  p={r4['p-value']}  {r4['Outcome']}")
    log_print(f"  High-friction MRs (≥3 round trips): "
              f"pre={high_friction_pre:.1f}%  post={high_friction_post:.1f}%")
    log_print(f"  CAUTION: pre n={len(pre_rt)} — descriptive only, test underpowered")
else:
    log_print(f"  SKIPPED — pre-period sample too small (n={len(pre_rt)}) for testing")

# ── H5: Reviewer Concentration (Gini) ────────────────────────────────
log_print("\n── H5: Reviewer Concentration ───────────────────────────────")

senior_notes = df_notes[df_notes['reviewer_id'].isin(senior_ids)].copy()
# Drop pre-existing period col from review_notes (Pre-AI/Post-AI labels) before
# merging in the script-computed period ('pre'/'post') from df_mr
senior_notes = senior_notes.drop(columns=['period'], errors='ignore').merge(
    df_mr[['mr_id', 'project_id', 'period']],
    on=['mr_id', 'project_id'], how='left'
)

pre_loads  = (senior_notes[senior_notes['period'] == 'pre']
              .groupby('reviewer_id').size().values)
post_loads = (senior_notes[senior_notes['period'] == 'post']
              .groupby('reviewer_id').size().values)

gini_pre  = gini_coefficient(pre_loads)
gini_post = gini_coefficient(post_loads)

# Quartile distribution
def quartile_shares(loads):
    loads = np.sort(loads)[::-1]
    total = loads.sum()
    n     = len(loads)
    q     = max(1, n // 4)
    shares = {
        'Q4 (top)' : loads[:q].sum()      / total * 100,
        'Q3'       : loads[q:2*q].sum()   / total * 100,
        'Q2'       : loads[2*q:3*q].sum() / total * 100,
        'Q1 (bot)' : loads[3*q:].sum()    / total * 100,
    }
    return shares

log_print(f"  Gini pre={gini_pre:.3f}  post={gini_post:.3f}  "
          f"Δ={gini_post-gini_pre:+.3f}")
log_print(f"  Quartile shares (pre)  : {quartile_shares(pre_loads)}")
log_print(f"  Quartile shares (post) : {quartile_shares(post_loads)}")

# Bootstrap significance test for Gini difference
n_boot = 2000
boot_diffs = []
all_loads  = np.concatenate([pre_loads, post_loads])
for _ in range(n_boot):
    s1 = np.random.choice(all_loads, len(pre_loads),  replace=True)
    s2 = np.random.choice(all_loads, len(post_loads), replace=True)
    boot_diffs.append(gini_coefficient(s2) - gini_coefficient(s1))

observed_diff = gini_post - gini_pre
p_gini = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))

r5 = {
    'Hypothesis'     : "H5",
    'Metric'         : "Reviewer Gini coefficient",
    'Pre Mean (SD)'  : f"{gini_pre:.3f}",
    'Post Mean (SD)' : f"{gini_post:.3f}",
    'Δ%'             : f"{(gini_post-gini_pre)/gini_pre*100:+.1f}%",
    'Pre Median'     : gini_pre,
    'Post Median'    : gini_post,
    'Normality (pre)': "Bootstrap test used",
    'Test Used'      : f"Bootstrap permutation (n={n_boot})",
    'Statistic'      : round(observed_diff, 4),
    'p-value'        : round(p_gini, 4),
    'Significant'    : "Yes" if p_gini < ALPHA_ADJUSTED else "No",
    'Rank-biserial r': "N/A",
    'Effect Size'    : effect_label(abs(observed_diff)),
    'Outcome'        : ("SUPPORTED" if p_gini < ALPHA_ADJUSTED
                        and gini_post > gini_pre else "NOT SUPPORTED"),
}
results.append(r5)
log_print(f"  Bootstrap p={p_gini:.4f}  {r5['Outcome']}")

# ── H6: Code Churn ───────────────────────────────────────────────────
log_print("\n── H6: Code Churn Ratio ─────────────────────────────────────")
log_print("  SKIPPED — lines_added / lines_deleted not available in GitLab export")
log_print("  (GitLab REST API requires per-MR diff calls; data not collected)")

# ── Summary Table IV ──────────────────────────────────────────────────
log_print("\n" + "=" * 65)
log_print("TABLE IV — HYPOTHESIS SUMMARY")
log_print("=" * 65)

df_results = pd.DataFrame(results)
summary_cols = ['Hypothesis', 'Metric', 'Δ%', 'p-value',
                'Significant', 'Effect Size', 'Outcome']
print(df_results[summary_cols].to_string(index=False))
df_results.to_csv("results_summary.csv", index=False)

# ══════════════════════════════════════════════════════════════════════
# 4. INTERRUPTED TIME SERIES ANALYSIS
# ══════════════════════════════════════════════════════════════════════

log_print("\n" + "=" * 65)
log_print("INTERRUPTED TIME SERIES ANALYSIS")
log_print("=" * 65)

def run_its(df, metric, period_unit='W', label=""):
    """
    ITS regression:
        Y = β0 + β1*time + β2*intervention + β3*time_post + ε
        β1 = pre-trend (slope before adoption)
        β2 = level change at adoption (immediate effect)
        β3 = slope change after adoption (gradual accumulation)
    """
    ts = (df.set_index('created_at')[metric]
          .dropna()
          .resample(period_unit)
          .mean()
          .reset_index())

    ts['time']          = np.arange(len(ts))
    ts['intervention']  = (ts['created_at'] >= ADOPTION_DATE).astype(int)
    ts['time_post']     = ts['time'] * ts['intervention']
    ts['time_post']    -= ts[ts['intervention'] == 1]['time'].min() * \
                          ts['intervention']  # centre post-adoption time

    ts = ts.dropna(subset=[metric])
    X  = add_constant(ts[['time', 'intervention', 'time_post']])
    y  = ts[metric]
    model = OLS(y, X).fit()

    log_print(f"\n  ITS: {label}")
    log_print(f"    β0 (intercept)       = {model.params['const']:.3f}")
    log_print(f"    β1 (pre-trend)       = {model.params['time']:.3f}  "
              f"p={model.pvalues['time']:.4f}")
    log_print(f"    β2 (level change)    = {model.params['intervention']:.3f}  "
              f"p={model.pvalues['intervention']:.4f}")
    log_print(f"    β3 (slope change)    = {model.params['time_post']:.3f}  "
              f"p={model.pvalues['time_post']:.4f}")
    log_print(f"    R²                   = {model.rsquared:.3f}")

    return ts, model

# MR volume ITS
df_vol = df_mr[['created_at']].copy()
df_vol['volume_val'] = 1
df_vol = df_vol.set_index('created_at').resample('W').sum().reset_index()

its_vol_ts, its_vol_model = run_its(df_vol, 'volume_val', 'W', "PR Volume")

# Turnaround time ITS
df_tt = df_mr[['created_at', 'time_to_first_review_hrs']].dropna().copy()
its_tt_ts, its_tt_model = run_its(
    df_tt, 'time_to_first_review_hrs', 'W', "Time to First Review"
)

# ══════════════════════════════════════════════════════════════════════
# 5. FIGURE 1 — INTERRUPTED TIME SERIES PLOT
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
fig.suptitle("Figure 1: Interrupted Time Series — Pre vs Post AI Tool Adoption",
             fontsize=13, fontweight='bold')

for ax, ts, model, metric, ylabel, title in [
    (axes[0], its_vol_ts, its_vol_model, 'volume_val',
     'MR Volume (per week)', 'H1: Pull Request Volume'),
    (axes[1], its_tt_ts, its_tt_model, 'time_to_first_review_hrs',
     'Hours', 'H3: Time to First Review'),
]:
    ts_plot = ts.rename(columns={'created_at': 'date'}) \
                 if 'created_at' in ts.columns else ts

    pre_ts  = ts_plot[ts_plot['date'] < ADOPTION_DATE]
    post_ts = ts_plot[ts_plot['date'] >= ADOPTION_DATE]

    ax.scatter(pre_ts['date'],  pre_ts[metric],
               color='steelblue', s=20, alpha=0.6, label='Pre-adoption (observed)')
    ax.scatter(post_ts['date'], post_ts[metric],
               color='tomato',    s=20, alpha=0.6, label='Post-adoption (observed)')

    # Fitted lines
    ax.plot(pre_ts['date'],
            model.fittedvalues[:len(pre_ts)],
            color='steelblue', linewidth=2, label='Pre-adoption trend')
    ax.plot(post_ts['date'],
            model.fittedvalues[len(pre_ts):],
            color='tomato', linewidth=2, label='Post-adoption trend')

    ax.axvline(ADOPTION_DATE, color='black', linestyle='--',
               linewidth=1.5, label=f'Adoption date ({ADOPTION_DATE.date()})')
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("its_plot.png", dpi=300, bbox_inches='tight')
log_print("\n  Saved: its_plot.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# 6. FIGURE 2 — LORENZ CURVES (Reviewer Concentration)
# ══════════════════════════════════════════════════════════════════════

def lorenz_curve(values):
    values = np.sort(np.array(values, dtype=float))
    cumsum = np.cumsum(values)
    cumsum = cumsum / cumsum[-1]
    x = np.linspace(0, 1, len(cumsum))
    return x, cumsum

fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle("Figure 2: Lorenz Curves — Senior Engineer Review Load Concentration",
             fontsize=12, fontweight='bold')

x_pre,  y_pre  = lorenz_curve(pre_loads)
x_post, y_post = lorenz_curve(post_loads)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect equality')
ax.plot(x_pre,  y_pre,  color='steelblue', linewidth=2,
        label=f'Pre-adoption  (Gini = {gini_pre:.3f})')
ax.plot(x_post, y_post, color='tomato',    linewidth=2,
        label=f'Post-adoption (Gini = {gini_post:.3f})')

ax.fill_between(x_pre,  y_pre,  [0]*len(x_pre),
                alpha=0.08, color='steelblue')
ax.fill_between(x_post, y_post, [0]*len(x_post),
                alpha=0.08, color='tomato')

ax.set_xlabel("Cumulative share of senior engineers\n(ranked by review volume, ascending)")
ax.set_ylabel("Cumulative share of total reviews")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("lorenz_curves.png", dpi=300, bbox_inches='tight')
log_print("  Saved: lorenz_curves.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# 7. DESCRIPTIVE STATISTICS TABLE (Table I)
# ══════════════════════════════════════════════════════════════════════

log_print("\n" + "=" * 65)
log_print("TABLE I — DATASET COMPOSITION")
log_print("=" * 65)

pre_days  = (pre['created_at'].max()  - pre['created_at'].min()).days
post_days = (post['created_at'].max() - post['created_at'].min()).days

table1 = pd.DataFrame({
    'Dimension'           : ['Duration (days)', 'Total MRs',
                             'Active contributors', 'Senior engineers (SE)',
                             'Repositories'],
    'Pre-Adoption (P₀)'   : [pre_days,  len(pre),
                             pre['author_id'].nunique(),  len(senior_ids),
                             pre['project_id'].nunique()],
    'Post-Adoption (P₁)'  : [post_days, len(post),
                             post['author_id'].nunique(), len(senior_ids),
                             post['project_id'].nunique()],
})
log_print(table1.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# 8. WRITE FULL LOG
# ══════════════════════════════════════════════════════════════════════

with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(log))

print(f"\nDone. Outputs written:")
print(f"  results_summary.csv  — Table IV for paper")
print(f"  its_plot.png         — Figure 1")
print(f"  lorenz_curves.png    — Figure 2")
print(f"  {OUTPUT_FILE}        — Full statistical log")
