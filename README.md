# Replication Package: The Reviewer's Burden

**Paper:** *The Reviewer's Burden: Code Review Load on Senior Engineers After AI Tool Adoption*
**Author:** Gaurav Bhatnagar
**Venue:** PeerJ Computer Science (under review)
**DOI:** *(to be added upon acceptance)*

---

## Overview

This repository contains the anonymized dataset and analysis scripts supporting the above paper. The study quantifies the impact of organizational AI coding tool adoption on code review workload borne by senior engineers, using a quasi-experimental before-after design applied to 12 months of enterprise Git repository telemetry.

**Key findings:**
- MR volume increased 204% post-adoption (Cohen's d = 2.07, p = 2.99 × 10⁻⁵)
- Time to merge increased 105% (Mann-Whitney U = 32,686, p = 7.63 × 10⁻⁵)
- Review coverage expanded 5.8-fold: 4.7% → 26.9% of MRs entering formal senior review
- Reviewer Gini coefficient increased from 0.083 → 0.261 (directional; n = 6 too small to reach significance)

---

## Repository Structure

```
reviewer-burden-replication/
├── data/
│   ├── mr_metrics.csv            # One row per merge request
│   ├── review_notes.csv          # One row per review note/comment
│   ├── approvals.csv             # One row per MR approval event
│   └── contributor_profiles.csv  # One row per contributor (seniority classification)
├── figures/
│   ├── figure1_its_volume.png    # Figure 1: ITS for MR volume (H1)
│   └── figure2_lorenz.png        # Figure 2: Lorenz curves (H5)
├── statisticalAnalysis.py        # Full hypothesis testing (H1–H6)
├── create_figures.py             # Generates publication-quality figures
├── requirements.txt
└── README.md
```

---

## Data Description

All contributor usernames have been anonymized. Senior engineers are labeled `SE_01`–`SE_05` plus the corresponding author (`gbhatnagar`). Non-senior contributors are labeled `NS_01`–`NS_06`. Numeric author IDs are preserved for joining across tables. MR titles and source branch names have been removed as they contained internal ticket identifiers.

### `data/mr_metrics.csv`

| Column | Type | Description |
|:---|:---|:---|
| period | string | `Pre-AI` or `Post-AI` |
| mr_id | int | Merge request ID |
| project_id | int | Repository ID |
| author_id | int | Anonymized contributor ID |
| author_username | string | Anonymized username (SE_0N, NS_0N, or gbhatnagar) |
| created_at | datetime | MR creation timestamp (UTC) |
| merged_at | datetime | MR merge timestamp (UTC) |
| lines_added | int | Lines added (null — not collected via API) |
| lines_deleted | int | Lines deleted (null — not collected via API) |
| files_changed | int | Files changed (null — not collected via API) |
| target_branch | string | Target integration branch |
| time_to_merge_hrs | float | Hours from creation to merge |
| churn_ratio | float | lines_added / lines_deleted (null) |
| first_review_at | datetime | Timestamp of first review event (null if no review recorded) |
| time_to_first_review_hrs | float | Hours from creation to first review |
| review_round_trips | int | Number of review approval cycles |

### `data/review_notes.csv`

| Column | Type | Description |
|:---|:---|:---|
| period | string | `Pre-AI` or `Post-AI` |
| mr_id | int | Merge request ID |
| project_id | int | Repository ID |
| reviewer_id | int | Anonymized reviewer ID (joins to author_id in contributor_profiles) |
| created_at | datetime | Review note timestamp (UTC) |
| resolvable | bool | Whether the note is a resolvable thread |
| resolved | bool | Whether the thread was resolved |

### `data/approvals.csv`

| Column | Type | Description |
|:---|:---|:---|
| period | string | `Pre-AI` or `Post-AI` |
| mr_id | int | Merge request ID |
| project_id | int | Repository ID |
| approver_id | int | Anonymized approver ID (joins to author_id in contributor_profiles) |

### `data/contributor_profiles.csv`

| Column | Type | Description |
|:---|:---|:---|
| author_id | int | Anonymized contributor ID (primary key) |
| author_username | string | Anonymized label (SE_0N, NS_0N, or gbhatnagar) |
| mrs_authored | int | Total MRs authored in study window |
| approvals_given | int | Total approval events given |
| notes_given | int | Total review notes posted |
| approval_to_author_ratio | float | approvals_given / mrs_authored |
| review_to_author_ratio | float | notes_given / mrs_authored |
| project_breadth | int | Count of distinct repositories active in |
| score | int | Count of seniority proxy signals satisfied (0–5) |
| is_senior | bool | True if score ≥ 3 |

---

## Seniority Classification

Senior engineer status is derived from five proxy signals (score ≥ 3 = senior):

1. Approval authority: ≥ 20 recorded approval events
2. Approval-to-author ratio: ≥ 1.0
3. Cross-repository breadth: ≥ 3 repositories
4. Authoring volume: ≥ 60 MRs authored
5. Review-to-author ratio: ≥ 1.5

Classification is applied retrospectively and held constant across both periods.

---

## Intervention Boundary

- **Adoption date:** January 1, 2026
- **Washout window:** ±15 days (December 17, 2025 – January 15, 2026 excluded)
- **Pre-adoption period (P₀):** July 1 – December 16, 2025 (168 days, 171 MRs)
- **Post-adoption period (P₁):** January 16 – June 30, 2026 (164 days, 480 MRs)

---

## Requirements

```
python >= 3.9
pandas
numpy
scipy
statsmodels
matplotlib
```

Install:

```bash
pip install -r requirements.txt
```

---

## Reproducing the Analysis

```bash
# Clone the repository
git clone https://github.com/gaurav-bhat/reviewer-burden-replication.git
cd reviewer-burden-replication

# Install dependencies
pip install -r requirements.txt

# Run full statistical analysis
python statisticalAnalysis.py
# → writes full_results.txt and results_summary.csv

# Regenerate publication figures
python create_figures.py
# → writes figure1_its_volume.png, figure2_lorenz.png, figure1_its_ttfr.png
```

---

## Ethics and Anonymization

Git telemetry data was collected from enterprise repositories with organizational authorization. No human subjects research was conducted. All contributor usernames have been replaced with anonymized labels (SE_0N / NS_0N) prior to deposit. MR titles and source branch names have been removed as they contained internal ticket identifiers. Numeric IDs (author_id, mr_id, project_id) are preserved for data integrity; they do not directly identify individuals without access to the originating GitLab instance.

---

## Citation

If you use this dataset or code, please cite:

```
Bhatnagar, G. (2026). The Reviewer's Burden: Code Review Load on Senior Engineers
After AI Tool Adoption. PeerJ Computer Science. DOI: [to be added]
```

---

## License

- **Code** (`statisticalAnalysis.py`, `create_figures.py`): [MIT License](LICENSE)
- **Data** (`data/`): [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
