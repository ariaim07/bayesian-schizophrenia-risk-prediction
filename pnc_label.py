"""
Psychosis-spectrum (PS) label for the Philadelphia Neurodevelopmental Cohort
(dbGaP phs000607, table pht003445 Subject_Phenotypes).

Use once you have authorized access to the individual-level table.

APPROXIMATION of the published PNC psychosis-spectrum definition
(Calkins et al., Penn BBL). Before publishing, replace the thresholds below with
the exact algorithm from the paper, ideally confirmed with the Penn group.

Codes (from the dbGaP variable report):
  PRIME items SIP003-SIP014: 0 Definitely disagree ... 6 Definitely agree; NA/9 = missing
  Gate items PSY001/020/029/050/060/070/071: 0 No, 1 Yes, 9 Unknown, NA
  PSY086 (conviction): 1 = sure at the time
  PSY101 (ill) / PSY103 (drugs): 1 Never, 2 Sometimes, 3 Always
  INT_TYPE: MP = self-report 11-17, AP = self-report 18+, YPI = parent report for 8-10,
            MI = parent report about an 11-17 year old (a second record for that child)
"""
import numpy as np
import pandas as pd

PRIME = [f"SIP{i:03d}" for i in range(3, 15)]
HALLUCINATION_GATES = ["PSY001", "PSY020", "PSY029", "PSY050", "PSY060"]
DELUSION_GATES = ["PSY070", "PSY071"]

# Never use these as predictors: they are consequences or parts of the outcome.
LEAKY = (["SCR001", "SCR003", "SCR004", "SCR005", "SCR006", "SCR007", "SCR008",
          "GAF001", "GAF002", "GAF003", "PSY106"]
         + [c for c in [f"SIP{i:03d}" for i in range(1, 45)]]
         + [f"PSY{i:03d}" for i in range(1, 161)])


def _num(s):
    """Coded values -> float; NA / 9 (unknown) -> NaN."""
    x = pd.to_numeric(s, errors="coerce")
    return x.where(x != 9)


def one_record_per_person(df):
    """Self-report for 11+ (MP/AP); parent report for 8-10 (YPI). Drops MI records,
    which are a parent's second report on an 11-17 year old (use for sensitivity only)."""
    keep = df[df["INT_TYPE"].isin(["MP", "AP", "YPI"])]
    if keep["SUBJID"].duplicated().any():
        raise ValueError("more than one primary record for some participants")
    return keep.set_index("SUBJID")


def ps_label(df):
    """Returns a Series: 1 = psychosis spectrum, 0 = clearly negative, NaN = excluded
    (in between, or symptoms explained by drugs / physical illness, or too much missing)."""
    d = one_record_per_person(df)
    prime = d[PRIME].apply(_num)
    halluc = d[HALLUCINATION_GATES].apply(_num)
    delus = d[DELUSION_GATES].apply(_num)
    sure = _num(d["PSY086"]) == 1
    drugs_or_ill = (_num(d["PSY103"]) >= 2) | (_num(d["PSY101"]) >= 2)

    subthreshold = (prime == 6).sum(axis=1).ge(1) | (prime >= 5).sum(axis=1).ge(3)
    threshold = (halluc == 1).any(axis=1) | ((delus == 1).any(axis=1) & sure)
    positive = (subthreshold | threshold) & ~drugs_or_ill

    clean_neg = ((prime <= 3).all(axis=1) & (halluc == 0).all(axis=1)
                 & (delus == 0).all(axis=1))
    too_missing = prime.isna().sum(axis=1) > 3

    y = pd.Series(np.nan, index=d.index)
    y[clean_neg] = 0
    y[positive] = 1
    y[too_missing | (drugs_or_ill & (subthreshold | threshold))] = np.nan
    return y


if __name__ == "__main__":
    # Smoke test on a made-up table with the real column names and codes
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"SUBJID": np.arange(n),
                       "INT_TYPE": rng.choice(["MP", "AP", "YPI"], n, p=[.55, .2, .25])})
    for c in PRIME:
        df[c] = rng.choice([0, 1, 2, 3, 4, 5, 6, 9], n, p=[.7, .07, .04, .07, .06, .03, .025, .005])
    for c in HALLUCINATION_GATES + DELUSION_GATES:
        df[c] = rng.choice([0, 1, 9], n, p=[.93, .065, .005])
    df["PSY086"] = rng.choice([0, 1], n)
    df["PSY101"] = rng.choice([1, 2, 3], n, p=[.9, .06, .04])
    df["PSY103"] = rng.choice([1, 2, 3], n, p=[.96, .03, .01])
    mi = df.sample(500, random_state=0).assign(INT_TYPE="MI")
    y = ps_label(pd.concat([df, mi]))
    print(y.value_counts(dropna=False).rename({1.0: "PS", 0.0: "negative"}))
