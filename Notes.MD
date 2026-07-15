# Financial Sentinel — Master Notes

**Purpose:** single source of truth for revision, separate from the working notebook. No code outputs here — just concepts, in plain language. Revisit every 2-3 days: cover a heading, explain it out loud first, then check below.

## Table of Contents
1. EDA (Day 16-17)
2. Correlation & Boxplots
3. Preprocessing (Day 16, Session 2)
4. Model Evaluation (Day 21, Session 2)
5. Quick Recall Self-Test

---

## 1. EDA — Day 16-17

### Dataset Verification
- Shape: (284807, 31) — confirmed, matches expected
- `.info()` confirms ZERO null values across all 31 columns
- All columns numeric (float64 except Class, which is int64)

### Column Meanings
- `Time` — seconds elapsed since first transaction
- `V1–V28` — PCA-anonymized features (privacy/legal requirement)
- `Amount` — transaction value, NOT anonymized
- `Class` — target variable. 0 = legit, 1 = fraud

### Class Imbalance — the central challenge
- Legit: 284,315 (99.83%) | Fraud: 492 (0.17%)
- Less than 1 in 500 transactions is fraud
- A model predicting "always legit" gets 99.83% accuracy while catching **zero** frauds
- **Lesson:** accuracy is the wrong metric here — need Precision/Recall instead

### Amount Distribution
- Mean: 88.35 | Median: 22.00 | Max: 25,691.16 | Min: 0.00
- Mean >> Median → right-skewed distribution, a few extreme outliers pull the mean up
- Fraud median (9.25) < legit median (22.00), but fraud mean (122) > legit mean (88) → fraud has its own outliers, possibly two fraud strategies (small testing amounts + occasional large hits)

### Habits Locked In
- Always verify `.shape` after loading — catches silent load failures
- Column names are case-sensitive (`df['class']` ≠ `df['Class']`)
- Check `.columns.tolist()` before referencing column names
- Form a hypothesis before running code, then compare to reality

---

## 2. Correlation & Boxplots

### Correlation
- Range: -1 to +1, measures **straight-line** relationships only
- 0 ≠ "no relationship" — only means "no LINEAR relationship." Curved patterns are invisible to this metric
- Top fraud-correlated features: V17 (-0.326), V14 (-0.302), V12 (-0.260)
- No single feature is strongly correlated (all under 0.33 in magnitude) → fraud needs multiple features combined

### Boxplot — 5 parts
1. Min (bottom whisker)
2. 25th percentile (bottom of box)
3. Median (line inside box)
4. 75th percentile (top of box)
5. Max (top whisker)
   + Outlier dots = extreme values beyond the whiskers

### V17 / V14 Findings
- Legit (Class 0): tiny box near 0, hundreds of outlier dots scattered above/below (large sample size → outliers show as individual dots instead of stretching the box)
- Fraud (Class 1): bigger box, shifted lower, fewer outlier dots (small sample size → spread gets absorbed into the box itself)
- **Important:** full ranges of fraud vs legit overlap heavily — no single feature gives certainty on its own

### Big Picture
No single feature can perfectly separate fraud from legit. This is exactly why fraud detection needs an ML model combining many features — not a simple rule.

---

## 3. Preprocessing — Day 16, Session 2

### Why Preprocessing is Needed
1. Scale mismatch: Amount (0–25,691) vs V1–V28 (-5 to +5)
2. No train/test separation yet
3. Severe class imbalance (492 fraud vs 284,315 legit)

### Feature Scaling — StandardScaler
> `scaled_value = (original_value - mean) / std`

Called **standardization** (mean=0, std=1) — not normalization (which squeezes into a fixed 0–1 range via a different method, e.g. MinMaxScaler).

Scaling repositions data, does **not** fix skew/shape — Amount stayed right-skewed even after scaling.

Only `Amount` and `Time` needed scaling — V1–V28 are already PCA-transformed, naturally centered near 0.

**Fit vs transform:**
- `fit()` — calculates and stores mean/std inside the scaler object
- `transform()` — applies that stored mean/std
- `fit_transform()` — does both in one call (used during training)
- Calling `fit()` again on the same scaler OVERWRITES the previous mean/std — use separate scaler objects per column
- At deployment: only ever `.transform()` new data, reusing the saved training scaler — never re-fit

```python
amount_scaler = StandardScaler()
df['Amount_scaled'] = amount_scaler.fit_transform(df[['Amount']])

time_scaler = StandardScaler()
df['Time_scaled'] = time_scaler.fit_transform(df[['Time']])
```

(Double brackets `df[['Amount']]` needed as sklearn input — requires 2D shape. Single brackets fine for naming new columns on the left of `=`.)

### Train/Test Split
**Why:** testing on the same data used for training tells you nothing — the model could just memorize answers (overfitting) instead of learning real patterns.

**The imbalance risk in plain random splitting:** with only 492 fraud rows, random chance could create an uneven split (e.g. most fraud landing in one set).

**Fix — `stratify` parameter:** forces both train and test sets to keep the same class proportion as the original dataset.

```python
from sklearn.model_selection import train_test_split

X = df.drop(['Class', 'Time', 'Amount'], axis=1)   # drop raw, unscaled columns
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Verified:** both train (0.173% fraud) and test (0.172% fraud) closely matched the original 0.17% ratio.

**Real bug hit:** forgot to drop original unscaled `Time`/`Amount` before splitting — model saw duplicate signal (scaled + unscaled), caused a convergence warning. Fixed by dropping both raw columns from `X`.

### Handling Class Imbalance — SMOTE
**Problem:** a model trained on imbalanced data has little incentive to learn fraud patterns — predicting "always legit" barely hurts its score.

**Why not undersampling:** throwing away 227,000+ legit rows to match 394 fraud rows = massive information loss.

**Why not simple duplication:** copy-pasting the same 394 fraud rows repeatedly causes the model to memorize those exact rows instead of the general pattern.

**SMOTE:** generates new *synthetic* fraud examples by interpolating between real fraud points in feature space — not copies, plausible new variations.

**Critical rule:** SMOTE applies only to training data, never to test data. The test set must stay 100% real, preserving the true real-world imbalance.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**Verified:**
- `y_train_resampled`: 227,451 legit / 227,451 fraud (balanced)
- `y_test`: 56,864 legit / 98 fraud (untouched, original imbalance preserved)

### Final Pipeline State
```
X_train_resampled, y_train_resampled  →  balanced, used to TRAIN
X_test, y_test                         →  real imbalance, used to EVALUATE
```

---

## 4. Model Evaluation — Day 21, Session 2

### The Guard Story (core mental model — return here whenever confused)
A security guard stands at a door. 56,962 people walk through (test set). 98 are actual thieves (fraud). 56,864 are honest (legit).

The guard points at 1,548 people he thinks are thieves.
- Of those 1,548 accusations, only 90 were actual thieves.
- Of the 98 real thieves, he caught 90, missed 8.

This guard = the trained Logistic Regression model.

### Confusion Matrix
Four possible outcomes for any prediction:
1. Predicted fraud, actually fraud → **True Positive** (correct catch)
2. Predicted legit, actually legit → **True Negative** (correct pass)
3. Predicted fraud, actually legit → **False Positive** (false alarm)
4. Predicted legit, actually fraud → **False Negative** (missed fraud — the dangerous one)

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

|                  | Predicted Legit | Predicted Fraud |
|------------------|:----------------:|:----------------:|
| **Actual Legit** | 55,406            | 1,458             |
| **Actual Fraud** | 8                  | 90                 |

### Recall — "out of all REAL thieves, how many did I catch?"
> Recall = TP / (TP + FN) = 90 / 98 = **0.918 (91.8%)**

Strong — caught 92% of real fraud.

### Precision — "out of all my ACCUSATIONS, how many were correct?"
> Precision = TP / (TP + FP) = 90 / 1548 = **0.058 (5.8%)**

Weak — only 6% of fraud accusations were actually correct. 1,458 innocent transactions wrongly flagged.

### The Tradeoff
A more aggressive guard (wider net) catches more real thieves (higher recall) but wrongly accuses more innocent people too (lower precision). Being more cautious flips this the other way. Can't max both without a genuinely better model — there's always a tradeoff to manage, not eliminate.

### F1-score — punishes lopsidedness
> F1 = 2 × (Precision × Recall) / (Precision + Recall)

Does NOT average the two numbers evenly — drags the score toward whichever is WEAKER. 92% recall + 6% precision does not land at "50/50" — it gets dragged down near the bad number.

**My own analogy:** like a DSA OA — no matter how good your dev skills are, if you fail the OA, you don't get hired. The final outcome doesn't average your strengths with your weak spot; the weak spot gates the result.

**Result:** F1 (fraud class) = **0.11** — confirms severe precision weakness despite strong recall.

### Why Accuracy is Meaningless Here (proven, not theoretical)
- Lazy "always predict legit" model: 99.83% accuracy, 0 frauds caught
- Real, working model: 97.42% accuracy, 90 frauds caught
- The WORSE-accuracy model is the one that's actually useful

### classification_report() — automates the by-hand math above
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```
Same precision/recall numbers as calculated by hand, just formatted — no new concept, just a printout.

### Baseline Established
**Logistic Regression** — Recall: 91.8% | Precision: 5.8% | F1 (fraud): 0.11
This is the reference point every future model gets compared against.

**Next step:** try Random Forest (better suited for non-linear patterns) and/or threshold tuning, then compare against this baseline using Recall/Precision/F1 — never accuracy.

---

## 5. Quick Recall Self-Test
*(Cover the sections above. Answer out loud first. Then check.)*

1. Why can a model with 99.83% accuracy still be completely useless?
2. What does a correlation of -0.33 actually tell you, and what is it BLIND to?
3. Why can fraud and legit boxplots overlap even on the strongest feature?
4. Why use `StandardScaler` instead of leaving Amount/Time raw?
5. Why is `fit()` only called on training data, never on test data?
6. Why does SMOTE only touch the training set, never the test set?
7. In the guard story, what's the difference between Recall and Precision?
8. Why does F1-score punish a lopsided guard instead of averaging the two numbers?
9. In one breath: why is accuracy the wrong metric for this project?