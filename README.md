# Predicting 30-Day Hospital Readmission for Diabetes Patients

I wanted to practice building a prediction model on real hospital data and, more importantly, check whether it would be useful. The project predicts which diabetes hospital visits will end in a readmission within 30 days, compares three models, checks the model for fairness, and estimates what a follow-up program could save.

This is a learning project. The data is public and de-identified.

## Data
Diabetes 130-US Hospitals (1999 to 2008), from the UCI Machine Learning Repository (CC BY 4.0). I used a copy on Kaggle. Each row is one hospital visit, not one patient, and some patients appear several times. About 11% of visits ended in a readmission within 30 days.

## What I did
1. Checked data quality. Several columns were mostly empty (`weight` was missing for 97%), so I dropped it.
2. Split train and test by patient, so one patient's visits never appear on both sides. Otherwise the test score would look better than it really is.
3. Removed 2,423 visits where the patient died or went to hospice (discharge codes 11, 13, 14, 19, 20, 21), because these patients cannot be readmitted, or almost never are. I only noticed this when my feature importance check showed `discharge_disposition_id` near the top. I checked the codes against the dataset's lookup file and reran everything. That left 99,343 visits.
4. Compared three models: logistic regression, gradient boosting and random forest, all weighted to pay attention to the rare readmissions.
5. Tuned the gradient boosting model, checked which columns the model relies on, and checked results across race and age groups.
6. Ran a what-if cost simulation using published cost figures.

## Results (19,802 test visits, 2,216 readmissions)
| Model | Recall | Precision | F1 | AUC |
|---|---|---|---|---|
| Logistic regression | 0.562 | 0.178 | 0.271 | 0.657 |
| Gradient boosting | 0.580 | 0.178 | 0.273 | 0.674 |
| Random forest | 0.349 | 0.218 | 0.268 | 0.667 |
| Gradient boosting, tuned | 0.597 | 0.178 | 0.274 | 0.672 |

Always predicting "no readmission" would be about 89% accurate but would catch none of the readmissions, so I looked at recall, precision, F1 and AUC instead of accuracy. The best model caught about 6 in 10 readmissions, but only about 18% of the visits it flagged were real readmissions. That is a modest model: it beats guessing, but it produces many false alarms. Tuning did not improve it, so I think the limit is the data, not the settings.

## What the model relies on
`number_inpatient` (hospital admissions in the previous year) mattered most, then `discharge_disposition_id` (where the patient went after leaving). `age` mattered very little.

## Fairness check
For the large race groups, recall was similar (0.604 for Caucasian, 0.588 for African American). The small groups had too few readmissions to interpret. By age, the model flags older patients much more often (27% of visits at 40 to 50 versus 51% at 80 to 90) while the real readmission rate only rises from about 10% to 12%, so older patients get more false alarms. Recall was lowest at ages 40 to 50 (0.470). This is a screening check, not a full audit.

## Cost simulation (what-if)
Assumptions: a readmission costs \$15,200 (HCUP Statistical Brief #304, endocrine/metabolic, 2020), a follow-up costs \$298 (a published Care Transitions Intervention evaluation), and a follow-up prevents 20% of readmissions (an assumption, close to the 18% average in a meta-analysis of 42 trials by Leppin et al., 2014).

| Visits followed up | Net benefit |
|---|---|
| Top 30% by risk (5,940 visits) | \$1,692,440 |
| Top 50% by risk (9,901 visits) | \$1,831,422 |
| Everyone (19,802 visits) | \$835,644 |

Following up only with the highest-risk visits roughly doubles the savings compared with following up with everyone. The result depends heavily on the assumptions. When follow-ups are cheap or very effective, following up with everyone is almost as good and the model adds little. This is a what-if, not a measured saving.

## What I learned
- Accuracy is misleading when the outcome is rare. A model that always says "no" looks 89% accurate and is useless.
- Leakage can come from two places: the same patient in train and test, and columns that contain the answer or describe something that happens after it.
- Checking feature importance can reveal a flaw in the data, as it did here with patients who had died.
- A modest model can still be valuable if the cost of acting on a prediction is high enough that choosing who to act on matters.

## Limitations
- The data is from 1999 to 2008 and from US hospitals, so it may not reflect current practice.
- AUC is about 0.67, so the model is far from reliable on its own.
- I compared models on one train/test split, and used the test set to compare them, which is acceptable for a learning project but not for a final evaluation.
- The cost simulation relies on an assumed follow-up effectiveness and on cost figures that are not specific to this population.
- Some race and age groups were too small to draw conclusions from.

## Run it
Open the notebook on Kaggle with the Diabetes 130-US Hospitals dataset attached, then run all cells from the top. The cohort filter and the drop of `weight` are in Step 4. Notebook: https://www.kaggle.com/code/paromitadas26/diabetes-readmission
