# Study 2: expectations and analysis plan, fixed before any Study 2 model was run

Recorded: 23 November 2026, after inspecting only variable names, category labels and missingness in the
OPM files (no outcome-by-predictor tabulation, no model). Any change after this point is reported as a deviation.

## Population and outcome
- Base: all employees in the 30 September 2023 Employment snapshot who are full-time (work schedule
  "FULL-TIME") and in tenure group 1 or 2 (career or career-conditional). 10% random sample (seed 42).
- Cases: separations with category "QUIT", effective October 2023 to September 2024, meeting the same
  full-time and tenure-group filters, with length of service at separation minus elapsed months >= 0.
- Design: case-base; one-year quit risk = 0.10 x odds(case vs base).

## Predictors (all present in both files)
Age bracket (midpoint); length of service (adjusted to the snapshot for cases); log annual pay (missing when
redacted) plus a redaction indicator; grade (numeric); step; education (5 ordered levels); supervisory status;
veteran; tenure group 2 (career-conditional, i.e. early-career probationary status); bargaining-unit status;
FLSA exempt; STEM / health / other occupation; occupational category; pay plan group; agency (top 15);
duty-station state (top 12).

## Expectations
- S2-E2 (rewards): lower pay and lower grade raise quit risk, with the pay effect steepest at the low end.
- S2-E4 (ease of movement): younger employees and those with shorter service have higher risk; STEM
  occupations, whose skills transfer readily to the private sector, have higher risk than other occupations.
- S2-E5 (embeddedness): career-conditional (tenure group 2) employees have higher risk; supervisors have
  lower risk.

## Analysis
Identical protocol to Study 1 with the fixed XGBoost configuration: 5-fold cross-fitting (LR and XGBoost),
SHAP on up to 20,000 employees per fold, construct grouping, shadow-feature screen (30 refits on 60,000-row
subsamples; cut-offs 70/80/90%), decile dependence for pay, age and service, age-band calibration audit,
Monte Carlo cost analysis with the Study 1 input distributions.
