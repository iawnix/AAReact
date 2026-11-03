# Selected-model interpretation

## Model

- Model: `XGB@RDKit+xTB+ACSF`
- Split: `seed=1`, `test_size=0.20`
- Train RMSE/R2: `0.0790` / `0.9928`
- Test RMSE/R2: `0.2184` / `0.9427`

## Sign prediction by split

- Train: `264/265` correct among nonzero targets (`99.6%`); true-zero rows excluded: `61`
- Test: `63/71` correct among nonzero targets (`88.7%`); true-zero rows excluded: `11`

## ee-ddG convention

The original ee is defined as `(R - S) / (R + S)`. The dataset uses:

```text
DDG = DeltaG_S^ddagger - DeltaG_R^ddagger
DDG = R * (TEMP + 273.15) * ln((1 + ee) / (1 - ee))
ee = tanh(DDG / (2RT))
```

Therefore, positive ee corresponds to positive DDG, and negative ee corresponds to negative DDG.

Formula check from `/home/iaw/DATA2/AAReact/DataSet/Data_All/full_data_436-20260723.csv`:

- Valid rows: `436`
- Max absolute formula error: `4.877e-12`
- Mean absolute formula error: `1.030e-12`
- Opposite-sign non-zero ee/DDG rows: `0`

## SHAP interpretation rule

- Positive SHAP value raises DDG and pushes the prediction toward R-major / positive ee.
- Negative SHAP value lowers DDG and pushes the prediction toward S-major / negative ee.

## Contribution by descriptor source

- RDKit: 65.5%
- xTB: 14.1%
- ACSF: 11.9%
- Condition: 8.5%

## Contribution by molecular role / condition

- Catalyst features: 70.4%
- Reactant features: 17.9%
- Pressure: 8.1%
- Solvent features: 3.2%
- Temperature: 0.4%

## Top SHAP features

- 1. `CAT_SlogP_VSA6` (RDKit, mean |SHAP| = 0.1192)
- 2. `PRESSURE` (Condition, mean |SHAP| = 0.1058)
- 3. `CAT_VSA_EState6` (RDKit, mean |SHAP| = 0.0848)
- 4. `REA_BCUT2D_MWHI` (RDKit, mean |SHAP| = 0.0843)
- 5. `REA_MaxAbsEStateIndex` (RDKit, mean |SHAP| = 0.0804)
- 6. `CAT_ACSF8` (ACSF, mean |SHAP| = 0.0683)
- 7. `CAT_XTB17` (xTB, mean |SHAP| = 0.0504)
- 8. `CAT_VSA_EState1` (RDKit, mean |SHAP| = 0.0479)
- 9. `CAT_FractionCSP3` (RDKit, mean |SHAP| = 0.0389)
- 10. `CAT_VSA_EState8` (RDKit, mean |SHAP| = 0.0235)
