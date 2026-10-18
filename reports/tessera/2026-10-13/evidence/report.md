# TESSERA compatibility pilot

348 parcels; 174 train west, 174 test east; year 2024.
Model+run migration falsifier, not an operational result.

| Head | Route | Macro-F1 | Balanced acc | Churn vs old features | ECE-10 |
|---|---|---:|---:|---:|---:|
| logreg_balanced | old_features | 0.164 | 0.184 | 0.000 | 0.355 |
| logreg_balanced | new_unmapped | 0.033 | 0.107 | 0.931 | 0.690 |
| logreg_balanced | new_procrustes | 0.116 | 0.129 | 0.603 | 0.142 |
| logreg_balanced | new_ridge | 0.136 | 0.151 | 0.322 | 0.332 |
| logreg_balanced | new_head_refit | 0.142 | 0.181 | 0.471 | 0.271 |
| rf_balanced | old_features | 0.127 | 0.160 | 0.000 | 0.104 |
| rf_balanced | new_unmapped | 0.061 | 0.100 | 0.270 | 0.180 |
| rf_balanced | new_procrustes | 0.060 | 0.099 | 0.264 | 0.053 |
| rf_balanced | new_ridge | 0.124 | 0.150 | 0.098 | 0.128 |
| rf_balanced | new_head_refit | 0.111 | 0.147 | 0.178 | 0.111 |

RF macro-F1 bootstrap mean [2.5%,97.5%]:
- new_ridge: 0.125 [0.101, 0.153]
- new_procrustes: 0.061 [0.053, 0.073]
- old_features: 0.128 [0.104, 0.156]
- new_unmapped: 0.062 [0.054, 0.073]
- new_head_refit: 0.112 [0.090, 0.133]
