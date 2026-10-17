"""narrabeen-duck-20261014: baselines then learned error heads, CPU only."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
FEATURES = ['tide', 'swh', 'mwp', 'pp1d', 'dir_sin', 'dir_cos', 'month_sin', 'month_cos',
            'setup', 'slope', 'sensor_L5', 'sensor_L7', 'sensor_L8', 'sensor_S2']
RETAIN = 0.70


def load(site):
    f = HERE / ('matched_observations_DUCK.csv' if site == 'DUCK' else 'matched_observations.csv')
    df = pd.read_csv(f, low_memory=False)
    return df[df.site == site].copy()


def scene_scores(df, score):
    g = df.groupby('scene', sort=False)[score].mean()
    return g


def retained(df, score_col, frac=RETAIN):
    g = scene_scores(df, score_col).sort_values()
    keep = set(g.index[: max(1, int(np.floor(frac * len(g))))])
    return df[df.scene.isin(keep)]


def rmse(df):
    return float(np.sqrt(np.mean(np.square(df.error))))


def baseline_scores(df):
    return {
        'unfiltered': pd.Series(0.0, index=df.index),
        'tide_only': df.tide.abs(),
        'wave_only': df.swh,
        'sensor_S2_pref': (~df.sensor.eq('S2')).astype(float),  # keep S2 scenes first
        'disagreement_only': df.disagreement.abs(),
    }


def fit_head(fit, features, quantile, seed):
    counts = fit.groupby('scene')['scene'].transform('count')
    w = 1.0 / counts
    m = HistGradientBoostingRegressor(loss='quantile', quantile=quantile, max_iter=100,
                                    max_leaf_nodes=15, learning_rate=0.07, min_samples_leaf=20,
                                    l2_regularization=1.0, early_stopping=False, random_state=seed)
    m.fit(fit[features], fit.error, sample_weight=w)
    return m


def run_fold(fit_site, test_site, seeds=(0, 1, 2)):
    fit_all, test_all = load(fit_site), load(test_site)
    fit = fit_all[fit_all.period == 'fit']
    cal = fit_all[fit_all.period == 'cal']
    test = test_all[test_all.period == 'test']
    out = {'fit_site': fit_site, 'test_site': test_site,
           'n': {'fit_rows': len(fit), 'fit_scenes': int(fit.scene.nunique()),
                 'cal_rows': len(cal), 'cal_scenes': int(cal.scene.nunique()),
                 'test_rows': len(test), 'test_scenes': int(test.scene.nunique())}}

    # cheap baselines: score each partition, retain ~70%, evaluate retained RMSE
    for name in ['unfiltered', 'tide_only', 'wave_only', 'sensor_S2_pref', 'disagreement_only']:
        cal_s = baseline_scores(cal)[name]
        test_s = baseline_scores(test)[name]
        if name == 'disagreement_only':
            cal_s = cal_s.fillna(np.inf); test_s = test_s.fillna(np.inf)
        cal_r = retained(cal.assign(_s=cal_s.values), '_s')
        test_r = retained(test.assign(_s=test_s.values), '_s')
        out[f'baseline_{name}'] = {'cal_rmse': rmse(cal_r), 'test_rmse': rmse(test_r),
                                   'test_scenes_kept': int(test_r.scene.nunique()),
                                   'test_coverage': test_r.scene.nunique() / test.scene.nunique()}
    # calibration-selected baseline: lowest cal retained RMSE
    best = min((k for k in out if k.startswith('baseline_')),
               key=lambda k: out[k]['cal_rmse'])
    out['calibration_selected_baseline'] = best
    out['baseline_test_rmse'] = out[best]['test_rmse']

    # learned heads
    for cond, feats in [('metadata', FEATURES), ('metadata+disagreement', FEATURES + ['disagreement'])]:
        for seed in seeds:
            q50 = fit_head(fit, feats, 0.5, seed)
            q05 = fit_head(fit, feats, 0.05, seed)
            q95 = fit_head(fit, feats, 0.95, seed)
            for split, df in [('cal', cal), ('test', test)]:
                d = df.copy()
                d['_pred_err'] = q50.predict(d[feats])
                d['_lo'], d['_hi'] = q05.predict(d[feats]), q95.predict(d[feats])
                d['_abs_pred'] = d._pred_err.abs()
                r = retained(d, '_abs_pred')
                out[f'{cond}_seed{seed}_{split}'] = {
                    'retained_rmse': rmse(r), 'scenes_kept': int(r.scene.nunique()),
                    'coverage': r.scene.nunique() / df.scene.nunique(),
                    'median_width': float((r._hi - r._lo).median()),
                    'interval_coverage': float(np.mean((r.error >= r._lo) & (r.error <= r._hi)))}
        s0 = out[f'{cond}_seed0_test']['retained_rmse']
        out[f'{cond}_rel_reduction'] = 1 - s0 / out['baseline_test_rmse']
    return out


def main():
    t0 = time.time()
    results = {'protocol': 'narrabeen-duck-20261014', 'retention': RETAIN,
               'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0))}
    for name, f, t in [('fold_A_fit_duck_test_narrabeen', 'DUCK', 'NARRABEEN'),
                       ('fold_B_fit_narrabeen_test_duck', 'NARRABEEN', 'DUCK')]:
        results[name] = run_fold(f, t)
    reds = [results[k]['metadata+disagreement_rel_reduction'] for k in results if k.startswith('fold_')]
    results['primary_outcome'] = {
        'mean_relative_reduction': float(np.mean(reds)), 'per_fold': reds,
        'prediction': '>=0.15 on both new test sites (project gate)',
        'passes': all(r >= 0.15 for r in reds)}
    results['elapsed_seconds'] = time.time() - t0
    results['model_fit_performed'] = True
    (HERE / 'fold_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results['primary_outcome'], indent=2))


if __name__ == '__main__':
    main()
