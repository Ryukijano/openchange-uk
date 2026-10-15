"""Repository-only, single-site feasibility screen; see protocol.md."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "sds-benchmark"
PROFILES = ["PF1", "PF2", "PF4", "PF6", "PF8"]
SEEDS = [0, 1, 2]
FEATURES = ["tide", "swh", "mwp", "pp1d", "dir_sin", "dir_cos", "month_sin",
            "month_cos", "setup", "slope", "sensor_L5", "sensor_L7", "sensor_L8", "sensor_S2"]


def utc(values):
    return pd.to_datetime(values, utc=True, format="mixed").astype("datetime64[ns, UTC]")


def contour(x, z):
    """MSL chainage on benchmark's 1 m grid; no ambiguous/extrapolated labels."""
    data = pd.DataFrame({"x": x, "z": z}).dropna().groupby("x").z.mean().sort_index()
    if len(data) < 3:
        return None
    x, z = data.index.to_numpy(), data.to_numpy()
    crossings = np.flatnonzero((z[:-1] >= 0) & (z[1:] < 0))
    if len(crossings) != 1:
        return None
    i = crossings[0]
    direct = float(x[i] + z[i] / (z[i] - z[i + 1]) * (x[i + 1] - x[i]))
    grid = np.arange(x.min(), x.max() + 0.01, 1)
    zg = np.interp(grid, x, z)
    k = np.flatnonzero((zg[:-1] >= 0) & (zg[1:] < 0))
    if len(k) != 1:
        return None
    i = k[0]
    value = float(grid[i] + zg[i] / (zg[i] - zg[i + 1]))
    return value, direct


def load_surveys():
    p = REPO / "datasets/NARRABEEN/raw/Narrabeen_Profiles.csv"
    raw = pd.read_csv(p)
    rows, excluded = [], 0
    for (profile, date), group in raw.groupby(["Profile ID", "Date"]):
        result = contour(group.Chainage, group.Elevation)
        if result is None:
            excluded += 1
            continue
        rows.append({"profile": profile, "survey_date": date,
                     "truth": result[0], "direct_truth": result[1]})
    out = pd.DataFrame(rows)
    out["survey_date"] = utc(out.survey_date)
    out.to_csv(HERE / "survey_contours.csv", index=False)
    return out, excluded


def load_submission(team, profile, s2=False, raw=False):
    folder = "raw_timeseries" if raw else "tidally_corrected_timeseries_MSL"
    if s2:
        folder += "_S2"
    tail = "raw" if raw else "tidally_corrected"
    p = REPO / f"submissions/{team}/NARRABEEN/{folder}/{profile}_timeseries_{tail}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["date", "position", "sensor", "profile"])
    frame = pd.read_csv(p)
    frame = frame.rename(columns={"dates": "date", profile: "position", "satname": "sensor"})
    frame["date"] = utc(frame.date)
    frame["position"] = pd.to_numeric(frame.position, errors="coerce")
    if s2:
        frame["sensor"] = "S2"
    frame["profile"] = profile
    return frame[["date", "position", "sensor", "profile"]].dropna(subset=["position"])


def match_surveys(frame, survey, hours=48):
    batches = []
    for profile, group in frame.groupby("profile"):
        truth = survey[survey.profile.eq(profile)].drop(columns="profile").sort_values("survey_date")
        matched = pd.merge_asof(group.sort_values("date"), truth, left_on="date", right_on="survey_date",
                                direction="nearest", tolerance=pd.Timedelta(hours=hours))
        batches.append(matched)
    out = pd.concat(batches, ignore_index=True).dropna(subset=["truth"])
    out["gap_hours"] = (out.date - out.survey_date).dt.total_seconds().abs() / 3600
    out["error"] = out.position - out.truth
    return out


def scene_summary(frame):
    return {"rows": len(frame), "scenes": int(frame.scene.nunique()),
            "surveys": int(frame.survey_date.nunique()), "profiles": int(frame.profile.nunique())}


def build_data(survey):
    audits = []
    coast = []
    for team in ["team_CoastSat", "team_SHOREX", "team_CASSIE"]:
        for sensor_set in [False, True]:
            frame = pd.concat([load_submission(team, pf, sensor_set) for pf in PROFILES], ignore_index=True)
            if frame.empty:
                continue
            frame["scene"] = frame.date.astype(str) + "_" + frame.sensor.astype(str)
            matched = match_surveys(frame, survey)
            audits.append({"team": team, "series": "S2" if sensor_set else "Landsat_or_mixed",
                           "available_rows": len(frame), "available_scenes": frame.scene.nunique(),
                           "matches_48h": scene_summary(matched),
                           "matches_36h": scene_summary(match_surveys(frame, survey, 36))})
            if team == "team_CoastSat":
                coast.append(matched)
    data = pd.concat(coast, ignore_index=True).drop_duplicates(["profile", "date", "sensor"])
    spread, rawpos = [], []
    for _, r in data.iterrows():
        spread.append(np.nan)
        rawpos.append(np.nan)
    data["disagreement"] = spread
    data["raw_position"] = rawpos
    for pf in PROFILES:
        subset = data.profile.eq(pf)
        idx = data.index[subset]
        left = data.loc[subset, ["date", "sensor", "position"]].copy()
        left["idx"] = idx
        ref = load_submission("team_SHOREX", pf)
        if not ref.empty:
            right = ref[["date", "sensor", "position"]].rename(columns={"position": "ref_position"})
            pair = pd.merge_asof(left.sort_values("date"), right.sort_values("date"), on="date", by="sensor",
                                 direction="nearest", tolerance=pd.Timedelta(minutes=10))
            data.loc[pair.idx, "disagreement"] = np.abs(pair.position - pair.ref_position).to_numpy()
        for s2 in [False, True]:
            raw = load_submission("team_CoastSat", pf, s2, raw=True)
            if not raw.empty:
                right = raw[["date", "sensor", "position"]].rename(columns={"position": "raw_position"})
                pair = pd.merge_asof(left.sort_values("date"), right.sort_values("date"), on="date", by="sensor",
                                     direction="nearest", tolerance=pd.Timedelta(seconds=1))
                good = pair.raw_position.notna()
                data.loc[pair.loc[good, "idx"], "raw_position"] = pair.loc[good, "raw_position"].to_numpy()
    data = data.sort_values("date").reset_index(drop=True)
    for file, cols, tolerance in [("tides", ["tides"], "20min"),
                                   ("waves_ERA5", ["swh", "mwp", "pp1d", "mwd"], "60min")]:
        ref = pd.read_csv(REPO / f"datasets/NARRABEEN/NARRABEEN_{file}.csv", low_memory=False)
        ref["date"] = utc(ref.dates)
        for c in cols:
            ref[c] = pd.to_numeric(ref[c].astype(str).str.strip("[]"), errors="coerce")
        data = pd.merge_asof(data.sort_values("date"), ref[["date", *cols]].sort_values("date"), on="date",
                             direction="nearest", tolerance=pd.Timedelta(tolerance))
    data = data.rename(columns={"tides": "tide"})
    data["slope"] = 0.1
    data["setup"] = 0.35 * data.slope * np.sqrt(data.swh * 9.81 * data.pp1d**2 / (2 * np.pi))
    data["dir_sin"] = np.sin(np.deg2rad(data.mwd))
    data["dir_cos"] = np.cos(np.deg2rad(data.mwd))
    data["month_sin"] = np.sin(2 * np.pi * data.date.dt.dayofyear / 365.25)
    data["month_cos"] = np.cos(2 * np.pi * data.date.dt.dayofyear / 365.25)
    for name in ["L5", "L7", "L8", "S2"]:
        data[f"sensor_{name}"] = data.sensor.eq(name).astype(float)
    data["split"] = np.where(data.date.dt.year <= 2009, "fit",
                             np.where(data.date.dt.year <= 2016, "cal", "test"))
    return data, audits


def retain(frame, score, fraction=0.7):
    ranking = pd.DataFrame({"scene": frame.scene, "score": score}).groupby("scene").score.mean()
    rank = pd.DataFrame({"score": ranking})
    rank["tie"] = [hashlib.sha256(str(s).encode()).hexdigest() for s in rank.index]
    keep = rank.sort_values(["score", "tie"], na_position="last").index[:math.ceil(len(rank) * fraction)]
    return frame.scene.isin(keep).to_numpy()


def metrics(frame, error, keep=None):
    error = np.asarray(error)
    if keep is None:
        keep = np.ones(len(frame), dtype=bool)
    valid = keep & np.isfinite(error)
    return {"rmse": float(np.sqrt(np.mean(error[valid]**2))), "bias": float(error[valid].mean()),
            "rows": int(valid.sum()), "row_coverage": float(valid.mean()),
            "scene_coverage": float(frame.loc[valid, "scene"].nunique() / frame.scene.nunique())}


def scene_quantile(frame, scores, level=0.9):
    score = pd.DataFrame({"scene": frame.scene, "score": scores}).groupby("scene").score.max().to_numpy()
    rank = min(len(score), math.ceil((len(score) + 1) * level))
    return float(np.sort(score)[rank - 1])


def bootstrap(frame, model_error, model_keep, baseline_error, baseline_keep, n=1000):
    work = pd.DataFrame({"scene": frame.scene, "m": np.asarray(model_error)**2 * model_keep,
                         "mn": model_keep.astype(int), "b": np.asarray(baseline_error)**2 * baseline_keep,
                         "bn": baseline_keep.astype(int)})
    values = work.groupby("scene")[["m", "mn", "b", "bn"]].sum().to_numpy()
    rng = np.random.default_rng(42)
    samples = values[rng.integers(0, len(values), size=(n, len(values)))].sum(axis=1)
    reduction = 1 - np.sqrt(samples[:, 0] / samples[:, 1]) / np.sqrt(samples[:, 2] / samples[:, 3])
    return np.quantile(reduction, [0.025, 0.975]).tolist()


def pair_metrics(test, qlo, qhi):
    rows = []
    frame = test.copy()
    frame["elo"], frame["ehi"] = qlo, qhi
    for _, group in frame.groupby(["profile", "sensor"]):
        group = group.sort_values("date").drop_duplicates("survey_date")
        for i in range(0, len(group) - 1, 2):
            a, b = group.iloc[i], group.iloc[i + 1]
            raw_change = b.position - a.position
            lo, hi = raw_change - b.ehi + a.elo, raw_change - b.elo + a.ehi
            truth = b.truth - a.truth
            claim = -1 if hi < 0 else (1 if lo > 0 else 0)
            rows.append({"profile": a.profile, "sensor": a.sensor, "a": str(a.date), "b": str(b.date),
                         "lo": lo, "hi": hi, "truth": truth, "claim": claim,
                         "wrong": bool(claim and claim * truth <= 0)})
    pair = pd.DataFrame(rows)
    if pair.empty:
        return {}, pair
    claims = pair.claim.ne(0)
    return {"pairs": len(pair), "claims": int(claims.sum()), "claim_coverage": float(claims.mean()),
            "false_direction_rate": float(pair.loc[claims, "wrong"].mean()) if claims.any() else None,
            "interval_coverage": float((pair.truth.ge(pair.lo) & pair.truth.le(pair.hi)).mean()),
            "note": "Correlated/exploratory pairs, not independent beach or 90% joint coverage proof."}, pair


def run():
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__, "repo_commit": subprocess.check_output(
               ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
           "seeds": SEEDS, "input_sha256": {}}
    inputs = [REPO / "datasets/NARRABEEN/raw/Narrabeen_Profiles.csv", HERE / "protocol.md", HERE / "run_stage0.py"]
    inputs += list((REPO / "datasets/NARRABEEN").glob("*.csv"))
    for team in ["team_CoastSat", "team_SHOREX", "team_CASSIE"]:
        inputs += list((REPO / f"submissions/{team}/NARRABEEN").glob("*/*.csv"))
    for p in inputs:
        env["input_sha256"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / "environment.lock").write_text(json.dumps(env, indent=2))
    survey, excluded = load_surveys()
    data, audits = build_data(survey)
    data.to_csv(HERE / "matched_observations.csv", index=False)
    print("MATCH AUDIT", json.dumps(audits), flush=True)
    print("SPLITS", data.groupby("split").size().to_dict(), flush=True)
    fit, cal, test = [data[data.split.eq(s)].copy().reset_index(drop=True) for s in ["fit", "cal", "test"]]
    assert all(len(f) > 30 for f in [fit, cal, test])
    assert not set(fit.scene) & set(cal.scene) and not set(fit.scene) & set(test.scene)
    assert not set(cal.scene) & set(test.scene)
    median = float(fit.error.median())
    def cheap(frame):
        return {"published": (frame.error.to_numpy(), np.zeros(len(frame))),
                "median_bias": ((frame.error - median).to_numpy(), np.zeros(len(frame))),
                "setup": ((frame.error + frame.setup / frame.slope).to_numpy(), np.zeros(len(frame))),
                "tide_rank": (frame.error.to_numpy(), frame.tide.abs().to_numpy()),
                "disagreement_rank": (frame.error.to_numpy(), frame.disagreement.to_numpy())}
    cal_baseline = {name: metrics(cal, err, retain(cal, score)) for name, (err, score) in cheap(cal).items()}
    best = min(cal_baseline, key=lambda key: cal_baseline[key]["rmse"])
    baseline = {name: {"all": metrics(test, err), "retained": metrics(test, err, retain(test, score))}
                for name, (err, score) in cheap(test).items()}
    print("BASELINES", json.dumps(baseline), "CHOSEN ON CAL", best, flush=True)
    result = {"status": "one-site temporal screen only; four-site Stage 0 blocked by missing surveys",
              "survey_excluded": excluded, "survey_rows": len(survey),
              "max_label_grid_difference_m": float((survey.truth - survey.direct_truth).abs().max()),
              "audits": audits, "splits": {s: scene_summary(f) for s, f in [("fit", fit), ("cal", cal), ("test", test)]},
              "feature_missing_fraction": data[FEATURES + ["disagreement"]].isna().mean().to_dict(),
              "calibration_baselines": cal_baseline, "chosen_baseline": best, "test_baselines": baseline,
              "raw_uncorrected_test": metrics(test, test.raw_position - test.truth), "models": []}
    weights = 1 / fit.scene.map(fit.scene.value_counts()).to_numpy()
    weights = weights / weights.mean()
    be, bs = cheap(test)[best]
    bk = retain(test, bs)
    for mode in ["metadata", "metadata_disagreement"]:
        features = FEATURES + (["disagreement"] if mode.endswith("disagreement") else [])
        for seed in SEEDS:
            preds_cal, preds_test = [], []
            for quantile in [0.05, 0.5, 0.95]:
                model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=100,
                          max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.07,
                          l2_regularization=1, early_stopping=False, random_state=seed)
                model.fit(fit[features].to_numpy(), fit.error.to_numpy(), sample_weight=weights)
                preds_cal.append(model.predict(cal[features].to_numpy()))
                preds_test.append(model.predict(test[features].to_numpy()))
            # Sort to enforce ordered quantiles without test-label access.
            cq, tq = np.sort(np.array(preds_cal), axis=0), np.sort(np.array(preds_test), axis=0)
            padding = scene_quantile(cal, np.maximum.reduce([cq[0] - cal.error, cal.error - cq[2], np.zeros(len(cal))]))
            lo, med, hi = tq[0] - padding, tq[1], tq[2] + padding
            err, keep = test.error.to_numpy() - med, retain(test, hi - lo)
            record = {"mode": mode, "seed": seed, "all": metrics(test, err), "retained": metrics(test, err, keep),
                      "calibration_padding_m": padding, "interval_coverage": float((test.error.ge(lo) & test.error.le(hi)).mean()),
                      "interval_width_m": float(np.mean(hi - lo)),
                      "raw_interval_coverage": float((test.error.ge(tq[0]) & test.error.le(tq[2])).mean()),
                      "retained_reduction_vs_cal_selected_baseline": 1 - metrics(test, err, keep)["rmse"] / metrics(test, be, bk)["rmse"],
                      "reduction_scene_bootstrap_95CI": bootstrap(test, err, keep, be, bk),
                      "per_sensor": {sensor: metrics(group, err[group.index]) for sensor, group in test.groupby("sensor")}}
            covered = test.error.ge(lo) & test.error.le(hi)
            record["scene_simultaneous_coverage"] = float(covered.groupby(test.scene).all().mean())
            record["retained_interval_coverage"] = float(covered[keep].mean())
            pairs, pair_frame = pair_metrics(test, lo, hi)
            record["pair_metrics"] = pairs
            result["models"].append(record)
            out = test[["date", "scene", "profile", "sensor", "survey_date", "position", "truth", "error"]].copy()
            out["qlo"], out["qmed"], out["qhi"], out["kept"] = lo, med, hi, keep
            out.to_csv(HERE / f"predictions_{mode}_{seed}.csv", index=False)
            pair_frame.to_csv(HERE / f"pairs_{mode}_{seed}.csv", index=False)
            print("MODEL", json.dumps(record), flush=True)
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
