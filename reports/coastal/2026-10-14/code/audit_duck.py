"""Realised Duck counts audit for narrabeen-duck-20261014. No model fitting."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import audit_surveys
from audit_surveys import (
    build_observations,
    projected_chainage,
    transect_geometry,
    unique_crossing,
)

audit_surveys.PROFILES['DUCK'] = ['-91', '1', '1006', '1097']
audit_surveys.SLOPES['DUCK'] = .1

HERE = Path(__file__).resolve().parent
PROFILES = ['-91', '1', '1006', '1097']
MSL_NAVD88 = -0.128


def load_duck():
    project, geometries = transect_geometry('DUCK', 32119)
    rows, rejected, examples = [], 0, []
    files = sorted((HERE / 'raw/DUCK').glob('*.nc'))
    for file in files:
        survey_date = pd.Timestamp(datetime.strptime(file.stem[-8:], '%Y%m%d').replace(tzinfo=timezone.utc))
        with Dataset(file) as ds:
            prof = np.asarray(ds['profileNumber'][:], dtype=float)
            lon = np.asarray(ds['lon'][:], dtype=float)
            lat = np.asarray(ds['lat'][:], dtype=float)
            elev = np.asarray(ds['elevation'][:], dtype=float)
        for name in PROFILES:
            mask = prof == float(name)
            if mask.sum() < 3:
                rejected += 1
                continue
            chain, off = projected_chainage(lon[mask], lat[mask], project, geometries[name])
            z = elev[mask]
            valid = np.isfinite(chain) & np.isfinite(z)
            x, z = chain[valid], z[valid]
            crossing = unique_crossing(x, z, MSL_NAVD88) if len(x) >= 3 else None
            if crossing is None:
                rejected += 1
                continue
            rows.append({'profile': name, 'survey_date': survey_date,
                         'truth': crossing,
                         'max_off_axis_m': float(np.nanmax(np.abs(off[valid])))})
            if len(examples) < 15:
                examples.append((x, z, crossing, str(survey_date.date()), name))
    surveys = pd.DataFrame(rows)
    return surveys, {'files': len(files), 'rejected_profile_dates': rejected,
                     'profile_dates_checked': rejected + len(rows)}, examples


def main():
    surveys, record, _ = load_duck()
    print('survey contour rows:', len(surveys), record)
    surveys.to_csv(HERE / 'survey_contours_DUCK.csv', index=False)
    data, counts = build_observations('DUCK', surveys)
    data.to_csv(HERE / 'matched_observations_DUCK.csv', index=False)
    record['survey_rows'] = len(surveys)
    record['counts'] = counts
    (HERE / 'data_audit_duck.json').write_text(json.dumps({'DUCK': record}, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
