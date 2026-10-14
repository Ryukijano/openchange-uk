"""Independent stdlib-only verification of the audit's match-count gate."""

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / 'sds-benchmark'
PROFILES = ['-400', '-300', '-200', '-100']


def stamp(value):
    out = datetime.fromisoformat(value)
    return out.replace(tzinfo=UTC) if out.tzinfo is None else out.astimezone(UTC)


def main():
    all_dates = [stamp(path.stem[-10:]) for path in (HERE / 'raw/TRUCVERT').glob('Monitoring*.nc')]
    contour_dates = {profile: [] for profile in PROFILES}
    with (HERE / 'survey_contours_TRUCVERT.csv').open() as source:
        for row in csv.DictReader(source):
            contour_dates[row['profile']].append(stamp(row['survey_date']))
    observations = set()
    for profile in PROFILES:
        for folder in ['tidally_corrected_timeseries_MSL', 'tidally_corrected_timeseries_MSL_S2']:
            path = REPO / f'submissions/team_CoastSat/TRUCVERT/{folder}/{profile}_timeseries_tidally_corrected.csv'
            if not path.exists():
                continue
            with path.open() as source:
                for row in csv.DictReader(source):
                    try:
                        position = float(row[profile])
                    except ValueError:
                        continue
                    if math.isfinite(position):
                        observations.add((stamp(row['dates']), 'S2' if folder.endswith('_S2') else row['satname'], profile))
    rows, potential = [], []
    for date, sensor, profile in sorted(observations):
        raw_gap = min(abs((date - survey).total_seconds()) for survey in all_dates)
        gap = min(abs((date - survey).total_seconds()) for survey in contour_dates[profile])
        if raw_gap <= 48 * 3600:
            potential.append((date, sensor, profile))
        if gap <= 48 * 3600:
            rows.append({'date': date.isoformat(), 'sensor': sensor, 'profile': profile, 'gap_hours': gap / 3600,
                         'period': 'fit' if date.year <= 2009 else 'cal' if date.year <= 2016 else 'test'})
    counts = {period: {'rows': len([r for r in rows if r['period'] == period]),
                      'scenes': len({(r['date'], r['sensor']) for r in rows if r['period'] == period})}
              for period in ['fit', 'cal', 'test']}
    primary = json.loads((HERE / 'data_audit.json').read_text())['TRUCVERT']['counts']
    assert len(rows) == primary['matched_48h_rows']
    for period, result in counts.items():
        assert result['rows'] == primary['periods'][period]['rows']
        assert result['scenes'] == primary['periods'][period]['scenes']
    output = {'independent_implementation': 'csv+datetime; no pandas/merge_asof',
              'potential_rows_before_contour_quality': len(potential),
              'potential_scene_groups_before_contour_quality': len({r[:2] for r in potential}),
              'matched_rows': rows, 'counts': counts, 'agrees_with_primary_audit': True,
              'registered_folds': [
                  {'test_site': 'NARRABEEN', 'blocking_truc_partition': 'cal', 'eligible': False},
                  {'test_site': 'TRUCVERT', 'blocking_truc_partition': 'test', 'eligible': False},
                  {'test_site': 'TORREYPINES', 'blocking_truc_partition': 'fit', 'eligible': False}],
              'status': 'insufficient data; no model comparison or primary outcome computed'}
    (HERE / 'independent_count_check.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
