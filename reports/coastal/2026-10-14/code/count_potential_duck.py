"""Counts-first audit: potential Duck survey-date x CoastSat scene matches.
Metadata only: survey dates come from the public CHL THREDDS catalog filenames;
scene dates come from on-disk benchmark submission CSVs. No payloads downloaded.
Survey dates carry no time-of-day; matching uses date-midnight UTC, so counts are
an upper bound for the same ±48 h contract."""
import csv, json, math, re, urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PROFILES={'DUCK':['-91','1','1006','1097']}
REPO = Path('/home/ubuntu/airr-masterplan/eo-task-research/sds-benchmark')
OUT = Path('/home/ubuntu/airr-masterplan/eo-task-research/coastal-three-site/duck_potential_counts.json')

def stamp(v):
    d = datetime.fromisoformat(v)
    return d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)

url = 'https://chlthredds.erdc.dren.mil/thredds/catalog/frf/geomorphology/elevationTransects/survey/data/catalog.html'
html = urllib.request.urlopen(url, timeout=60).read().decode('utf8', 'ignore')
survey_dates = sorted({datetime.strptime(m, '%Y%m%d').replace(tzinfo=UTC)
                       for m in re.findall(r'survey_(\d{8})\.nc', html)})

obs = set()
for team in ['team_CoastSat']:
    for folder in ['tidally_corrected_timeseries_MSL', 'tidally_corrected_timeseries_MSL_S2']:
        for f in sorted((REPO / f'submissions/{team}/DUCK/{folder}').glob('*_timeseries_tidally_corrected.csv')):
            profile = f.name.split('_')[0]
            if profile not in PROFILES['DUCK']:
                continue
            with f.open() as src:
                for row in csv.DictReader(src):
                    try:
                        v = float(row[profile])
                    except (ValueError, KeyError):
                        continue
                    if math.isfinite(v):
                        obs.add((stamp(row['dates']), 'S2' if folder.endswith('_S2') else row['satname'], profile))

rows = []
for date, sensor, profile in sorted(obs):
    gap = min(abs((date - s).total_seconds()) for s in survey_dates)
    if gap <= 48 * 3600:
        rows.append({'date': date.isoformat(), 'sensor': sensor, 'profile': profile, 'gap_hours': gap / 3600,
                     'period': 'fit' if date.year <= 2009 else 'cal' if date.year <= 2016 else 'test'})
counts = {p: {'rows': sum(1 for r in rows if r['period'] == p),
              'scenes': len({(r['date'], r['sensor']) for r in rows if r['period'] == p})}
          for p in ['fit', 'cal', 'test']}
out = {'catalog_url': url, 'survey_dates': len(survey_dates),
       'first_survey': survey_dates[0].isoformat(), 'last_survey': survey_dates[-1].isoformat(),
       'coastsat_observations': len(obs), 'matched_48h_upper_bound': len(rows),
       'matched_48h_scenes': len({(r['date'], r['sensor']) for r in rows}),
       'counts': counts, 'years': dict(Counter(r['date'][:4] for r in rows)),
       'caveat': 'upper bound: survey dates have no time-of-day (midnight assumed); contour-quality exclusions not applied'}
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2)[:2000])
