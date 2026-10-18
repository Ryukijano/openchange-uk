"""Survey/geometry/count audit only. No training or outcome selection."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import netCDF4
import numpy as np
import pandas as pd
import pyproj
import sklearn
from netCDF4 import Dataset
from pyproj import Transformer

matplotlib.use('Agg')
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / 'sds-benchmark'
PROFILES = {'NARRABEEN': ['PF1', 'PF2', 'PF4', 'PF6', 'PF8'],
            'TRUCVERT': ['-400', '-300', '-200', '-100'],
            'TORREYPINES': ['PF525', 'PF535', 'PF585', 'PF595']}
SLOPES = {'NARRABEEN': .1, 'TRUCVERT': .05, 'TORREYPINES': .035}


def utc(values):
    return pd.to_datetime(values, utc=True, format='mixed').astype('datetime64[ns, UTC]')


def values(variable):
    return np.asarray(np.ma.asarray(variable[:], dtype=float).filled(np.nan))


def unique_crossing(x, z, level=0):
    """Do not bridge missing cells or choose among multiple crossings."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    valid_x = np.isfinite(x)
    x, z = x[valid_x], z[valid_x]
    order = np.argsort(x)
    x, z = x[order], z[order] - level
    if len(x) < 3 or len(np.unique(x)) != len(x):
        return None
    valid = np.isfinite(z[:-1]) & np.isfinite(z[1:])
    crossings = np.flatnonzero(valid & (z[:-1] >= 0) & (z[1:] < 0))
    upward = np.flatnonzero(valid & (z[:-1] < 0) & (z[1:] >= 0))
    if len(crossings) != 1 or len(upward):
        return None
    i = crossings[0]
    return float(x[i] + z[i] / (z[i] - z[i + 1]) * (x[i + 1] - x[i]))


def transect_geometry(site, epsg):
    features = json.loads((REPO / f'datasets/{site}/{site}_transects.geojson').read_text())['features']
    project = Transformer.from_crs(4326, epsg, always_xy=True)
    out = {}
    for feature in features:
        name = feature['properties']['name']
        origin = np.array(project.transform(*feature['geometry']['coordinates'][0]))
        end = np.array(project.transform(*feature['geometry']['coordinates'][-1]))
        unit = (end - origin) / np.linalg.norm(end - origin)
        out[name] = (origin, unit)
    return project, out


def projected_chainage(lon, lat, project, geometry):
    east, north = project.transform(lon, lat)
    origin, unit = geometry
    xy = np.column_stack([east, north]) - origin
    return xy @ unit, xy @ np.array([-unit[1], unit[0]])


def load_truc():
    root = HERE / 'raw/TRUCVERT'
    project, geometries = transect_geometry('TRUCVERT', 2154)
    with Dataset(root / 'Grids.nc') as grid:
        gx, gy, lat, lon = [values(grid[k]) for k in ['xlg', 'ylg', 'latg', 'long']]
    slots, geo = {}, []
    for profile in PROFILES['TRUCVERT']:
        i = int(np.argmin(abs(gy[0, :] - int(profile))))
        assert abs(gy[0, i] - int(profile)) < .001
        chain, lateral = projected_chainage(lon[:, i], lat[:, i], project, geometries[profile])
        assert np.nanmax(abs(lateral)) < 1
        assert abs(chain[0]) < .5 and np.all(np.diff(chain) > 0)
        slots[profile] = (i, chain)
        geo.append({'site': 'TRUCVERT', 'profile': profile, 'grid_first_x': float(gx[0, i]),
                    'projected_first_chainage': float(chain[0]), 'max_off_axis_m': float(np.max(abs(lateral))),
                    'grid_minus_chainage_min_m': float((gx[:, i] - chain).min()),
                    'grid_minus_chainage_max_m': float((gx[:, i] - chain).max())})
    rows, rejected, examples = [], 0, []
    files = sorted(root.glob('Monitoring_TrucVert_*.nc'))
    for file_index, path in enumerate(files):
        date = pd.Timestamp(path.stem[-10:], tz='UTC')
        with Dataset(path) as dataset:
            z = values(dataset['zg'])
            assert z.shape == gx.shape
            assert dataset['zg'].standard_name == 'height_above_mean_sea_level'
            assert '0.4' in dataset['zg'].comment
        z[z < -100] = np.nan
        for profile, (i, chain) in slots.items():
            result = unique_crossing(chain, z[:, i])
            grid_result = unique_crossing(gx[:, i], z[:, i])
            if result is None:
                rejected += 1
                continue
            rows.append({'site': 'TRUCVERT', 'profile': profile, 'survey_date': date, 'truth': result,
                         'native_grid_contour': grid_result, 'datum_level': 0.0})
            if file_index in [0, len(files) // 2, len(files) - 1]:
                examples.append(('TRUCVERT', profile, date, chain, z[:, i], 0., result))
    return pd.DataFrame(rows), {'files': len(files), 'rejected_profile_dates': rejected, 'geometry': geo}, examples


def load_narrabeen():
    raw = pd.read_csv(REPO / 'datasets/NARRABEEN/raw/Narrabeen_Profiles.csv')
    rows, rejected, examples = [], 0, []
    groups = list(raw.groupby(['Profile ID', 'Date']))
    for group_index, ((profile, date), group) in enumerate(groups):
        if profile not in PROFILES['NARRABEEN']:
            continue
        group = group.groupby('Chainage', as_index=False).Elevation.mean().sort_values('Chainage')
        chain, z = group.Chainage.to_numpy(), group.Elevation.to_numpy()
        result = unique_crossing(chain, z)
        if result is None:
            rejected += 1
            continue
        stamp = pd.Timestamp(date, tz='UTC')
        rows.append({'site': 'NARRABEEN', 'profile': profile, 'survey_date': stamp, 'truth': result,
                     'native_grid_contour': result, 'datum_level': 0.})
        if group_index % max(1, len(groups) // 15) == 0:
            examples.append(('NARRABEEN', profile, stamp, chain, z, 0., result))
    return pd.DataFrame(rows), {'rejected_profile_dates': rejected}, examples


def load_torrey():
    files = sorted((HERE / 'raw/TORREYPINES').glob('map*.nc'))
    if not files:
        return None, {'status': 'blocked: publisher download requires access/user upload'}, []
    project, geometry = transect_geometry('TORREYPINES', 26911)
    rows, geo, examples, rejected = [], [], [], 0
    for file_index, path in enumerate(files):
        date = pd.to_datetime(path.stem[3:11], format='%Y%m%d', utc=True)
        with Dataset(path) as dataset:
            z, profiles, lon, lat = [values(dataset[k]) for k in ['mapz', 'alg', 'lonc', 'latc']]
        z[z < -100] = np.nan
        for profile in PROFILES['TORREYPINES']:
            match = np.flatnonzero(profiles[0, :] == int(profile[2:]))
            assert len(match) == 1, (path.name, profile)
            i = int(match[0])
            chain, lateral = projected_chainage(lon[:, i], lat[:, i], project, geometry[profile])
            assert np.nanmax(abs(lateral)) < 5, (profile, np.nanmax(abs(lateral)))
            # Coordinates are ordered seaward-to-landward in this archive.
            result = unique_crossing(chain, z[:, i], .774)
            if file_index == 0:
                geo.append({'site': 'TORREYPINES', 'profile': profile, 'max_off_axis_m': float(np.nanmax(abs(lateral))),
                            'landward_chainage_m': float(np.nanmin(chain)), 'datum_level_NAVD88': .774})
            if result is None:
                rejected += 1
                continue
            rows.append({'site': 'TORREYPINES', 'profile': profile, 'survey_date': date, 'truth': result,
                         'native_grid_contour': result, 'datum_level': .774})
            if file_index in [0, len(files) // 2, len(files) - 1]:
                examples.append(('TORREYPINES', profile, date, chain, z[:, i], .774, result))
    return pd.DataFrame(rows), {'files': len(files), 'rejected_profile_dates': rejected, 'geometry': geo}, examples


def load_submission(site, team, profile, s2=False, raw=False):
    folder = 'raw_timeseries' if raw else 'tidally_corrected_timeseries_MSL'
    if s2:
        folder += '_S2'
    suffix = 'raw' if raw else 'tidally_corrected'
    path = REPO / f'submissions/{team}/{site}/{folder}/{profile}_timeseries_{suffix}.csv'
    if not path.exists():
        return pd.DataFrame({'date': pd.Series(dtype='datetime64[ns, UTC]'),
                             'position': pd.Series(dtype=float),
                             'sensor': pd.Series(dtype='str'), 'profile': pd.Series(dtype='str')})
    frame = pd.read_csv(path).rename(columns={'dates': 'date', profile: 'position', 'satname': 'sensor'})
    frame['date'] = utc(frame.date)
    frame['position'] = pd.to_numeric(frame.position, errors='coerce')
    if s2:
        frame['sensor'] = 'S2'
    frame['profile'] = profile
    return frame[['date', 'position', 'sensor', 'profile']].dropna(subset=['position'])


def match_surveys(frame, surveys, hours=48):
    frame, surveys = frame.copy(), surveys.copy()
    frame['date'] = utc(frame.date)
    surveys['survey_date'] = utc(surveys.survey_date)
    groups = []
    for profile, group in frame.groupby('profile'):
        truth = surveys[surveys.profile.eq(profile)][['survey_date', 'truth']].sort_values('survey_date')
        matched = pd.merge_asof(group.sort_values('date'), truth, left_on='date', right_on='survey_date',
                                direction='nearest', tolerance=pd.Timedelta(hours=hours))
        groups.append(matched)
    result = pd.concat(groups, ignore_index=True).dropna(subset=['truth'])
    result['gap_hours'] = (result.date - result.survey_date).dt.total_seconds().abs() / 3600
    assert result.gap_hours.le(hours).all()
    return result


def build_observations(site, surveys):
    coast = pd.concat([load_submission(site, 'team_CoastSat', p, s2) for p in PROFILES[site]
                       for s2 in [False, True]], ignore_index=True).drop_duplicates(['date', 'sensor', 'profile'])
    coast['scene'] = site + '_' + coast.date.astype(str) + '_' + coast.sensor.astype(str)
    data = match_surveys(coast, surveys)
    sensitivity = match_surveys(coast, surveys, 36)
    data['site'] = site
    data['disagreement'], data['raw_position'] = np.nan, np.nan
    for profile, group in data.groupby('profile'):
        left = group[['date', 'sensor', 'position']].copy()
        left['row'] = group.index
        ref = pd.concat([load_submission(site, 'team_SHOREX', profile, s2) for s2 in [False, True]], ignore_index=True)
        if not ref.empty:
            ref = ref.rename(columns={'position': 'ref_position'})
            pair = pd.merge_asof(left.sort_values('date'), ref[['date', 'sensor', 'ref_position']].sort_values('date'),
                                 on='date', by='sensor', direction='nearest', tolerance=pd.Timedelta(minutes=10))
            data.loc[pair.row, 'disagreement'] = abs(pair.position - pair.ref_position).to_numpy()
        raw = pd.concat([load_submission(site, 'team_CoastSat', profile, s2, True) for s2 in [False, True]], ignore_index=True)
        if not raw.empty:
            raw = raw.rename(columns={'position': 'raw_position'})
            pair = pd.merge_asof(left.sort_values('date'), raw[['date', 'sensor', 'raw_position']].sort_values('date'),
                                 on='date', by='sensor', direction='nearest', tolerance=pd.Timedelta(seconds=1))
            data.loc[pair.row, 'raw_position'] = pair.raw_position.to_numpy()
    for file, columns, tolerance in [('tides', ['tides'], '20min'), ('waves_ERA5', ['swh', 'mwp', 'pp1d', 'mwd'], '60min')]:
        reference = pd.read_csv(REPO / f'datasets/{site}/{site}_{file}.csv', low_memory=False)
        reference['date'] = utc(reference.dates)
        for column in columns:
            reference[column] = pd.to_numeric(reference[column].astype(str).str.strip('[]'), errors='coerce')
        data = pd.merge_asof(data.sort_values('date'), reference[['date', *columns]].sort_values('date'),
                             on='date', direction='nearest', tolerance=pd.Timedelta(tolerance))
    data = data.rename(columns={'tides': 'tide'})
    data['slope'] = SLOPES[site]
    data['setup'] = .35 * data.slope * np.sqrt(data.swh * 9.81 * data.pp1d**2 / (2 * np.pi))
    for name, vals in [('dir', np.deg2rad(data.mwd)), ('month', 2 * np.pi * data.date.dt.dayofyear / 365.25)]:
        data[f'{name}_sin'], data[f'{name}_cos'] = np.sin(vals), np.cos(vals)
    for sensor in ['L5', 'L7', 'L8', 'S2']:
        data[f'sensor_{sensor}'] = data.sensor.eq(sensor).astype(float)
    data['error'] = data.position - data.truth
    data['period'] = np.where(data.date.dt.year <= 2009, 'fit', np.where(data.date.dt.year <= 2016, 'cal', 'test'))
    counts = {'available_rows': len(coast), 'available_scenes': int(coast.scene.nunique()),
              'matched_48h_rows': len(data), 'matched_48h_scenes': int(data.scene.nunique()),
              'matched_36h_rows': len(sensitivity), 'matched_36h_scenes': int(sensitivity.scene.nunique()),
              'periods': {name: {'rows': len(group), 'scenes': int(group.scene.nunique()),
                                'survey_dates': int(group.survey_date.nunique()), 'sensors': sorted(group.sensor.unique())}
                         for name, group in data.groupby('period')},
              'metadata_missing_fraction': data[['tide', 'swh', 'mwp', 'pp1d', 'disagreement']].isna().mean().to_dict()}
    return data, counts


def snapshot():
    paths = [HERE / 'protocol.md', HERE / 'sources.yaml', HERE / 'audit_surveys.py', HERE / 'acquire_surveys.py']
    paths += list((HERE / 'raw').rglob('*.nc'))
    paths += [REPO / 'datasets/NARRABEEN/raw/Narrabeen_Profiles.csv']
    for site in PROFILES:
        paths += list((REPO / f'datasets/{site}').glob('*.csv'))
        paths += [REPO / f'datasets/{site}/{site}_transects.geojson']
        for team in ['team_CoastSat', 'team_SHOREX']:
            paths += list((REPO / f'submissions/{team}/{site}').glob('*/*.csv'))
    import subprocess
    return {'created_utc': datetime.now(UTC).isoformat(), 'python': platform.python_version(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'netCDF4': netCDF4.__version__,
            'sklearn': sklearn.__version__, 'pyproj': pyproj.__version__,
            'sds_git_sha': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
            'uncommitted_research_scripts': True,
            'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def plot_examples(examples):
    # Fixed first/middle/last survey selections, never selected by model residuals.
    for site in PROFILES:
        items = [row for row in examples if row[0] == site][:15]
        if not items:
            continue
        figure, axes = plt.subplots(5, 3, figsize=(13, 15), constrained_layout=True)
        for axis, row in zip(axes.flat, items, strict=False):
            _, profile, date, x, z, level, crossing = row
            axis.plot(x, z, linewidth=1)
            axis.axhline(level, color='black', linestyle='--')
            axis.axvline(crossing, color='red', linestyle=':')
            axis.set_title(f'{profile} {date.date()}')
            axis.set_xlabel('Along benchmark transect (m)')
            axis.set_ylabel('Elevation in source datum (m)')
        for axis in list(axes.flat)[len(items):]:
            axis.set_visible(False)
        figure.suptitle(f'{site}: survey-derived contours, not model predictions')
        figure.savefig(HERE / f'pilot_contours_{site}.png', dpi=120)
        plt.close(figure)


def main():
    environment = snapshot()
    (HERE / 'environment.lock').write_text(json.dumps(environment, indent=2))
    frames, audit, examples = [], {}, []
    for site, loader in [('NARRABEEN', load_narrabeen), ('TRUCVERT', load_truc), ('TORREYPINES', load_torrey)]:
        surveys, record, samples = loader()
        audit[site] = record
        if surveys is None:
            continue
        assert not surveys.duplicated(['profile', 'survey_date']).any()
        surveys.to_csv(HERE / f'survey_contours_{site}.csv', index=False)
        data, counts = build_observations(site, surveys)
        frames.append(data)
        record['survey_rows'], record['counts'] = len(surveys), counts
        examples.extend(samples)
        print(site, json.dumps(record), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(HERE / 'matched_observations.csv', index=False)
    (HERE / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    plot_examples(examples)
    print('Audit complete; no model fitted.', flush=True)


if __name__ == '__main__':
    main()
