"""Offline descriptive audit only; does not create a training manifest or split."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import shapely

ROOT = Path('/home/ubuntu/repos/openchange-uk')
RUN = ROOT / 'outputs/jepa-metadata-local20261011T143651Z-42766'
FETCH = json.loads((RUN / 'fetch-result.json').read_text())


def count(mask):
    return int(np.count_nonzero(mask))


def quantiles(x):
    values = np.asarray(x)
    finite = values[np.isfinite(values)]
    return {str(q): float(np.quantile(finite, q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]}


def main():
    for record in FETCH['files']:
        assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() == record['sha256']
    tables = [pq.read_table(record['path']) for record in FETCH['files']]
    schemas = [t.schema for t in tables]
    table = pa.concat_tables(tables)
    n = table.num_rows
    def col(k):
        return table[k].combine_chunks().to_numpy(zero_copy_only=False)
    ids = col('sample_id')
    split = col('split')
    lon, lat = col('center_lon'), col('center_lat')
    s1 = np.stack([col(f'S1_time_{i}') for i in range(4)], axis=1)
    s2 = np.stack([col(f'S2_time_{i}') for i in range(4)], axis=1)
    cloud = np.stack([col(f'cloud_cover_{i}') for i in range(4)], axis=1)
    valid_dates = ~np.isnat(s1) & ~np.isnat(s2)
    gap = np.abs((s1 - s2) / np.timedelta64(1, 'D'))
    geo_crs = [pyproj.CRS.from_json_dict(json.loads(s.metadata[b'geo'])['columns']['geometry']['crs'])
               for s in schemas]
    assert all(crs.to_epsg() == 4326 for crs in geo_crs)
    geometries = shapely.from_wkb(table['geometry'].to_pylist())
    native_bounds = np.array(table['bounds'].to_pylist())
    uk_box = shapely.box(-11, 49, 2.5, 61.5)
    uk = shapely.intersects(geometries, uk_box)
    center_uk = (lon >= -11) & (lon <= 2.5) & (lat >= 49) & (lat <= 61.5)
    centers = shapely.points(lon, lat)
    center_inside = shapely.covers(geometries, centers)
    crs_codes = col('crs')
    errors = np.zeros(n)
    for code in np.unique(crs_codes):
        idx = np.flatnonzero(crs_codes == code)
        transformer = pyproj.Transformer.from_crs(4326, int(code), always_xy=True)
        transformed = shapely.transform(geometries[idx], transformer.transform, interleaved=False)
        errors[idx] = np.max(np.abs(shapely.bounds(transformed) - native_bounds[idx]), axis=1)
    train_idx = np.flatnonzero(split == 'train')
    val_idx = np.flatnonzero(split == 'val')
    tree = shapely.STRtree(geometries[train_idx])
    cross = tree.query(geometries[val_idx], predicate='intersects')
    areas = shapely.area(shapely.intersection(geometries[val_idx[cross[0]]],
                                             geometries[train_idx[cross[1]]]))
    fractions = areas / np.minimum(shapely.area(geometries[val_idx[cross[0]]]),
                                  shapely.area(geometries[train_idx[cross[1]]]))
    largest = int(np.argmax(fractions))
    i, j = val_idx[cross[0, largest]], train_idx[cross[1, largest]]
    example = {'train_location_id': ids[j], 'val_location_id': ids[i],
               'train_geometry_wkt': shapely.to_wkt(geometries[j], rounding_precision=12),
               'val_geometry_wkt': shapely.to_wkt(geometries[i], rounding_precision=12),
               'overlap_fraction_smaller': float(fractions[largest])}
    (RUN / 'overlap-example.json').write_text(json.dumps(example, indent=2) + '\n')
    nearest = np.abs((s2[:, :, None] - s1[:, None, :]) / np.timedelta64(1, 'D'))
    nearest_s1 = np.argmin(nearest, axis=2)
    nearest_gap = np.min(nearest, axis=2)
    cutoff = np.datetime64('2020-12-31T23:59:59.999999999')
    embargo_end = np.datetime64('2021-01-31T23:59:59.999999999')
    gap7 = valid_dates & (gap <= 7)
    pilot_candidate = gap7 & (~uk[:, None])
    time_before = pilot_candidate & (s1 <= cutoff) & (s2 <= cutoff)
    time_after = pilot_candidate & (s1 > embargo_end) & (s2 > embargo_end)
    strata = {}
    for threshold in [0, .01, .05, .1, .5, .8]:
        above = cloud > threshold
        strata[str(threshold)] = {'observations_above': count(above),
                                 'locations_with_any_above': count(np.any(above, axis=1)),
                                 'outside_uk_gap7_observations_above': count(above & pilot_candidate),
                                 'outside_uk_gap7_locations_with_any_above': count(np.any(above & pilot_candidate, axis=1))}
    pairing = {}
    for days in [1, 3, 7, 14, 30]:
        mask = valid_dates & (gap <= days)
        pairing[str(days)] = {'index_pairs': count(mask), 'locations_with_any_index_pair': count(mask.any(axis=1)),
                              'outside_uk_index_pairs': count(mask & ~uk[:, None]),
                              'outside_uk_locations_with_any_index_pair': count(np.any(mask & ~uk[:, None], axis=1))}
    buffers = {'0': count(uk)}
    nearby = np.flatnonzero(shapely.intersects(geometries, shapely.box(-13, 47, 4.5, 63.5)))
    transform = pyproj.Transformer.from_crs(4326, 3035, always_xy=True)
    dense_box = shapely.segmentize(uk_box, .02)
    projected_box = shapely.transform(dense_box, transform.transform, interleaved=False)
    projected = shapely.transform(geometries[nearby], transform.transform, interleaved=False)
    distance = shapely.distance(projected, projected_box)
    for km in [5, 10]:
        buffers[str(km)] = count(distance <= km * 1000)
    bins = [-90, -60, -30, 0, 30, 60, 90]
    latitude = {f'[{a},{b})': {'all_locations': count((lat >= a) & (lat < b)),
                 'outside_uk_gap7_locations': count((lat >= a) & (lat < b) & pilot_candidate.any(axis=1))}
                for a, b in zip(bins[:-1], bins[1:])}
    tar_stats = {}
    for membership in ['train', 'val']:
        subset = col('tar')[split == membership]
        tars, sizes = np.unique(subset, return_counts=True)
        tar_stats[membership] = {'shards': len(tars), 'min_locations': int(sizes.min()),
                                'max_locations': int(sizes.max()), 'locations': int(sizes.sum())}
    result = {
        'status': 'descriptive_metadata_audit_not_a_training_manifest',
        'source_revision': FETCH['files'][0]['revision'],
        'input_checksums': {Path(r['path']).name: r['sha256'] for r in FETCH['files']},
        'packages': {p: importlib.metadata.version(p) for p in ['pyarrow','numpy','shapely','pyproj']},
        'columns': table.column_names, 'null_counts': {k: table[k].null_count for k in table.column_names},
        'rows': n, 'unique_location_ids': len(np.unique(ids)),
        'duplicate_member_keys': n - len(set(zip(split.tolist(), col('tar').tolist(), col('zarr').tolist()))),
        'provider_memberships': {str(k): int(v) for k,v in zip(*np.unique(split, return_counts=True))},
        's1_timestamp_range': [str(s1.min()), str(s1.max())],
        's2_timestamp_range': [str(s2.min()), str(s2.max())],
        'timestamps_missing': count(~valid_dates),
        's1_nonmonotonic_locations': count(np.any(s1[:,1:] < s1[:,:-1],axis=1)),
        's2_nonmonotonic_locations': count(np.any(s2[:,1:] < s2[:,:-1],axis=1)),
        'within_location_duplicate_s1_dates': count(np.any(np.diff(np.sort(s1,axis=1),axis=1)==np.timedelta64(0,'ns'),axis=1)),
        'within_location_duplicate_s2_dates': count(np.any(np.diff(np.sort(s2,axis=1),axis=1)==np.timedelta64(0,'ns'),axis=1)),
        'index_gap_days_quantiles': quantiles(gap), 'gap_thresholds_days': pairing,
        'nearest_other_index_than_seasonal_index': count(nearest_s1 != np.arange(4)[None,:]),
        'nearest_gap_days_quantiles': quantiles(nearest_gap),
        'nearest_mapping_reuses_s1_locations': count(np.any(np.diff(np.sort(nearest_s1,axis=1),axis=1)==0,axis=1)),
        'cloud_fraction_quantiles': quantiles(cloud), 'cloud_nonfinite': count(~np.isfinite(cloud)),
        'cloud_outside_0_1': count((cloud<0)|(cloud>1)), 'cloud_exact_zero': count(cloud == 0),
        'cloud_thresholds': strata, 'uk_box': [-11,49,2.5,61.5], 'uk_box_footprint_intersections': count(uk),
        'uk_box_centers': count(center_uk), 'footprint_only_uk_exclusions': count(uk & ~center_uk),
        'uk_exclusion_buffer_sensitivity_km': buffers,
        'geometry_crs': 'EPSG:4326', 'native_raster_crs_unique': len(np.unique(crs_codes)),
        'invalid_geometry': count(~shapely.is_valid(geometries)),
        'empty_geometry': count(shapely.is_empty(geometries)),
        'centers_not_covered_by_footprint': count(~center_inside),
        'native_bbox_width_m': quantiles(native_bounds[:,2]-native_bounds[:,0]),
        'native_bbox_height_m': quantiles(native_bounds[:,3]-native_bounds[:,1]),
        'geometry_vs_native_bounds_max_error_m': float(errors.max()),
        'provider_train_val_footprint_intersection_pairs': int(cross.shape[1]),
        'provider_val_locations_overlapping_train': len(np.unique(cross[0])),
        'provider_train_val_positive_area_pairs': count(areas > 0),
        'provider_val_locations_positive_area_overlap': len(np.unique(cross[0][areas > 0])),
        'overlap_smaller_footprint_fraction_quantiles': {
            str(q): float(np.quantile(fractions, q)) for q in [0, .5, .95, 1]},
        'latitude_bands': latitude, 'tar_location_inventory': tar_stats,
        'exploratory_time_policy': {'cutoff': str(cutoff), 'validation_after': str(embargo_end),
            'maximum_gap_days': 7, 'uk_buffer_km': 0, 'accepted': False,
            'pre_cutoff_candidate_pairs': count(time_before),
            'pre_cutoff_candidate_locations': count(time_before.any(axis=1)),
            'post_embargo_candidate_pairs': count(time_after),
            'post_embargo_candidate_locations': count(time_after.any(axis=1)),
            'locations_with_both_eras': count(time_before.any(axis=1) & time_after.any(axis=1))},
        'limitations': ['No imagery quality, mask polarity, spectral units or shard range access checked.',
                        'Cloud fields are provider estimates, not manually verified cloud labels.',
                        'Timestamp columns are timezone-naive; differences use full nanosecond instants as stored.',
                        'UK buffer sensitivities use densified WGS84 box projected to EPSG:3035; not a signed buffer policy.',
                        'Other benchmark footprints and geographical split buffers are not yet applied.',
                        'Counts are descriptive; no manifest, train/validation assignment or source approval for imagery.']
    }
    (RUN / 'metadata-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['columns','null_counts','limitations','input_checksums','packages']}, indent=2))


if __name__ == '__main__':
    main()
