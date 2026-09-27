"""Harvest 13 Malaysian states for 2017 and 2024 into a local CSV.

Install: python -m pip install earthengine-api
Authenticate once: earthengine authenticate
Run: python Malaysia_harvester_2018_2024.py

Adapted from Malaysia_harvester_multistate.py. State-wide statistics, not
paddy-masked statistics. Median_NDVI retains the legacy column name: it is
the spatial mean of the annual per-pixel median NDVI, NOT peak NDVI.
No synthetic fallback measurements or yield/production estimates are made.
"""

import argparse
import calendar
import csv
import json
import time
from pathlib import Path

import ee

yearStarts=2017
yearEndsBefore=2025

STATES = [
    'Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang',
    'Perak', 'Perlis', 'Pulau Pinang', 'Selangor', 'Terengganu', 'Sabah',
    'Sarawak',
]
ALIASES = {
    'Johor': ['Johor', 'Johore'],
    'Melaka': ['Melaka', 'Malacca'],
    'Negeri Sembilan': ['Negeri Sembilan', 'Negri Sembilan'],
    'Pulau Pinang': ['Pulau Pinang', 'Pinang', 'Penang'],
    'Terengganu': ['Terengganu', 'Trengganu'],
}
FIELDS = ['state', 'year', 'Median_NDVI', 'Solar_Radiation_MJ',
          'Mean_Temp_C', 'Annual_Rain_mm', 'Soil_pH', 'Soil_Clay_Pct']


def fetch(obj):
    """Retry transient server failures; never replace an error with a value."""
    for attempt in range(3):
        try:
            return obj.getInfo()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def mean_over(image, geometry, scale):
    return image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=scale,
        maxPixels=1_000_000_000, tileScale=4,
    )


def ndvi(image):
    # Preserve the supplied multistate script's scene-level cloud filter.
    # No new pixel cloud mask is introduced for only the new years.
    return image.normalizedDifference(['B8', 'B4']).rename('Median_NDVI')


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.csv')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='cogent-sweep-399309')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent / 'output' / 'Malaysia_Paddy_States_GEE_2017_2024.csv'
    )
    args = parser.parse_args()
    try:
        ee.Initialize(project=args.project)
    except Exception as exc:
        raise RuntimeError(
            'Earth Engine initialization failed. Run earthengine authenticate '
            'and confirm access to the project, or pass --project YOUR_PROJECT.'
        ) from exc

    boundaries = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(
        ee.Filter.eq('ADM0_NAME', 'Malaysia'))
    available = fetch(boundaries.aggregate_array('ADM1_NAME'))
    geometries = {}
    for state in STATES:
        matches = [name for name in ALIASES.get(state, [state]) if name in available]
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely resolve {state}. Available names: {available}')
        selected = boundaries.filter(ee.Filter.eq('ADM1_NAME', matches[0]))
        geometries[state] = selected.geometry()

    soil = ee.Image.cat([
        ee.Image('projects/soilgrids-isric/phh2o_mean')
        .select('phh2o_0-5cm_mean').multiply(0.1).rename('Soil_pH'),
        ee.Image('projects/soilgrids-isric/clay_mean')
        .select('clay_0-5cm_mean').multiply(0.1).rename('Soil_Clay_Pct'),
    ])
    rows, diagnostics = [], []
    partial = args.output.with_name(args.output.stem + '_partial.csv')
    for state in STATES:
        geometry = geometries[state]
        soil_values = fetch(mean_over(soil, geometry, 500))
        for year in range (yearStarts, yearEndsBefore):
            print(f'Harvesting {state}, {year} ...', flush=True)
            # filterDate end is exclusive: include December 31.
            start, end = f'{year}-01-01', f'{year + 1}-01-01'
            weather = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                       .filterBounds(geometry).filterDate(start, end))
            days = fetch(weather.size())
            if days != 365 + int(calendar.isleap(year)):
                raise ValueError(f'Incomplete weather coverage: {state} {year}: {days} days')
            climate = ee.Image.cat([
                weather.select('temperature_2m').mean().subtract(273.15)
                .rename('Mean_Temp_C'),
                weather.select('total_precipitation_sum').sum().multiply(1000)
                .rename('Annual_Rain_mm'),
                # Annual mean DAILY solar energy: MJ / m2 / day.
                weather.select('surface_solar_radiation_downwards_sum').mean()
                .divide(1_000_000).rename('Solar_Radiation_MJ'),
            ])
            values = fetch(mean_over(climate, geometry, 10000))
            s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(geometry).filterDate(start, end)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 45)))
            scenes = fetch(s2.size())
            vegetation = (fetch(mean_over(s2.map(ndvi).median(), geometry, 1000))
                          if scenes else {})
            row = dict.fromkeys(FIELDS)
            row.update(state=state, year=year)
            row.update({key: soil_values.get(key) for key in ('Soil_pH', 'Soil_Clay_Pct')})
            row.update({key: values.get(key) for key in
                        ('Mean_Temp_C', 'Annual_Rain_mm', 'Solar_Radiation_MJ')})
            row['Median_NDVI'] = vegetation.get('Median_NDVI')
            # Retain source script precision for existing metrics; solar is unrounded.
            for key in ('Soil_pH', 'Soil_Clay_Pct', 'Mean_Temp_C', 'Annual_Rain_mm'):
                if row[key] is not None:
                    row[key] = round(row[key], 2)
            if row['Median_NDVI'] is not None:
                row['Median_NDVI'] = round(row['Median_NDVI'], 4)
            missing = [key for key in FIELDS[2:] if row[key] is None]
            if missing:
                print(f'  Missing measurements left blank: {missing}', flush=True)
            rows.append(row)
            diagnostics.append(dict(state=state, year=year, weather_days=days,
                                    sentinel_scenes=scenes, missing_columns=missing))
            save_csv(partial, rows)

    assert len(rows) == (yearEndsBefore-yearStarts)*len(STATES) #change this number
    assert len({(r['state'], r['year']) for r in rows}) == len(rows)
    save_csv(args.output, rows)
    args.output.with_suffix('.metadata.json').write_text(json.dumps({
        'rows': len(rows), 'diagnostics': diagnostics,
        'method': 'State-wide GAUL boundaries; no paddy mask',
        'Median_NDVI': 'Legacy name: state mean of annual median NDVI; scene clouds <45%',
        'Solar_Radiation_MJ': 'Annual average daily downward solar energy, MJ/m2/day',
        'Annual_Rain_mm': 'Annual rainfall total spatially averaged over state, mm',
        'soil': 'Static SoilGrids 0-5 cm; pH and clay percentage',
        'changes_from_original': 'Full calendar years; missing data blank; solar added',
        'derived_yield_and_production': 'Not computed: formula absent from supplied scripts',
    }, indent=2), encoding='utf-8')
    print(f'Saved {len(rows)} rows to {args.output.resolve()}')


if __name__ == '__main__':
    main()
