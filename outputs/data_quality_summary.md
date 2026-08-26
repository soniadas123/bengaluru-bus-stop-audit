# Bengaluru Bus Stop Audit - Data Quality Summary

## Shape
- Rows: 406
- Columns: 86

## Column types
- `str`: 50 columns
- `int64`: 24 columns
- `float64`: 12 columns

## Missingness overview
- Columns with 0% missing: 53
- Columns fully empty (100% missing): 6

Top 20 columns by % missing:

| Column | % missing |
|---|---|
| Relationship between Bus Stop and Pedestrians | 100.0% |
| _notes | 100.0% |
| _submitted_by | 100.0% |
| _tags | 100.0% |
| _validation_status | 100.0% |
| Check this box if the bus stop's kerb is higher than 15cms or has steps to access | 100.0% |
| If Location field is not working, please enter coordinates from google maps or paste the location URL | 81.0% |
| Is the space behind the bus shelter usable? | 76.6% |
| Photo of any other key observation 2 | 70.2% |
| Photo of any other key observation 2 _URL | 70.2% |
| Photo of any other key observation 1 _URL | 50.5% |
| Photo of any other key observation 1 | 50.5% |
| Top 1–2 improvements you would recommend for this stop | 36.5% |
| Is the bus stop big enough for the number of people who typically wait here? | 36.0% |
| If seated at the bus stop, can you clearly see an approaching bus? | 36.0% |
| Does the bus shelter carry advertisements? | 36.0% |
| Does the bus stop have a roof that shelters people from rain? | 36.0% |
| Is the seating provided comfortable? | 36.0% |
| How many bus shelters are placed close to each other? | 36.0% |
| Condition of Bus Shelter | 36.0% |

## Data quality issues

### 1. Fully-empty columns (safe to drop)
- `Relationship between Bus Stop and Pedestrians`
- `_notes`
- `_submitted_by`
- `_tags`
- `_validation_status`
- `Check this box if the bus stop's kerb is higher than 15cms or has steps to access`

### 2. Skip-logic missingness (not true gaps)
These questions are only asked when a bus shelter/signboard exists. Their ~30-36% missingness lines up with stops recorded as 'No Shelter and No Signboard' or 'Only "Bus Stop" Signboard - No Bus Shelter', not with data collection errors.

| Column | % missing | % of missing rows with no shelter/signboard |
|---|---|---|
| Condition of Bus Shelter | 36.0% | 100.0% |
| Does the bus stop have a roof that shelters people from rain? | 36.0% | 100.0% |
| Does the bus shelter carry advertisements? | 36.0% | 100.0% |
| Is the seating provided comfortable? | 36.0% | 100.0% |
| Is the bus stop big enough for the number of people who typically wait here? | 36.0% | 100.0% |
| If seated at the bus stop, can you clearly see an approaching bus? | 36.0% | 100.0% |
| How many bus shelters are placed close to each other? | 36.0% | 100.0% |
| Is the name of the bus stop clearly displayed? | 31.8% | 100.0% |
| Is bus route information displayed at the stop? | 31.8% | 100.0% |

### 3. Missing GPS coordinates
5 rows have no usable coordinates (`location_flag == 'NO_LOCATION'`) and are excluded from the map and any geo-based analysis:

- `773910562` - Kadubisinahalli (towards Marathahahlli)
- `774174237` - CMRIT College (Towards Kundalahalli Colony)
- `776408127` - Innovity Multiplex (towards Kalamandir)
- `793887115` - Not sure
- `812345457` - Royal Meenakshi Mall

### 4. Coordinate reliability
Coordinate source breakdown:

- device_gps: 326
- gmaps_link: 73
- nan: 5
- typed_coordinates: 2

- GPS precision (`_Survey Location_precision`, meters) is only recorded for device-collected fixes (326 rows, matching the `device_gps` count above). 3 of those have a reported error greater than 100m, i.e. low-confidence fixes.

### 5. Duplicate coordinates with different stop names
Same coordinates recorded under different stop names - likely a duplicate submission, or a paired directional stop that should be reviewed manually rather than auto-merged:

- (12.882883, 77.629785): 'Hiranandani Apartment,Yelenahalli', 'Shantiniketan layout'
- (12.896606, 77.720101): 'Kodati Gate (Towards Kempegowda Bus Station)', 'Kodati Gate (Towards Kempegowda Bus Station)'
- (12.927598, 77.583618): 'Jayanagar 4th block (towards 5th block)', 'Jayangar 4th block (towards 5th block)'
- (12.927950, 77.583456): 'Jayanagar 4th block ( towards lalbagh)', 'Jayangar 4th block (towards 3rd block)'
- (12.936313, 77.580223): 'South end (towards Banashankari)', 'South end circle'
- (12.971845, 77.610590): 'Mayo Hall', 'Brigade Road (Towards Dairy Circle/Christ College/ Koramangala)'

### 6. Duplicate stop names
14 stop names appear more than once (case-insensitive exact match), covering 22 rows. Spot checks show some are spelling variants of the same stop (e.g. 'Jayanagar' vs 'Jayangar') rather than true repeats - listed here for manual review, not auto-merged.
