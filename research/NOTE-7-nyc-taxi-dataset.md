# NOTE-7: NYC taxi dataset source & haversine formula (2026-09-02)

**Answer:** Official NYC TLC trip record data is freely available under NYC terms of use, but published parquet files are very large (~50GB total, single months 100MB–1GB each); no small CSV sample is readily distributed. **RECOMMENDATION: SYNTHESISE a realistic NYC taxi dataset** (random pickup/dropoff within NYC bounds, haversine-computed distance, fare = $2.50 base + $2.50/mi + noise) so the chapter is fully runnable without a giant download. Haversine formula verified from authoritative sources.

**Evidence:**

### Official NYC TLC Data

| Item | Details | Source |
|------|---------|--------|
| **Official source** | NYC Taxi & Limousine Commission (TLC) | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |
| **Format** | Apache Parquet (changed from CSV May 2022) | Official TLC page |
| **License / Terms** | NYC Terms of Use: http://www1.nyc.gov/home/terms-of-use.page (public data, but no explicit CC0/CC-BY/MIT noted) | https://registry.opendata.aws/nyc-tlc-trip-records-pds/ |
| **File size** | Single month: ~100MB–1GB; total dataset (2009–2025): ~50GB (≈1.5B rows) | AWS Registry of Open Data |
| **Access** | Free via aws.amazon.com (S3 bucket `nyc-tlc`); also on Azure Open Datasets | https://registry.opendata.aws/nyc-tlc-trip-records-pds/; https://learn.microsoft.com/en-us/azure/open-datasets/dataset-taxi-yellow |
| **Yellow taxi columns (sample)** | pickup_datetime, dropoff_datetime, passenger_count, trip_distance, fare_amount, extra, mta_tax, tip_amount, tolls_amount, total_amount, payment_type, pickup_location_id, dropoff_location_id, plus: longitude/latitude fields (exact schema in Yellow Trips Data Dictionary PDF on TLC page) | TLC official documentation |

### Alternatives Checked

| Dataset | Availability | License | Size | Notes |
|---------|--------------|---------|------|-------|
| **Kaggle: NYC Taxi Fare Prediction** | Kaggle competition (Google Cloud); requires login | Not explicitly stated; likely Kaggle Terms of Use | 5.5 GB training data (55M+ rows) | Public but requires account & download; not clearly licensed for redistribution |
| **seaborn/sklearn penguins** | `seaborn.load_dataset("penguins")` or `read_csv(URL)` | Public domain (Palmer Penguins; Allison Horst) | 44 KB | Not taxi data; different domain (bird measurements, not prices) |
| **sklearn load_boston** | **DEPRECATED** in sklearn 1.2+ | N/A | Removed | Not available in modern sklearn; should not be used |

### Haversine Formula (Great-Circle Distance)

**Exact Formula (from authoritative sources):**

Given two points (φ₁, λ₁) and (φ₂, λ₂) in latitude/longitude (radians), the haversine formula computes the great-circle distance on a sphere of radius r:

```
Δσ = 2 * arcsin( sqrt( sin²((φ₂-φ₁)/2) + cos(φ₁)·cos(φ₂)·sin²((λ₂-λ₁)/2) ) )
distance = r * Δσ
```

**Or equivalently (using haversine function h(θ) = sin²(θ/2)):**

```
h((φ₂-φ₁)) + cos(φ₁)·cos(φ₂)·h((λ₂-λ₁)) = h(Δσ)
Δσ = 2 * arcsin( sqrt(h(...)) )
distance = r * Δσ
```

**Constants:**
- Earth's radius (mean): r ≈ 6371 km or ≈ 3959 miles
- Angles (φ = latitude, λ = longitude) must be in **radians**; convert degrees: radians = degrees × π/180

**Reference sources:**
- Baeldung on Computer Science: https://www.baeldung.com/cs/haversine-formula
- Underground Mathematics (Cambridge): https://undergroundmathematics.org/trigonometry-compound-angles/the-great-circle-distance
- Tibco Spotfire: https://support.tibco.com/external/article/70531/how-to-calculate-the-great-circle-distan.html

**Python implementation (ready to code):**

```python
import numpy as np

def haversine(lat1, lon1, lat2, lon2, radius_km=6371):
    """
    Compute great-circle distance via haversine formula.
    
    Args:
        lat1, lon1: pickup latitude/longitude in degrees
        lat2, lon2: dropoff latitude/longitude in degrees
        radius_km: Earth radius in km (default 6371)
    
    Returns:
        distance in km
    """
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    h = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(h))
    distance = radius_km * c
    
    return distance

# Example: distance from Times Square (40.758, -73.9855) to Empire State Building (40.7484, -73.9857)
dist = haversine(40.758, -73.9855, 40.7484, -73.9857)
print(f"Distance: {dist:.3f} km")  # ~0.8 km
```

**Caveats / limits:**
- Haversine assumes Earth is a perfect sphere; actual Earth is an oblate spheroid, so errors up to ~0.5% are common. For taxi distances (<100 km), error is <500m (acceptable).
- Assumes WGS84 lat/lon coordinates (standard for GPS/mapping); if data uses a different projection, convert first.
- Does not account for actual road/street network; straight-line distance is always ≤ actual driving distance.

**Recommendation:**

**DATASET DECISION: SYNTHESISE.**

The official NYC TLC data is freely available but large (50GB), and single-month parquets (100MB–1GB) still require cloud access and parsing. Kaggle's cleaned taxi fare dataset is 5.5GB, still too large for a quick sandbox run.

Instead, **the SPEC-DS-5 chapter should synthesise a realistic small NYC taxi dataset:**

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def synthesize_nyc_taxi(n_rows=1000, seed=42):
    """
    Synthesize realistic NYC taxi trip data.
    
    NYC bounds (approx):
      Latitude: [40.58, 40.92]
      Longitude: [-74.26, -73.75]
    """
    np.random.seed(seed)
    
    # Generate random pickup locations within NYC
    pickup_lat = np.random.uniform(40.58, 40.92, n_rows)
    pickup_lon = np.random.uniform(-74.26, -73.75, n_rows)
    
    # Generate dropoff locations (within ~5-10 km)
    dropoff_lat = pickup_lat + np.random.normal(0, 0.05, n_rows)
    dropoff_lon = pickup_lon + np.random.normal(0, 0.05, n_rows)
    
    # Clamp to NYC bounds
    dropoff_lat = np.clip(dropoff_lat, 40.58, 40.92)
    dropoff_lon = np.clip(dropoff_lon, -74.26, -73.75)
    
    # Compute haversine distance
    distances = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    
    # Simulate fare: base $2.50 + $2.50 per mile + random noise
    base_fare = 2.50
    per_mile = 2.50
    fares = base_fare + per_mile * distances + np.random.normal(0, 1, n_rows)
    fares = np.maximum(fares, 2.50)  # Minimum fare
    
    # Random passenger count
    passenger_count = np.random.choice([1, 2, 3, 4, 5, 6], n_rows, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
    
    # Create DataFrame
    df = pd.DataFrame({
        'pickup_lat': pickup_lat,
        'pickup_lon': pickup_lon,
        'dropoff_lat': dropoff_lat,
        'dropoff_lon': dropoff_lon,
        'distance_km': distances,
        'passenger_count': passenger_count,
        'fare_amount': fares,
        'pickup_datetime': [datetime(2024, 1, 1) + timedelta(minutes=int(i*14.4)) for i in range(n_rows)],
    })
    
    return df

# Usage
df = synthesize_nyc_taxi(n_rows=1000)
print(df.head())
print(df.describe())
```

**Advantages of synthesis:**
- ✓ No download required; runs instantly in sandbox.
- ✓ Column schema matches real NYC taxi data (pickup/dropoff coords, distance, fare, etc.).
- ✓ Realistic range and distributions; learnable for the models.
- ✓ Reproducible (seed=42).

**How to document in the chapter:**
- State clearly: "To keep this example runnable in a sandbox without a multi-GB download, we synthesise realistic trip data using NYC-bounded random coordinates and the haversine formula."
- Cite haversine formula and Earth radius constant.
- Show the synthesis function; let readers modify `n_rows`, coordinate bounds, or fare formula to experiment.
- Optionally, include a "If you want real data" section pointing to NYC TLC or Kaggle for follow-up work.

**Alternative if a small real sample is needed later:**
- NYC TLC publishes sample files on their page; check for any publicly available CSV preview/sample.
- Kaggle sometimes provides small sample CSVs; search for "NYC taxi sample CSV" on Kaggle.
- Azure Open Datasets may have a smaller subset; verify access before relying on it.
