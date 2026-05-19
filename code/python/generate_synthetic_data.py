"""
Generate synthetic G03/C01/C02/C10 panels with identical schema to the
real proprietary data, but with simulated values drawn from independent
distributions.

This dataset is for CODE EXECUTION VERIFICATION ONLY.

It allows reviewers to confirm that all R and Python analysis scripts run
end-to-end without errors. Coefficients, p-values, and figures produced
from this synthetic dataset are statistically meaningless — they do NOT
reproduce the findings reported in the paper.

The proprietary prescription panel is under a Data Use Agreement and cannot
be redistributed. See data/README.md for access information.

Usage:
    python code/python/generate_synthetic_data.py                    # default: small (100 rows each)
    python code/python/generate_synthetic_data.py --full             # full size (matches real N)
    python code/python/generate_synthetic_data.py --output data/synthetic
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Real-data row counts (for --full mode)
N_FULL = {"G03": 22_737, "C01": 12_377, "C02": 20_113, "C10": 21_841}

# Eight southeastern U.S. states in the study
STATE_ABBR = ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"]
STATE_NAME = {
    "AL": "Alabama", "FL": "Florida", "GA": "Georgia", "KY": "Kentucky",
    "MS": "Mississippi", "NC": "North Carolina", "SC": "South Carolina", "TN": "Tennessee",
}
# Approximate FIPS prefixes for each state
STATE_FIPS_PREFIX = {
    "AL": 1, "FL": 12, "GA": 13, "KY": 21,
    "MS": 28, "NC": 37, "SC": 45, "TN": 47,
}

# Columns common to all four ATC categories (C01/C02/C10 stop here).
COMMON_COLS = [
    "fips", "sld_dt", "republican_percentage", "is_brand",
    "dspns_qty_sum", "dspns_qty_mean", "dspns_qty_count",
    "rx_prc_amt_sum", "rx_prc_amt_mean", "rx_prc_amt_median",
    "transaction_count", "loc_id_nunique",
    "insurance_coverage_mean", "patient_coverage_mean",
    "year", "healthrisk_impact",
    "republican_percentage_std", "republicanStd_x_healthrisk",
    "brand_healthrisk", "republicanStd_x_brand", "republicanStd_x_brand_x_healthrisk",
    "avg_prc",
    "county_name", "state_name", "state_abbr",
    "total_county_population",
    "edu_less_than_hs", "edu_hs_graduate", "edu_some_college",
    "edu_bachelors", "edu_graduate", "edu_bachelor_above",
    "male_percentage", "female_percentage", "over_65_percentage",
    "income_low_percentage", "income_lower_middle_percentage",
    "income_middle_percentage", "income_upper_middle_percentage",
    "income_high_percentage", "household_median",
    "race_white", "race_black", "race_native", "race_asian",
    "race_pacific", "race_other", "race_two_or_more",
    "unemployment_rate", "employment_rate",
    "urban_rural_value_lessBetter", "metro_indicator",
    "cases", "deaths",
    "week", "month", "weekday",
    "pharmacy_density",
]

# Extra interaction columns present in C01/C02/C10 only.
NONSTD_INTERACTION_COLS = [
    "republican_x_brand", "republican_x_brand_x_healthrisk", "republican_x_healthrisk",
]

# G03-only drug subcategory volume indicators.
G03_DRUG_COLS = [
    "drug_ANDROGENS_ANABOLIC_sum", "drug_ESTROGENS_sum",
    "drug_PROGESTINS_sum", "drug_CONTRACEPTIVES_sum",
]


def simulate_panel(category: str, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate one synthetic panel matching the schema of the real data."""
    state = rng.choice(STATE_ABBR, size=n_rows)
    fips = np.array([STATE_FIPS_PREFIX[s] * 1000 + rng.integers(1, 200) for s in state])

    base_date = pd.Timestamp("2019-04-01")
    day_offset = rng.integers(0, 91, size=n_rows)
    year_flag = rng.integers(0, 2, size=n_rows)
    sld_dt = [base_date + pd.Timedelta(days=int(d) + int(y) * 365) for d, y in zip(day_offset, year_flag)]
    year = np.array([dt.year for dt in sld_dt])
    healthrisk = (year == 2020).astype(int)

    republican = rng.uniform(15.0, 85.0, size=n_rows)
    republican_std = (republican - republican.mean()) / republican.std()
    is_brand = rng.integers(0, 2, size=n_rows)

    dspns_count = rng.integers(1, 50, size=n_rows)
    dspns_mean = rng.lognormal(mean=3.4, sigma=0.7, size=n_rows)
    dspns_sum = dspns_mean * dspns_count

    rx_mean = rng.lognormal(mean=3.5, sigma=1.2, size=n_rows)
    rx_sum = rx_mean * dspns_count

    df = pd.DataFrame({
        "fips": fips,
        "sld_dt": [dt.strftime("%Y-%m-%d") for dt in sld_dt],
        "republican_percentage": republican,
        "is_brand": is_brand,
        "dspns_qty_sum": dspns_sum,
        "dspns_qty_mean": dspns_mean,
        "dspns_qty_count": dspns_count,
        "rx_prc_amt_sum": rx_sum,
        "rx_prc_amt_mean": rx_mean,
        "rx_prc_amt_median": rx_mean * rng.uniform(0.5, 1.2, size=n_rows),
        "transaction_count": dspns_count,
        "loc_id_nunique": rng.integers(1, 16, size=n_rows),
        "insurance_coverage_mean": rng.uniform(0.0, 1.0, size=n_rows),
        "patient_coverage_mean": rng.uniform(0.0, 1.0, size=n_rows),
        "year": year,
        "healthrisk_impact": healthrisk,
        "republican_percentage_std": republican_std,
        "republicanStd_x_healthrisk": republican_std * healthrisk,
        "brand_healthrisk": is_brand * healthrisk,
        "republicanStd_x_brand": republican_std * is_brand,
        "republicanStd_x_brand_x_healthrisk": republican_std * is_brand * healthrisk,
        "avg_prc": rng.lognormal(mean=2.0, sigma=1.3, size=n_rows),
        "county_name": [f"County_{rng.integers(1, 200)}" for _ in range(n_rows)],
        "state_name": [STATE_NAME[s] for s in state],
        "state_abbr": state,
        "total_county_population": rng.integers(10_000, 2_700_000, size=n_rows),
        "edu_less_than_hs": rng.uniform(5.0, 31.0, size=n_rows),
        "edu_hs_graduate": rng.uniform(16.0, 50.0, size=n_rows),
        "edu_some_college": rng.uniform(15.0, 37.0, size=n_rows),
        "edu_bachelors": rng.uniform(3.0, 31.0, size=n_rows),
        "edu_graduate": rng.uniform(2.5, 22.0, size=n_rows),
        "edu_bachelor_above": rng.uniform(7.0, 46.0, size=n_rows),
        "male_percentage": rng.uniform(45.0, 59.0, size=n_rows),
        "female_percentage": rng.uniform(41.0, 55.0, size=n_rows),
        "over_65_percentage": rng.uniform(4.0, 16.0, size=n_rows),
        "income_low_percentage": rng.uniform(11.0, 41.0, size=n_rows),
        "income_lower_middle_percentage": rng.uniform(15.0, 37.0, size=n_rows),
        "income_middle_percentage": rng.uniform(22.0, 38.0, size=n_rows),
        "income_upper_middle_percentage": rng.uniform(5.0, 21.0, size=n_rows),
        "income_high_percentage": rng.uniform(2.0, 25.0, size=n_rows),
        "household_median": rng.integers(32_000, 84_000, size=n_rows),
        "race_white": rng.uniform(27.0, 94.0, size=n_rows),
        "race_black": rng.uniform(1.0, 72.0, size=n_rows),
        "race_native": rng.uniform(0.0, 2.0, size=n_rows),
        "race_asian": rng.uniform(0.1, 7.0, size=n_rows),
        "race_pacific": rng.uniform(0.0, 0.3, size=n_rows),
        "race_other": rng.uniform(0.0, 11.0, size=n_rows),
        "race_two_or_more": rng.uniform(0.5, 11.0, size=n_rows),
        "unemployment_rate": rng.uniform(0.5, 7.0, size=n_rows),
        "employment_rate": rng.uniform(21.0, 68.0, size=n_rows),
        "urban_rural_value_lessBetter": rng.integers(1, 10, size=n_rows),
        "metro_indicator": rng.integers(0, 2, size=n_rows),
        "cases": rng.poisson(lam=200, size=n_rows) * healthrisk,
        "deaths": rng.poisson(lam=10, size=n_rows) * healthrisk,
        "week": rng.integers(14, 28, size=n_rows),
        "month": rng.integers(4, 7, size=n_rows),
        "weekday": rng.integers(0, 7, size=n_rows),
        "pharmacy_density": rng.uniform(0.001, 1.0, size=n_rows),
    })

    if category == "G03":
        for col in G03_DRUG_COLS:
            df[col] = rng.poisson(lam=2.0, size=n_rows)
        column_order = (
            COMMON_COLS[:12]
            + G03_DRUG_COLS
            + COMMON_COLS[12:]
        )
    else:
        for col in NONSTD_INTERACTION_COLS:
            df[col] = df["republican_percentage"] * df["is_brand"] * (df["healthrisk_impact"] if "healthrisk" in col else 1)
        column_order = COMMON_COLS[:46] + NONSTD_INTERACTION_COLS + COMMON_COLS[46:]

    return df[column_order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic",
                        help="Output directory (default: data/synthetic)")
    parser.add_argument("--full", action="store_true",
                        help="Generate full-size synthetic data matching real N. Default is 100 rows each.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in ("G03", "C01", "C02", "C10"):
        n = N_FULL[cat] if args.full else 100
        df = simulate_panel(cat, n, rng)
        out_path = out_dir / f"synthetic_merged_sldDt_{cat}_daily_covid.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path}  ({n:,} rows, {df.shape[1]} cols)")

    print()
    print(f"Synthetic panels written to {out_dir.resolve()}")
    print("WARNING: These data are for CODE EXECUTION VERIFICATION ONLY.")
    print("Statistical results from synthetic data are NOT meaningful.")


if __name__ == "__main__":
    main()
