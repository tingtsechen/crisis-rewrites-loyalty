import pandas as pd
import numpy as np
from scipy import stats
import os


def calculate_panel_c(df):
    """COVID impact on brand-generic choices across political groups"""
    df = df.copy()  # Create a copy to avoid modifying original data
    df["covid_period"] = df["year"] == 2020

    # Calculate transaction changes for each group
    def calc_transaction_changes(data, drug_type):
        changes = {}
        for group in data["political_group"].unique():
            # Calculate total transaction for pre and post periods
            pre = data[
                (data["political_group"] == group)
                & (data["brand_generic"] == drug_type)
                & (~data["covid_period"])
            ].shape[0]

            post = data[
                (data["political_group"] == group)
                & (data["brand_generic"] == drug_type)
                & (data["covid_period"])
            ].shape[0]

            # Calculate percentage change
            changes[group] = ((post - pre) / pre * 100) if pre != 0 else 0
        return changes

    # Calculate brand share changes with standard errors
    def calc_brand_share_change(group):
        # Calculate brand share for pre and post periods
        pre_brand = (group[~group["covid_period"]]["brand_generic"] == "B").mean() * 100
        post_brand = (group[group["covid_period"]]["brand_generic"] == "B").mean() * 100

        # Calculate standard errors for pre and post periods
        n_pre = len(group[~group["covid_period"]])
        n_post = len(group[group["covid_period"]])

        # Standard error for proportion
        se_pre = np.sqrt((pre_brand / 100 * (1 - pre_brand / 100)) / n_pre) * 100
        se_post = np.sqrt((post_brand / 100 * (1 - post_brand / 100)) / n_post) * 100

        # Standard error for the difference
        se_diff = np.sqrt(se_pre**2 + se_post**2)

        return pd.Series({"change": post_brand - pre_brand, "se": se_diff})

    # Calculate changes for both drug types
    generic_changes = calc_transaction_changes(df, "G")
    brand_changes = calc_transaction_changes(df, "B")
    brand_share_changes = df.groupby("political_group").apply(calc_brand_share_change)

    print("\n generic transaction changes:")
    print(generic_changes)

    print("\n brand transaction changes:")
    print(brand_changes)

    print("\nBrand Share Changes with Standard Errors:")
    print("----------------------------------------")
    for group in brand_share_changes.index:
        if group in [
            "Democrat-leaning",
            "Republican-leaning",
        ]:  # Only print Democrat and Republican
            change = brand_share_changes.loc[group, "change"]
            se = brand_share_changes.loc[group, "se"]
            print(f"{group:.<20} {change:>8.2f} ± {se:>6.2f} percentage points")
    print("----------------------------------------")

    # Create comparison table
    print("\nComparison Table: Democrat vs Republican Changes")
    print("-" * 60)
    print(
        f"{'Drug Type':<15} {'Democrat-leaning':>15} {'Republican-leaning':>20} {'Difference':>15}"
    )
    print("-" * 60)

    # Calculate and print Generic row
    dem_generic = generic_changes["Democrat-leaning"]
    rep_generic = generic_changes["Republican-leaning"]
    diff_generic = dem_generic - rep_generic
    print(
        f"{'Generic':<15} {dem_generic:>14.2f}% {rep_generic:>19.2f}% {diff_generic:>14.2f}%"
    )

    # Calculate and print Brand row
    dem_brand = brand_changes["Democrat-leaning"]
    rep_brand = brand_changes["Republican-leaning"]
    diff_brand = dem_brand - rep_brand
    print(f"{'Brand':<15} {dem_brand:>14.2f}% {rep_brand:>19.2f}% {diff_brand:>14.2f}%")
    print("-" * 60)


def add_political_groups(df):
    """Categorize counties based on political ideology"""
    terciles = df["republican_percentage"].quantile([0, 0.33, 0.67, 1])
    labels = ["Democrat-leaning", "Competitive", "Republican-leaning"]
    df["political_group"] = pd.cut(
        df["republican_percentage"],
        bins=terciles,
        labels=labels,
        include_lowest=True,
    )
    return df


# File path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
data_path = os.path.join(
    DATA_DIR,
    "processed",
    "merged_data_Gclass_SldDt_04-06_2019-2020_8_states_drugCategory_county_election2020_ODY-loc.csv",
)

# Process data
sexual_drug_G03 = ["ANDROGENS-ANABOLIC", "CONTRACEPTIVES", "ESTROGENS", "PROGESTINS"]

# Read and process data
df = pd.read_csv(data_path)
df_G03 = add_political_groups(df[df["gpi_2_desc"].isin(sexual_drug_G03)])

# Run the analysis
calculate_panel_c(df_G03)
