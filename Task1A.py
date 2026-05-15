import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Load datasets
games_file_path = "/Users/sparshverma/Documents/Wharton_Files/games_2022 - games_2022.csv"
regions_file_path = "/Users/sparshverma/Documents/Wharton_Files/Team Region Groups - Sheet1.csv"

df_games = pd.read_csv(games_file_path)
df_regions = pd.read_csv(regions_file_path)

# Standardizing column names
df_games.columns = df_games.columns.str.lower().str.replace(" ", "_", regex=True)
df_regions.columns = df_regions.columns.str.lower().str.replace(" ", "_", regex=True)

# Data Cleaning Function
def cleaning(df):
    df_cleaned = df.drop_duplicates()

    # Handling missing values
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in ['int64', 'float64']:
            df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)
        elif df_cleaned[column].dtype == 'object':
            df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else "Unknown", inplace=True)

    return df_cleaned

df_games = cleaning(df_games)

# Add calculated features safely (avoid division by zero)
df_games['fgr_2'] = df_games["fgm_2"] / df_games["fga_2"].replace(0, np.nan)
df_games['fgr_3'] = df_games["fgm_3"] / df_games["fga_3"].replace(0, np.nan)
df_games['ftr'] = df_games["ftm"] / df_games["fta"].replace(0, np.nan)

# Merge Region Data
df_games = df_games.merge(df_regions, on="team", how="left")

# Compute team-wise average (excluding non-numeric columns)
numeric_cols = df_games.select_dtypes(include=[np.number]).columns.tolist()
team_avg_df = df_games.groupby("team")[numeric_cols].mean().reset_index()

# Merge region data into the team-level dataframe
df_team_avg = team_avg_df.merge(df_regions, on="team", how="left")

# OVR Function with Team Average Scaling
def ovr(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Initialize scalers
    robust_scaler = RobustScaler()
    minmax_scaler = MinMaxScaler()

    # Apply scaling to team averages
    team_avg_scaled = df.copy()

    for col in numeric_cols:
        if col in team_avg_scaled.columns:
            col_data = df[col].fillna(0).to_numpy().reshape(-1, 1)  # Replace NaNs with 0
            robust_scaled = robust_scaler.fit_transform(col_data)
            final_scaled = minmax_scaler.fit_transform(robust_scaled)
            team_avg_scaled[f'{col}_final_scaled'] = final_scaled.flatten()

    return team_avg_scaled

# Apply OVR Function to team-level dataset
df_team_avg = ovr(df_team_avg)

# Define weighting function using CatBoost
def weighting(features, target, df):
    df = df.dropna(subset=[target])  # Ensure target column has no NaN values

    if target not in df.columns:
        return f"Error: Target variable '{target}' not found."

    # Filter features that exist in the dataset
    features = [feat for feat in features if feat in df.columns]

    if not features:
        return f"Error: No valid features found for target '{target}'"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
    model.fit(X_train, y_train)

    return dict(zip(features, model.feature_importances_))

# Compute Weighting
offense_feats = ["fgr_2_final_scaled", "fgr_3_final_scaled", "ftr_final_scaled", "ast_final_scaled", 
                 "largest_lead_final_scaled", "tov_final_scaled", "oreb_final_scaled", "dreb_final_scaled"]
defense_feats = ["dreb_final_scaled", "blk_final_scaled", "stl_final_scaled", "f_tech_final_scaled", "f_personal_final_scaled"]
ext_feats = ["rest_days_final_scaled", "ot_length_min_tot_final_scaled", "tz_dif_h_e_final_scaled", "prev_game_dist_final_scaled", "home_away_ns_final_scaled", "travel_dist_final_scaled"]

offense_results = weighting(offense_feats, "team_score_final_scaled", df_team_avg)
defense_results = weighting(defense_feats, "opponent_team_score_final_scaled", df_team_avg)
ext_results = weighting(ext_feats, "team_score_final_scaled", df_team_avg)

# Compute total feature importance for each category
total_offense_weight = sum(offense_results.values()) if offense_results else 1
total_defense_weight = sum(defense_results.values()) if defense_results else 1
total_ext_weight = sum(ext_results.values()) if ext_results else 1

# Normalize the weights to sum to 1
total_weight = total_offense_weight + total_defense_weight + total_ext_weight
W_o = total_offense_weight / total_weight
W_d = total_defense_weight / total_weight
W_e = total_ext_weight / total_weight

# Compute Ratings
def make_rtg(final_df, weights):
    modified_values = []

    for row in final_df.itertuples(index=False):
        weighted_sum = sum(getattr(row, col, 0) * weights.get(col, 0) for col in weights)
        modified_values.append(weighted_sum)

    # Scale ratings between 0 and 100
    scaled_rtg = MinMaxScaler(feature_range=(0, 100)).fit_transform(np.array(modified_values).reshape(-1, 1))
    
    return scaled_rtg.flatten()

df_team_avg["offense_rating"] = make_rtg(df_team_avg, offense_results)
df_team_avg["defense_rating"] = make_rtg(df_team_avg, defense_results)
df_team_avg["extra_rating"] = make_rtg(df_team_avg, ext_results)

# Compute OVR using weighted feature importance
df_team_avg["OVR"] = (W_o * df_team_avg["offense_rating"]) + \
                      (W_d * df_team_avg["defense_rating"]) + \
                      (W_e * df_team_avg["extra_rating"])

# Save updated dataset
df_team_avg.to_csv("Team_Average_with_Weighted_OVR.csv", index=False)

# Print adjusted weights
print(f"🔹 Offense Weight: {W_o:.3f}, Defense Weight: {W_d:.3f}, Extra Weight: {W_e:.3f}")

# Segregate by Region
regions = df_team_avg["region"].dropna().unique()
region_dfs = {region: df_team_avg[df_team_avg["region"] == region] for region in regions}

# Save regional data
for region, region_df in region_dfs.items():
    region_df.to_csv(f"{region}_Team_Avg.csv", index=False)

# Print OVR Ratings of Each Region
for region, region_df in region_dfs.items():
    print(f"\n🔹 {region} Region Ratings (Weighted OVR):\n")
    print(region_df[["team", "offense_rating", "defense_rating", "extra_rating", "OVR"]].sort_values(by="OVR", ascending=False).head())

import os

# Define the folder where the CSVs will be saved
output_folder = "/Users/sparshverma/Documents/Wharton_Files"

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save regional data as CSV files in the output folder
for region, region_df in region_dfs.items():
    filename = f"{output_folder}/{region}_Team_Avg.csv"
    region_df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename}")

print("\nAll regional files have been successfully saved! 🎉")