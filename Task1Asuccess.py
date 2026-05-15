import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import os

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
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in ['int64', 'float64']:
            df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)
        elif df_cleaned[column].dtype == 'object':
            df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else "Unknown", inplace=True)
    return df_cleaned

df_games = cleaning(df_games)

# Compute Win/Loss
df_games["win"] = (df_games["team_score"] > df_games["opponent_team_score"]).astype(int)

# Compute Team Metrics
df_games['fgr_2'] = df_games["fgm_2"] / df_games["fga_2"].replace(0, np.nan)
df_games['fgr_3'] = df_games["fgm_3"] / df_games["fga_3"].replace(0, np.nan)
df_games['ftr'] = df_games["ftm"] / df_games["fta"].replace(0, np.nan)

# Merge Region Data
df_games = df_games.merge(df_regions, on="team", how="left")

# Compute team-wise statistics
numeric_cols = df_games.select_dtypes(include=[np.number]).columns.tolist()
team_avg_df = df_games.groupby("team")[numeric_cols].mean().reset_index()

# Compute Win Percentage
df_team_win_pct = df_games.groupby("team")["win"].mean().reset_index().rename(columns={"win": "win_pct"})

# Merge statistics and win percentage
df_team_avg = team_avg_df.merge(df_team_win_pct, on="team", how="left").merge(df_regions, on="team", how="left")

# Scaling Function
def scale_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    robust_scaler = RobustScaler()
    minmax_scaler = MinMaxScaler()
    df_scaled = df.copy()

    for col in numeric_cols:
        if col in df_scaled.columns:
            col_data = df[col].fillna(0).to_numpy().reshape(-1, 1)
            robust_scaled = robust_scaler.fit_transform(col_data)
            final_scaled = minmax_scaler.fit_transform(robust_scaled)
            df_scaled[f'{col}_final_scaled'] = final_scaled.flatten()

    return df_scaled

df_team_avg = scale_features(df_team_avg)

# Compute Performance Rating (Win% scaled 0-100)
scaler = MinMaxScaler(feature_range=(0, 100))
df_team_avg["performance_rating"] = scaler.fit_transform(df_team_avg[["win_pct"]])

# Compute Feature Importance for Offense, Defense, and External Factors
offense_feats = ["fgr_2_final_scaled", "fgr_3_final_scaled", "ftr_final_scaled", "ast_final_scaled", 
                 "largest_lead_final_scaled", "tov_final_scaled", "oreb_final_scaled", "dreb_final_scaled"]
defense_feats = ["dreb_final_scaled", "blk_final_scaled", "stl_final_scaled", "f_tech_final_scaled", "f_personal_final_scaled"]
ext_feats = ["rest_days_final_scaled", "ot_length_min_tot_final_scaled", "tz_dif_h_e_final_scaled", "prev_game_dist_final_scaled", "home_away_ns_final_scaled", "travel_dist_final_scaled"]

def get_feature_importance(features, target, df):
    df = df.dropna(subset=[target])
    features = [feat for feat in features if feat in df.columns]

    if not features:
        return {}

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
    model.fit(X_train, y_train)

    return dict(zip(features, model.feature_importances_))

offense_results = get_feature_importance(offense_feats, "team_score_final_scaled", df_team_avg)
defense_results = get_feature_importance(defense_feats, "opponent_team_score_final_scaled", df_team_avg)
ext_results = get_feature_importance(ext_feats, "team_score_final_scaled", df_team_avg)

# Compute Ratings
def compute_rating(df, weights):
    scores = [sum(getattr(row, col, 0) * weights.get(col, 0) for col in weights) for row in df.itertuples(index=False)]
    return MinMaxScaler(feature_range=(0, 100)).fit_transform(np.array(scores).reshape(-1, 1)).flatten()

# Compute Weightage for OVR using Win/Loss as Target
def compute_rtg_weights(df):
    features = ["offense_rating", "defense_rating", "extra_rating"]
    target = "win"
    df = df.dropna(subset=[target])
    
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
    model.fit(X_train, y_train)
    
    feature_importances = dict(zip(features, model.feature_importances_))
    total_importance = sum(feature_importances.values())
    return {feat: imp / total_importance for feat, imp in feature_importances.items()}

# Compute Ratings Per Region
regions = df_team_avg["region"].dropna().unique()
region_dfs = {region: df_team_avg[df_team_avg["region"] == region].sort_values(by="team").reset_index(drop=True) for region in regions}

alpha = 0.6  # Weight for Static OVR in Hybrid OVR

# Apply Computations to Each Region
for region, region_df in region_dfs.items():
    region_df["offense_rating"] = compute_rating(region_df, offense_results)
    region_df["defense_rating"] = compute_rating(region_df, defense_results)
    region_df["extra_rating"] = compute_rating(region_df, ext_results)

    # Compute Dynamic OVR Weights
    rtg_weights = compute_rtg_weights(region_df)
    W_o, W_d, W_e = [rtg_weights.get(f, 0) for f in ["offense_rating", "defense_rating", "extra_rating"]]

    # Compute OVR
    region_df["OVR"] = (W_o * region_df["offense_rating"]) + (W_d * region_df["defense_rating"]) + (W_e * region_df["extra_rating"])

    # Compute Hybrid OVR
    region_df["Hybrid_OVR"] = alpha * region_df["OVR"] + (1 - alpha) * region_df["performance_rating"]

    # Save to CSV
    region_df.to_csv(f"/Users/sparshverma/Documents/Wharton_Files/{region}_Teams_with_Hybrid_OVR_success.csv", index=False)

print("✅ All regional team Hybrid OVR files have been successfully saved! 🎉")