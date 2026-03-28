"""
SageMaker Training Script — Call Center Daily Forecasting
Trains ensemble of HistGradientBoosting + Ridge per portfolio per metric.
Saves models and predictions to /opt/ml/model/
"""
import argparse
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---- Feature Engineering (must match notebook) ----

US_HOLIDAYS = pd.to_datetime([
    '2024-01-01', '2024-01-15', '2024-02-19', '2024-05-27', '2024-06-19',
    '2024-07-04', '2024-09-02', '2024-10-14', '2024-11-11', '2024-11-28',
    '2024-12-25',
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-06-19',
    '2025-07-04', '2025-09-01', '2025-10-13', '2025-11-11', '2025-11-27',
    '2025-12-25',
])

PORTFOLIOS = ['A', 'B', 'C', 'D']
TARGET_METRICS = ['Call Volume', 'CCT', 'Abandon Rate']


def build_features(df, staffing_df=None, portfolio=None):
    feat = pd.DataFrame(index=df.index)
    feat['Date'] = df['Date']

    feat['day_of_week'] = df['Date'].dt.dayofweek
    feat['day_of_month'] = df['Date'].dt.day
    feat['month'] = df['Date'].dt.month
    feat['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    feat['quarter'] = df['Date'].dt.quarter
    feat['year'] = df['Date'].dt.year
    feat['is_weekend'] = (feat['day_of_week'] >= 5).astype(int)
    feat['is_monday'] = (feat['day_of_week'] == 0).astype(int)
    feat['is_friday'] = (feat['day_of_week'] == 4).astype(int)

    feat['dow_sin'] = np.sin(2 * np.pi * feat['day_of_week'] / 7)
    feat['dow_cos'] = np.cos(2 * np.pi * feat['day_of_week'] / 7)
    feat['dom_sin'] = np.sin(2 * np.pi * feat['day_of_month'] / 31)
    feat['dom_cos'] = np.cos(2 * np.pi * feat['day_of_month'] / 31)
    feat['month_sin'] = np.sin(2 * np.pi * feat['month'] / 12)
    feat['month_cos'] = np.cos(2 * np.pi * feat['month'] / 12)

    feat['is_holiday'] = df['Date'].isin(US_HOLIDAYS).astype(int)

    dates = df['Date'].values
    holiday_arr = US_HOLIDAYS.values
    days_since = []
    days_until = []
    for d in dates:
        past = holiday_arr[holiday_arr <= d]
        future = holiday_arr[holiday_arr >= d]
        ds = (d - past[-1]) / np.timedelta64(1, 'D') if len(past) > 0 else 365
        du = (future[0] - d) / np.timedelta64(1, 'D') if len(future) > 0 else 365
        days_since.append(ds)
        days_until.append(du)
    feat['days_since_holiday'] = days_since
    feat['days_until_holiday'] = days_until

    feat['is_month_start_week'] = (feat['day_of_month'] <= 5).astype(int)
    feat['is_month_end_week'] = (feat['day_of_month'] >= 26).astype(int)
    feat['is_first_of_month'] = (feat['day_of_month'] == 1).astype(int)

    day_index = (df['Date'] - df['Date'].min()).dt.days
    for k in range(1, 4):
        feat[f'fourier_weekly_sin_{k}'] = np.sin(2 * np.pi * k * day_index / 7)
        feat[f'fourier_weekly_cos_{k}'] = np.cos(2 * np.pi * k * day_index / 7)
    for k in range(1, 4):
        feat[f'fourier_yearly_sin_{k}'] = np.sin(2 * np.pi * k * day_index / 365.25)
        feat[f'fourier_yearly_cos_{k}'] = np.cos(2 * np.pi * k * day_index / 365.25)

    for metric in TARGET_METRICS:
        if metric in df.columns:
            feat[f'{metric}_lag7'] = df[metric].shift(7)
            feat[f'{metric}_lag14'] = df[metric].shift(14)
            feat[f'{metric}_lag28'] = df[metric].shift(28)
            feat[f'{metric}_lag365'] = df[metric].shift(365)
            feat[f'{metric}_roll7'] = df[metric].rolling(7).mean()
            feat[f'{metric}_roll14'] = df[metric].rolling(14).mean()
            feat[f'{metric}_roll30'] = df[metric].rolling(30).mean()
            feat[f'{metric}_std7'] = df[metric].rolling(7).std()
            feat[f'{metric}_ewm7'] = df[metric].ewm(span=7).mean()

    if staffing_df is not None and portfolio is not None:
        staff_col = f'Staff_{portfolio}'
        if staff_col in staffing_df.columns:
            staff_merge = staffing_df[['Date', staff_col]].rename(columns={staff_col: 'num_agents'})
            feat = feat.merge(staff_merge, on='Date', how='left')
            feat['agents_change'] = feat['num_agents'].diff()

    for metric in TARGET_METRICS:
        if metric in df.columns:
            feat[f'target_{metric}'] = df[metric]

    return feat


def get_feature_cols(feat_df):
    """Get columns that are features (not Date or targets)."""
    exclude = ['Date'] + [f'target_{m}' for m in TARGET_METRICS]
    return [c for c in feat_df.columns if c not in exclude]


def train_models(feat_df, quantile=0.55):
    """Train ensemble models for each metric. Returns dict of models and validation scores."""
    feature_cols = get_feature_cols(feat_df)
    models = {}
    scores = {}

    # Use data with lags available (drop first 365 rows for lag_365)
    valid_mask = feat_df[feature_cols].notna().all(axis=1)
    clean_df = feat_df[valid_mask].copy()

    # Time series split: train on everything before July 2025, validate on July 2025
    train_mask = clean_df['Date'] < '2025-07-01'
    val_mask = (clean_df['Date'] >= '2025-07-01') & (clean_df['Date'] < '2025-08-01')

    X_train = clean_df.loc[train_mask, feature_cols].values
    X_val = clean_df.loc[val_mask, feature_cols].values

    for metric in TARGET_METRICS:
        target_col = f'target_{metric}'
        if target_col not in clean_df.columns:
            continue

        y_train = clean_df.loc[train_mask, target_col].values
        y_val = clean_df.loc[val_mask, target_col].values

        # Model 1: HistGradientBoosting with quantile loss (asymmetric bias)
        hgb = HistGradientBoostingRegressor(
            loss='quantile',
            quantile=quantile,
            max_iter=500,
            max_depth=6,
            learning_rate=0.05,
            l2_regularization=1.0,
            min_samples_leaf=10,
            random_state=42,
        )
        hgb.fit(X_train, y_train)

        # Model 2: Ridge with same features
        ridge = Ridge(alpha=1.0)
        # Fill NaN for Ridge (shouldn't happen after cleaning but just in case)
        X_train_r = np.nan_to_num(X_train, 0)
        X_val_r = np.nan_to_num(X_val, 0)
        ridge.fit(X_train_r, y_train)

        # Ensemble predictions on validation
        pred_hgb = hgb.predict(X_val)
        pred_ridge = ridge.predict(X_val_r)
        pred_ensemble = 0.6 * pred_hgb + 0.4 * pred_ridge

        mae = mean_absolute_error(y_val, pred_ensemble)
        rmse = np.sqrt(mean_squared_error(y_val, pred_ensemble))

        models[metric] = {'hgb': hgb, 'ridge': ridge, 'feature_cols': feature_cols}
        scores[metric] = {'mae': mae, 'rmse': rmse}

        print(f"  {metric}: Val MAE={mae:.2f}, RMSE={rmse:.2f}")

    return models, scores


def predict_august(feat_df, models, daily_data_df):
    """Generate daily predictions for August 2025."""
    predictions = {}

    # Build August 2025 dates
    aug_dates = pd.date_range('2025-08-01', '2025-08-31', freq='D')

    for metric in TARGET_METRICS:
        if metric not in models:
            continue

        model_info = models[metric]
        feature_cols = model_info['feature_cols']

        # For August, we need features. Use the feature DataFrame which includes Aug dates
        aug_mask = (feat_df['Date'] >= '2025-08-01') & (feat_df['Date'] <= '2025-08-31')
        X_aug = feat_df.loc[aug_mask, feature_cols]

        # Handle NaN in lag features for August (fill with last known values)
        X_aug_filled = X_aug.fillna(method='ffill').fillna(method='bfill').fillna(0)

        pred_hgb = model_info['hgb'].predict(X_aug_filled.values)
        pred_ridge = model_info['ridge'].predict(np.nan_to_num(X_aug_filled.values, 0))
        pred_ensemble = 0.6 * pred_hgb + 0.4 * pred_ridge

        # Also compute baseline: Aug 2024 scaled by YoY growth
        aug24 = daily_data_df[(daily_data_df['Date'].dt.month == 8) & (daily_data_df['Date'].dt.year == 2024)]
        jan_jul_24 = daily_data_df[(daily_data_df['Date'].dt.year == 2024) &
                                    (daily_data_df['Date'].dt.month >= 1) &
                                    (daily_data_df['Date'].dt.month <= 7)]
        jan_jul_25 = daily_data_df[(daily_data_df['Date'].dt.year == 2025) &
                                    (daily_data_df['Date'].dt.month >= 1) &
                                    (daily_data_df['Date'].dt.month <= 7)]

        if len(jan_jul_24) > 0 and len(jan_jul_25) > 0 and len(aug24) > 0:
            growth = jan_jul_25[metric].mean() / jan_jul_24[metric].mean() if jan_jul_24[metric].mean() > 0 else 1.0
            # Match Aug 2024 days to Aug 2025 by day-of-week alignment
            aug24_vals = aug24[metric].values
            baseline = np.zeros(31)
            for i, d in enumerate(aug_dates):
                # Find matching day-of-week in Aug 2024
                target_dow = d.dayofweek
                aug24_dows = aug24['Date'].dt.dayofweek.values
                matching = np.where(aug24_dows == target_dow)[0]
                if len(matching) > 0:
                    baseline[i] = aug24_vals[matching].mean() * growth
                else:
                    baseline[i] = aug24_vals.mean() * growth
        else:
            baseline = pred_ensemble  # fallback

        # Final ensemble with baseline
        final = 0.50 * pred_ensemble + 0.25 * pred_ensemble + 0.25 * baseline[:len(pred_ensemble)]
        # Simplifies to: 0.75 * ensemble + 0.25 * baseline
        predictions[metric] = final

    return aug_dates[:len(list(predictions.values())[0])], predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantile', type=float, default=0.55)
    parser.add_argument('--bias-factor', type=float, default=1.05)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    args = parser.parse_args()

    print(f"Training with quantile={args.quantile}, bias_factor={args.bias_factor}")
    print(f"Data dir: {args.data_dir}")
    print(f"Model dir: {args.model_dir}")

    # Load pre-processed data from CSV (uploaded by notebook)
    all_models = {}
    all_predictions = {}
    all_scores = {}

    for p in PORTFOLIOS:
        print(f"\n{'='*60}")
        print(f"Training Portfolio {p}")
        print(f"{'='*60}")

        daily_path = os.path.join(args.data_dir, f'daily_{p}.csv')
        staffing_path = os.path.join(args.data_dir, 'staffing.csv')

        daily_df = pd.read_csv(daily_path, parse_dates=['Date'])
        staffing_df = pd.read_csv(staffing_path, parse_dates=['Date'])

        # Build features
        feat_df = build_features(daily_df, staffing_df, p)

        # Train models
        models, scores = train_models(feat_df, quantile=args.quantile)
        all_models[p] = models
        all_scores[p] = scores

        # Predict August 2025
        aug_dates, preds = predict_august(feat_df, models, daily_df)
        all_predictions[p] = {'dates': aug_dates, 'predictions': preds}

        for metric, vals in preds.items():
            print(f"  Aug 2025 {metric}: mean={np.mean(vals):.1f}, min={np.min(vals):.1f}, max={np.max(vals):.1f}")

    # Save everything
    output = {
        'predictions': {},
        'scores': all_scores,
        'quantile': args.quantile,
        'bias_factor': args.bias_factor,
    }
    for p in PORTFOLIOS:
        output['predictions'][p] = {
            'dates': all_predictions[p]['dates'].strftime('%Y-%m-%d').tolist(),
            'Call Volume': all_predictions[p]['predictions'].get('Call Volume', np.zeros(31)).tolist(),
            'CCT': all_predictions[p]['predictions'].get('CCT', np.zeros(31)).tolist(),
            'Abandon Rate': all_predictions[p]['predictions'].get('Abandon Rate', np.zeros(31)).tolist(),
        }

    # Save as JSON for easy loading
    with open(os.path.join(args.model_dir, 'predictions.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Save models as pickle
    with open(os.path.join(args.model_dir, 'models.pkl'), 'wb') as f:
        pickle.dump(all_models, f)

    print("\nTraining complete. Artifacts saved to", args.model_dir)
