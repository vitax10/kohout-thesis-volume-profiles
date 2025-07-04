import os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import MAE
import pickle
import uuid
import glob
import torch
import gc
import logging
import sys
from contextlib import contextmanager
import warnings
import tensorflow as tf
import pytorch_lightning as pl

# Suppressing warnings
warnings.filterwarnings("ignore")

# Suppressing Python warnings
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppressing TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disabling oneDNN custom operations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppressing TensorFlow logs
tf.get_logger().setLevel("ERROR")

# Suppressing PyTorch Lightning and NeuralForecast output
pl.utilities.seed.seed_everything = lambda *args, **kwargs: None
pl.trainer.trainer.Trainer._log_hyperparams = lambda *args, **kwargs: None
os.environ["NEURALFORECAST_LOG_LEVEL"] = "ERROR"

# Configuring PyTorch to minimize output
torch.set_printoptions(threshold=0)
torch.autograd.set_detect_anomaly(False)
torch.set_float32_matmul_precision("medium")

# Configuring logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("neuralforecast").setLevel(logging.CRITICAL)

# Disabling PyTorch Lightning progress bar
pl.trainer.trainer.Trainer.enable_progress_bar = False

# Setting PyTorch memory configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Context manager
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Directories
base_dir = "/home/jupyter-kohv04@vse.cz/kohv04/backtesting_final/"
metadata_dir = os.path.join(base_dir, "metadata")
metadata_file = os.path.join(metadata_dir, "nasdaq100_ticker_dataset.json")

# Loading tickers
if not os.path.exists(metadata_file):
    raise FileNotFoundError(f"Metadata file {metadata_file} not found")
nasdaq100_tickers = pd.read_json(metadata_file)["Ticker"].tolist()[17:]

# Time parameters
trading_start = pd.to_datetime("09:30:00").time()
trading_end = pd.to_datetime("16:00:00").time()
premarket_start = pd.to_datetime("08:30:00").time()
end_date = pd.to_datetime("2025-02-14")
training_end_date = end_date - timedelta(days=180)  # August 18, 2024
start_date = training_end_date - timedelta(days=90)  # May 20, 2024
holdout_start_date = training_end_date  # August 18, 2024

def create_volume_prediction_dir(ticker):
    """Create volume_prediction directory for a given ticker."""
    volume_pred_dir = os.path.join(base_dir, f"ticker={ticker}_standardized/volume_prediction")
    for d in [volume_pred_dir, os.path.join(volume_pred_dir, "models"), 
              os.path.join(volume_pred_dir, "predictions"), os.path.join(volume_pred_dir, "metrics")]:
        os.makedirs(d, exist_ok=True)
    return volume_pred_dir

def preprocess_ticker_data(ticker, n_lags=15, is_holdout=False):
    """Preprocess data for a specific ticker, for training or hold-out period."""
    logger.info(f"Preprocessing data for ticker {ticker}, holdout={is_holdout}")
    ticker_std = dd.read_parquet(os.path.join(base_dir, f"ticker={ticker}_standardized/*.parquet"))
    
    # Date range
    if is_holdout:
        start = holdout_start_date
        end = end_date
    else:
        start = start_date
        end = training_end_date
    
    # Filtering by date and time
    ticker_std = ticker_std[(ticker_std["timestamp"] >= start) & 
                           (ticker_std["timestamp"] <= end) &
                           ((ticker_std["timestamp"].dt.time >= premarket_start) & 
                            (ticker_std["timestamp"].dt.time <= trading_end))]
    ticker_std = ticker_std.compute()
    logger.info(f"Loaded {len(ticker_std)} rows for ticker {ticker}")
    
    ticker_regime = pd.read_csv(os.path.join(base_dir, f"ticker={ticker}/{ticker}_regimes.csv"))
    ticker_regime["date"] = pd.to_datetime(ticker_regime["date"])
    ticker_std["timestamp"] = pd.to_datetime(ticker_std["timestamp"])
    
    # Adding time-based features
    ticker_std["hour"] = ticker_std["timestamp"].dt.hour
    ticker_std["day_of_week"] = ticker_std["timestamp"].dt.dayofweek
    ticker_std["minute"] = ticker_std["timestamp"].dt.minute
    ticker_std["time_since_open"] = ((ticker_std["timestamp"] - 
                                    ticker_std["timestamp"].dt.floor("D") - 
                                    pd.Timedelta(hours=9, minutes=30)).dt.total_seconds() / 60)
    ticker_std["is_trading"] = (ticker_std["timestamp"].dt.time >= trading_start).astype(int)
    ticker_std["date"] = ticker_std["timestamp"].dt.date
    ticker_std["intraday_minute"] = (((ticker_std["timestamp"] - 
                                    ticker_std["timestamp"].dt.floor("D") - 
                                    pd.Timedelta(hours=9, minutes=30)).dt.total_seconds() / 60).astype(int) + 1)
    ticker_regime["date"] = ticker_regime["date"].dt.date
    
    # Merging with regime data
    ticker_std = ticker_std.merge(ticker_regime[["date", "volatility_regime", "trend_regime", 
                                               "liquidity_regime", "news_impact"]].rename(
                                               columns={"news_impact": "news_impact_regime"}), 
                                on="date", how="left")
    
    # Adding lag features
    if n_lags > 0:
        lag_cols = {f"volume_lag_{lag}": ticker_std["volume"].shift(lag) for lag in range(1, n_lags + 1)}
        lag_cols.update({f"close_lag_{lag}": ticker_std["close"].shift(lag) for lag in range(1, n_lags + 1)})
        ticker_std = pd.concat([ticker_std, pd.DataFrame(lag_cols, index=ticker_std.index)], axis=1)
    
    # Scaling continuous features (training period only)
    volume_pred_dir = create_volume_prediction_dir(ticker)
    if not is_holdout:
        scaler = StandardScaler()
        cont_cols = ["open", "high", "low", "close", "volume", "estimated_obd", 
                    "estimated_bid_ask_spread", "prev_session_high", "prev_session_low", "50_day_sma"]
        if n_lags > 0:
            cont_cols += [f"volume_lag_{i}" for i in range(1, n_lags + 1)] + [f"close_lag_{i}" for i in range(1, n_lags + 1)]
        
        ticker_std["raw_volume"] = ticker_std["volume"]
        ticker_std[cont_cols] = scaler.fit_transform(ticker_std[cont_cols])
        pd.to_pickle(scaler, os.path.join(volume_pred_dir, "scaler.pkl"))
        
        # Scaling log volume
        ticker_std["log_volume"] = np.log1p(ticker_std["raw_volume"])
        log_volume_scaler = StandardScaler()
        ticker_std["log_volume"] = log_volume_scaler.fit_transform(ticker_std[["log_volume"]]) * 2 - 1
        pd.to_pickle(log_volume_scaler, os.path.join(volume_pred_dir, "log_volume_scaler.pkl"))
        
        # Encoding categorical features
        for col in ["volatility_regime", "trend_regime", "liquidity_regime"]:
            le = LabelEncoder()
            ticker_std[col] = le.fit_transform(ticker_std[col].astype(str))
            pd.to_pickle(le, os.path.join(volume_pred_dir, f"{col}_encoder.pkl"))
    
    # Calculating returns
    ticker_std["returns"] = ticker_std["close"].pct_change()
    
    # Imputing NaNs
    def impute_nans(group):
        if n_lags > 0:
            for lag in range(1, n_lags + 1):
                group[f"volume_lag_{lag}"] = group[f"volume_lag_{lag}"].fillna(
                    group["volume"].iloc[0] if group["timestamp"].dt.time.iloc[0] < trading_start else group["volume"])
                group[f"close_lag_{lag}"] = group[f"close_lag_{lag}"].fillna(
                    group["close"].iloc[0] if group["timestamp"].dt.time.iloc[0] < trading_start else group["close"])
        group["returns"] = group["returns"].fillna(0)
        return group
    ticker_std = ticker_std.groupby(ticker_std["timestamp"].dt.date).apply(impute_nans).reset_index(drop=True)
    
    # NeuralForecast data (training period only)
    if not is_holdout:
        nf_data = ticker_std[["ticker", "timestamp", "log_volume"]].rename(
            columns={"ticker": "unique_id", "timestamp": "ds", "log_volume": "y"})
        exogenous_cols = ["open", "high", "low", "close", "volume", "estimated_obd", 
                        "estimated_bid_ask_spread", "prev_session_high", "prev_session_low", 
                        "50_day_sma", "hour", "day_of_week", "minute", "time_since_open", 
                        "is_trading", "volatility_regime", "trend_regime", "liquidity_regime", 
                        "news_impact_regime", "returns", "intraday_minute"]
        if n_lags > 0:
            exogenous_cols += [f"volume_lag_{i}" for i in range(1, n_lags + 1)] + [f"close_lag_{i}" for i in range(1, n_lags + 1)]
        
        nf_data = nf_data.join(ticker_std[exogenous_cols])
        nf_data = nf_data[nf_data["is_trading"] == 1]
        ticker_std = ticker_std[ticker_std["is_trading"] == 1]
        
        return nf_data, ticker_std
    else:
        ticker_std["true_volume"] = ticker_std["volume"]
        ticker_std = ticker_std[ticker_std["is_trading"] == 1]
        return None, ticker_std

def train_and_predict(ticker, nf_data, ticker_std):
    """Train TFT model and generate predictions for a specific ticker."""
    volume_pred_dir = create_volume_prediction_dir(ticker)
    metrics_dir = os.path.join(volume_pred_dir, "metrics")
    
    # Model config
    lag_configs = [15]
    futr_exog_list = ["hour", "day_of_week", "minute", "time_since_open", "is_trading"]
    base_hist_exog_list = ["open", "high", "low", "close", "volume", "estimated_obd", 
                          "estimated_bid_ask_spread", "prev_session_high", "prev_session_low", 
                          "50_day_sma", "returns", "volatility_regime", "trend_regime", 
                          "liquidity_regime", "news_impact_regime", "intraday_minute"]
    configs = [{
        "input_size": 60,
        "hidden_size": 128,
        "n_head": 2,
        "learning_rate": 0.00046234800833302813,
        "max_steps": 1000,
        "batch_size": 64,
        "windows_batch_size": 128,
        "scaler_type": "standard",
        "futr_exog_list": futr_exog_list,
        "hist_exog_list": base_hist_exog_list,
        "stat_exog_list": [],
        "early_stop_patience_steps": 5,
        "val_check_steps": 50,
        "random_seed": 13
    }]
    
    all_metrics = []
    
    # Training TFT model and collecting metrics
    for n_lags in lag_configs:
        columns_to_keep = ["unique_id", "ds", "y"] + futr_exog_list + base_hist_exog_list
        if n_lags > 0:
            columns_to_keep += [f"volume_lag_{i}" for i in range(1, n_lags + 1)] + [f"close_lag_{i}" for i in range(1, n_lags + 1)]
        nf_data_subset = nf_data[columns_to_keep]
        hist_exog_list = base_hist_exog_list + ([f"volume_lag_{i}" for i in range(1, n_lags + 1)] + 
                                               [f"close_lag_{i}" for i in range(1, n_lags + 1)] if n_lags > 0 else [])
        
        for config_idx, tft_config in enumerate(configs):
            tft_config["hist_exog_list"] = hist_exog_list
            models = [TFT(h=15, loss=MAE(), **tft_config)]
            nf = NeuralForecast(models=models, freq="1min")
            cv_results = nf.cross_validation(df=nf_data_subset, n_windows=3, step_size=60, 
                                          val_size=15, refit=True)
            
            model_name = "TFT"
            for h in [5, 10, 15]:
                cv_results[f"{model_name}-{h}"] = np.nan
            
            for cutoff, group in cv_results.groupby('cutoff'):
                if 'TFT' in group.columns and len(group) >= 15:
                    group = group.sort_values('ds')
                    group_subset = group.iloc[:15]
                    for h in [5, 10, 15]:
                        if h <= 15:
                            cv_results.loc[group_subset.index[h-1], f"{model_name}-{h}"] = group_subset.iloc[h-1]['TFT']
            
            for h in [5, 10, 15]:
                col_name = f"{model_name}-{h}"
                if col_name in cv_results.columns and cv_results[col_name].notna().any():
                    pred_values = cv_results[col_name][cv_results[col_name].notna() & cv_results["y"].notna()]
                    true_values = cv_results["y"][cv_results[col_name].notna() & cv_results["y"].notna()]
                    if len(pred_values) > 0:
                        scaled_mae = np.mean(np.abs(pred_values - true_values))
                        scaled_rmse = np.sqrt(np.mean((pred_values - true_values) ** 2))
                        scaled_r2 = r2_score(true_values, pred_values)
                        
                        pred_values = (pred_values + 1) / 2
                        true_values = (true_values + 1) / 2
                        log_volume_scaler = pd.read_pickle(os.path.join(volume_pred_dir, "log_volume_scaler.pkl"))
                        pred_values = log_volume_scaler.inverse_transform(pred_values.values.reshape(-1, 1)).flatten()
                        true_values = log_volume_scaler.inverse_transform(true_values.values.reshape(-1, 1)).flatten()
                        pred_values = np.expm1(pred_values)
                        true_values = np.expm1(true_values)
                        unscaled_r2 = r2_score(true_values, pred_values)
                        
                        all_metrics.append({
                            "ticker": ticker,
                            "model": model_name,
                            "config": f"Config_{config_idx+1}",
                            "horizon": h,
                            "scaled_MAE": scaled_mae,
                            "scaled_RMSE": scaled_rmse,
                            "scaled_R2": scaled_r2,
                            "MAE": np.mean(np.abs(pred_values - true_values)),
                            "RMSE": np.sqrt(np.mean((pred_values - true_values) ** 2)),
                            "R2": unscaled_r2,
                            "n_lags": n_lags
                        })
    
    # TFT metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(metrics_dir, "tft_metrics.csv"), index=False)
    
    # Splitting data
    total_rows = len(nf_data)
    train_size = int(0.85 * total_rows)
    val_size = int(0.05 * total_rows)
    train_data = nf_data.iloc[:train_size]
    val_data = nf_data.iloc[train_size:train_size + val_size]
    test_data = nf_data.iloc[train_size + val_size:]
    
    # Training and saving TFT model
    nf.fit(df=train_data, val_size=len(val_data))
    nf.save(path=os.path.join(volume_pred_dir, f"models/{ticker}_model"), model_index=None, overwrite=True)
    
    # Generating predictions for test set
    predictions_list = []
    input_size = configs[0]["input_size"]
    for idx in range(train_size + val_size, total_rows - 15 + 1):
        input_data = nf_data.iloc[max(0, idx - input_size):idx]
        if len(input_data) < input_size:
            logger.warning(f"Skipping index {idx} for ticker {ticker}: input data length {len(input_data)} < {input_size}")
            continue
        futr_data = nf.make_future_dataframe(df=input_data)
        futr_data = futr_data[(futr_data['ds'].dt.time >= trading_start) & 
                            (futr_data['ds'].dt.time <= trading_end)]
        if len(futr_data) != 15:
            logger.warning(f"Skipping index {idx} for ticker {ticker}: future data length {len(futr_data)} != 15")
            continue
        daily_exog = nf_data[['ds', 'volatility_regime', 'trend_regime', 
                            'liquidity_regime', 'news_impact_regime']].copy()
        daily_exog['date'] = daily_exog['ds'].dt.date
        daily_exog = daily_exog.drop(columns=['ds']).drop_duplicates(subset=['date'])
        futr_data['date'] = futr_data['ds'].dt.date
        futr_data = futr_data.merge(daily_exog, on='date', how='left')
        futr_data['hour'] = futr_data['ds'].dt.hour
        futr_data['day_of_week'] = futr_data['ds'].dt.dayofweek
        futr_data['minute'] = futr_data['ds'].dt.minute
        futr_data['time_since_open'] = ((futr_data['ds'] - 
                                       futr_data['ds'].dt.floor('D') - 
                                       pd.Timedelta(hours=9, minutes=30)).dt.total_seconds() / 60)
        futr_data['is_trading'] = 1
        futr_data = futr_data[['unique_id', 'ds'] + futr_exog_list]
        pred = nf.predict(df=input_data, futr_df=futr_data)
        pred['horizon'] = (pred['ds'] - pred.groupby('unique_id')['ds'].transform('first')).dt.total_seconds() / 60 + 1
        pred = pred[pred['horizon'] == 15].drop(columns=['horizon']).rename(
            columns={'TFT': 'predicted_volume_15min', 'unique_id': 'ticker', 'ds': 'timestamp'})
        predictions_list.append(pred)
    
    # Processing predictions
    predictions = pd.concat(predictions_list).merge(
        test_data[["unique_id", "ds", "y"]].rename(
            columns={"unique_id": "ticker", "ds": "timestamp", "y": "true_volume"}),
        on=["ticker", "timestamp"], how="left")
    predictions = predictions.dropna(subset=["predicted_volume_15min", "true_volume"])
    
    logger.info(f"Generated {len(predictions)} TFT predictions for ticker {ticker}")
    
    predictions["predicted_volume_15min_scaled"] = predictions["predicted_volume_15min"]
    predictions["true_volume_scaled"] = predictions["true_volume"]
    log_volume_scaler = pd.read_pickle(os.path.join(volume_pred_dir, "log_volume_scaler.pkl"))
    predictions["predicted_volume_15min"] = np.expm1(log_volume_scaler.inverse_transform(
        ((predictions["predicted_volume_15min"] + 1) / 2).values.reshape(-1, 1)).flatten())
    predictions["true_volume"] = np.expm1(log_volume_scaler.inverse_transform(
        ((predictions["true_volume"] + 1) / 2).values.reshape(-1, 1)).flatten())
    clip_max = np.percentile(predictions["true_volume"], 99)
    predictions["predicted_volume_15min"] = np.clip(predictions["predicted_volume_15min"], 0, clip_max)
    
    # Saving set predictions
    predictions.to_csv(os.path.join(volume_pred_dir, f"predictions_{ticker}.csv"), index=False)
    
    return predictions


# Main execution wrapped in suppress_output:
with suppress_output():
    for ticker in nasdaq100_tickers:
        try:
            # Processing training period
            nf_data, ticker_std = preprocess_ticker_data(ticker, n_lags=15, is_holdout=False)
            predictions = train_and_predict(ticker, nf_data, ticker_std)
            
            # Cleaning up memory
            del nf_data, ticker_std, predictions
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            continue

    # Aggregating metrics after processing all tickers
    def aggregate_metrics():
        """Aggregate metrics across all tickers."""
        all_metrics = []
        for ticker in nasdaq100_tickers:
            metrics_dir = os.path.join(base_dir, f"ticker={ticker}_standardized/volume_prediction/metrics")
            for metric_file in glob.glob(os.path.join(metrics_dir, "*.csv")):
                metrics_df = pd.read_csv(metric_file)
                all_metrics.append(metrics_df)
        if all_metrics:
            pd.concat(all_metrics).to_csv(os.path.join(base_dir, "aggregated_metrics.csv"), index=False)
            logger.info("Aggregated metrics saved to aggregated_metrics.csv")
        else:
            logger.warning("No metrics found to aggregate")

    aggregate_metrics()