# Implementation of a Trading Algorithm with a Focus on Volume Profiles

This repository contains the complete codebase and implementation for the bachelor's thesis, **"Implementation of a trading algorithm with a focus on volume profiles."** The research investigates the incremental value of integrating advanced analytical methods into classic intraday trading strategies. It systematically evaluates the impact of adding volume-based parameters and a predictive deep learning model for volume forecasting.

The entire framework is developed in **Python** and rigorously backtested on historical 1-minute data for **NASDAQ-100 securities**.

***

## Thesis Abstract

The thesis investigates the incremental value of integrating advanced analytics into classical intraday trading strategies. To systematically assess the impact of increasing analytical complexity, a tiered framework of nine strategies was developed spanning baseline (price-only), volume-enhanced, and deep learning-enhanced archetypes. A **Temporal Fusion Transformer (TFT)** model is implemented to provide 15-minute ahead volume forecasts for the most advanced strategies, and everything is rigorously backtested on NASDAQ-100 securities.

The empirical results demonstrate a key trade-off: while incorporating volume-based parameters did not significantly increase profitability, it offered a powerful, statistically significant advantage in risk management by consistently reducing maximum drawdowns. Conversely, the hypothesis that a sophisticated transformer-based forecast would further improve performance was rejected, as the strategies operating on predicted volume failed to outperform their simpler, heuristic-based counterparts. Analysis also validates that strategy performance is highly regime-dependent, particularly for mean-reversion and momentum approaches.

It is concluded that the primary contribution of adding volume-based complexity in this context is defensive, serving to preserve capital rather than amplify returns, and highlights a critical trade-off between the benefits and implementation costs of deploying advanced predictive models in intraday trading strategies on higher frequency data.

***

## Project Pipeline & Structure

The project follows a sequential pipeline, with each notebook and script representing a key stage in the research process, from data collection to final evaluation.

#### 1. Data Collection & Preprocessing
- **`1.1_listed_stocks_nasdaq.ipynb`**: Fetches and cleans the complete list of NASDAQ-listed symbols.
- **`1.2_data_collect.ipynb`**: Gathers 5 years of 1-minute historical data for NASDAQ-100 tickers from **Polygon.io** and performs initial feature engineering.
- **`1.3_news_calendar.ipynb`**: Aggregates a comprehensive news calendar from various sources, including macroeconomic data (**FRED**), **SEC filings**, and other corporate events, to create a news-impact flag.
- **`1.3_2_product_launches_generator.py`**: Uses **Google's Gemini 2.5 Pro** model to generate a synthetic dataset of corporate product launches for the news calendar.

#### 2. Backtesting Dataset Creation
- **`2.backtesting_subset_nasdaq100.ipynb`**: Creates a clean, unified, and time-indexed dataset for the NASDAQ-100 subset, ready for backtesting. It handles missing data through interpolation, filters for trading hours, and merges the news impact indicator.

#### 3. Market Regime Classification
- **`3_Regime_classification.ipynb`**: Classifies each trading day into **Volatility**, **Trend**, and **Liquidity** regimes using indicators like the Stochastic Oscillator, ADX, and Bollinger Bands with dynamic, rolling quantile-based thresholds.

#### 4. Volume Prediction with Temporal Fusion Transformer (TFT)
- **`4_Volume_prediction_modelling.ipynb`** & **`4_1_volume_prediction.py`**: Trains a **Temporal Fusion Transformer (TFT)** model to predict 15-minute ahead trading volume.
- **`4_2_Volume_imputation.ipynb`**: Applies the trained TFT model to generate volume predictions for the optimization and simulation periods.

#### 5. Indicator Calculation
- **`5_Indicators.ipynb`**: Calculates a wide array of technical indicators required for the trading strategies, including VWAP, developing Volume Profile metrics (VAH, VAL, dPOC), and signals for the three distinct strategy families.

#### 6. Strategy Optimization
- **`6_Optimization.ipynb`**: Conducts a large-scale **Batched Random Search** to find optimal hyperparameters for the 9 trading strategies, maximizing the **Calmar Ratio** on a 3-month in-sample dataset.

#### 7. Trading Simulation
- **`7_Trading_simulation.ipynb`**: Runs a full backtest on a 3-month hold-out dataset using the generalized best parameters. The simulation includes realistic market frictions like **transaction fees (0.01%)**, **slippage (0.02%)**, and a **0.5% order rejection probability**.

#### 8. Evaluation & Hypothesis Testing
- **`8_Evaluation.ipynb`**: Analyzes simulation results, calculates performance metrics, and identifies the best-performing ticker-strategy combinations.
- **`9_Hypothesis_testing.ipynb`**: Performs statistical **paired t-tests** and **Welch's t-tests** to validate the core research hypotheses.

***

## Dependencies

The project was developed using Python 3.10. To set up the environment, first create and activate a conda environment, then install the required packages from the `requirements.txt` file.

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n trading_env python=3.10
    conda activate trading_env
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

***

## Usage

The code is organized into a series of Jupyter notebooks that should be run in sequential order to replicate the research pipeline.

-   **`1.1` to `1.3`**: Scripts for data collection. **Note:** These require API keys for Polygon.io, SEC-API, and Google Cloud, which must be configured locally.
-   **`2.backtesting_subset_nasdaq100.ipynb`**: Processes the raw data into a clean Parquet file for backtesting.
-   **`3_Regime_classification.ipynb`**: Generates the market regime classifications.
-   **`4_Volume_prediction_modelling.ipynb`** to **`4_2_Volume_imputation.ipynb`**: Train the TFT model and generate predictions. This is computationally intensive and may require a GPU.
-   **`5_Indicators.ipynb`** to **`9_Hypothesis_testing.ipynb`**: Run the optimization, simulation, and final analysis.

> **Disclaimer:** The code is provided for educational and demonstration purposes. To run it successfully, you must have access to the necessary data sources and APIs. The initial raw data is not included in this repository due to its large size.

***

## Methodology

### 1. Tiered Strategy Framework
The research evaluates nine distinct intraday strategies, categorized into three levels of increasing analytical complexity:

-   **Baseline Strategies**: Rely purely on price-based indicators.
-   **Volume-Enhanced Strategies**: Augment baseline strategies with signals from historical volume data (e.g., VWAP, developing Value Area).
-   **Deep Learning-Enhanced Strategies**: Incorporate the TFT model's predicted future volume as a forward-looking confirmation signal.

### 2. Key Findings
-   **Volume as a Defensive Filter**: Adding volume data provided a **statistically significant advantage in risk management** by consistently reducing maximum portfolio drawdown.
-   **Predictive Model Performance**: The sophisticated TFT model **did not improve performance** over simpler, heuristic-based volume rules, a hypothesis that was unequivocally rejected. This was likely due to the static nature of the model used in the backtest.
-   **Market Regime Dependency**: The research statistically confirmed that **mean-reversion** strategies perform better in low-volatility environments, while **momentum** strategies thrive in trending, high-volatility markets.

***

## 🔗 Links

-   **Thesis PDF**: [Implementation of a trading algorithm with a focus on volume profiles.pdf](<INSERT_LINK_TO_YOUR_PDF_HERE>)
-   **Dataset**: The NASDAQ-100 dataset used for this research is available for download on the [Releases page](<INSERT_LINK_TO_YOUR_GITHUB_RELEASE_HERE>).
