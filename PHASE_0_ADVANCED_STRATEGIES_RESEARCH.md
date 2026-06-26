# PHASE_0: Advanced Strategies, Labeling, and Signal Generation Research

**Date:** 2026-06-27
**Focus:** EUR/USD M5 scalping — labeling methods, dynamic TP/SL, SMC/ICT as ML features, order flow, regime detection, news filtering, spread analysis, position sizing, anti-clustering, signal quality scoring
**Sources:** 60+ sources including academic papers, GitHub repos (51.8k+ stars), MQL5 articles, TradingView, QuantInsti, arXiv papers, forex research

---

## 1. Triple-Barrier Method Improvements

### 1.1 Standard Triple-Barrier (Lopez de Prado)
The triple-barrier method labels trades using three barriers:
- **Upper barrier (TP)**: Price hits profit target → label = +1
- **Lower barrier (SL)**: Price hits stop loss → label = -1
- **Vertical barrier (time)**: Neither hit within N bars → label based on final price

### 1.2 Better Alternatives for M5 Forex

**A. Adaptive Triple-Barrier (Volatility-Adjusted)**
- Instead of fixed multipliers, use ATR-scaled barriers that adapt to current volatility
- TP = entry ± C1 × ATR(14), SL = entry ± C2 × ATR(14)
- For 3:1 TP:SL: C1 = 4.5, C2 = 1.5 (as specified in AGENTS.md)
- Research shows adaptive barriers outperform fixed-pip barriers by 15-25% in risk-adjusted terms

**B. Dynamic Barrier with Time Decay**
- As the vertical barrier approaches, tighten the TP/SL proportionally
- Prevents labels from being dominated by time-expiry neutral trades
- Formula: effective_TP = TP × (1 - t/T)^0.5 where t = elapsed bars, T = total horizon

**C. Meta-Labeling (Primary + Secondary Model)**
- **Primary model**: Predicts direction (LONG/SHORT) using triple-barrier labels
- **Secondary model**: Predicts whether to TAKE the primary signal (binary: take/skip)
- Research from MQL5 (2026): Meta-labeling reduced max drawdown by 57% on EUR/USD H1
- Key insight: Meta-labeling's drawdown reduction comes from exposure control, not predictive accuracy
- The secondary model only needs >50% accuracy to add value through trade filtering

**D. Genetic-Algorithm Optimized Barriers**
- Research (MDPI, 2024): GA-optimized barriers outperform static barriers
- GA searches for optimal TP/SL multipliers that maximize Sharpe ratio
- Produces two label types: High Risk/High Profit (HRHP) and Low Risk/Low Profit (LRLP)
- 51.42% increase in profitability vs traditional methods

**E. Regime-Conditional Labeling**
- Use different barrier multipliers for different market regimes
- Trending regime: wider TP (6x ATR), tighter SL (1.0x ATR)
- Ranging regime: tighter TP (3x ATR), wider SL (2.0x ATR)
- Regime detected via ADX threshold or HMM classification

### 1.3 Recommended for EUR/USD M5
Use **Adaptive Triple-Barrier with Meta-Labeling**:
1. Primary labels: ATR-scaled (4.5x ATR TP, 1.5x ATR SL, 20-bar horizon)
2. Secondary model: Binary classifier (take/skip signal)
3. Train secondary on primary model's outputs + market context features

---

## 2. Dynamic TP/SL Optimization

### 2.1 ATR-Based Dynamic TP/SL (Best for Scalping)
- **SL = 1.5 × ATR(14) on M5** — adapts to current volatility
- **TP = 4.5 × ATR(14) on M5** — maintains 3:1 ratio
- Why ATR: A 10-pip stop is "a joke on GBP/JPY London open and excessive on EUR/CHF in Asia" (ChartSnipe, 2026)

### 2.2 Optimal TP/SL Ratios from Research
| Ratio | Win Rate Needed | Typical Use |
|-------|----------------|-------------|
| 1:1 | >50% | Scalping (fast exits) |
| 1:1.5 | 45-50% | EMA pullback scalping |
| 1:2 | 40% | Standard day trading |
| 1:3 | 25%+ | Target system (our setup) |
| 1:5 | 17%+ | Trend following |

### 2.3 Advanced TP/SL Methods

**A. Structure-Based SL**
- Place SL beyond the last swing high/low (not arbitrary ATR multiple)
- More logical — respects market structure
- Combine with ATR cap: SL = min(structure_level, 1.5 × ATR)

**B. Breakeven + Trailing Stop Protocol**
- Move SL to breakeven after +1R profit (e.g., +4.5 pips if SL = 1.5 pips)
- Activate trailing stop after +2R
- Trail by 1.0 × ATR(14)
- This "locks in" profits while letting winners run

**C. Dynamic ATR-Adaptive TP**
```python
# Volatility-adaptive TP multiplier
if current_atr > atr_percentile_75:
    tp_multiplier = 5.0  # High vol: wider TP
    sl_multiplier = 1.5
elif current_atr < atr_percentile_25:
    tp_multiplier = 3.5  # Low vol: tighter TP
    sl_multiplier = 1.2
else:
    tp_multiplier = 4.5  # Normal vol
    sl_multiplier = 1.5
```

### 2.4 Transaction Cost Integration
- Use p95 spread (not mean) for cost modeling — "your strategy will occasionally enter during illiquid windows" (MQL5, 2026)
- Total round-trip cost = spread (p95) + slippage (0.3-0.7 pips) + commission
- For EUR/USD: ~1.67 pips total cost at p95 spread
- **Minimum TP must exceed 2× total round-trip cost** for positive expectancy

---

## 3. Session-Based Trading

### 3.1 Optimal Sessions for EUR/USD Scalping

| Session | Time (UTC) | Spreads | Volume | Best For |
|---------|------------|---------|--------|----------|
| **London-NY Overlap** | **13:00-16:00** | **Tightest (0.1-0.3 pips)** | **~50% daily volume** | **SCALPING (BEST)** |
| London Open | 07:00-08:00 | Tight | High | Breakout strategies |
| NY Morning | 13:00-17:00 | Tight | High | Trend following |
| Asian | 00:00-08:00 | Wide (2-4 pips) | Low | Range trading (avoid for scalping) |
| Late NY | 17:00-21:00 | Wide | Low | Value destroyer — avoid |

### 3.2 Key Research Findings
- **London-NY overlap wins on P&L per trade** across thousands of journal entries (TradersSecondBrain, 2026)
- London Open wins on win rate (clean directional moves)
- NY Afternoon is "consistently the value-destroyer — negative average P&L, highest max losses"
- **57% of all global forex turnover** flows through London + New York centers

### 3.3 ICT Kill Zones (for SMC/ICT strategies)
- **Asian Kill Zone**: 20:00-00:00 EST (accumulation/manipulation)
- **London Kill Zone**: 02:00-05:00 EST (manipulation/distribution)
- **NY Kill Zone**: 07:00-10:00 EST (distribution)
- **London-NY Overlap**: 08:00-12:00 EST (maximum liquidity)

### 3.4 Implementation
```python
# Session filter
def is_trading_session(utc_hour):
    """London-NY overlap only for scalping"""
    return 13 <= utc_hour < 16  # UTC

# Sub-session precision (strongest moves)
def is_peak_session(utc_hour):
    """First 2 hours of overlap — strongest directional runs"""
    return 13 <= utc_hour < 15  # UTC
```

---

## 4. Smart Money Concepts (SMC) as ML Features

### 4.1 Can SMC Be Quantified? YES
Multiple open-source libraries implement SMC detection in Python/MQL5:

**Libraries:**
- `smartmoneyconcepts` (1761 stars) — Python, detects OB, FVG, BOS/CHoCH, liquidity
- `SMC_ICT_Library` (MQL5) — 15 ONNX ML training scripts for SMC concepts
- `ouroboros.v2` — SMC-based algorithmic system with LLM gate, walk-forward optimizer
- `falcon-ai` — SMC detection + AI confluence scoring

### 4.2 SMC Features That Can Be Quantified

| SMC Concept | How to Quantify | ML Feature |
|-------------|----------------|------------|
| **Order Blocks (OB)** | Distance from current price to nearest unmitigated OB | `ob_distance_pct`, `ob_type` (bull/bear), `ob_validity` (fresh/tested/mitigated) |
| **Fair Value Gaps (FVG)** | Gap size / ATR, fill probability, age | `fvg_size_atr`, `fvg_fill_prob`, `fvg_age_bars`, `fvg_type` |
| **BOS/CHoCH** | Binary events + distance to structural level | `bos_recent` (3 bars), `choch_recent` (3 bars), `structure_direction` |
| **Liquidity Sweeps** | Distance swept, volume at sweep | `sweep_magnitude`, `sweep_side`, `sweep_confirmed` |
| **Premium/Discount Zone** | Price position within range | `zone_position` (0=discount, 0.5=equilibrium, 1=premium) |
| **OTE Zone** | Whether price is in 0.618-0.786 Fib zone | `in_ote_zone` (binary), `ote_distance` |

### 4.3 SMC Confluence Scoring (Research-Backed)
From Falcon AI and SM Radar implementations:
- Each SMC element gets a quality score (0-100)
- **FVG quality factors**: Gap size vs ATR (sweet spot 0.3-1.5x), displacement strength, volume, trend alignment
- **OB quality factors**: OB size (0.3-1.0 ATR optimal), post-OB displacement, volume, structure alignment
- **Confluence multiplier**: FVG + OB overlap + BOS/CHoCH alignment = highest quality
- Standalone FVGs have 30-45% fail rate; with confluence, win rate exceeds 70%

### 4.4 Recommended SMC Feature Set for ML
```python
smc_features = {
    # Order Blocks
    'ob_distance': float,      # % distance to nearest OB
    'ob_type': int,            # 1=bull, -1=bear, 0=none
    'ob_validity': float,      # 1=fresh, 0.5=tested, 0=mitigated
    
    # Fair Value Gaps
    'fvg_present': int,        # 1 if FVG exists within 10 bars
    'fvg_size_atr': float,     # Gap size / ATR
    'fvg_fill_probability': float,  # Estimated fill prob
    'fvg_direction': int,      # 1=bullish, -1=bearish
    
    # Market Structure
    'bos_recent': int,         # 1 if BOS in last 3 bars
    'choch_recent': int,       # 1 if CHoCH in last 3 bars
    'structure_trend': int,    # 1=bullish, -1=bearish
    
    # Liquidity
    'liquidity_sweep': int,    # 1 if sweep detected
    'sweep_magnitude': float,  # How far past the level
    
    # Zone
    'premium_discount': float, # 0-1 position in range
    'in_ote_zone': int,        # 1 if in 0.618-0.786
    
    # Confluence
    'smc_confluence_score': float,  # 0-100 composite score
}
```

---

## 5. ICT Concepts as Features

### 5.1 Break of Structure (BOS)
- **Definition**: Candle close beyond prior swing high (bullish) or swing low (bearish)
- **Quantification**: 
  - `bos_direction`: +1 (bullish BOS), -1 (bearish BOS), 0 (none)
  - `bos_distance`: Distance from entry to BOS level
  - `bos_confirmed`: Whether BOS is confirmed (close beyond) vs wick-only

### 5.2 Change of Character (CHoCH)
- **Definition**: First break AGAINST prevailing trend — reversal warning
- **Quantification**:
  - `choch_detected`: Binary flag
  - `choch_type`: Internal vs swing-level CHoCH
  - `choch_displacement`: Whether displacement candles accompany the break (MSS confirmation)
- **Key finding**: "A 5M CHoCH at random is noise. A 5M CHoCH inside an unmitigated 4H order block after a liquidity sweep is a sniper entry" (Quantum Algo, 2026)

### 5.3 Optimal Trade Entry (OTE)
- **Definition**: Fibonacci 0.618-0.786 retracement zone after displacement
- **Quantification**:
  - `in_ote_zone`: Binary (price within 0.618-0.786 of impulse leg)
  - `ote_midpoint_distance`: Distance from 70.5% level
  - `ote_valid`: Only valid after confirmed displacement move

### 5.4 Market Structure Shift (MSS)
- **Definition**: Structural break WITH displacement (FVG present in break leg)
- **More significant than CHoCH** — indicates institutional commitment
- **Quantification**: `mss_confirmed` = CHoCH + FVG present in displacement leg

### 5.5 Recommended ICT Feature Set
```python
ict_features = {
    'bos_direction': int,      # +1/-1/0
    'bos_distance_pips': float,
    'choch_detected': int,     # Binary
    'choch_with_displacement': int,  # MSS confirmation
    'ote_in_zone': int,        # In 0.618-0.786
    'ote_distance': float,     # Distance from 70.5%
    'mss_confirmed': int,      # CHoCH + FVG displacement
    'structure_timeframe': int,  # Which TF the structure is on
    'htf_alignment': int,      # Does LTF structure align with HTF?
}
```

---

## 6. Order Flow Analysis as ML Features

### 6.1 Key Insight: Forex CVD is Synthetic
- Forex has **no consolidated tape** — CVD uses tick volume proxy, not real volume
- Each tick classified as "buy" or "sell" based on price direction (not actual aggressor)
- Less reliable than crypto CVD (which uses exchange trade tape)
- Still useful as a feature, but with lower confidence weighting

### 6.2 Order Flow Features for ML

| Feature | Calculation | ML Use |
|---------|-------------|--------|
| **Delta** | Buy volume - Sell volume per bar | Directional pressure |
| **CVD (Cumulative Delta)** | Running total of delta | Persistent buying/selling campaigns |
| **CVD Divergence** | Price makes new high but CVD doesn't | Exhaustion signal |
| **Delta Momentum** | Rate of change of delta | Acceleration/deceleration |
| **Absorption** | Large volume with no price movement | Institutional defense of level |
| **Volume Spike** | Volume > 2x 20-period average | Institutional participation |
| **Tick Velocity** | Ticks per second | Market activity level |

### 6.3 Implementation (MT5)
From `mqt-microstructure` library:
```python
# CVD calculation from tick data
cvd = CumulativeVolumeDelta()
for tick in ticks:
    cvd.add_tick(tick.price, tick.volume, tick.price > prev_price)
    
# Features
delta = cvd.get_bar_delta()
cumulative = cvd.get_cumulative()
divergence = cvd.detect_divergence(price_highs, lookback=20)
```

### 6.4 Important Caveat
"Forex CVD on 1-hour charts is about the minimum useful resolution because tick data needs time to accumulate meaningful patterns" (Kalena, 2026). For M5, CVD features should be used with caution — they add signal but are noisier than on crypto.

---

## 7. Volatility Regime Detection

### 7.1 Three Regimes That Matter (from research)
1. **Trending** (~38% of time): ADX > 25, correlations stay positive, volatility expands gradually → trend-following works
2. **Mean Reversion** (~49% of time): Volatility contracts, ranges hold, correlations mean-revert → range trading works
3. **Crisis** (~13% of time): All correlations go to 1 or -1, volatility explodes, liquidity vanishes → cut position sizes by 75%

### 7.2 Detection Methods

**A. HMM (Hidden Markov Models)**
- Most academically rigorous
- 3 hidden states: Calm, Turbulent, Crisis
- arXiv (2026): Triple-timeframe MS-GARCH for EUR/USD — "statistically superior volatility forecasting"
- Time-varying transition probabilities (TVTP) strongly justified at 4H and 1H scales

**B. Hybrid HMM + Wasserstein Clustering**
- Combines temporal memory (HMM) with distributional geometry (Wasserstein)
- More stable than pure HMM during structural breaks
- Labels: Trending, Range, Choppy, Transitional

**C. Rule-Based (Simple, Effective)**
```python
def detect_regime(df, adx_period=14, atr_period=14):
    adx = compute_adx(df, adx_period)
    atr_ratio = atr_short / atr_long  # e.g., ATR(10) / ATR(50)
    
    if adx > 25 and atr_ratio > 1.0:
        return 'TRENDING'
    elif adx < 20 and atr_ratio < 0.8:
        return 'RANGING'
    elif atr_ratio > 1.5:
        return 'CRISIS'
    else:
        return 'TRANSITIONAL'
```

**D. LLM-Based Regime Labeling (QuantInsti, 2026)**
- DeepSeek LLM labels regimes from compact numeric summaries
- Input: mean return, volatility, trend score, ATR proxy, z-score, drawdown proxy
- Outperformed KMeans clustering in OOS testing
- "The LLM acts as a flexible classifier that makes 'soft' judgments from multiple signals at once"

### 7.3 Recommended for EUR/USD M5
Use **Rule-Based + HMM ensemble**:
1. Primary: ADX + ATR ratio rule-based (fast, interpretable)
2. Confirmation: HMM with 3 states (calm/turbulent/crisis)
3. Regime as a feature: Include regime label as categorical input to ML model
4. Regime-conditional: Different model weights per regime

---

## 8. News Event Filtering

### 8.1 High-Impact Events to Filter (EUR/USD)

| Event | Typical Move | Window Before | Window After |
|-------|-------------|---------------|--------------|
| **NFP** | 50-150 pips | 30 min | 60 min |
| **FOMC** | 30-100 pips | 30 min | 90 min |
| **ECB Rate** | 30-80 pips | 30 min | 60 min |
| **CPI** | 30-80 pips | 30 min | 30 min |
| **GDP** | 20-60 pips | 15 min | 15 min |
| **PMI** | 10-30 pips | 15 min | 15 min |

### 8.2 Implementation Options

**A. MQL5 Native Calendar (Recommended for MT5)**
```cpp
bool IsNewsBlackout(int minutesBefore = 30, int minutesAfter = 30) {
    datetime now = TimeGMT();
    MqlCalendarValue values[];
    int count = CalendarValueHistory(values, now - minutesBefore*60, now + minutesAfter*60);
    for(int i = 0; i < count; i++) {
        MqlCalendarEvent event;
        CalendarEventById(values[i].event_id, event);
        if(event.importance == CALENDAR_IMPORTANCE_HIGH)
            return true;
    }
    return false;
}
```

**B. ForexFactory API (Python)**
- Scrape weekly calendar, parse event times and impact levels
- Cache for 60 minutes, refresh periodically
- Currency-aware: only filter events affecting EUR or USD

**C. Hybrid Approach**
- Use MQL5 calendar for live trading
- Use ForexFactory CSV for backtesting (MQL5 calendar not available in tester)

### 8.3 Key Rules
- **Tier 1 (NFP, FOMC, ECB)**: 30 min before, 60 min after
- **Tier 2 (CPI, GDP)**: 15 min before, 15 min after
- **Tier 3 (PMI, Retail)**: Optional — 5 min window if conservative
- **Always skip**: 5 min before to 15 min after for high-impact EUR/USD events
- **Never trade**: During the 30-minute window around NFP or FOMC

### 8.4 Backtesting Limitation
"The economic calendar is not available in the backtester, so you need to skip that check during optimization runs. Your historical results will include trades that the live filter would have blocked" (MQL5, 2026). Use blackout zone lists for backtesting exclusion.

---

## 9. Spread Analysis

### 9.1 Spread Impact on Scalping

| Metric | Value | Impact |
|--------|-------|--------|
| EUR/USD avg spread (overlap) | 0.1-0.3 pips | Minimal |
| EUR/USD avg spread (off-hours) | 2-4 pips | Fatal for scalping |
| EUR/USD spread during NFP | 5-10x normal | Blocks all entries |
| Target: 5 pips, spread: 1.5 pips | 30% cost eaten before start | Edge destroyed |

### 9.2 Spread Filter Rules
- **Hard rule**: Only enter when spread ≤ 20% of pip target
  - 5-pip target → max spread 1.0 pip
  - 10-pip target → max spread 2.0 pips
- **Dynamic filter**: Only enter when spread ≤ 1.2× historical average spread
- **Session-aware**: Track spread by hour, use hourly mean instead of global mean

### 9.3 Spread as ML Feature
```python
spread_features = {
    'current_spread_pips': float,
    'spread_vs_average': float,  # current / 20-period average
    'spread_percentile': float,  # Where in historical distribution
    'spread_stable': int,        # 1 if spread < 1.2x average
    'spread_widening': int,      # 1 if spread > previous bar's spread
}
```

### 9.4 Backtesting with Realistic Spreads
- Use **p95 spread** (not mean) for cost modeling
- Use **variable spread** in Strategy Tester (not fixed)
- Include spread widening around news events
- "MT4 does not natively show historical spread data in standard backtests — re-run with Variable Spread enabled" (MT4Programming, 2026)

---

## 10. Position Sizing for Scalping

### 10.1 Methods Comparison

| Method | Complexity | Adapts to Equity | Adapts to Volatility | Max DD | Recommendation |
|--------|-----------|-----------------|---------------------|--------|----------------|
| **Fixed Fractional** | Low | Yes | Indirect (via SL) | 10-20% | **DEFAULT CHOICE** |
| Fixed Lot | Zero | No | No | Variable | Demo/test only |
| ATR-Based | Medium | Yes | Yes | 10-20% | Swing + multi-pair |
| Full Kelly | High | Yes (aggressive) | No | 60-90% | Theory only |
| **Quarter Kelly** | High | Yes (moderate) | No | 10-20% | Advanced, after 500+ trades |

### 10.2 Recommended for EUR/USD Scalping
**Fixed Fractional: 0.5-1% risk per trade**
- Why not Kelly: "Full Kelly is practically unusable — produces extreme drawdowns" (ForexMechanics, 2026)
- Why not 2%: "For scalping, 0.5-1% is the defensible range" (AIFinHub, 2026)
- Quarter Kelly only after: documented edge, 500+ paired live trades, win rate within ±2% across rolling windows

### 10.3 Position Size Formula
```python
def calculate_position_size(account_balance, risk_pct, sl_pips, pip_value=10):
    """
    Fixed fractional position sizing
    """
    risk_amount = account_balance * (risk_pct / 100)
    position_size_lots = risk_amount / (sl_pips * pip_value)
    return round(position_size_lots, 2)

# Example: $10,000 account, 1% risk, 5-pip SL
# risk_amount = $100
# position_size = 100 / (5 * 10) = 2.0 lots
```

### 10.4 Portfolio Heat Limit
- Maximum 3-5% total portfolio risk at any time
- Maximum 1-2 open positions on EUR/USD
- If already exposed, reduce new position size proportionally

---

## 11. Anti-Clustering (Prevent Over-Trading)

### 11.1 The Problem
- Scalping EAs can fire multiple entries in rapid succession during volatile periods
- Correlated entries compound risk exponentially
- "Most retail losses on event days come from one thing: trading into a news spike" (MQL5, 2026)

### 11.2 Anti-Clustering Methods

**A. Time-Based Cooldown**
```python
class AntiCluster:
    def __init__(self, cooldown_bars=6):
        self.cooldown_bars = cooldown_bars
        self.last_entry_bar = -cooldown_bars
    
    def can_trade(self, current_bar):
        return (current_bar - self.last_entry_bar) >= self.cooldown_bars
```
- Minimum 3-6 bars between entries on same direction
- Configurable per symbol and regime

**B. Spatial Exclusion Zone**
- Create a "box" around each open position
- No new entries within X pips of existing position
- Three modes: Conservative (large zone), Moderate (medium), Aggressive (small)

**C. Maximum Open Positions**
- Hard cap: 1-2 simultaneous positions on EUR/USD
- Netting mode: Block all new entries while any position is open

**D. Hourly/Daily Trade Cap**
- Maximum 3-5 trades per hour
- Maximum 8-12 trades per day
- Daily loss limit: close all and stop if drawdown > X%

**E. Correlation Control Filter (CCF)**
- If open trade on EUR/USD, block trades on GBP/USD, USD/CHF (correlated)
- Maximum 1 trade per currency group (USD, EUR)

### 11.3 Recommended Configuration
```python
anti_cluster_config = {
    'cooldown_bars': 6,           # 30 minutes on M5
    'max_open_positions': 1,      # Single position mode
    'max_trades_per_hour': 3,
    'max_trades_per_day': 10,
    'max_daily_loss_pct': 2.0,    # Stop trading after -2% daily
    'min_distance_pips': 10,      # Between positions
}
```

---

## 12. Signal Quality Scoring

### 12.1 Multi-Factor Confidence Score (0-100)

From AlgoKing (2026), Falcon AI, and thrive-fi research:

| Factor | Weight | What It Measures |
|--------|--------|-----------------|
| **ML Model Confidence** | 30% | Raw prediction probability from XGBoost/LightGBM |
| **Confluence Score** | 25% | Number of confirming indicators (MTF alignment, SMC, volume) |
| **Regime Fitness** | 15% | Does the signal match the current regime? (trend signal in trending market) |
| **Spread Quality** | 15% | Current spread vs historical baseline |
| **Session Quality** | 10% | Is it during optimal trading hours? |
| **Data Freshness** | 5% | How recent is the input data? |

### 12.2 Scoring Thresholds
| Score | Classification | Action |
|-------|---------------|--------|
| 80-100 | **Exceptional** | Full position size, high conviction |
| 70-79 | **High** | 75% position size, worth trading |
| 60-69 | **Moderate** | 50% position size, needs confirmation |
| 50-59 | **Low** | 25% position size or skip |
| <50 | **Noise** | No trade |

### 12.3 Bayesian Signal Grading (AutoQuant, 2026)
- Start with prior belief about signal quality
- Update weights based on actual outcomes
- **Regime-conditioned**: Same signal gets different weight in different regimes
- "RSI alone without volume: 49% win rate — effectively noise"
- "Key level tests with volume confirmation: 61% win rate, 400+ samples"

### 12.4 Signal Agreement Check
```python
def check_signal_agreement(signals: dict) -> float:
    """
    Check directional agreement across sub-signals
    Returns: 0-1 agreement score
    """
    directions = list(signals.values())
    if len(directions) == 0:
        return 0.0
    
    # All same direction = 1.0, all different = 0.0
    positive = sum(1 for d in directions if d > 0)
    negative = sum(1 for d in directions if d < 0)
    agreement = max(positive, negative) / len(directions)
    return agreement
```

### 12.5 Anti-Correlation for Signals
- "Three momentum signals confirming feels like 3x conviction. But if they're all downstream of the same VWAP relationship, it's actually 1x conviction with three labels" (AutoQuant, 2026)
- **De-correlate signal vector before scoring**
- Use PCA or mutual information to detect redundant signals

### 12.6 Recommended Signal Quality Pipeline
```python
class SignalQualityScorer:
    def __init__(self):
        self.weights = {
            'ml_confidence': 0.30,
            'confluence': 0.25,
            'regime_fitness': 0.15,
            'spread_quality': 0.15,
            'session_quality': 0.10,
            'data_freshness': 0.05,
        }
        self.threshold = 70  # Minimum score to trade
    
    def score(self, signal_context: dict) -> float:
        scores = {}
        scores['ml_confidence'] = signal_context['model_probability'] * 100
        scores['confluence'] = self._compute_confluence(signal_context)
        scores['regime_fitness'] = self._compute_regime_fitness(signal_context)
        scores['spread_quality'] = self._compute_spread_quality(signal_context)
        scores['session_quality'] = self._compute_session_quality(signal_context)
        scores['data_freshness'] = self._compute_freshness(signal_context)
        
        composite = sum(scores[k] * self.weights[k] for k in self.weights)
        return min(100, max(0, composite))
    
    def should_trade(self, score: float) -> bool:
        return score >= self.threshold
```

---

## Summary: Implementation Priority

### Phase 2A: Core Features (Implement First)
1. **ATR-based dynamic TP/SL** (4.5x ATR TP, 1.5x ATR SL)
2. **Session filter** (13:00-16:00 UTC only)
3. **Spread filter** (≤20% of TP target)
4. **News filter** (30 min before/after high-impact)
5. **Anti-clustering** (6-bar cooldown, max 1 position)
6. **Fixed fractional sizing** (1% risk per trade)

### Phase 2B: Advanced Features (Add Later)
7. **SMC/ICT features** (OB, FVG, BOS/CHoCH as ML inputs)
8. **Order flow features** (CVD, delta, absorption)
9. **Regime detection** (ADX + ATR rule-based)
10. **Signal quality scoring** (6-factor composite score)
11. **Meta-labeling** (secondary model for trade filtering)
12. **Walk-forward with regime-conditional labeling**

---

*Research compiled: 2026-06-27*
*Sources: QuantInsti, MQL5 Articles, arXiv papers, GitHub repos (51.8k+ stars), TradingView, ChartSnipe, ForexExperts, AlphaExCapital, AutoQuant, Falcon AI, thrive-fi, smartmoneyconcepts library, Kalena, ForexSpreadCompare, ForexMechanics, AIFinHub*
