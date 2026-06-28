# XAUUSD Implementation Plan — 7 Improvements

## Goal
- R:R >= 2:1 on every trade
- 4,000 pips/month target
- Best accuracy, best setups
- H4→M15→M5 ICT cascade

## Improvement 1: Trailing Stop (Trails Upward, No Fixed TP)

**Current (R:R 1.13):**
- BE at 1R → exits at breakeven too early
- Trail at 1R behind peak → captures ~50% of MFE
- No TP exit → winners give back profit

**New (R:R >= 2:1):**
- No fixed TP — trailing IS the exit strategy
- BE at 1.5R → gives trade more room to breathe
- Trail at 1.5R behind peak → captures ~65% of MFE
- SL only moves forward, never backward

**Logic:**
```
if MFE >= 1.5R and not BE hit:
    SL = entry + $0.10 (breakeven + 1 pip)
    BE hit = True

if MFE >= 2.5R:
    new_SL = entry + (MFE - 1.5R) * SL_distance
    if new_SL > current_SL:
        SL = new_SL  (only moves forward)
```

**Example (LONG):**
- Entry: $2000, SL: $1985 (SL distance = $15)
- Price → $2022.50 (MFE = 1.5R) → SL = $2000.10 (BE)
- Price → $2037.50 (MFE = 2.5R) → SL = $2015 (lock 1R)
- Price → $2052.50 (MFE = 3.5R) → SL = $2030 (lock 2R)
- Price → $2082.50 (MFE = 5.5R) → SL = $2060 (lock 4R)
- Price pulls back to $2060 → exit at $2060 = **+4R profit**

## Improvement 2: Wire M5 Confirmation into Pipeline

**Current:** `m5_confirmation.py` exists but is orphaned from pipeline.
**New:** Import and call `compute_m5_confirmation()` in pipeline after M15 signals.

**M5 provides:**
- `m5_bull_confirm` / `m5_bear_confirm` — MSS/rejection/engulfing + strong body + volume spike
- Use as additional quality gate: M15 signal + M5 confirmation = higher confidence

## Improvement 3: H4→M15→M5 Cascade

**Current:** Pipeline runs on single M15 timeframe.
**New:** Three-layer cascade:

1. **H4 layer** (already exists in `layer1.py`):
   - Regime detection (Hurst+ADX+Choppiness)
   - BOS/CHoCH for direction
   - EMA 50/200 filter
   - Output: LONG_ONLY / SHORT_ONLY / NO_BIAS

2. **M15 layer** (already exists in pipeline):
   - PD Array Matrix (OB, FVG, BPR, Breaker, IFVG, Unicorn, Rejection)
   - Premium/Discount zones
   - OTE, Liquidity Pools, VWAP, Asian Range
   - Output: Which PD array is price at?

3. **M5 layer** (new — wire m5_confirmation.py):
   - MSS/CHoCH confirmation
   - Rejection candle
   - Engulfing pattern
   - Volume spike
   - Output: Precise entry candle

**Cascade rule:** H4 bias → M15 setup → M5 trigger. All three must align.

## Improvement 4: R:R >= 2:1 Filter

**Current:** min_rr=2.0 in BacktestConfig but not enforced in pipeline.
**New:** Enforce in pipeline:
- Calculate TP2 = 2x SL distance
- Check if next structural level (liquidity pool, swing high/low) is >= TP2
- If not, reject trade
- This ensures every trade has room to reach 2:1

## Improvement 5: Remove Breakeven-at-1R (Move to 1.5R)

**Current:** BE at 1R → too aggressive, exits trades that would have been winners.
**New:** BE at 1.5R → gives trade 50% more room before locking capital.

## Improvement 6: M5 Data Alignment

**Current:** M5 data exists (100K bars) but not used.
**New:** 
- Load M5 data alongside M15
- For each M15 signal, find the corresponding M5 bar
- Use M5 confirmation at that exact bar
- M5 entry = first M5 bar after M15 signal with confirmation

## Improvement 7: Confluence Scoring Update

**Current:** 10 factors, M5 not included.
**New:** Add M5 confirmation as factor:
- M5 MSS aligned: +5 points
- M5 rejection candle: +3 points
- M5 engulfing: +3 points
- M5 volume spike: +2 points
- Max M5 contribution: 8 points (rebalance other factors)

## Implementation Order

1. Save plan ✅
2. Fix backtest engine (trailing stop + TP logic)
3. Wire M5 confirmation into pipeline
4. Implement H4→M15→M5 cascade
5. Add R:R >= 2:1 filter
6. Add M5 data alignment
7. Update confluence scoring
8. Run test backtest (5K bars)
9. Run full backtest (100K bars)
10. Commit and push

## Files to Modify

| File | Change |
|------|--------|
| `xauusd/backtest/engine.py` | Trailing stop rewrite, TP exit option |
| `xauusd/pipeline.py` | Wire M5, cascade, R:R filter |
| `xauusd/execution/confluence.py` | Add M5 factor |
| `xauusd/entry/m5_confirmation.py` | May need minor fixes |
| `scripts/backtest_full_pipeline.py` | Update for new pipeline |

## Success Criteria

- R:R >= 2:1 on all trades
- Win rate >= 45%
- PF >= 1.5
- 4,000 pips/month on test set
- M5 confirmation fires on >= 30% of M15 signals

*Created: 2026-06-29*
