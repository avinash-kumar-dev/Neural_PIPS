import pandas as pd
import numpy as np
from datetime import datetime, timezone


class LiveFeatureEngineer:
    def __init__(self, feature_engineer, feature_pipeline):
        self.engineer = feature_engineer
        self.pipeline = feature_pipeline

    def compute_for_latest_bar(self, m5_df, m15_df, h1_df, h4_df):
        all_features = self.engineer.compute_all(m5_df, m15_df, h1_df, h4_df)

        last_idx = all_features.index[-1]
        last_row = all_features.loc[last_idx].copy()
        last_row['time'] = m5_df['time'].iloc[-1]

        raw_60 = last_row[self.engineer.FEATURE_VECTOR].values.reshape(1, -1)
        raw_60 = np.nan_to_num(raw_60, nan=0.0, posinf=0.0, neginf=0.0)

        feat_names = self.pipeline.feature_names_in_
        raw_dict = dict(zip(feat_names, raw_60[0]))

        scaled = self.pipeline.transform(raw_60)
        scaled_names = self.pipeline.feature_names_out_
        scaled_dict = dict(zip(scaled_names, scaled[0]))

        scaled_dict['time'] = last_row['time']

        return scaled_dict, raw_dict

    def compute_atr_pips(self, m5_df, period=14):
        tr = pd.concat([
            m5_df['high'] - m5_df['low'],
            abs(m5_df['high'] - m5_df['close'].shift(1)),
            abs(m5_df['low'] - m5_df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return atr * 10000
