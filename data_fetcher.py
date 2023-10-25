import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Fetcher(BaseEstimator, TransformerMixin):

    def __init__(self, building_id=122, meter=0, primary_use=99):  # 99 referes to none

        self.building_id = building_id
        self.meter = meter
        self.primary_use = primary_use

    def fit(self, x, y=None):
        return self  # nothing else to do

    def season_finder(self, month):  # 0 for sprint - 1 for summer - 2 for fall - 3 for winter
        if month in [3, 4, 5]:
            return 0
        elif month in [6, 7, 8]:
            return 1
        elif month in [9, 10, 11]:
            return 2
        elif month in [12, 1, 2]:
            return 3

    def transform(self, x):
        df = x.copy()
        df = df.drop(['Unnamed: 0', 'precip_depth_1_hr',
                      'cloud_coverage', 'site_id', 'square_feet'], axis=1)

        if self.primary_use == 99:

            df = df.query(
                f'building_id=={self.building_id} & meter=={self.meter}')
            new_df = df.drop(['building_id', 'meter', 'primary_use'], axis=1)
        else:
            df = df.query(
                f'building_id=={self.building_id} & meter=={self.meter} & primary_use =={self.primary_use}')
            new_df = df.drop(['building_id', 'meter', 'primary_use'], axis=1)
        timestamp = pd.to_datetime(new_df["timestamp"])
        new_df.loc[:, ['season']] = new_df.month.apply(self.season_finder)
        new_df.loc[:, ['weekend']] = timestamp.dt.dayofweek > 4
        new_df.loc[:, ['day_of_the_week']] = timestamp.dt.dayofweek
        new_df.set_index('timestamp', inplace=True)

        return new_df


class Do_nothing(BaseEstimator, TransformerMixin):

    def __init__(self):  # 99 referes to none
        return

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x):
        return x.astype(int)
