import pandas as pd


@pd.api.extensions.register_dataframe_accessor("custom")
class CustomDataframeAccessor:
    def __init__(self, df):
        self._df = df

    def scolumns(self):
        return sorted(self._df.columns.to_list())

    def columns_na(self):
        return [col for col in self._df.columns if self._df[col].isna().any() and self._df[col].dtype != "object"]

    def scolumns_na(self):
        return sorted(self.columns_na())

    def sunique(self):
        return sorted(self._df.unique())

    def sinfo(self):
        return self._df.sort_index(axis=1).info()

    def describe(self):
        numeric_df = self._df.select_dtypes(include=["number"])  # Select only numeric columns
        desc = numeric_df.describe().drop(["25%", "50%", "75%"])  # Drop percentiles
        medians = numeric_df.median().rename("median")  # Add median
        desc = pd.concat([desc, medians.to_frame().T])  # Use concat instead of append
        return desc.round(2)

    def print(self):
        print(self._df)
        return None


# Usage
# df = pd.DataFrame({'A': [1, 2, 3], 'C': [4, 5, 6], 'B': [7, 8, 9]})
# print(df.custom.columns())


@pd.api.extensions.register_series_accessor("custom")
class CustomSeriesAccessor:
    def __init__(self, series):
        self._series = series

    def sunique(self):
        """Access the double method of the custom accessor."""
        return sorted(self._series.astype("category").cat.categories.to_numpy())
