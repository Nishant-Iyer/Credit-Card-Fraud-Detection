import numpy as np
import pandas as pd
import pytest
from src.features.transformers import CyclicalTimeEncoder, DataFrameRobustScaler

def test_cyclical_time_encoder():
    # Test dataset with specific times
    # 86400 seconds in a day:
    # 0 sec (midnight) -> sin=0, cos=1
    # 21600 sec (6 AM) -> sin=1, cos=0
    # 43200 sec (12 PM) -> sin=0, cos=-1
    df = pd.DataFrame({
        "Time": [0.0, 21600.0, 43200.0]
    })
    
    encoder = CyclicalTimeEncoder(time_col="Time", period=86400.0, drop_original=True)
    df_transformed = encoder.fit_transform(df)
    
    assert "Time_sin" in df_transformed.columns
    assert "Time_cos" in df_transformed.columns
    assert "Time" not in df_transformed.columns
    
    # Assert values match trigonometry
    np.testing.assert_allclose(df_transformed["Time_sin"].values, [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(df_transformed["Time_cos"].values, [1.0, 0.0, -1.0], atol=1e-5)


def test_dataframe_robust_scaler():
    df = pd.DataFrame({
        "Amount": [10.0, 20.0, 100.0], # Median = 20.0, IQR = 90.0 - 15.0 approx (depends on interpolation)
        "Other": [1.0, 2.0, 3.0]
    })
    
    scaler = DataFrameRobustScaler(scale_cols=["Amount"])
    df_scaled = scaler.fit_transform(df)
    
    # Column should still be present
    assert "Amount" in df_scaled.columns
    assert "Other" in df_scaled.columns
    
    # "Other" column should not be modified
    pd.testing.assert_series_equal(df_scaled["Other"], df["Other"])
    
    # Amount column should be scaled (median should become 0)
    assert df_scaled["Amount"].iloc[1] == 0.0
