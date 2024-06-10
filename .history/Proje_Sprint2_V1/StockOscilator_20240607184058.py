# chaikin_oscillator.py

import pandas as pd

def calculate_chaikin_oscillator(data, periods_short=3, periods_long=10, high_col='high', low_col='low', close_col='close', vol_col='volume'):
    """
    Calculate the Chaikin Oscillator for the given data.

    Parameters:
    - data: DataFrame containing the data with columns for high, low, close, and volume.
    - periods_short: Number of periods for the short EMA.
    - periods_long: Number of periods for the long EMA.
    - high_col: Name of the column containing high prices.
    - low_col: Name of the column containing low prices.
    - close_col: Name of the column containing close prices.
    - vol_col: Name of the column containing volume.

    Returns:
    - DataFrame: Original DataFrame with a new column 'ch_osc' containing the Chaikin Oscillator values.
    """
    ac = pd.Series(index=data.index, dtype='float64')
    val_last = 0

    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            val = val_last + ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (
                        row[high_col] - row[low_col]) * row[vol_col]
        else:
            val = val_last
        ac[index] = val
        val_last = val

    ema_long = ac.ewm(span=periods_long, adjust=False).mean()
    ema_short = ac.ewm(span=periods_short, adjust=False).mean()
    data['ch_osc'] = ema_short - ema_long

    return data