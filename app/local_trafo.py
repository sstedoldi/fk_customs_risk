import numpy as np

to_log_list = ['CIF_USD_EQUIVALENT', 'QUANTITY', 'GROSS.WEIGHT', 'TOTAL.TAXES.USD']

def num_to_log(df, num_col):
    log_numeric = []
    for var in num_col:
        # checking 0 values before log transformation
        if df[var].min() == 0:
            df.loc[df[var] == 0.0, var] = 0.001

        df['log_'+var]=np.log(df[var])
        log_numeric.append('log_'+var)

    return df, log_numeric