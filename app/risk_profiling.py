import pandas as pd
import itertools
import os

def risk_profiling(dataframe, to_comb, target, save=False, output_path = ''):
    """
    Apply target encoding and build tables with risk profiles.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame with historic data to analyze.
    to_comb : list
        Categorical variables to combine.
    target : str
        Target variable for encoding and risk profiling.
    save : bool, optional
        Indication whether to save risk profile tables, default is False.
    output_path : str, optional
        Output path for saving risk profile tables, default is ''.

    Returns:
    --------
    df_ : pandas.DataFrame
        DataFrame with target encoding and assigned risk profiles.
    risk_tables : list
        List of names of the risk profile tables.

    Notes:
    ------
    - The function modifies the input DataFrame (`dataframe`) by adding target encoding columns and risk profiles.
    - If `save` is True, it saves risk profile tables in the specified `output_path`.

    Examples:
    ---------
    >>> df, risk_tables = risk_profiling(data, ['Category', 'Type'], 'Illicit', save=True, output_path='output_profiles/')
    """

    df_ = dataframe.copy() #To prevent chages in the data set

    risk_tables = []
    #------------------------------
    #------------------------------

    # Deleting nulls in the columns to process
    df_.dropna(subset = to_comb, inplace = True)
    df_.dropna(subset = [target], inplace = True)

    # Ensuring that categorical variables to_comb are str
    df_.loc[:,to_comb] = df_.loc[:,to_comb].astype(str)
    
    # Creating tuples of categotical to_comb
    combinations = list(itertools.combinations(to_comb, 2))

    # Loop to concatenate combinations date with '&&' 
    for (cat1, cat2) in combinations:
        colName = cat1 + '&&' + cat2
        df_.loc[:,colName] = df_.loc[:,cat1]+'&&'+df_.loc[:,cat2]
    
    profile_candidates = ['OFFICE', 'IMPORTER.TIN','TARIFF.CODE', 'ORIGIN.CODE'] \
                        + [col for col in df_.columns if '&&' in col]
    #print("profile_candidates",profile_candidates)
    #------------------------------
    
    ## Target encoding regarding sum of illicit and its rate: 'Encoded_sum' & 'Encoded_Mean'

    for feature in profile_candidates: #going over features to assing risk
        #DF with the data enconded
        df_encoded = df_[target].groupby(df_[feature]).agg(['sum','mean'])
        # print(df_encoded)
        df_encoded.rename(columns = {'sum': f'{feature}_{target}_sum', 
                                     'mean': f'{feature}_{target}_rate'}, inplace=True)
        #MERGE left to bring encoded data to the df_ trated 
        df_ = df_.merge(df_encoded, how='left', on=feature)
    
        #Saving Target encoding list for feature
        csv_name = f'{feature}_hist_sum&rate.csv'
       
        if save:
            path_name = os.path.join(output_path,csv_name)
            df_encoded.to_csv(path_name)
           
        risk_tables += [csv_name] #name of the new risk table

    #------------------------------
    ## Risk profiles according to the quartil of the sum (Qsum) and the rate (Qrate) of the illicit

    for feature in profile_candidates: #going over features to assing risk
        df_[f'{feature}_{target}_Qsum'] = pd.qcut(df_[f'{feature}_{target}_sum'],
                                                   q=[0, .25, .50, .75, 1], 
                                                   labels = False, duplicates = 'drop')   

        df_[f'{feature}_{target}_Qrate'] = pd.qcut(df_[f'{feature}_{target}_rate'], 
                                                   q=[0, .25, .50, .75, 1], 
                                                   labels = False, duplicates = 'drop') 

        ## Saving risk profiling in csv format
        df_profile = df_.drop_duplicates(subset=[feature]) #Subset of feature unique values
        df_profile.reset_index(drop=True, inplace=True)
        # Concatenation between feature_column and profile_columns
        df_profile = pd.concat([df_profile[feature], df_profile.filter(like=f'{feature}_{target}_Q')], axis=1) 
        
        # Naming and saving historic risk profile
        csv_name = feature+'_hist_Qsum&Qrate.csv'
        
        if save:
            path_name = os.path.join(output_path, csv_name)
            df_profile.to_csv(path_name)
            
        risk_tables += [csv_name] #name of the new risk table


    return df_, risk_tables