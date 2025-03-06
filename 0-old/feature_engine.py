# Dependencies 
import os
import pandas as pd
import itertools
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Feature_engine(BaseEstimator, TransformerMixin):

    '''
    Feature_engine class for transforming new data.

    Parameters:
    - tariff_code: str, default='HS06'
        Tariff code to analyze.
    - IQR_fact: float, default=1.5
        Interquartile Range (IQR) factor for statistical analysis.
    - lower_IQR: bool, default=False
        Flag to include lower IQR limit in statistical analysis.
    - save_stats: bool, default=False
        Flag to save statistics to a CSV file.
    - tariff_k: int, default=5
        Number of clusters for KMeans clustering.
    - to_comb: list, default=[]
        List of categorical features to combine.
    - risk_tables: list, default=[]
        List of historical risk tables to work with.
    - risk_tables_path: str, default=''
        relative directory where the risk tables are saved
    - random_seed: int, default=1
        Random seed for reproducibility.

    Methods:
    - basic_num_features(X):
        Creates new basic numerical features.
    - date_features(X):
        Creates date-related features.
    - activity_analize(X):
        Analyzes activity information.
    - activity_features(X):
        Creates new features from activity information.
    - stats_analyze(X, col_to_process):
        Analyzes statistical information.
    - stats_col_name(var):
        Generates column names for statistical features.
    - clustering(X, cols, var):
        Performs clustering using KMeans.
    - fit(X, y=None):
        Fits the feature engineering model to the data.
    - transform(X, y=None):
        Transforms the data using the fitted model.

    Attributes:
    - tariff_code_stats: DataFrame
        Statistical information for tariff codes.
    - code_activity: dict
        Activity information for specified variables.
    - fitted: bool
        Flag indicating if the model has been fitted.

    Examples:
    ```python
    # Instantiate the Feature_engine class
    feature_engine = Feature_engine()

    # Fit the model to the data
    feature_engine.fit(train_data)

    # Transform new data using the fitted model
    transformed_data = feature_engine.transform(new_data)
    ```

    Note:
    The class is designed for feature engineering and transformation of new data based on statistical analysis and clustering.
    '''

    def __init__(self, tariff_code='HS06', 
                IQR_fact=1.5, lower_IQR=False, 
                save_stats=False, tariff_k=5,
                to_comb=[], risk_tables=[],
                stats_info_path='',
                risk_tables_path='',
                random_seed=1):

        # tariff code to analyse 
        self.tariff_code = tariff_code
        ## stats parameters to use
        self.IQR_fact = IQR_fact
        self.lower_IQR = lower_IQR
        self.stats_vars = ['Unitprice', 'WUnitprice', 'TaxRatio']
        # tariff code stats
        self.tariff_code_stats = pd.DataFrame()
        self.save_stats = save_stats
        # tariff code clusters
        self.tariff_k = tariff_k
        # activity information
        self.code_activity = {}
        self.activity_vars = ['OFFICE', 'IMPORTER.TIN', 'ORIGIN.CODE']
        # categorical to combine 
        self.to_comb = to_comb
        # risk tables to work with
        self.risk_tables = risk_tables
        # risk tables path
        self.risk_tables_path = risk_tables_path
        # random seed
        self.random_seed = random_seed
        # status
        self.fitted = False

    def basic_num_features(self, X):
        # print('Creating new basic numerical features')

        X[self.tariff_code] = X['TARIFF.CODE'].str[:6]
        X.loc[:, 'Unitprice'] = X.loc[:,'CIF_USD_EQUIVALENT'] / X.loc[:,'QUANTITY']
        X.loc[:, 'WUnitprice'] = X.loc[:,'CIF_USD_EQUIVALENT'] / X.loc[:,'GROSS.WEIGHT']
        X.loc[:, 'TaxRatio'] = X.loc[:,'TOTAL.TAXES.USD'] / X.loc[:,'CIF_USD_EQUIVALENT']
        X.loc[:, 'TaxUnitquantity'] = X.loc[:,'TOTAL.TAXES.USD'] / X.loc[:,'QUANTITY']

        return X

    def date_features(self, X):
        
        X['formatted_date'] = pd.to_datetime(X[['year', 'month', 'day']], errors='coerce')
        X['formatted_date'] = X['formatted_date'].fillna(method='ffill')
        
        X['day_of_week'] = X['formatted_date'].dt.dayofweek

        return X

    def activity_analize(self, X):
        # fit method
        code_activity = {}

        for var in self.activity_vars:
            code_activity[var] = X[var].value_counts().reset_index()
            code_activity[var].columns = [var, f'{var}_act']        
        
        return code_activity
    
    def activity_features(self, X):
        # transform method
        for var in self.activity_vars:
            X = pd.merge(X, self.code_activity[var], on=var, how='left')
            X[f'{var}_act_Q'] = pd.qcut(X[f'{var}_act'], q=[0, .25, .50, .75, 1], labels = False, duplicates = 'drop')
        return X

    def stats_analyze(self, X, col_to_process):
        # fit method
        print(f'Extracting stats info from {col_to_process}')

        tariff_codes_used = list(X[self.tariff_code].unique())
        # stats description per each tariff code regarding var
        dic_code_stats = {}

        for code in tqdm(tariff_codes_used, total=len(tariff_codes_used), mininterval=0.1):
            df_code = X[col_to_process].loc[X[self.tariff_code] == code]
            col_stats = df_code.describe()
            col_stats['p10'] = df_code.quantile(0.1)
            col_stats['p90'] = df_code.quantile(0.9)
            dic_code_stats[code] = col_stats

        # Tariff code var description
        df_code_stats = pd.DataFrame(dic_code_stats).T
        # IQR
        df_code_stats.loc[:, 'IQR'] = df_code_stats.loc[:,'75%'] - df_code_stats.loc[:,'25%']
        # upper limit
        df_code_stats.loc[:, f'3Q+{str(self.IQR_fact)}*IQR'] = df_code_stats.loc[:,'75%'] + self.IQR_fact*df_code_stats.loc[:, 'IQR']
        # lower limit # this column is not useful because of the chi2 distribution of this variables
        if self.lower_IQR:
            df_code_stats.loc[:, f'1Q-{str(self.IQR_fact)}*IQR'] = df_code_stats.loc[:,'25%'] - self.IQR_fact*df_code_stats.loc[:, 'IQR']

        # imputing missing std for cases with not enough
        df_code_stats.fillna(0, inplace=True)

        return df_code_stats

    def stats_col_name(self, var):
        # fit method
        new_col_name = {
                # 'count': var+'_count',
                'mean': var+'_mean',
                'std': var+'_std',
                'min': var+'_min',
                '25%': var+'_p25',
                '50%': var+'_p50',
                '75%': var+'_p75',
                'max': var+'_max',
                'p10': var+'_p10',
                'p90': var+'_p90',
                f'3Q+{str(self.IQR_fact)}*IQR': var+'_'+f'3Q+{str(self.IQR_fact)}*IQR',
                f'1Q-{str(self.IQR_fact)}*IQR': var+'_'+f'1Q-{str(self.IQR_fact)}*IQR',
            }
        
        return new_col_name

    def clustering(self, X, cols, var):
        # fit method
        print(f'Getting clusters for {var}')
        features = X[cols]
        features_scaled = StandardScaler().fit_transform(features)

        kmeans = KMeans(n_clusters=self.tariff_k, init='k-means++',
                        random_state=self.random_seed)

        X[f'{var}_cluster'] = kmeans.fit_predict(features_scaled)

        return X

    def activity_features(self, X):
        for var in self.activity_vars:
            X = pd.merge(X, self.code_activity[var], on=var, how='left')
            X[f'{var}_act_Q'] = pd.qcut(X[f'{var}_act'], q=[0, .25, .50, .75, 1], labels = False, duplicates = 'drop')
        
        return X

    def stats_features(self, X):
        X = X.merge(self.tariff_code_stats, how='left', 
                    left_on=self.tariff_code, right_index=True)
        
        print('Creating new features from stats info')
        name = self.tariff_code
        
        for var in self.stats_vars:
            print(f'Working on {var} stats')
            # count of occurrences per tariff code
            X[f'{name}_activity'] = X['count']
            # difference between the variable and its mean
            X[f'{name}_{var}_diff_mean'] = X[var] - X[f'{var}_mean']
            # difference between the variable and its median
            X[f'{name}_{var}_diff_median'] = X[var] - X[f'{var}_p50']
            # indicating whether the variable is under the 10th percentile
            X[f'{name}_{var}_under_p10'] = X[var] < X[f'{var}_p10']
            # indicating whether the variable is over the 90th percentile
            X[f'{name}_{var}_over_p90'] = X[var] > X[f'{var}_p90']
            # indicating whether the variable is over a calculated threshold based on IQR
            X[f'{name}_{var}_over_{str(self.IQR_fact)}*IQR'] = X[var] > X[f'{var}_3Q+{str(self.IQR_fact)}*IQR']
            # categorical column ('{name}_{var}_Q') indicating quartile ranges
            X[f'{name}_{var}_Q'] = pd.qcut(X[var], q=[0, .25, .50, .75, 1], labels = False, duplicates = 'drop')

        # dropping unuseful columns
        stats_cols = list(self.tariff_code_stats.columns)[:-3]
        X.drop(columns = stats_cols, inplace=True)
        
        return X
    
    def cat_comb_features(self, X):
        # transform method
        # making sure that to_comb features are str
        X.loc[:,self.to_comb] = X.loc[:,self.to_comb].astype(str)
        
        # creating combinations
        combinations = list(itertools.combinations(self.to_comb, 2))
        
        # Loop to combine features with a double & simbol 
        for (cat1, cat2) in combinations:
            colName = cat1 + '&&' + cat2
            X.loc[:,colName] = X.loc[:,cat1]+'&&'+X.loc[:,cat2]

        return X
    
    def hist_risk_features(self, X):
        # transform method
        for table in self.risk_tables:
            # Reading each table in the risk_tables list
            PATH_INPUT_TABLE = os.path.join(self.risk_tables_path, table)
            df_table = pd.read_csv(PATH_INPUT_TABLE)
            # Patch to eliminate the index column (unnamed) in some of the tables
            df_table.drop(df_table.columns[df_table.columns.str.contains('unnamed',case = False)], axis = 1, inplace = True)
            # Key to JOIN dataframes
            key = df_table.columns[0]
            # Making sure that risk table data are str
            df_table.loc[:,key] = df_table.loc[:,key].astype(str)
            # Merge
            X = pd.merge(X, df_table, left_on=key, right_on=key, how='left')

            # Join
            # X = X.join(df_table.set_index(key), on = key)
        
        return X

    def fit(self, X, y=None):
        
        X_ = X.copy()

        print('# Fitting started\n')

        print('## Calculating basic numeric feature to fit\n')
        X_ = self.basic_num_features(X_)

        print(f'## Processing activity information')
        print(f'Variables: {self.activity_vars}\n')

        self.code_activity = self.activity_analize(X_)

        print('## Processing tariff codes stats information')
        print(f'Codes: {self.tariff_code}')
        print(f'Variables: {self.stats_vars}\n')

        self.tariff_code_stats = pd.DataFrame()

        # tariff code stats from vars
        for var in self.stats_vars:

            df_tariff_var = self.stats_analyze(X_, col_to_process = var)

            new_col_name = self.stats_col_name(var)
            df_tariff_var.rename(columns=new_col_name, inplace=True)

            if self.tariff_code_stats.empty:
                self.tariff_code_stats = df_tariff_var
            else:
                self.tariff_code_stats = pd.merge(self.tariff_code_stats, 
                                                df_tariff_var.drop(columns='count'), 
                                                left_index=True, right_index=True)

        self.tariff_code_stats.sort_index(inplace=True)

        # clustering
        print(f'\nCreating clusters from {self.tariff_code} stats info')

        for var in self.stats_vars:
            input_cols = [col for col in self.tariff_code_stats.columns if col.startswith(var)]
            self.tariff_code_stats = self.clustering(self.tariff_code_stats, input_cols, var)

        if (not self.tariff_code_stats.empty) & self.save_stats:
            self.tariff_code_stats.to_csv(os.path.join(self.stats_info_path,'tariff_stats.csv'))

        if (not self.tariff_code_stats.empty) & (len(self.code_activity)>0):
            self.fitted = True
            print('\nFitting finished succesfully')
        else:
            print('\nFitting finished unsuccesfully')

        return
        
    def transform(self, X, y=None):
        if self.fitted:
            print('# Transformation started\n')

            X_ = X.copy()

            # creating new basic numeric features
            print('## Creating new basic numerical features\n')
            X_ = self.basic_num_features(X_)

            # creating date related features
            print('## Creating date related features\n')
            X_ = self.date_features(X_)

            # creating new features from activity info
            print('## Creating activity features')
            print(f'Variables: {self.activity_vars}\n')
            X_ = self.activity_features(X_)

            # creating new features from tariff codes stats info
            print('## Creating tariff codes stats features')
            print(f'Codes: {self.tariff_code}')
            print(f'Variables: {self.stats_vars}\n')        
            X_ = self.stats_features(X_)

            # creating categical combined features
            print('\n## Creating new caterical features combining the existing ones')
            print(f'Variables to combine: {self.to_comb}\n')
            X_ = self.cat_comb_features(X_)

            # creating hist risk feature
            print('## Matching new training codes with hisrical risk profiles\n')
            X_ = self.hist_risk_features(X_)
          
            # X_.reset_index(drop=True,inplace=True)

            print('Transformation finished')

            return X_
        
        else:
            print('Transformation is not possible without a previous fit')
            return