import pandas as pd

def preprocesser(df, feature_engine, num_to_log, model_features, std_scaler, minmax_scaler):

    target_info = ['illicit','RAISED_TAX_AMOUNT_USD']
    target = 'illicit'
    ori_numeric = df.drop(['year','month','day']+target_info, axis = 1).select_dtypes('number').columns.tolist()
    last_year=df.year.max()    
    df_new = df[(df.year == last_year) & (df.month > 6)]
    df_new = df_new[:-int(4*len(df_new)/100)]
    X = df_new.drop(target_info, axis=1)
    y = df_new[target]
    X = feature_engine.transform(X)
    X, _ = num_to_log(X, ori_numeric)
    X = X[model_features]
    X.fillna(0, inplace=True)
    X = X[std_scaler.feature_names_in_]
    X[std_scaler.feature_names_in_] = std_scaler.transform(X[std_scaler.feature_names_in_])
    # minmax scaler
    X = X[minmax_scaler.feature_names_in_]
    X[minmax_scaler.feature_names_in_] = minmax_scaler.transform(X[minmax_scaler.feature_names_in_])

    return X, y

class DDT_cleaner():
    def __init__(self):
        self.to_drop = ['Sumario Contencioso s-n', # target info
                        'Decision Final',
                        'Comiso s-n','Observaciones',
                        'Codigo del Verificador',
                        'Cinfcst',
                        # 'infraccion',
                        'Minfperjui SUM',
                        'Indicador deudas S-N', # 100% null
                        'Canal de Seleccion  - DDT', # All R
                        'Desc Destinacion', # Text unnecesary
                        'Codigo de la destinacion',
                        'Precio Oficial en dolares - ART SUM', # Numeric with all 0
                        'Tipo Agente', # All DESP
                        'Tipo Agente.1', # All IMEX
                        'Monto Fob en U$S - ART SUM_x', # preserving _y column
                        'Fecha de Embarque - DDT', 
                        'Fecha de Cumplido - DDT',
                        'Pais de Destinacion - DDT']
        self.date_col=['Fecha de Oficializacion - DDT',
                        # 'Fecha de Cumplido - DDT',
                        # 'Fecha de Embarque - DDT',
                        'Fec. Inicio Act. DESP',
                        'Fech. alta Sist. DESP',
                        'Fec. Inicio Act. IMEX',
                        'Fech. alta Sist. IMEX']
        self.new_names=[{'Fec. Inicio Act.' : 'Fec. Inicio Act. DESP'},
                        {'Fech. alta Sist' : 'Fech. alta Sist. DESP'},
                        {'Tipo Empresa' : 'Tipo DESP'},
                        {'Tipo Empresa.1' : 'Tipo IMEX'},
                        {'Fec. Inicio Act..1' : 'Fec. Inicio Act. IMEX'},
                        {'Fech. alta Sist.1' : 'Fech. alta Sist. IMEX'},
                        {'Monto Fob en U$S - ART SUM_y' : 'Monto Fob en U$S - ART SUM'},
                        {'DDT_item' : 'Codigo de la destinacion + item N°'}]
        self.to_shadow=['Indicador embargos S-N', 
                        'Tonalidad Canal Rojo - DDT',
                        # 'Fecha de Embarque - DDT',
                        # 'Fecha de Cumplido - DDT',
                        # 'Pais de Destinacion - DDT',
                        'Via Medio de Transporte - DDT',
                        'Lugar operativo - DDT']
    def simplification(self,X):
        print("## Simplifying codes\n")
        try:
            X['Destinacion_XX'] = X['Destinacion - DDT'].str[0:2]
        except Exception as e:
                print(e)
        return X
    def drop_duplicates(self,X):
        print("## Dropping duplicates\n")
        X = X.drop_duplicates(keep='last')
        return X
    def drop_useless(self,X):
        print("## Dropping useless columns\n")
        for col in self.to_drop:
            try:
                X.drop(col, axis=1, inplace=True)
            except Exception as e:
                print(e)
        return X
    def rename_cols(self,X):
        print("## Renaming columns\n")
        for col in self.new_names:
            try:
                X.rename(columns = col, inplace=True)
            except Exception as e:
                print(e)
        return X
    def date_format(self,X):
        print("## Date formarting\n")
        for col in self.date_col:
            try:
                X[col] = pd.to_datetime(X[col])
            except Exception as e:
                print(e)
        return X
    def null_imputation(self,X):
        print('## Null imputation\n')
        for col in self.to_shadow:
            try:
                X[col + '_shadow'] = X[col].isnull()
            except Exception as e:
                print(e)
        # binary variables
        X['Tonalidad Canal Rojo - DDT'] = X['Tonalidad Canal Rojo - DDT'].fillna(False).replace('VALOR', True)
        X['Indicador embargos S-N'] = X['Indicador embargos S-N'].fillna(False).replace({'S':True,'N':False})
        ## transpor means
        transpor_means = {'2': 'AIRE',
                          '4': 'TIERRA',
                          '8': 'MAR',
                          '1': 'OTRO',
                          '3': 'OTRO',
                          '5': 'OTRO',
                          '6': 'OTRO',
                          '7': 'OTRO',
                          '9': 'OTRO',
                          'A': 'OTRO'}
        X['Via Medio de Transporte - DDT'] = X['Via Medio de Transporte - DDT'].fillna('HIDRO').replace(transpor_means) 
        ## operative places
        X['Lugar operativo - DDT'] = X['Lugar operativo - DDT'].fillna('SIN LUGAR')
        # cheking nulls
        if X.isnull().sum().sum() != 0: 
            print('-- Unsuccessful imputation!\n')
        
        return X
            
    def preprocess(self, X):
        X_=X.copy()
        print('# Preprocessing\n')
        X_ = self.simplification(X_)
        print('------------------------')
        X_ = self.drop_duplicates(X_)
        print('------------------------')
        X_ = self.drop_useless(X_)
        print('------------------------')
        X_ = self.rename_cols(X_)
        print('------------------------')
        X_ = self.date_format(X_)
        print('------------------------')
        X_ = self.null_imputation(X_)
        print('Preprocessing finished')
        print('------------------------')
        print('------------------------')
        return X_
    
