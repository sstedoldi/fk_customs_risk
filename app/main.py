### dependencies
## gral
import os
from datetime import datetime
## application
from flask import Flask, jsonify, request, render_template
## pre-process
import pandas as pd
from local_prepro import preprocesser#,DDT_cleaner
## transformations
from local_trafo import to_log_list, num_to_log
import joblib
## model
from xgboost import XGBClassifier
import json
## explanation
import lime
import lime.lime_tabular
import dill as pickle
## local
from feature_engine import Feature_engine # to read and transform
from feature_selected import feature_selected_list # to read list of features
# from onehot_features import onehot_feature_list # to read columns need for one-hot features
from risk_tables import risk_tables_list # to read list of risk tables
## future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

### app instance
app = Flask(__name__, static_folder='images')

### context
with app.app_context():
    ## context parameters
    current_directory = os.path.dirname(__file__)
    # risk info
    risk_profiles_path = "risk_profiles"
    risk_tables = risk_tables_list
    # stats info
    stats_info_path = "stats_info"
    # act info
    act_info_path = "act_info"
    # model
    model_path = os.path.join(current_directory, "models/XGB_synthetic_17-03-2024_17-24-36.json")
    model_meta_path = os.path.join(current_directory, "models/XGB_synthetic_17-03-2024_17-24-36_metadata.json")
    explainer_path = os.path.join(current_directory, "explainers/XGB_synthetic_18-03-2024_10-56-12_explainer.pkl")    
    # transformations
    minmax_scaler_path = os.path.join(current_directory, "scalers/XGB_synthetic_17-03-2024_17-24-36_minmax_scaler.pkl")
    std_scaler_path = os.path.join(current_directory, "scalers/XGB_synthetic_17-03-2024_17-24-36_standar_scaler.pkl")
    # encoding
    cate_to_targetEncoding = ['OFFICE', 'IMPORTER.TIN', 'TARIFF.CODE', 'ORIGIN.CODE']
    cate_to_oneHotEncoding = []
    to_log = to_log_list
    # feature selected
    feature_selected = feature_selected_list
    # output paths
    output_pred_path = "predictions"
    ## context objects
    # cleaner
    # cleaner = DDT_cleaner()
    # feature engine
    feature_engine = Feature_engine(to_comb=cate_to_targetEncoding, 
                        risk_tables=risk_tables,
                        stats_and_act_work='read', # not processing stats info nor activity
                        save_stats=False, 
                        save_act=False,
                        stats_tables_path=stats_info_path,
                        act_tables_path=act_info_path,
                        risk_tables_path=risk_profiles_path)
    feature_engine.fit()
    # scalers
    std_scaler = joblib.load(std_scaler_path)
    minmax_scaler = joblib.load(minmax_scaler_path)
    # model
    model = XGBClassifier()
    model.load_model(model_path)
    # print(model)
    # model features
    with open(model_path, 'r') as model_file:
        model_info = json.load(model_file)
        model_features = model_info['learner']['feature_names']
    model_file.close()
    # model meta
    with open(model_meta_path, 'r') as metadata_file:
        model_meta = json.load(metadata_file)
    metadata_file.close()
    print(model_meta)
    best_thr = float(model_meta['Optimal threshold'])
    # explainer
    # with open(explainer_path, 'rb') as exp_file:
    #     explainer = pickle.load(exp_file)
    # exp_file.close()
    dtypes_dict = {
                    'TARIFF.CODE':str,
                    'CIF_USD_EQUIVALENT': float,
                    'QUANTITY': float,
                    'GROSS.WEIGHT': float,
                    'TOTAL.TAXES.USD': float,
                    'RAISED_TAX_AMOUNT_USD':float,
                    'illicit':bool
                    }
    df_raw = pd.read_csv('import_data.csv', sep=",", dtype=dtypes_dict)
    X_new, y_new = preprocesser(df_raw, feature_engine, 
                                num_to_log,
                                model_features,
                                std_scaler,
                                minmax_scaler)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_new,
                                                   mode='classification',
                                                   training_labels = y_new, 
                                                   class_names=['Legal', 'Fraud'], 
                                                   feature_names=list(X_new.columns),
                                                   discretize_continuous=False,
                                                   verbose=True,
                                                   random_state=42)
    del df_raw, X_new, y_new

### root
@app.route('/')
def index():
    app_name = 'Synth_risk App Server'
    program_name = 'CustomPortal Project'
    author_name = 'SST'
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return render_template('index.html', app_name=app_name, program_name=program_name, 
                           author_name=author_name, current_date=current_date)

### metadata
@app.route('/metadata', methods=['POST'])
def metadata():
    try:
        # getting request
        print('# Model metadata requested #')
        # reading metadata
        with open(model_meta_path, 'r') as file:
            metadata = json.load(file)
        print(metadata)
        
        return jsonify(metadata)
    
    except Exception as e:
        print(e)
    return jsonify({'error': 'Internal Server Error'}), 500
    
### predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # getting request
        X_request = request.get_json(force=True)
        print('------------------------')
        print('------------------------')
        print('# New request #')
        # request as pandas df
        X = pd.DataFrame(X_request, index=[0])
        print(X)
        # import checking
        X_ = X.copy()
        # cleaning and preprocess
        ###
        # feature engine transformation
        X_ = feature_engine.transform(X_)
        # one-hot encoding
        ###
        # log transformation
        X_, _ = num_to_log(X_, to_log)
        # var selection
        X_ = X_[feature_selected]
        # nulls imputation
        X_.fillna(0, inplace=True)
        # standarization
        # re-order the features for standarization
        X_ = X_[std_scaler.feature_names_in_]
        X_[std_scaler.feature_names_in_] = std_scaler.transform(X_[std_scaler.feature_names_in_])
        # minmax scaler
        X_ = X_[minmax_scaler.feature_names_in_]
        X_[minmax_scaler.feature_names_in_] = minmax_scaler.transform(X_[minmax_scaler.feature_names_in_])
        # prediction
        # re-order the features for prediction
        X_ = X_[model_features]
        proba = model.predict_proba(X_)
        # explanaition
        exp = explainer.explain_instance(X_.loc[0], model.predict_proba, num_features=10)
        dict_exp = dict(exp.as_list())
        # response
        X['proba'] = proba[:, 1]
        print('# Explanation #')
        print(dict_exp)
        print('------------------------')
        print('------------------------')
        if proba is not None:
            response_data = {'input': str(X),
                            'expla': dict_exp,
                            'proba': str(proba[:, 1])}
        else:
            response_data = {'error': 'No prediction generated'}

        return jsonify(response_data)
    
    except Exception as e:
        print(e)
        return jsonify({'error': 'Internal Server Error'}), 500

### predict
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    print('Predict batch')
    try:
        # getting request
        X_request = request.get_json(force=True)
        print('------------------------')
        print('------------------------')
        print('# New request #')
        # request as pandas df
        X = pd.DataFrame(X_request)
        print(X)
        # import checking
        X_ = X.copy()
        # cleaning and preprocess
        ###
        # feature engine transformation
        X_ = feature_engine.transform(X_)
        # one-hot encoding
        ###
        # log transformation
        X_, _ = num_to_log(X_, to_log)
        # var selection
        X_ = X_[feature_selected]
        # nulls imputation
        X_.fillna(0, inplace=True)
        # standarization
        # re-order the features for standarization
        X_ = X_[std_scaler.feature_names_in_]
        X_[std_scaler.feature_names_in_] = std_scaler.transform(X_[std_scaler.feature_names_in_])
        # minmax scaler
        X_ = X_[minmax_scaler.feature_names_in_]
        X_[minmax_scaler.feature_names_in_] = minmax_scaler.transform(X_[minmax_scaler.feature_names_in_])
        # prediction
        # re-order the features for prediction
        X_ = X_[model_features]
        proba = model.predict_proba(X_)
        # response
        X['proba'] = proba[:, 1]
        print('# Result #')
        print(X[['proba']])
        print('------------------------')
        print('------------------------')
        if proba is not None:
            response_data = X.to_dict()
        else:
            response_data = {'error': 'No prediction generated'}

        return jsonify(response_data)
    
    except Exception as e:
        print(e)
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)