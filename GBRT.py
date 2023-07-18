from padelpy import padeldescriptor
import pandas as pd
import numpy as np
import joblib
import warnings
import os
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

models = {
    'e_pred':'model/GBRT/Emsision_Model_for_Predict.m',
    'a_pred':'model/GBRT/Absorption_Model_for_Predict.m',
    'p_pred':'model/GBRT/PLQY_Model_for_Classification.m'
}

fp_file = {
    "CDK":"model/GBRT/Fingerprinter.xml",
    "Ext":"model/GBRT/ExtendedFingerprinter.xml",
    "Estat":"model/GBRT/EStateFingerprinter.xml",
    "SF":"model/GBRT/SubstructureFingerprinter.xml",
    "SFC":"model/GBRT/SubstructureFingerprintCount.xml",
}
# solvent change
solvent_dict = {
    'O': '63.1,0.681,0.997,1.062,0.025',
    'ClC(Cl)(Cl)Cl': '32.4,0.768,0.000,0.000,0.044',
    'S=C=S': '32.8,1.000,0.000,0.000,0.104',
    'ClC(Cl)Cl': '39.1,0.783,0.614,0.047,0.071',
    'ClCCl': '40.7,0.761,0.769,0.040,0.178',
    'N=CO': '55.9,0.814,1.006,0.549,0.414',
    'C[N+](=O)[O-]': '46.3,0.710,0.954,0.078,0.236',
    'CO': '55.4,0.608,0.904,0.605,0.545',
    'N#CCCl': '46.4,0.763,1.024,0.445,0.184',
    'CC#N': '45.6,0.645,0.974,0.044,0.286',
    'CC[N+](=O)[O-]': '43.7,0.706,0.902,0.000,0.234',
    'CCO': '51.8,0.633,0.783,0.400,0.658',
    'CS(C)=O': '45.1,0.830,1.000,0.072,0.647',
    'CCC#N': '43.6,0.668,0.888,0.030,0.365',
    'CC(C)=O': '42.3,0.651,0.907,0.000,0.475',
    'C=CCO': '51.9,0.705,0.839,0.415,0.585',
    'COC(C)=O': '38.7,0.645,0.637,0.000,0.527',
    'COC(=O)OC': '38.8,0.653,0.531,0.064,0.433',
    'CN(C)C=O': '43.2,0.759,0.977,0.031,0.613',
    'CCCO': '50.5,0.658,0.748,0.367,0.782',
    'CC(C)O': '48.4,0.633,0.808,0.283,0.830',
    'COP(=O)(OC)OC': '43.5,0.707,0.909,0.000,0.522',
    'CC1COC(=O)O1': '46.0,0.746,0.942,0.106,0.341',
    'CCCC#N': '42.5,0.689,0.864,0.000,0.384',
    'CCC(C)=O': '41.3,0.669,0.872,0.000,0.520',
    'C1CCOC1': '37.5,0.714,0.634,0.000,0.591',
    'CCOC(C)=O': '38.0,0.656,0.603,0.000,0.542',
    'CCCCCl': '36.9,0.693,0.529,0.000,0.138',
    'CCCCN': '37.6,0.690,0.296,0.000,0.944',
    'CCCCO': '49.7,0.674,0.655,0.341,0.809',
    'CCC(C)O': '47.1,0.656,0.706,0.221,0.888',
    'CCOCC': '34.5,0.617,0.385,0.000,0.562',
    'c1ccncc1': '40.5,0.842,0.761,0.033,0.581',
    'CN1CCCC1=O': '42.2,0.812,0.959,0.024,0.613',
    'CC1CCCO1': '36.5,0.700,0.768,0.000,0.584',
    'CCC(=O)CC': '39.9,0.692,0.785,0.000,0.557',
    'C1CCNCC1': '36.3,0.754,0.365,0.000,0.933',
    'CCC(C)C': '30.8,0.581,0.000,0.000,0.053',
    'CCCCC': '30.9,0.593,0.000,0.000,0.073',
    'CCCCOC': '34.8,0.647,0.345,0.000,0.505',
    'COC(C)(C)C': '34.5,0.622,0.422,0.000,0.567',
    'CCCCCO': '49.3,0.687,0.587,0.319,0.860',
    'CN(C)C(=O)N(C)C': '40.9,0.778,0.878,0.000,0.624',
    'Fc1c(F)c(F)c(F)c(F)c1F': '34.2,0.623,0.252,0.000,0.119',
    'Clc1ccccc1Cl': '38.0,0.869,0.676,0.033,0.144',
    'Brc1ccccc1': '36.6,0.875,0.497,0.000,0.192',
    'Clc1ccccc1': '36.8,0.833,0.537,0.000,0.182',
    'Fc1ccccc1': '37.0,0.761,0.511,0.000,0.113',
    'O=[N+]([O-])c1ccccc1': '41.2,0.891,0.873,0.056,0.240',
    'c1ccccc1': '34.3,0.793,0.270,0.000,0.124',
    'Nc1ccccc1': '44.4,0.924,0.956,0.132,0.264',
    'CC(=O)N(C)C': '42.9,0.763,0.987,0.028,0.650',
    'O=C1CCCCC1': '40.1,0.766,0.745,0.000,0.482',
    'CN1CCCCC1': '32.6,0.708,0.116,0.000,0.836',
    'CCN(CC)C(C)=O': '41.4,0.748,0.918,0.000,0.660',
    'CCCCCC': '30.9,0.616,0.000,0.000,0.056',
    'CC(C)OC(C)C': '34.0,0.625,0.324,0.000,0.657',
    'CCCCCCO': '48.9,0.698,0.552,0.315,0.879',
    'CCN(CC)CC': '32.1,0.660,0.108,0.000,0.885',
    'FC(F)(F)c1ccccc1': '38.7,0.694,0.663,0.014,0.073',
    'N#Cc1ccccc1': '41.5,0.851,0.852,0.047,0.281',
    'Cc1ccccc1': '33.9,0.782,0.284,0.000,0.128',
    'OCc1ccccc1': '50.7,0.861,0.788,0.409,0.461',
    'CCCCCCC': '30.9,0.635,0.000,0.000,0.083',
    'CC(=O)c1ccccc1': '40.6,0.848,0.808,0.044,0.365',
    'Cc1ccc(C)cc1': '33.1,0.778,0.175,0.000,0.160',
    'OCCc1ccccc1': '49.7,0.849,0.793,0.376,0.523',
    'CCCCCCCC': '31.0,0.650,0.000,0.000,0.079',
    'CCCCOCCCC': '33.0,0.672,0.175,0.000,0.637',
    'CCCCCCCCO': '48.3,0.713,0.454,0.299,0.923',
    'Cc1cc(C)cc(C)c1': '32.9,0.775,0.155,0.000,0.190',
    'CCCCCCCCC': '30.8,0.660,0.000,0.000,0.053',
    'C1CCC2CCCCC2C1': '31.1,0.753,0.000,0.000,0.056',
    'CCCCCCCCCC': '30.8,0.669,0.000,0.000,0.066',
    'CCCCCCCCCCO': '48.0,0.722,0.383,0.259,0.912',
    'CC12CCC(CC1)C(C)(C)O2': '34.0,0.736,0.343,0.000,0.737',
    'Cc1cccc2ccccc12': '35.3,0.908,0.510,0.000,0.156',
    'CCCCCCCCCCCC': '31.0,0.683,0.000,0.000,0.086',
    'CCCCN(CCCC)CCCC': '32.1,0.689,0.060,0.000,0.854',
    'c1ccc(COCc2ccccc2)cc1': '36.4,0.877,0.509,0.000,0.330',
    'C1CCCCC1': '30.9,0.683,0,0,0.073',
    'CCN(CC)CCe': '32.1,0.66,0.108,0,0.885',
    'C1COCCO': '36,0.737,0.312,0,0.444',
    'CC(C)=CCO[P](O)(=O)O[P](O)(O)=O': '42.9,0.763,0.987,0.028,0.65',
    'C[S](C)=O': '45.1,0.83,1,0.072,0.647',
    'CNC=O': '54.1,0.759,0.977,0.031,0.613'
}

def get_sol(smi):
    if smi in solvent_dict:
        return solvent_dict[smi]
    else:
        return None
    
def pred(test_df, models=models):
    try:
        test_df['plqy'] = test_df['plqy'] / 100
        pred_df = test_df.copy()
        test_df['a_pred'] = test_df['absorption/nm'].values
        test_df['e_pred'] = test_df['emission/nm'].values
        test_df['p_pred'] = test_df['plqy'].values
        test_df['pred'] = 0
        pred_df['sol'] = list(map(get_sol, pred_df.solvent))
        pred_df = pred_df[pred_df.sol.str.len() >0]
        test_df.loc[pred_df.index, ['pred']] = 1
        smi_dict = dict([(k,v) for v,k in enumerate(set(pred_df.smiles.values))])
        with open('data/GBRT/test.smi', 'w') as file:
            for smi, idx in smi_dict.items():
                file.write(f'{smi} {idx}\n')
        for key, file in fp_file.items():
            print('get ',key)
            padeldescriptor(mol_dir='data/GBRT/test.smi', 
                            d_file= 'data/GBRT/fp.csv', #'Substructure.csv'
                            #descriptortypes='SubstructureFingerprint.xml', 
                            descriptortypes= file,
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=10,
                            removesalt=True,
                            log=True,
                            fingerprints=True)
            with open('data/GBRT/fp.csv', 'r')  as file:
                fp_list = file.read().strip().split('\n')[1:]
                fp_dict = dict([(v.split('",')[0][1:], v.split('",')[1]) for v in fp_list ])
            pred_df[key] = [ fp_dict[str(smi_dict[smi])] for smi in pred_df.smiles]
        pred_df = pred_df[pred_df.SFC.str.len()>1220]
        for name, model in models.items():
            clf = joblib.load(model)
            X_pre = np.array([ f'{r.sol},{r.CDK},{r.Ext},{r.Estat},{r.SF},{r.SFC}'.split(',') for i,r in pred_df.iterrows()])
            y_pre = clf.predict(X_pre)
            pred_df[name] = y_pre
        test_df.loc[pred_df.index, ['a_pred', 'e_pred', 'p_pred']] = pred_df.loc[:,['a_pred', 'e_pred', 'p_pred']]
        test_df['a_err'] = np.abs(test_df['absorption/nm'] - test_df['a_pred'])
        test_df['e_err'] = np.abs(test_df['emission/nm'] - test_df['e_pred'])
        test_df['p_err'] = np.abs(test_df['plqy'] - test_df['p_pred'])
        return True
    except Exception as r:
        print(r)
        return False
    return False

def df2inp(test_df):
    try:
        df = test_df.copy()
        df['sol'] = list(map(get_sol, df.solvent))
        df = df[df.sol.str.len() >0]
        smi_dict = dict([(k,v) for v,k in enumerate(set(df.smiles.values))])
        with open('data/GBRT/test.smi', 'w') as file:
            for smi, idx in smi_dict.items():
                file.write(f'{smi} {idx}\n')
        for key, file in fp_file.items():
            print('get ',key)
            padeldescriptor(mol_dir='data/GBRT/test.smi', 
                            d_file= 'data/GBRT/fp.csv', #'Substructure.csv'
                            #descriptortypes='SubstructureFingerprint.xml', 
                            descriptortypes= file,
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=10,
                            removesalt=True,
                            log=True,
                            fingerprints=True)
            with open('data/GBRT/fp.csv', 'r')  as file:
                fp_list = file.read().strip().split('\n')[1:]
                fp_dict = dict([(v.split('",')[0][1:], v.split('",')[1]) for v in fp_list ])
            df[key] = [ fp_dict[str(smi_dict[smi])] for smi in df.smiles]
        return df
    except Exception as r:
        print(r)
        return None
    return None

def train(df, path='model/GBRT', target='AE'):
    if target.find('AE') == -1 and target.find('PE') == -1:
        print('Error 1.')
        return False
    print('getting Data')
    train_df = df2inp(df)
    train_df = train_df[train_df.SFC.str.len()>1220]
    X_train = np.array([ f'{r.sol},{r.CDK},{r.Ext},{r.Estat},{r.SF},{r.SFC}'.split(',') for i,r in train_df.iterrows()]).astype('float')
    y_train_a = train_df['absorption/nm']
    y_train_e = train_df['emission/nm']
    y_train_p = train_df['plqy']
    if target.find('AE') > -1:
        e_clf = GradientBoostingRegressor(learning_rate=0.05,
                                    max_depth=31,
                                    max_features=300,
                                    min_samples_leaf=20,
                                    n_estimators=1000)
        a_clf = GradientBoostingRegressor(learning_rate=0.05,
                                    max_depth=31,
                                    max_features=300,
                                    min_samples_leaf=20,
                                    n_estimators=1000)

        print('Training ABS model')
        a_clf.fit(X_train, y_train_a)
        print('Training EMI model')
        e_clf.fit(X_train, y_train_e)
        joblib.dump(a_clf, f'{path}/GBRT_ABS_train.m')
        joblib.dump(e_clf, f'{path}/GBRT_EMI_train.m')
        print(f'done, saved to {path}/GBRT_ABS_train.m and {path}/GBRT_EMI_train.m')
    elif target.find('PE') > -1:
        p_clf = lgb.LGBMRegressor(n_estimators=600,
                        learning_rate=0.1,
                        max_depth=70,
                        num_leaves=45,
                        objective='binary')
        print('Training PLQY model')
        p_clf.fit(X_train, y_train_p)
        joblib.dump(p_clf, f'{path}/GBRT_PLQY_train.m')
    return True