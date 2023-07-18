# load data
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from multiprocessing import Pool
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys, AllChem
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

def smi2fp(data):
    smi, sol = data
    mol = Chem.MolFromSmiles(smi)
    maccs = MACCSkeys.GenMACCSKeys(mol).ToList()
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=33).ToList()
    smol = Chem.MolFromSmiles(sol)
    smaccs = MACCSkeys.GenMACCSKeys(smol).ToList()
    smorgan = AllChem.GetMorganFingerprintAsBitVect(smol, 2, nBits=33).ToList()
    return maccs+morgan+smaccs+smorgan

def smiles2inp(smiles, solvents, thd=10):
    worker = Pool(thd)
    fp_list = worker.map(smi2fp, zip(smiles, solvents))
    worker.close()
    worker.join()
    return np.array(fp_list)

def get_dataset(df, pro, batch_size=32, shuffle=False):
    if pro == "AE":
        e_name = 'emission/nm'
        a_name = 'absorption/nm'
        Y = np.dstack([df[a_name], df[e_name]])[0]
    elif pro == "PE":
        e_name = 'e/m-1cm-1'
        p_name = 'plqy'
        Y = np.dstack([df[p_name], df[e_name]])[0]
    else:
        print("key Error!")
        return None
    X = smiles2inp(df.smiles, df.solvent)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).prefetch(-1)

def FCNN():
    model = keras.models.Sequential()
    l5 = keras.layers.Dense(512, activation='relu')
    l6 = keras.layers.Dropout(rate=0.2)
    l7 = keras.layers.Dense(128, activation='relu')
    l8 = keras.layers.Dense(30, activation='relu')
    l9 = keras.layers.Dense(2)
    layer = [l5, l6, l7, l8, l9]
    for i in range(len(layer)):
        model.add(layer[i])
    adam = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])
    return model

def get_sch():
    def scheduler(epoch, lr):
        if epoch > 0 and epoch % 500 == 0:
            return lr * 0.1
        else:
            return lr
    return tf.keras.callbacks.LearningRateScheduler(scheduler)

def show_his(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="valid loss")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("LOSS", fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    
def test_model(model, test_df, pro):
    x_test = smiles2inp(test_df.smiles, test_df.solvent)
    y_pred = model.predict(x_test)
    
    if pro == "AE":
        e_name = 'emission/nm'
        a_name = 'absorption/nm'
        y_true = np.dstack([test_df[a_name], test_df[e_name]])[0]
        for j in range(y_pred.shape[1]):
            err = np.abs(y_true[:,j] - y_pred[:,j])
            mse = mean_squared_error(y_true[:,j], y_pred[:,j])
            rmse = np.sqrt(mse)
            print(f'{a_name if j == 0 else e_name}')
            print(f"MAE: {np.mean(err):.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            plt.hist(err, range=(min(err),max(err)),bins=10, rwidth=0.8)
            plt.show()
            plt.scatter(list(list(zip(*y_true))[j]), list(list(zip(*y_pred))[j]), s=1)
            plt.show()
    elif pro == "PE":
        e_name = 'e/m-1cm-1'
        p_name = 'plqy'
        y_true = np.dstack([test_df[p_name], test_df[e_name]])[0] / 100
        y_pred /= 100
        for j in range(y_pred.shape[1]):
            err = np.abs(y_true[:,j] - y_pred[:,j])
            mse = mean_squared_error(y_true[:,j], y_pred[:,j])
            rmse = np.sqrt(mse)
            print(f'{p_name if j == 0 else e_name}')
            print(f"MAE: {np.mean(err):.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            plt.hist(err, range=(min(err),max(err)),bins=10, rwidth=0.8)
            plt.show()
            plt.scatter(list(list(zip(*y_true))[j]), list(list(zip(*y_pred))[j]), s=1)
            plt.show()
    else:
        print("key Error!")
        return None
    
def evaluate_model(model, test_df, pro, tag='split', splits=['sol', 'ran']):
    x_test = smiles2inp(test_df.smiles, test_df.solvent)
    y_pred = model.predict(x_test)
    if pro == "AE":
        test_df['a_pred'] = y_pred[:,0]
        test_df['e_pred'] = y_pred[:,1]
        test_df['a_err'] = np.abs(test_df['absorption/nm'] - test_df['a_pred'])
        test_df['e_err'] = np.abs(test_df['emission/nm'] - test_df['e_pred'])
        
        for split in splits:
            a_mae = np.mean(test_df[test_df[tag].str.find(split)==0].a_err)
            a_mse = mean_squared_error(test_df[test_df[tag].str.find(split) == 0]['absorption/nm'], 
                                       test_df[test_df[tag].str.find(split) == 0]['a_pred'])
            a_rmse = np.sqrt(a_mse)
            print(f"{split} Absorption MAE: {a_mae:.4f}")
            print(f"{split} Absorption MSE: {a_mse:.4f}")
            print(f"{split} Absorption RMSE: {a_rmse:.4f}")
            print()
            e_mae = np.mean(test_df[test_df[tag].str.find(split)==0].e_err)
            e_mse = mean_squared_error(test_df[test_df[tag].str.find(split) == 0]['emission/nm'], 
                                       test_df[test_df[tag].str.find(split) == 0]['e_pred'])
            e_rmse = np.sqrt(e_mse)
            print(f"{split} Emission MAE: {e_mae:.4f}")
            print(f"{split} Emission MSE: {e_mse:.4f}")
            print(f"{split} Emission RMSE: {e_rmse:.4f}")
            print()
    elif pro == "PE":
        e_name = 'e/m-1cm-1'
        p_name = 'plqy'
        y_true = np.dstack([test_df[p_name], test_df[e_name]])[0] /100
        y_pred /= 100
        test_df['p_pred'] = y_pred[:,0]
        test_df['e_pred'] = y_pred[:,1]
        test_df['p_err'] = np.abs(y_true[:,0] - test_df['p_pred'])
        test_df['e_err'] = np.abs(y_true[:,1] - test_df['e_pred'])
        for split in splits:
            y_true = np.dstack([test_df[test_df[tag].str.find(split) == 0][p_name], test_df[test_df[tag].str.find(split) == 0][e_name]])[0]
            y_true = y_true / 100
            p_mae = np.mean(test_df[test_df[tag].str.find(split) == 0]['p_err'])
            p_mse = mean_squared_error(y_true[:,0], test_df[test_df[tag].str.find(split) == 0]['p_pred'])
            p_rmse = np.sqrt(p_mse)
            print(f"{split} PLQY MAE: {p_mae:.4f}")
            print(f"{split} PLQY MSE: {p_mse:.4f}")
            print(f"{split} PLQY RMSE: {p_rmse:.4f}")
            print()

            e_mae = np.mean(test_df[test_df[tag].str.find(split) == 0]['e_err'])
            e_mse = mean_squared_error(y_true[:,1], test_df[test_df[tag].str.find(split) == 0]['e_pred'])
            e_rmse = np.sqrt(e_mse)
            print(f"{split} Epsilon MAE: {e_mae:.4f}")
            print(f"{split} Epsilon MSE: {e_mse:.4f}")
            print(f"{split} Epsilon RMSE: {e_rmse:.4f}")
            print()
    else:
        print("key Error!")
        return None