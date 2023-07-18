import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from utils import molecule_from_smiles, graphs_from_smiles, MPNNDataset
from sklearn.metrics import mean_squared_error

# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), initializer="zeros", name="bias",
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features


class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])
        for i in range(self.steps):
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated

class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, molecule_indicator = inputs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)


class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
    ):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        x, padding_mask = inputs
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return proj_output

class MPNN(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        batch_size=32
        message_units=64
        message_steps=5
        num_attention_heads=16
        dense_units=512
        self.TER1 = TransformerEncoderReadout(num_attention_heads, message_units, dense_units, batch_size)
        self.TER2 = TransformerEncoderReadout(num_attention_heads, message_units, dense_units, batch_size)
        self.MP = MessagePassing(message_units, message_steps)
        self.PP = PartitionPadding(batch_size)

    def call(self, inputs):
        atom_features, bond_features, pair_indices, molecule_indicator = inputs
        features = self.MP([atom_features, bond_features, pair_indices])
        features = self.PP([features, molecule_indicator])
        padding_mask = tf.reduce_any(tf.not_equal(features, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        features = self.TER1([features, padding_mask])
        features = self.TER2([features, padding_mask])
        return features

class FLAM_AE:
    def __init__(self, atom_dim=35, bond_dim=12, sol_dim=74, lr=1e-3):
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.sol_dim = sol_dim
        self.lr = lr
        self.his = None
        self.MPNN = MPNN()
        self.model = self.get_predictor()
        
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
        
    def show_his(self):
        if not self.his:
            print('not trained yet')
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.his.history["loss"], label="train loss")
        plt.plot(self.his.history["val_loss"], label="valid loss")
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("LOSS", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
    
    def train(self, train_dataset, valid_dataset, epoch=30):
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.02)
        self.his = self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epoch,
            callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)],
        )
    
    def test(self, test_df, a_name='absorption/nm', e_name='emission/nm'):
        x_test = graphs_from_smiles(test_df.smiles, test_df.solvent)
        y_true = np.dstack([test_df[a_name], test_df[e_name]])[0]
        test_dataset = MPNNDataset(x_test, (y_true))
        y_pred = self.model.predict(test_dataset)
        eval_res = self.model.evaluate(test_dataset)
        print(eval_res)
        legends = [f"{a_name}: t/p = {y_true[i][0]:.1f} / {y_pred[i][0]:.1f}" for i in range(len(y_true))]
        for j in range(y_pred.shape[1]):
            err = np.abs(y_true[:,j] - y_pred[:,j])
            mse = mean_squared_error(y_true[:,j], y_pred[:,j])
            rmse = np.sqrt(mse)
            print(f'{a_name if j == 0 else e_name}:')
            print(f"MAE : {np.mean(err):.4f}")
            print(f"MSE : {mse:.4f}")
            print(f"RMSE : {rmse:.4f}")
            plt.hist(err, range=(min(err),max(err)),bins=10, rwidth=0.8)
            plt.show()
            plt.scatter(list(list(zip(*y_true))[j]), list(list(zip(*y_pred))[j]), s=1)
            plt.show()
            
    def evaluate(self, test_df, p_name='plqy', e_name='e/m-1cm-1', splits=['sol', 'ran'], tag='split'):
        y_pred = self.pred(test_df)
        test_df['a_pred'] = y_pred[:,0]
        test_df['e_pred'] = y_pred[:,1]
        test_df['a_err'] = np.abs(test_df['absorption/nm'] - test_df['a_pred'])
        test_df['e_err'] = np.abs(test_df['emission/nm'] - test_df['e_pred'])
        for split in splits:
            a_mse = mean_squared_error(test_df[test_df[tag].str.find(split) == 0]['absorption/nm'], 
                                       test_df[test_df[tag].str.find(split) == 0]['a_pred'])
            a_rmse = np.sqrt(a_mse)
            print(f"{split} Absorption MAE:", np.mean(test_df[test_df[tag].str.find(split)==0].a_err))
            print(f"{split} Absorption MSE:", a_mse)
            print(f"{split} Absorption RMSE:", a_rmse)
            print()
            e_mse = mean_squared_error(test_df[test_df[tag].str.find(split) == 0]['emission/nm'], 
                                       test_df[test_df[tag].str.find(split) == 0]['e_pred'])
            e_rmse = np.sqrt(e_mse)
            print(f"{split} Emission MAE:", np.mean(test_df[test_df[tag].str.find(split)==0].e_err))
            print(f"{split} Emission MSE:", e_mse)
            print(f"{split} Emission RMSE:", e_rmse)
            print()
            
    def pred(self, test_df):
        x_test = graphs_from_smiles(test_df.smiles, test_df.solvent)
        test_dataset = MPNNDataset(x_test, np.zeros((len(test_df), 2)))
        y_pred = self.model.predict(test_dataset)
        return y_pred
    
    def get_predictor(self):
        atom_features = layers.Input((self.atom_dim), dtype="float32", name="atom_features")
        bond_features = layers.Input((self.bond_dim), dtype="float32", name="bond_features")
        pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
        molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
        gfeature = self.MPNN([atom_features, bond_features, pair_indices, molecule_indicator])
        solvent = layers.Input((1), dtype="float32", name="solvent")
        sfeature = layers.Embedding(self.sol_dim, 16)(solvent)
        sfeature = layers.Dense(64)(sfeature)
        features = layers.Attention()([sfeature, gfeature])
        features = layers.GlobalAveragePooling1D()(features)
        gfeature = layers.GlobalAveragePooling1D()(gfeature)
        x = layers.Dense(256)(features)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64)(x)
        x = layers.LayerNormalization()(gfeature+x)
        x = layers.Dropout(0.02)(x)
        x = layers.Dense(64)(x)
        ae = layers.Dense(2, name='AE')(x)
        model = keras.Model(
            inputs=[atom_features, bond_features, pair_indices, molecule_indicator, solvent],
            outputs=[ae],
        )
        model.compile(
            metrics=['mae'],
            loss=keras.losses.MeanAbsoluteError(reduction="auto", name="mean_absolute_error"),
            optimizer=keras.optimizers.SGD(learning_rate=self.lr),
        )
        return model
    
class FLAM_PE:
    def __init__(self, atom_dim=35, bond_dim=12, sol_dim=74, lr=5e-3):
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.sol_dim = sol_dim
        self.lr = lr
        self.his = None
        self.MPNN = MPNN()
        self.model = self.get_predictor()
        
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
        
    def show_his(self):
        if not self.his:
            print('not trained yet')
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.his.history["loss"], label="train loss")
        plt.plot(self.his.history["val_loss"], label="valid loss")
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("LOSS", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
    
    def train(self, train_dataset, valid_dataset, epoch=30):
        def scheduler(epoch, lr):
            if epoch < 50:
                return lr
            else:
                return lr * tf.math.exp(-0.02)
        self.his = self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epoch,
            callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)],
        )
    
    def test(self, test_df, p_name='plqy', e_name='e/m-1cm-1'):
        x_test = graphs_from_smiles(test_df.smiles, test_df.solvent)
        y_true = np.dstack([test_df[p_name], test_df[e_name]])[0]
        test_dataset = MPNNDataset(x_test, (y_true))
        y_pred = self.model.predict(test_dataset)
        y_true = y_true / 100
        y_pred = y_pred / 100
        eval_res = self.model.evaluate(test_dataset)
        print(eval_res)
        legends = [f"{p_name}: t/p = {y_true[i][0]:.1f} / {y_pred[i][0]:.1f}" for i in range(len(y_true))]
        for j in range(y_pred.shape[1]):
            err = np.abs(y_true[:,j] - y_pred[:,j])
            mse = mean_squared_error(y_true[:,j], y_pred[:,j])
            rmse = np.sqrt(mse)
            print(f'{p_name if j == 0 else e_name}:')
            print(f"MAE : {np.mean(err):.4f}")
            print(f"MSE : {mse:.4f}")
            print(f"RMSE : {rmse:.4f}")
            plt.hist(err, range=(min(err),max(err)),bins=10, rwidth=0.8)
            plt.show()
            plt.scatter(list(list(zip(*y_true))[j]), list(list(zip(*y_pred))[j]), s=1)
            plt.show()
            
    def evaluate(self, test_df, p_name='plqy', e_name='e/m-1cm-1', splits=['sol', 'ran'], tag='split'):
        y_true = np.dstack([test_df[p_name], test_df[e_name]])[0]
        y_pred = self.pred(test_df)
        y_true = y_true / 100
        y_pred = y_pred / 100
        test_df['p_pred'] = y_pred[:,0]
        test_df['e_pred'] = y_pred[:,1]
        test_df['p_err'] = np.abs(y_true[:,0] - test_df['p_pred'])
        test_df['e_err'] = np.abs(y_true[:,1] - test_df['e_pred'])
        for split in splits:
            y_true = np.dstack([test_df[test_df[tag].str.find(split) == 0][p_name], test_df[test_df[tag].str.find(split) == 0][e_name]])[0]
            y_true = y_true/100
            p_mae=np.mean(test_df[test_df[tag].str.find(split) == 0]['p_err'])
            p_mse = mean_squared_error(y_true[:,0], test_df[test_df[tag].str.find(split) == 0]['p_pred'])
            p_rmse = np.sqrt(p_mse)
            print(f"{split} PLQY MAE:{p_mae:.4f}")
            print(f"{split} PLQY MSE: {p_mse:.4f}")
            print(f"{split} PLQY RMSE: {p_rmse:.4f}")
            print()

            e_mae=np.mean(test_df[test_df[tag].str.find(split) == 0]['e_err'])
            e_mse = mean_squared_error(y_true[:,1], test_df[test_df[tag].str.find(split) == 0]['e_pred'])
            e_rmse = np.sqrt(e_mse)
            print(f"{split} Epsilon MAE: {e_mae:.4f}")
            print(f"{split} Epsilon MSE: {e_mse:.4f}")
            print(f"{split} Epsilon RMSE: {e_rmse:.4f}")
            print()
        
    def pred(self, test_df):
        x_test = graphs_from_smiles(test_df.smiles, test_df.solvent)
        test_dataset = MPNNDataset(x_test, np.zeros((len(test_df), 2)))
        y_pred = self.model.predict(test_dataset)
        return y_pred
    
    def get_predictor(self):
        atom_features = layers.Input((self.atom_dim), dtype="float32", name="atom_features")
        bond_features = layers.Input((self.bond_dim), dtype="float32", name="bond_features")
        pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
        molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
        gfeature = self.MPNN([atom_features, bond_features, pair_indices, molecule_indicator])
        solvent = layers.Input((1), dtype="float32", name="solvent")
        sfeature = layers.Embedding(self.sol_dim, 16)(solvent)
        sfeature = layers.Dense(64)(sfeature)
        features = layers.Attention()([sfeature, gfeature])
        features = layers.GlobalAveragePooling1D()(features)
        gfeature = layers.GlobalAveragePooling1D()(gfeature)
        gfeature = layers.Dense(64)(gfeature)
        x = layers.Dense(256)(features)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64)(x)
        x = layers.LayerNormalization()(gfeature+x)
        x = layers.Dropout(0.05)(x)
        x = layers.Dense(64)(x)
        ae = layers.Dense(2, name='AE')(x)
        model = keras.Model(
            inputs=[atom_features, bond_features, pair_indices, molecule_indicator, solvent],
            outputs=[ae],
        )
        model.compile(
            loss=keras.losses.MeanAbsoluteError(reduction="auto", name="mean_absolute_error"),
            optimizer=keras.optimizers.SGD(learning_rate=self.lr),
        )
        return model

class FLAM_con_AE:
    def __init__(self, atom_dim=35, bond_dim=12, sol_dim=74, lr=1e-3):
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.sol_dim = sol_dim
        self.lr = lr
        self.his = None
        self.MPNN = MPNN()
        self.model = self.get_predictor()
        
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
        
    def show_his(self):
        if not self.his:
            print('not trained yet')
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.his.history["loss"], label="train loss")
        plt.plot(self.his.history["val_loss"], label="valid loss")
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("LOSS", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
    
    def train(self, train_dataset, valid_dataset, epoch=30):
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.02)
        self.his = self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epoch,
            callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)],
        )
    
    def test(self, test_df, a_name='absorption/nm', e_name='emission/nm'):
        x_test = graphs_from_smiles(test_df.smiles, test_df.solvent)
        y_true = np.dstack([test_df[a_name], test_df[e_name]])[0]
        test_dataset = MPNNDataset(x_test, (y_true))
        y_pred = self.model.predict(test_dataset)
        eval_res = self.model.evaluate(test_dataset)
        print(eval_res)
        legends = [f"{a_name}: t/p = {y_true[i][0]:.1f} / {y_pred[i][0]:.1f}" for i in range(len(y_true))]
        for j in range(y_pred.shape[1]):
            err = np.abs(y_true[:,j] - y_pred[:,j])
            mse = mean_squared_error(y_true[:,j], y_pred[:,j])
            rmse = np.sqrt(mse)
            print(f'{a_name if j == 0 else e_name}:')
            print(f"MAE : {np.mean(err):.4f}")
            print(f"MSE : {mse:.4f}")
            print(f"RMSE : {rmse:.4f}")
            plt.hist(err, range=(min(err),max(err)),bins=10, rwidth=0.8)
            plt.show()
            plt.scatter(list(list(zip(*y_true))[j]), list(list(zip(*y_pred))[j]), s=1)
            plt.show()
            
    def evaluate(self, test_df, p_name='plqy', e_name='e/m-1cm-1', splits=['sol', 'ran'], tag='split'):
        y_pred = self.pred(test_df)
        test_df['a_pred'] = y_pred[:,0]
        test_df['e_pred'] = y_pred[:,1]
        test_df['a_err'] = np.abs(test_df['absorption/nm'] - test_df['a_pred'])
        test_df['e_err'] = np.abs(test_df['emission/nm'] - test_df['e_pred'])
        for split in splits:
            a_mse = mean_squared_error(test_df[test_df[tag].str.find(split) == 0]['absorption/nm'], 
                                       test_df[test_df[tag].str.find(split) == 0]['a_pred'])
            a_rmse = np.sqrt(a_mse)
            print(f"{split} Absorption MAE:", np.mean(test_df[test_df[tag].str.find(split)==0].a_err))
            print(f"{split} Absorption MSE:", a_mse)
            print(f"{split} Absorption RMSE:", a_rmse)
            print()
            e_mse = mean_squared_error(test_df[test_df[tag].str.find(split) == 0]['emission/nm'], 
                                       test_df[test_df[tag].str.find(split) == 0]['e_pred'])
            e_rmse = np.sqrt(e_mse)
            print(f"{split} Emission MAE:", np.mean(test_df[test_df[tag].str.find(split)==0].e_err))
            print(f"{split} Emission MSE:", e_mse)
            print(f"{split} Emission RMSE:", e_rmse)
            print()
        
    def pred(self, test_df):
        x_test = graphs_from_smiles(test_df.smiles, test_df.solvent)
        test_dataset = MPNNDataset(x_test, np.zeros((len(test_df), 2)))
        y_pred = self.model.predict(test_dataset)
        return y_pred
    
    def get_predictor(self):
        atom_features = layers.Input((self.atom_dim), dtype="float32", name="atom_features")
        bond_features = layers.Input((self.bond_dim), dtype="float32", name="bond_features")
        pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
        molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
        gfeature = self.MPNN([atom_features, bond_features, pair_indices, molecule_indicator])
        solvent = layers.Input((1), dtype="float32", name="solvent")
        sfeature = layers.Embedding(self.sol_dim, 16)(solvent)
        sfeature = layers.Dense(64)(sfeature)
        sfeature = layers.GlobalAveragePooling1D()(sfeature)
        gfeature = layers.GlobalAveragePooling1D()(gfeature)
        features = layers.Concatenate()([sfeature, gfeature])
        x = layers.Dense(512)(features)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64)(x)
        ae = layers.Dense(2, name='AE')(x)
        model = keras.Model(
            inputs=[atom_features, bond_features, pair_indices, molecule_indicator, solvent],
            outputs=[ae],
        )
        model.compile(
            loss=keras.losses.MeanAbsoluteError(),
            metrics=['mae'],
            optimizer=keras.optimizers.SGD(learning_rate=self.lr),
        )
        return model

class FLAM_con_PE:
    def __init__(self, atom_dim=35, bond_dim=12, sol_dim=74, lr=1e-3):
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.sol_dim = sol_dim
        self.lr = lr
        self.his = None
        self.MPNN = MPNN()
        self.model = self.get_predictor()
        
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
        
    def show_his(self):
        if not self.his:
            print('not trained yet')
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.his.history["loss"], label="train loss")
        plt.plot(self.his.history["val_loss"], label="valid loss")
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("LOSS", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
    
    def train(self, train_dataset, valid_dataset, epoch=30):
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.02)
        self.his = self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epoch,
            callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)],
        )
    
    def test(self, test_df, p_name='plqy', e_name='e/m-1cm-1'):
        x_test = graphs_from_smiles(test_df.smiles, test_df.solvent)
        y_true = np.dstack([test_df[p_name], test_df[e_name]])[0]
        test_dataset = MPNNDataset(x_test, (y_true))
        y_pred = self.model.predict(test_dataset)
        y_true = y_true / 100
        y_pred = y_pred / 100
        eval_res = self.model.evaluate(test_dataset)
        print(eval_res)
        legends = [f"{p_name}: t/p = {y_true[i][0]:.1f} / {y_pred[i][0]:.1f}" for i in range(len(y_true))]
        for j in range(y_pred.shape[1]):
            err = np.abs(y_true[:,j] - y_pred[:,j])
            mse = mean_squared_error(y_true[:,j], y_pred[:,j])
            rmse = np.sqrt(mse)
            print(f'{p_name if j == 0 else e_name}:')
            print(f"MAE : {np.mean(err):.4f}")
            print(f"MSE : {mse:.4f}")
            print(f"RMSE : {rmse:.4f}")
            plt.hist(err, range=(min(err),max(err)),bins=10, rwidth=0.8)
            plt.show()
            plt.scatter(list(list(zip(*y_true))[j]), list(list(zip(*y_pred))[j]), s=1)
            plt.show()
            
    def evaluate(self, test_df, p_name='plqy', e_name='e/m-1cm-1', splits=['sol', 'ran'], tag='split'):
        y_true = np.dstack([test_df[p_name], test_df[e_name]])[0]
        y_pred = self.pred(test_df)
        y_true = y_true / 100
        y_pred = y_pred / 100
        test_df['p_pred'] = y_pred[:,0]
        test_df['e_pred'] = y_pred[:,1]
        test_df['p_err'] = np.abs(y_true[:,0] - test_df['p_pred'])
        test_df['e_err'] = np.abs(y_true[:,1] - test_df['e_pred'])
        for split in splits:
            y_true = np.dstack([test_df[test_df[tag].str.find(split) == 0][p_name], test_df[test_df[tag].str.find(split) == 0][e_name]])[0]
            y_true = y_true/100
            p_mae=np.mean(test_df[test_df[tag].str.find(split) == 0]['p_err'])
            p_mse = mean_squared_error(y_true[:,0], test_df[test_df[tag].str.find(split) == 0]['p_pred'])
            p_rmse = np.sqrt(p_mse)
            print(f"{split} PLQY MAE:{p_mae:.4f}")
            print(f"{split} PLQY MSE: {p_mse:.4f}")
            print(f"{split} PLQY RMSE: {p_rmse:.4f}")
            print()

            e_mae=np.mean(test_df[test_df[tag].str.find(split) == 0]['e_err'])
            e_mse = mean_squared_error(y_true[:,1], test_df[test_df[tag].str.find(split) == 0]['e_pred'])
            e_rmse = np.sqrt(e_mse)
            print(f"{split} Epsilon MAE: {e_mae:.4f}")
            print(f"{split} Epsilon MSE: {e_mse:.4f}")
            print(f"{split} Epsilon RMSE: {e_rmse:.4f}")
            print()
        
    def pred(self, test_df):
        x_test = graphs_from_smiles(test_df.smiles, test_df.solvent)
        test_dataset = MPNNDataset(x_test, np.zeros((len(test_df), 2)))
        y_pred = self.model.predict(test_dataset)
        return y_pred
    
    def get_predictor(self):
        atom_features = layers.Input((self.atom_dim), dtype="float32", name="atom_features")
        bond_features = layers.Input((self.bond_dim), dtype="float32", name="bond_features")
        pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
        molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
        gfeature = self.MPNN([atom_features, bond_features, pair_indices, molecule_indicator])
        solvent = layers.Input((1), dtype="float32", name="solvent")
        sfeature = layers.Embedding(self.sol_dim, 16)(solvent)
        sfeature = layers.Dense(64)(sfeature)
        sfeature = layers.GlobalAveragePooling1D()(sfeature)
        gfeature = layers.GlobalAveragePooling1D()(gfeature)
        features = layers.Concatenate()([sfeature, gfeature])
        x = layers.Dense(512)(features)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64)(x)
        ae = layers.Dense(2, name='PE')(x)
        model = keras.Model(
            inputs=[atom_features, bond_features, pair_indices, molecule_indicator, solvent],
            outputs=[ae],
        )
        model.compile(
            loss=keras.losses.MeanAbsoluteError(),
            optimizer=keras.optimizers.SGD(learning_rate=self.lr),
        )
        return model
