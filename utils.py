import os
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.metrics import mean_squared_error

# atom_featurizer.dim: 35
# bond_featurizer.dim: 12
# len(solvent_dict) 74

RDLogger.DisableLog("rdApp.*")

atom_list = ["B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Si", "Se"]

solvent_dict = {
    'unknow': 0,
    'O': 1, 'CO': 2, 'Cl': 3, 'CC#N': 4, 'N=CO': 5, 'CCO': 6, 'CCC#N': 7, 'CC(C)=O': 8, 'CN=CO': 9,
    'CC(=O)O': 10, 'CC(C)O': 11, 'CCCO': 12, 'OCCO': 13, 'CCCC#N': 14, 'CCC(C)=O': 15, 'C1CCOC1': 16,
    'CCCCC': 17, 'CCC(C)C': 18, 'CN(C)C=O': 19, 'COC(C)=O': 20, 'CCCCO': 21, 'CCOCC': 22, 'CC(C)(C)O': 23,
    'CC(C)CO': 24, 'CC(O)CO': 25, 'PBS': 26, 'CS(C)=O': 27, 'c1ccccc1': 28, 'c1ccncc1': 29,
    'ClCCl': 30, 'C1CCCCC1': 31, 'CC1CCCO1': 32, 'CCCCCC': 33, 'CCC(C)CC': 34, 'CC(=O)N(C)C': 35,
    'CCOC(C)=O': 36, 'C1COCCO1': 37, 'CCC(C)(C)O': 38, 'CCCCCO': 39, 'COCCOC': 40, 'OCC(O)CO': 41,
    'Cc1ccccc1': 42, 'O=S(=O)(O)O': 43, 'ClCCCl': 44, 'O=C1CCCCC1': 45, 'CC1CCCCC1': 46, 'CN1CCCC1=O': 47,
    'OCC(F)(F)F': 48, 'OC(F)(F)CF': 49, 'CC(=O)C(C)(C)C': 50, 'CCCCCCC': 51, 'CCN(CC)CC': 52,
    'CC1COC(=O)O1': 53, 'CCCCCCO': 54, 'CC(C)OC(C)C': 55, 'N#Cc1ccccc1': 56, 'OCCOCCO': 57,
    'Cc1ccc(C)cc1': 58, 'Clc1ccccc1': 59, 'O=C(O)C(F)(F)F': 60, 'CC(C)CC(C)(C)C': 61, 'CCCCOC(C)=O': 62,
    'CCCCCCCO': 63, 'ClC(Cl)Cl': 64, 'ClC=C(Cl)Cl': 65, 'CCCCCCCCO': 66, 'CCCCOCCCC': 67, 'Clc1ccccc1Cl': 68,
    'ClC(Cl)(Cl)Cl': 69, 'CCCCCCCCCCO': 70, 'CCCCCCCCCCCO': 71, 'CCCCCCCCCCCCCC': 72, 'CCCCCCCCCCCCCCCC': 73
}

def get_sol_idx(sol):
    if sol in solvent_dict:
        return solvent_dict[sol]
    else:
        return 0

def remove_ion(smis):
    remover = SaltRemover(defnData="[Zn,Ni,Cl,I,Na,Br,Co,Pd,Fe,Sb,Pt,Au,F,Li,K,Ru,Cd,Cs,Al,Mg,Ag,Cu,Sn,Mn,Bi,Pb,Tl,Hg,Ca]")
    ions = ['[O-][Cl+3]([O-])([O-])[O-]', 'F[P-](F)(F)(F)(F)[F+]',
     'O=C(O)C(F)(F)F', '[O-][Cl+3]([O-])([O-])O', 'F[P-](F)(F)(F)(F)F',
     'c1cc[cH-]c1', 'O=S(=O)([O-])C(F)(F)F',
     'F[B-](F)(F)F', 'CCCC[N+](CCCC)(CCCC)CCCC',
     'O=C([O-])C(F)(F)F', 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F',
     'CCP(CC)CC', 'O=S(=O)(O)C(F)(F)F',
     'O', 'CCN(CC)CC', '[H+]', '[OH-]', 'Cc1ccc(S(=O)(=O)[O-])cc1',
     '[O][Cl+3]([O-])([O-])[O-]', 'FC1=C=C(F)[C-](F)C(F)=C1F',
     'FB(F)F', '[N-]=C=S', 'N', 'I[I-]I', 'CN(C)C=O',
     'CC[O-]', 'ClCCl', '[N-]=O', 'ClC(Cl)Cl', 'P',
     '[Yb+3]', '[Lu+3]', '[Y+3]', '[In+3]']
    smis1 = []
    for smi in smis:
        if smi.find('.')>0:
            mol = Chem.MolFromSmiles(smi)
            mol = remover.StripMol(mol,dontRemoveEverything=True)
            smi = Chem.MolToSmiles(mol)
            if smi.find('.')>0:
                ss = set(smi.split('.'))
                for ion in ions:
                    if len(ss) == 1:
                        break
                    if ion in ss:
                        ss.remove(ion)
                if len(ss) == 1:
                    smi = ss.pop()
                else:
                    smi = ''
        smis1.append(smi)
    return smis1

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output

class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        
    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()
    
    def chiral(self, atom):
        return atom.GetChiralTag().name.lower()
    
    def ring(self, atom):
        if atom.GetIsAromatic():
            return 2
        if atom.IsInRing():
            return 1
        return 0


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()
    
    def stereo(self, bond):
        return bond.GetStereo().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()
    
def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule

atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": set(atom_list),
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
        "chiral": {"chi_unspecified", "chi_tetrahedral_cw", "chi_tetrahedral_ccw"},
        "ring": {0, 1, 2},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "stereo": {"stereonone", "stereoz", "stereoe", "stereocis", "stereotrans"},
        "conjugated": {True, False},
    }
)

def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


def graphs_from_smiles(smiles_list, solvent_list):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []
    solvent_list = list(map(get_sol_idx, solvent_list))

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    # Convert lists to ragged tensors for tf.data.Dataset later on
    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
        tf.ragged.constant(solvent_list, dtype=tf.int32),
    )

def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    atom_features, bond_features, pair_indices, solvents = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator, solvents), y_batch

def MPNNDataset(X, Y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)

def get_graphs(molecules):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for molecule in molecules:
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)
        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    # Convert lists to ragged tensors for tf.data.Dataset later on
    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )

def get_ae_dataset(df):
    e_name = 'emission/nm'
    a_name = 'absorption/nm'
    x = graphs_from_smiles(df.smiles, df.solvent)
    y = (np.dstack([df[a_name], df[e_name]])[0])
    dataset = MPNNDataset(x, y)
    return dataset


def get_pe_dataset(df):
    e_name = 'e/m-1cm-1'
    p_name = 'plqy'
    x = graphs_from_smiles(df.smiles, df.solvent)
    y = (np.dstack([df[p_name], df[e_name]])[0])
    dataset = MPNNDataset(x, y, shuffle=True)
    return dataset

def evaluate(model, test_df, pro):
    if pro == "AE":
        y_pred = model.pred(test_df)
        test_df['a_pred'] = y_pred[:,0]
        test_df['e_pred'] = y_pred[:,1]
        test_df['a_err'] = np.abs(test_df['absorption/nm'] - test_df['a_pred'])
        test_df['e_err'] = np.abs(test_df['emission/nm'] - test_df['e_pred'])
        splits = ['sol', 'sc', 'ran']
        for split in splits:
            a_mse = mean_squared_error(test_df[test_df['split'].str.find(split) == 0]['absorption/nm'], 
                                       test_df[test_df['split'].str.find(split) == 0]['a_pred'])
            a_rmse = np.sqrt(a_mse)
            print(f"{split} Absorption MAE:", np.mean(test_df[test_df['split'].str.find(split)==0].a_err))
            print(f"{split} Absorption MSE:", a_mse)
            print(f"{split} Absorption RMSE:", a_rmse)
            print()
            e_mse = mean_squared_error(test_df[test_df['split'].str.find(split) == 0]['emission/nm'], 
                                       test_df[test_df['split'].str.find(split) == 0]['e_pred'])
            e_rmse = np.sqrt(e_mse)
            print(f"{split} Emission MAE:", np.mean(test_df[test_df['split'].str.find(split)==0].e_err))
            print(f"{split} Emission MSE:", e_mse)
            print(f"{split} Emission RMSE:", e_rmse)
            print()
    elif pro == "PE":
        y_true = np.dstack([test_df[p_name], test_df[e_name]])[0]
        y_pred = model.pred(test_df)
        y_true = y_true / 100
        y_pred = y_pred / 100
        test_df['p_pred'] = y_pred[:,0]
        test_df['e_pred'] = y_pred[:,1]
        test_df['p_err'] = np.abs(y_true[:,0] - test_df['p_pred'])
        test_df['e_err'] = np.abs(y_true[:,1] - test_df['e_pred'])
        splits = ['sol', 'ran']
        for split in splits:
            y_true = np.dstack([test_df[test_df['split'].str.find(split) == 0][p_name], test_df[test_df['split'].str.find(split) == 0][e_name]])[0]

            y_true = y_true/100
            p_mae=np.mean(test_df[test_df['split'].str.find(split) == 0]['p_err'])
            p_mse = mean_squared_error(y_true[:,0], test_df[test_df['split'].str.find(split) == 0]['p_pred'])
            p_rmse = np.sqrt(p_mse)
            print(f"{split} PLQY MAE:{p_mae:.4f}")
            print(f"{split} PLQY MSE: {p_mse:.4f}")
            print(f"{split} PLQY RMSE: {p_rmse:.4f}")
            print()
            
            e_mae=np.mean(test_df[test_df['split'].str.find(split) == 0]['e_err'])
            e_mse = mean_squared_error(y_true[:,1], test_df[test_df['split'].str.find(split) == 0]['e_pred'])
            e_rmse = np.sqrt(e_mse)
            print(f"{split} Epsilon MAE: {e_mae:.4f}")
            print(f"{split} Epsilon MSE: {e_mse:.4f}")
            print(f"{split} Epsilon RMSE: {e_rmse:.4f}")
            print()
    else:
        print("Error Type!")