# FLAM: FLuorescence property prediction with Attention-driven Model

This is a repository of the article "[Predicting Photophysical Properties and Offering Molecular Design Insights with An Attention-Driven Model](#)".

The FluoDB dataset could be found in [figshare](https://figshare.com/s/1a8b8300a9e63f390e6d).

## Overview

FLAM based on an attention-driven model can effectively extract interacting information between the fluorophore and the solvent to improve optical property prediction accuracy. We constructed the largest open-source fluorophore database to date (FluoDB), containing 55,169 fluorophore-solvent pairs and related photophysical data. Benchmarking with different test sets demonstrates that FLAM can quickly and accurately predict the fluorophores' optical properties in different solvents. Both the reliability and potential of FLAM have been verified via molecular and atomic interpretability analysis, insights from which can further revolutionize the design of new fluorophores.

## System Requirements

### Hardware and OS requirements

This code has been tested on Linux(Ubuntu 22.04) and Windows(10,11), which only requires a standard computer with enough RAM to support the in-memory operations.

There is **no need** for a GPU to be involved in the computation, and the acceleration of the computation by the GPU is not significant.

### Python Dependencies
FLAM mainly depends on these packages.
```
pandas==1.1.5
numba==0.55.2
numpy==1.21.6
matplotlib==3.4.2
rdkit==2022.3.3
scikit-learn==1.0.2
tensorflow==2.9.1
```

### Set up the environment

We strongly recommand to install packages in [conda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) environment.

After install conda, create an environmnet
```
conda create -n flam python==3.8
```
activate the environment and install the packages

```
conda activate flam
cd /dir/to/this/repo # !cd the dir you git clone or download this repository
pip install -r requirements.txt
```

and then you could use FLAM by python shell or jupyter.

## FLAM Usage

We provide four models in utils.py, please follow the sample code to import the models. The following code takes the first model as an example, the other models just need to replace the model name.
- FLAM_AE
- FLAM_PE
- FLAM_con_AE
- FLAM_con_PE

The input data should be a pandas.DataFrame such as:
smiles     | solvent
-------- | -----
SMILES of Molecule  | SMILES of solvent

The solvent SMILES will be converted to a feature, so note that the model only supports 73 solvents (could be found in utils.py).

If you want to try on other solvents, a new model should be trained to learn the solvent features.

### Dataset

These example datasets could be found in data/train_data:
```
- data/train_data/ae_train.csv   #  dataset for AE model training
- data/train_data/ae_valid.csv   #  dataset for AE model validation
- data/train_data/ae_test.csv   #  dataset for AE model test
- data/train_data/pe_train.csv   #  dataset for PE model training
- data/train_data/pe_valid.csv   #  dataset for PE model validation
- data/train_data/pe_test.csv   #  dataset for PE model test
```


### Run Model

```
# Load model
from model import SAFLU_AE
model = SAFLU_AE()
model.load('model/FLAM/FLAM_AE.h5')

# Load data
import pandas as pd
# load data
test_df = pd.read_csv('data/train_data/ae_test.csv')
# load model
model.load('model/FLAM/FLAM_AE.h5')
y_pred = model.pred(test_df)
absorption_pred = y_pred[:,0]
emission_pred = y_pred[:,1]
```


### Train and evaluation FLAM
```
# Train example
import pandas as pd
from utils import get_ae_dataset
# load data
train_df = pd.read_csv('data/train_data/ae_train.csv')
train_data = get_ae_dataset(train_df)
valid_df = pd.read_csv('data/train_data/ae_valid.csv')
valid_data = get_ae_dataset(valid_df)
# train model
model.train(train_dataset, valid_dataset, epoch=200)
# show train plot
model.show_his()
# save model
model.save('model/temp.h5')
```

```
# Test example
import pandas as pd
test_df = pd.read_csv('data/train_data/ae_test.csv')
model.test(test_df)
```

```
# Evaluation example
import pandas as pd
test_df = pd.read_csv('data/train_data/ae_test.csv')
model.evaluate(test_df, tag='split', splits=['sol', 'ran'])
```


### Output of FLAM

The AE model directly gives predictions of absorption and emission in nm.

The PE model plqy and epsilon need to be converted using the following equations.

$$
PLQY = \frac{PLQY_{pred}}{100}\\
$$

$$
Epsilon(M^{-1}cm^{-1}) = 10^{\frac{Epsilon_{pred}}{100}}\\
$$


## Note

We'd love to hear from you. Please let us know if you have any comments or suggestions, or if you have questions about the code or the procedure. Also, if you'd like to contribute to this project please let us know.

## License 
This project is covered under the MIT License.