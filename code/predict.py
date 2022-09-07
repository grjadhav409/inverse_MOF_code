import vaemof
from vaemof import experiments
from vaemof import utils
from vaemof.vocabs import SELFIESVocab, MOFVocab, PropVocab
from vaemof import modules
from vaemof import training
from vaemof.model import VAEMOF
from vaemof import configs
from vaemof.utils import header_str
vaemof.experiments.plot_settings()

import os
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import rdkit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from itertools import product
from more_itertools import chunked
from collections import OrderedDict
from rdkit.Chem import PandasTools
from IPython.display import SVG, display
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams['font.size'] = 8

print(f'rdkit : {rdkit.__version__}')
print(f'torch : {torch.__version__}')
print(f'cuda? {torch.cuda.is_available()}')
tqdm.pandas()
utils.disable_rdkit_log()

WORK_DIR = 'model/'
hparams_file = os.path.join(WORK_DIR,'config.json')
hparams = configs.AttributeDict.from_jsonfile(hparams_file)
hparams['train_device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
utils.set_seed(hparams['train_seed'])
device = torch.device(hparams['train_device'])
configs.print_config(hparams)
model = VAEMOF.load(hparams)

smiles_column = 'branch_smiles'
testtrain_column = 'train/test'
df = experiments.get_generator_df(csv_file=hparams['files_data'],
                                  smiles_column=smiles_column,
                                  use_duplicates=hparams['vae_duplicate_smiles'],
                                  testing=False)
ids2mofs, mof2ids, mof_columns = experiments.get_mofdict(
    df, hparams['mof_encoding'])
df.head()

prop_df = experiments.get_prop_df(csv_file=hparams['files_prop'],
                                  targets=hparams['y_labels'],
                                  mof2ids=mof2ids,
                                  testing=False,
                                  smiles_column=smiles_column,
                                  compute_scscore=True)
prop_df.head()

train_index = np.array(df[df[testtrain_column] == 1].index.tolist())
test_index = np.array(df[df[testtrain_column] == 0].index.tolist())
prop_train_index = np.array(
    prop_df[prop_df[testtrain_column] == 1].index.tolist())
prop_test_index = np.array(
    prop_df[prop_df[testtrain_column] == 0].index.tolist())
print(f'Test  sizes: {len(test_index):7d} and {len(prop_test_index):7d}')

train_mof = model.df_to_tuples(df.loc[train_index], smiles_column)
test_mof = model.df_to_tuples(df.loc[test_index], smiles_column)
prop_train = model.df_to_tuples(prop_df.loc[prop_train_index], smiles_column)
prop_test = model.df_to_tuples(prop_df.loc[prop_test_index], smiles_column)
train_data = train_mof + prop_train
test_data = test_mof + prop_test

print(header_str('R^2 scores and MAEs'))
src_data = prop_test
batch_size = 256
n_loops = int(np.ceil(len(src_data) / batch_size))
y_pred, y_true = [], []
for chunk in  tqdm(chunked(src_data,batch_size),total=n_loops, desc='Pred'):
    batch = model.tuples_to_tensors(chunk)
    z = model.inputs_to_z(batch['x'],batch['mof'])
    outs = model.z_to_outputs(z)
    y_true.extend(model.vocab_y.inverse_transform(batch['y']))
    y_pred.extend(outs['y'])  
y_pred = np.stack(y_pred)
y_true = np.stack(y_true)
experiments.regression_statistics(y_true, y_pred, hparams['y_labels'])

print(header_str('Prior Check'))
n = 10000
gen_df = experiments.sample_model(model, n)
print('valid smiles: {} out of {} ({}%)'.format(gen_df['valid'].sum(),n,gen_df['valid'].sum()/n*100.0))
gen_df.head()

print(header_str('Posterior check'))
tries=10
sub_sample = 1000
src_data = train_data
src_data = random.sample(src_data,min(len(src_data),sub_sample))
valid_smiles=[]
recon_smiles=[]
n = len(src_data)
results=[]
mof_results=[]
for t in tqdm(src_data):
    batch = [t]*tries
    batch = model.tuples_to_tensors(batch)
    z = model.inputs_to_z(batch['x'],batch['mof'])
    outs = model.z_to_outputs(z)
    true_smiles = vaemof.vocabs.isosmiles(model.vocab.ids_to_string(t[0]))
    true_mof = model.vocab_mof.ids_to_mof(t[1])
    smiles_list = outs['x']
    mof_list = outs['mof']
    acc_mof = any([ all(i==true_mof) for i in np.array(mof_list)])
    valid_smiles = [si for si in set(smiles_list) if vaemof.vocabs.valid_smiles(si)]
    valid_smiles = [vaemof.vocabs.isosmiles(si) for si in valid_smiles]
    same_smiles = [si for si in valid_smiles if si==true_smiles]
    results.append([true_smiles, smiles_list[0], len(same_smiles)>0,len(valid_smiles)>0])
    mof_results.append(acc_mof)       
post_df = pd.DataFrame(results,columns=['smiles','recon_smiles','same','valid'])
print('valid: {} out of {} ({:.2f}%)'.format(post_df['valid'].sum(),n,post_df['valid'].sum()/n*100.0))
print('same : {} out of {} ({:.2f}%)'.format(post_df['same'].sum(),n,post_df['same'].sum()/n*100.0))
print('MOF recon acc: {}'.format(float(sum(mof_results))/float(len(mof_results))))

print(header_str('PCA analysis of the latent space'))
sub_sample = 100000
src_data = prop_train
src_data = random.sample(src_data,min(len(src_data),sub_sample))
n = len(src_data)
batch_size=64
n_loops = int(np.ceil(n/batch_size))
z=[]
y=[]
for chunk in tqdm(chunked(src_data,batch_size),total=n_loops, desc='Generating predictions'):
    batch = model.tuples_to_tensors(chunk)
    y.extend(model.vocab_y.inverse_transform(batch['y']))
    z_tensor = model.inputs_to_z(batch['x'],batch['mof'])
    z.extend(z_tensor.cpu().numpy())     
z = np.stack(z)
z_pca = PCA(2).fit_transform(z)
z_y_df = pd.DataFrame(y,columns=hparams.y_labels)
z_y_df['x']=z_pca[:,0]
z_y_df['y']=z_pca[:,1]
print(z.shape)
z_y_df.head()
cmap='viridis_r'
for index,col in enumerate(hparams.y_labels):
    print(header_str(col))
    plt.figure(figsize=(8,6.5))
    scatter = plt.scatter(z_y_df['x'],z_y_df['y'],c=z_y_df[col],s=2.5,lw=1.5,cmap=cmap,edgecolor=None)
    plt.colorbar(scatter)
    plt.title(col)
    plt.savefig('../results/%s-%s.png'%(col,cmap),dpi=300,transparent=True)
    
print(header_str('Sampling the neighboring space of known MOF'))
nu_1104 = [('sym_5_on_12','sym_12_mc_11','ftw','[Lr]c1ccc(C#Cc2ccc([Lr])cc2)cc1',0.0)]
nu_df = pd.DataFrame(nu_1104,columns=['organic_core','metal_node','topology','branch_smiles','dist'])
nu_tuple = model.df_to_tuples(nu_df, smiles_column)
nu_tensor = model.tuples_to_tensors(nu_tuple)
nu_z = model.inputs_to_z(nu_tensor['x'],nu_tensor['mof'])
i = 0
tries = 100
noise_max = 20
while i < tries:
    sample = []
    noise = random.uniform(0,noise_max)
    sample_z = experiments.perturb_z(nu_z, noise)
    try:
        outs = model.z_to_outputs(sample_z)
        smiles = outs['x'][0]
        if vaemof.vocabs.valid_smiles(smiles):
            sample_df = pd.DataFrame(nu_1104,columns=['organic_core','metal_node','topology','branch_smiles','dist'])
            dist = np.linalg.norm(nu_z-sample_z)
            sample_df['organic_core'] = outs['mof'][0][1]
            sample_df['metal_node'] = outs['mof'][0][0]
            sample_df['topology'] = outs['mof'][0][2]
            sample_df['branch_smiles'] = outs['x'][0]
            sample_df['dist'] = dist
            nu_df = nu_df.append(sample_df, ignore_index=True)
            nu_df.shape
            i = i + 1
    except:
        pass
nu_df = nu_df.sort_values(by=['dist'])
nu_df.head()
nu_df.to_csv('../results/sampling_neighbors.csv')

print(header_str('Interpolating between known MOF structures'))
nu_1104 = [('sym_5_on_12','sym_12_mc_11','ftw','[Lr]c1ccc(C#Cc2ccc([Lr])cc2)cc1',0.0)]
nu_1000 = [('sym_5_on_11','sym_8_mc_9','csq','[Lr]c1ccc([Lr])cc1',0.0)]
nu_df_1104 = pd.DataFrame(nu_1104,columns=['organic_core','metal_node','topology','branch_smiles','dist'])
nu_df_1000 = pd.DataFrame(nu_1000,columns=['organic_core','metal_node','topology','branch_smiles','dist'])
nu_tuple_1104 = model.df_to_tuples(nu_df_1104, smiles_column)
nu_tensor_1104 = model.tuples_to_tensors(nu_tuple_1104)
nu_z_1104 = model.inputs_to_z(nu_tensor_1104['x'],nu_tensor_1104['mof'])
nu_tuple_1000 = model.df_to_tuples(nu_df_1000, smiles_column)
nu_tensor_1000 = model.tuples_to_tensors(nu_tuple_1000)
nu_z_1000 = model.inputs_to_z(nu_tensor_1000['x'],nu_tensor_1000['mof'])
full_dist = np.linalg.norm(nu_z_1104-nu_z_1000)
nu_df_1000['dist'] = full_dist
interpolate_num = 5
nu_z = [(nu_z_1000-nu_z_1104)/interpolate_num*i+nu_z_1104 for i in range(1,interpolate_num)]
tries = 5
for z in nu_z:
    i = 0
    while i < tries:
        try:
            outs = model.z_to_outputs(z)
            smiles = outs['x']
            if vaemof.vocabs.valid_smiles(smiles[0]):
                interpolate_df = pd.DataFrame(nu_1104,columns=['organic_core','metal_node','topology','branch_smiles','dist'])
                dist = np.linalg.norm(nu_z_1104-z)
                interpolate_df['organic_core'] = outs['mof'][0][1]
                interpolate_df['metal_node'] = outs['mof'][0][0]
                interpolate_df['topology'] = outs['mof'][0][2]
                interpolate_df['branch_smiles'] = outs['x'][0]
                interpolate_df['dist'] = dist
                nu_df_1104 = nu_df_1104.append(interpolate_df, ignore_index=True)
                i = i + 1
        except:
            pass
full_df = nu_df_1104.append(nu_df_1000, ignore_index=True)
full_df.head()
full_df.to_csv('../results/interpolation_between_MOFs.csv')