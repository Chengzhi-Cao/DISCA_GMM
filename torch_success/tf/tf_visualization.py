import os, h5py, math
from tqdm import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,7"
import sys
sys.path.append('/code/DISCA_GMM')
from disca.DISCA_gmmu_cavi_llh_scanning_new import *
from disca_dataset.DISCA_visualization import *
from hist_filtering.filtering import *
from config import *



import sys,argparse
import pdb
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import pandas as pd
from sklearn.metrics import homogeneity_completeness_v_measure
from utils.plots import *
from utils.metrics import *
import h5py
from disca_dataset.DISCA_visualization import *
import pickle, mrcfile
import scipy.ndimage as SN
from PIL import Image
from collections import Counter
from disca.DISCA_gmmu_cavi_llh_scanning_new import *
from GMMU.gmmu_cavi_stable_new import CAVI_GMMU as GMM
from config import *
from tqdm import *
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
color=['#6A539D','#E6D7B2','#99CCCC','#FFCCCC','#DB7093','#D8BFD8','#6495ED',\
'#1E90FF','#7FFFAA','#FFFF00','#FFA07A','#FF1493','#B0C4DE','#00CED1','#FFDAB9','#DA70D6']
color=np.array(color)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"




parser = argparse.ArgumentParser()
#dataset options
parser.add_argument("--filtered_data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/data.h5')
parser.add_argument("--data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/original')

#stored path
parser.add_argument("--saving_path", type=str, default='/data/zfr888/EMD_4603/Results_tf0815')
parser.add_argument("--algorithm_name", type=str, default='gmmu_cavi_llh_hist')
parser.add_argument("--filtered_particle_saving_path", type=str, default='/data/zfr888/EMD_4603/Results_tf0815/filtered_particle')

parser.add_argument("--image_size", type=int, default=24)
parser.add_argument("--input_size", type=int, default=24)
parser.add_argument("--candidateKs", default=[7,8,9,10,11])

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--factor", default=2)
parser.add_argument("--lr", default=0.01)
parser.add_argument("--momentum", default=0.9)
parser.add_argument("--loss_function",type=int, default=2)
parser.add_argument("--optimizer", default='adam')
parser.add_argument("--hidden_num",type=int, default=32)

parser.add_argument("--reg_covar", type=int, default=0.000001)
parser.add_argument("--u_filter_rate", type=str, default=0.025)
parser.add_argument("--alpha",type=float, default=1.0)

parser.add_argument("--scanning_bottom",type=int, default=100)
parser.add_argument("--scanning_upper",type=int, default=20000)
parser.add_argument("--num_labels",type=int, default=10)

parser.add_argument("--scanning_num", type=int, default=1)
parser.add_argument("--DIVIDE", type=int,default=10)
parser.add_argument("--M", type=int,default=2)
parser.add_argument("--sub_epoch", type=int,default=2)
parser.add_argument("--subtomo_num_test",type=int, default=8000)
# parser.add_argument("--NN_visual",type=bool, default=False,action='store_false')
# parser.add_argument("--GMMU_visual",type=bool, default=False,action='store_false')
parser.add_argument("--NN_visual", action='store_false')
parser.add_argument("--GMMU_visual", action='store_false')
args = parser.parse_args()


label_path = args.saving_path+'/results'
model_path = args.saving_path+'/models'
label_names = ['labels_'+args.algorithm_name]
figures_path = args.saving_path+'/figures/'+label_names[0]
infos = pickle_load(args.data_path+'/info.pickle')
v = read_mrc_numpy_vol(args.data_path+'/emd_4603.map')
algorithms = ['classificationmodel']+args.algorithm_name.split('_')
v = (v - np.mean(v))/np.std(v)
vs = []
s = 32//2

#trained model
model_names = []
for model_name in os.listdir(model_path):
    algo = model_name.split('_')[:len(algorithms)]
    if algo == algorithms :
        model_names.append(os.path.splitext(model_name)[0])

#extracted particles
h5f = h5py.File(args.filtered_data_path,'r')                                                        
x_train = h5f['dataset_1'][16265-args.subtomo_num_test:] # only 'dataset_1'                              
h5f.close()
print(x_train.shape)

infos = infos[16265-args.subtomo_num_test:]
infonp = np.array(infos)
print(set(infonp[:,0]),infonp.shape)


print('args.NN_visual=',args.NN_visual)
print('args.GMMU_visual=',args.GMMU_visual)
if args.NN_visual == True:
    #visualization using classification NN
    print('#visualization using classification NN')
    for model_name in model_names:
        print('model_name=',model_name)
        classmodelpath = os.path.join(model_path,model_name)+'.h5'
        yopopath = os.path.join(model_path,'deltamodel_'+'_'.join(model_name.split('_')[1:]))+'.h5'
        #gpath = os.path.join(model_path,'gmmumodel_'+'_'.join(model_name.split('_')[1:]))+'.h5'
        figure_path = os.path.join(figures_path,'_'.join(model_name.split('_')[1:]))
        if not os.path.isdir(figure_path):
            os.makedirs(figure_path)
        
        yopo = tf.keras.models.load_model(yopopath, custom_objects={'CosineSimilarity': CosineSimilarity})
        classmodel = tf.keras.models.load_model(classmodelpath, custom_objects={'CosineSimilarity': CosineSimilarity,\
                                                                'SNN': SNN,\
                                                                'NSNN': NSNN})

        features = yopo.predict([x_train, x_train, x_train])[0]

        print('features[0]=',features[0])
        print('features[1]=',features[1])
        print('features[2]=',features[2])
        
        print('x_train=',x_train[0])
        labels_soft = classmodel.predict([features, features, features])[0]
        labels = np.array([np.argmax(labels_soft[q, :]) for q in range(len(labels_soft))])


        _unique_label = np.unique(labels)
        print('_unique_label=',_unique_label)
        print('## Cluster sizes:', [np.sum(labels == k) for  k in set(labels)])  # 返回每个标签的数量

        for i in tqdm(range(np.max(labels) + 1)):
            #print(model_name, i)
            locs = np.array(infos)[labels == i]
            v_i = np.zeros_like(v)
            for j in locs:
                if j[0] == 'emd_4603.map': #emd_4603_deconv_corrected.mrc / emd_4603.map
                    v_i[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s] = \
                    v[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s]
            save_png(cub_img(v_i[:,:,::15])['im'], os.path.join(figure_path, 'NN'+str(i) + model_name + '.png'))
            



if args.GMMU_visual == True:
    #visualization using GMMU
    print('#visualization using GMMU')
    for model_name in model_names:
        print('model_name=',model_name)
        classmodelpath = os.path.join(model_path,model_name)+'.h5'
        yopopath = os.path.join(model_path,'deltamodel_'+'_'.join(model_name.split('_')[1:]))+'.h5'
        #gpath = os.path.join(model_path,'gmmumodel_'+'_'.join(model_name.split('_')[1:]))+'.h5'
        figure_path = os.path.join(figures_path,'_'.join(model_name.split('_')[1:]))
        if not os.path.isdir(figure_path):
            os.makedirs(figure_path)
        
        yopo = tf.keras.models.load_model(yopopath, custom_objects={'CosineSimilarity': CosineSimilarity})
        classmodel = tf.keras.models.load_model(classmodelpath, custom_objects={'CosineSimilarity': CosineSimilarity,\
                                                                'SNN': SNN,\
                                                                'NSNN': NSNN})
        features = yopo.predict([x_train, x_train, x_train])[0]
        # you can set replacce args.candidateKs with some K
        labels_temp_proba, labels_temp, K, same_K, features_pca, gmm = \
                statistical_fitting_tf_split_merge(features = np.squeeze(features), \
                                                labels = None, candidateKs = args.candidateKs,\
                                                        K = None, reg_covar = args.reg_covar, it = 0,\
                                                        u_filter_rate=args.u_filter_rate, alpha = args.alpha)
        labels_soft = labels_temp_proba
        labels = labels_temp


        _unique_label = np.unique(labels)
        print('_unique_label=',_unique_label)
        print('## Cluster sizes:', [np.sum(labels == k) for  k in set(labels)])  # 返回每个标签的数量

        for i in tqdm(range(np.max(labels) + 1)):
            locs = np.array(infos)[labels == i]
            v_i = np.zeros_like(v)
            for j in locs:
                if j[0] == 'emd_4603.map': #emd_4603_deconv_corrected.mrc / emd_4603.map
                    v_i[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s] = \
                    v[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s]
            save_png(cub_img(v_i[:,:,::15])['im'], os.path.join(figure_path, 'GMMU'+str(i) + model_name + '.png'))