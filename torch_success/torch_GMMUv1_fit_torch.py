
# V1使用torch写的fitting函数，得到的效果不错
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn import init
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA    
import sys, multiprocessing, importlib, pickle, time
from multiprocessing.pool import Pool  
from tqdm.auto import tqdm
from torchvision.transforms import Normalize
import os, h5py, math
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import class_weight 

import argparse
import faulthandler
faulthandler.enable()
from disca.DISCA_gmmu_cavi_llh_scanning_new import statistical_fitting_tf_split_merge,statistical_fitting,DDBI_uniform,prepare_training_data,prepare_training_datav2
from disca.DISCA_gmmu_cavi_llh_scanning_new_torch import statistical_fitting_torch_split_merge
import pickle, mrcfile
# from utils import con_hist_filtering



def con_hist_filtering(X, scanning_bottom=100, scanning_upper=20000,saving_path = None):
    """
    X: the input data (w,h,l)
    saving_path: saving the path of each labels components
    neighbor_size: the neighbor arrays that are counted
    """
    dic_components_nums = {}
    output_index = []
    for label_i in np.unique(X):
        if label_i == 0:
            #print('remember to delete, label 0!')
            background_index = np.array(np.where(X == label_i)).T
            output_index.append(background_index[np.random.randint(background_index.shape[0], \
                                                size=min(background_index.shape[0], int(0.1*np.prod(X.shape)))),])
            #dic_components_nums[label_i] = components_nums
            continue

        # con_nums doesn't include the background 0
        con_labels, con_nums = skimage.measure.label(np.where(X == label_i, X, 0), return_num=True)
        #print('remember to delete',np.unique(con_labels), con_nums)
        components_nums = []
        for i in range(con_nums):
            components_nums.append(np.sum(con_labels == i+1))
        #
        dic_components_nums[label_i] = components_nums
        #

        qts_bottom = np.min([np.quantile(components_nums, 0.05),scanning_bottom]) # np.quantile(components_nums, 0.25)
        qts_up = np.min([np.quantile(components_nums, 0.85),scanning_upper]) #is the maximum 20000?
        for i in range(con_nums):
            if np.sum(con_labels == i+1) >= qts_bottom and np.sum(con_labels == i+1) <= qts_up:
                output_index.append(np.floor(np.mean(np.array(np.where(con_labels == i+1)).T, axis=0).reshape([1, -1])))
    
    if len(output_index)==0:
        background_index = np.array(np.where(X != 0)).T
        output_index.append(background_index[np.random.randint(background_index.shape[0], \
                                                size=min(background_index.shape[0], 10)),])
    output_index = np.concatenate(output_index)

    if saving_path is not None:
        np.save(saving_path, dic_components_nums)

    return output_index

class Subtomogram_Dataset:
    def __init__(self, train_data, label_one_hot):
        self.train_data = train_data
        self.label_one_hot = label_one_hot

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        features = self.train_data[index]
        labels = self.label_one_hot[index]

        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(labels)

        return features, labels

def align_cluster_index(ref_cluster, map_cluster):
    """                                                                                                                                                                            
    remap cluster index according the the ref_cluster.                                                                                                                                    
    both inputs must have same number of unique cluster index values.                                                                                                                      
    """

    ref_values = np.unique(ref_cluster)
    map_values = np.unique(map_cluster)

    if ref_values.shape[0] != map_values.shape[0]:
        print('error: both inputs must have same number of unique cluster index values.')
        return ()
    cont_mat = contingency_matrix(ref_cluster, map_cluster)

    row_ind, col_ind = linear_sum_assignment(len(ref_cluster) - cont_mat)

    map_cluster_out = map_cluster.copy()

    for i in ref_values:
        map_cluster_out[map_cluster == col_ind[i]] = i

    return map_cluster_out, col_ind



def DDBI(features, labels):
    """                                                                                                                                                                            
    compute the Distortion-based Davies-Bouldin index defined in Equ 1 of the Supporting Information.                                                                                                        
    """

    means_init = np.array([np.mean(features[labels == i], 0) for i in np.unique(labels)])
    precisions_init = np.array(
        [np.linalg.inv(np.cov(features[labels == i].T) + Config.reg_covar * np.eye(features.shape[1])) for i in
         np.unique(labels)])

    T = np.array([np.mean(np.diag(
        (features[labels == i] - means_init[i]).dot(precisions_init[i]).dot((features[labels == i] - means_init[i]).T)))
                  for i in np.unique(labels)])

    D = np.array(
        [np.diag((means_init - means_init[i]).dot(precisions_init[i]).dot((means_init - means_init[i]).T)) for i in
         np.unique(labels)])

    DBI_matrix = np.zeros((len(np.unique(labels)), len(np.unique(labels))))

    for i in range(len(np.unique(labels))):
        for j in range(len(np.unique(labels))):
            if i != j:
                DBI_matrix[i, j] = (T[i] + T[j]) / (D[i, j] + D[j, i])

    DBI = np.mean(np.max(DBI_matrix, 0))

    return DBI

class YOPOFeatureModel(nn.Module):

    def __init__(self):
        super(YOPOFeatureModel, self).__init__()

        self.dropout = nn.Dropout(0.5)
        self.m1 = self.get_block(1, 64)
        self.m2 = self.get_block(64, 80)
        self.m3 = self.get_block(80, 96)
        self.m4 = self.get_block(96, 112)
        self.m5 = self.get_block(112, 128)
        self.m6 = self.get_block(128, 144)
        self.m7 = self.get_block(144, 160)
        self.m8 = self.get_block(160, 176)
        self.m9 = self.get_block(176, 192)
        self.m10 = self.get_block(192, 208)
        # self.m11 = self.get_block(104, 117)
        # self.m12 = self.get_block(117, 140)
        # self.m13 = self.get_block(140, 150)
        self.batchnorm = torch.nn.BatchNorm3d(1360)
        self.linear = nn.Linear(
            in_features=1360,
            out_features=32
        )
       
        self.weight_init(self)
        
    '''
	Initialising the model with blocks of layers.
	'''

    @staticmethod
    def get_block(input_channel_size, output_channel_size):
        return nn.Sequential(
            torch.nn.Conv3d(in_channels=input_channel_size,
                            out_channels=output_channel_size,
                            kernel_size=(3, 3, 3),
                            padding=0,
                            dilation=(1, 1, 1)),  
            torch.nn.BatchNorm3d(output_channel_size),
            # torch.nn.ELU(inplace=True),
            torch.nn.ELU(inplace=False),
        )

    '''
	Initialising weights of the model.
	'''

    @staticmethod
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    '''
	Forward Propagation Pass.
	'''

    def forward(self, input_image):
        output = self.dropout(input_image)
        output = self.m1(output)
        o1 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m2(output)
        o2 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m3(output)
        o3 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m4(output)
        o4 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m5(output)
        o5 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m6(output)
        o6 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m7(output)
        o7 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m8(output)
        o8 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m9(output)
        o9 = F.max_pool3d(output, kernel_size=output.size()[2:])
        output = self.m10(output)
        o10 = F.max_pool3d(output, kernel_size=output.size()[2:])
        # print(output.size())
        """
		output = self.m11(output)
		o11 = F.max_pool3d(output, kernel_size=output.size()[2:])
		output = self.m12(output)
		o12 = F.max_pool3d(output, kernel_size=output.size()[2:])
		output = self.m13(output)
		o13 = F.max_pool3d(output, kernel_size=output.size()[2:])
		"""
        m = torch.cat((o1, o2, o3, o4, o5, o6, o7, o8, o9, o10), dim=1)
        # print(m.size())
        m = self.batchnorm(m)
        m = nn.Flatten()(m)
        m = self.linear(m)
        return m


def statistical_fitting(features, labels, candidateKs, K, reg_covar, i):
    """
    fitting a Gaussian mixture model to the extracted features from YOPO
    given current estimated labels, K, and a number of candidateKs. 

    reg_covar: non-negative regularization added to the diagonal of covariance.     
    i: random_state for initializing the parameters.
    """

    pca = PCA(n_components=16)  
    features_pca = pca.fit_transform(features) 

    labels_K = [] 
    BICs = [] 
                                                                                                                                                            
    for k in candidateKs: 
        if k == K: 
            try:
                weights_init = np.array([np.sum(labels == j)/float(len(labels)) for j in range(k)]) 
                means_init = np.array([np.mean(features_pca[labels == j], 0) for j in range(k)]) 
                precisions_init = np.array([np.linalg.inv(np.cov(features_pca[labels == j].T) + reg_covar * np.eye(features_pca.shape[1])) for j in range(k)]) 
 
                gmm_0 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i,  
                                        weights_init=weights_init, means_init=means_init, precisions_init=precisions_init, init_params = 'random') 
 
                gmm_0.fit(features_pca) 
                labels_k_0 = gmm_0.predict(features_pca)

            except:     
                gmm_0 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i, init_params = 'random') 
                gmm_0.fit(features_pca) 
                labels_k_0 = gmm_0.predict(features_pca) 
                         
         
            gmm_1 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i, init_params = 'random') 
            gmm_1.fit(features_pca) 
            labels_k_1 = gmm_1.predict(features_pca) 
             
            m_select = np.argmin([gmm_0.bic(features_pca), gmm_1.bic(features_pca)]) 
             
            if m_select == 0: 
                labels_K.append(labels_k_0) 
                 
                BICs.append(gmm_0.bic(features_pca)) 
             
            else: 
                labels_K.append(labels_k_1) 
                 
                BICs.append(gmm_1.bic(features_pca)) 
         
        else: 
            gmm = GaussianMixture(n_components=k, covariance_type='full', tol=0.0001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i, init_params = 'random') 
         
            gmm.fit(features_pca) 
            labels_k = gmm.predict(features_pca) 

            labels_K.append(labels_k) 
             
            BICs.append(gmm.bic(features_pca)) 
    
    labels_temp = remove_empty_cluster(labels_K[np.argmin(BICs)])                     
     
    K_temp = len(np.unique(labels_temp)) 
     
    if K_temp == K: 
        same_K = True 
    else: 
        same_K = False 
        K = K_temp     

    print('Estimated K:', K)
    
    return labels_temp, K, same_K, features_pca   



def convergence_check(i, M, labels_temp, labels, done):
    # if i > 0:
    #     if np.sum(labels_temp == labels) / float(len(labels)) > 0.999:
    #         done = True

    i += 1
    if i == M:
        done = True

    labels = labels_temp

    return i, labels, done



def pickle_dump(o, path, protocol=2):
    """                                                                                                                                                                            
    write a pickle file given the object o and the path.                                                                                                                      
    """ 
    with open(path, 'wb') as f:    pickle.dump(o, f, protocol=protocol)



def run_iterator(tasks, worker_num=multiprocessing.cpu_count(), verbose=True):
    """
    parallel multiprocessing for a given task, this is useful for speeding up the data augmentation step.
    """

    if verbose:		print('parallel_multiprocessing()', 'start', time.time())

    worker_num = min(worker_num, multiprocessing.cpu_count())

    for i,t in tasks.items():
        if 'args' not in t:     t['args'] = ()
        if 'kwargs' not in t:     t['kwargs'] = {}
        if 'id' not in t:   t['id'] = i
        assert t['id'] == i

    completed_count = 0 
    if worker_num > 1:

        pool = Pool(processes = worker_num)
        pool_apply = []
        for i,t in tasks.items():
            aa = pool.apply_async(func=call_func, kwds={'t':t})

            pool_apply.append(aa)


        for pa in pool_apply:
            yield pa.get(99999)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()

        pool.close()
        pool.join()
        del pool

    else:

        for i,t in tasks.items():
            yield call_func(t)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()
	
    if verbose:		print('parallel_multiprocessing()', 'end', time.time())


    
run_batch = run_iterator #alias



def call_func(t):

    if 'func' in t:
        assert 'module' not in t
        assert 'method' not in t
        func = t['func']
    else:
        modu = importlib.import_module(t['module'])
        func = getattr(modu, t['method'])

    r = func(*t['args'], **t['kwargs'])
    return {'id':t['id'], 'result':r}



def random_rotation_matrix():
    """
    generate a random 3D rigid rotation matrix.
    """
    m = np.random.random( (3,3) )
    u,s,v = np.linalg.svd(m)

    return u



def rotate3d_zyz(data, Inv_R, center=None, order=2):
    """
    rotate a 3D data using ZYZ convention (phi: z1, the: x, psi: z2).
    """
    # Figure out the rotation center
    if center is None:
        cx = data.shape[0] / 2
        cy = data.shape[1] / 2
        cz = data.shape[2] / 2
    else:
        assert len(center) == 3
        (cx, cy, cz) = center

    
    from scipy import mgrid
    grid = mgrid[-cx:data.shape[0]-cx, -cy:data.shape[1]-cy, -cz:data.shape[2]-cz]
    # temp = grid.reshape((3, np.int(grid.size / 3)))
    temp = grid.reshape((3, int(grid.size / 3)))
    temp = np.dot(Inv_R, temp)
    grid = np.reshape(temp, grid.shape)
    grid[0] += cx
    grid[1] += cy
    grid[2] += cz

    # Interpolation
    from scipy.ndimage import map_coordinates
    d = map_coordinates(data, grid, order=order)

    return d



def data_augmentation(x_train, factor = 2):
    """
    data augmentation given the training subtomogram data.
    if factor = 1, this function will return the unaugmented subtomogram data.
    if factor > 1, this function will return (factor - 1) number of copies of augmented subtomogram data.
    """

    if factor > 1:

        x_train_augmented = []
        
        x_train_augmented.append(x_train)

        for f in range(1, factor):
            ts = {}        
            for i in range(len(x_train)):                       
                t = {}                                                
                t['func'] = rotate3d_zyz                                   
                                                      
                # prepare keyword arguments                                                                                                               
                args_t = {}                                                                                                                               
                # args_t['data'] = x_train[i,:,:,:,0]   
                args_t['data'] = x_train[i,0,:,:,:]                                                                                                                 
                args_t['Inv_R'] = random_rotation_matrix()                                                   
                                                                                                                                                                                                                                           
                t['kwargs'] = args_t                                                  
                ts[i] = t                                                       
                                                                      
            rs = run_batch(ts, worker_num=48)
            # x_train_f = np.expand_dims(np.array([_['result'] for _ in rs]), -1)
            x_train_f = np.expand_dims(np.array([_['result'] for _ in rs]), 1)
            
            x_train_augmented.append(x_train_f)
            
        x_train_augmented = np.concatenate(x_train_augmented)
    
    else:
        x_train_augmented = x_train                        

        x_train[x_train == 0] = np.random.normal(loc=0.0, scale=1.0, size = np.sum(x_train == 0))

    return x_train_augmented



def one_hot(a, num_classes):
    """
    one-hot encoding. 
    """
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])   



def smooth_labels(labels, factor=0.1):
    """
    label smoothing. 
    """
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
 
    return labels           



def remove_empty_cluster(labels):
    """
    if there are no samples in a cluster,
    this function will remove the cluster and make the remaining cluster number compact. 
    """
    labels_unique = np.unique(labels)
    for i in range(len(np.unique(labels))):
        labels[labels == labels_unique[i]] = i

    return labels



# def prepare_training_data(x_train, labels, label_smoothing_factor):
#     """
#     training data preparation given the current training data x_train, labels, and label_smoothing_factor
#     """

#     label_one_hot = one_hot(labels, len(np.unique(labels))) 
     
#     index = np.array(range(x_train.shape[0] * 2)) 

#     np.random.shuffle(index)         
     
#     x_train_augmented = data_augmentation(x_train, 2) 
    
#     x_train_permute = x_train_augmented[index].copy() 

#     label_smoothing_factor *= 0.9 

#     labels_augmented = np.tile(smooth_labels(label_one_hot, label_smoothing_factor), (2,1))               

#     labels_permute = labels_augmented[index].copy() 

#     return label_one_hot, x_train_permute, label_smoothing_factor, labels_permute



class YOPOClassification(nn.Module):
    def __init__(self, num_labels, vector_size=32):
        super(YOPOClassification, self).__init__()
        self.main_input = nn.Linear(vector_size, num_labels)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.main_input(x)
        x = self.softmax(x)
        return x

class YOPO_Final_Model(nn.Module):
    def __init__(self, yopo_feature, yopo_classification):
        super(YOPO_Final_Model, self).__init__()
        self.feature_model = yopo_feature
        self.classification_model = yopo_classification
        
    def forward(self, input_image):
        features = self.feature_model(input_image)
        output = self.classification_model(features)
        return output

def image_normalization(img_list):
    ### img_list is a list cantains images, returns a list contains normalized images

    normalized_images = []
    print('Normalizing')
    for image in img_list:
        image = np.array(image)
        image = torch.tensor(image)
        normalize_single = Normalize(mean=[image.mean()], std=[image.std()])(image).tolist()
        normalized_images.append(normalize_single)
    print('Normalizing finished')
    return normalized_images


def pickle_load(path): 
    with open(path, 'rb') as f:     o = pickle.load(f, encoding='latin1') 

    return o 

def read_mrc_numpy_vol(path):
    with mrcfile.open(path) as mrc:
        v = mrc.data
        v = v.astype(np.float32).transpose([2,1,0])
    return v

def save_png(m, name, normalize=True, verbose=False):

    m = np.array(m, dtype=np.float32)

    mv = m[np.isfinite(m)]
    if normalize:
        # normalize intensity to 0 to 1
        if mv.max() - mv.min() > 0:
            m = (m - mv.min()) / (mv.max() - mv.min())
        else:
            m = np.zeros(m.shape)
    else:
        assert mv.min() >= 0
        assert mv.max() <= 1

    m = np.ceil(m * 65534)
    m = np.array(m, dtype=np.uint16)

    import png          # in pypng package
    png.from_array(m, 'L').save(name)


def cub_img(v, view_dir=2):
    if view_dir == 0:
        vt = np.transpose(v, [1,2,0])
    elif view_dir == 1:
        vt = np.transpose(v, [2,0,1])
    elif view_dir == 2:
        vt = v
    
    row_num = vt.shape[0] + 1
    col_num = vt.shape[1] + 1
    slide_num = vt.shape[2]
    disp_len = int( np.ceil(np.sqrt(slide_num)) )
    
    slide_count = 0
    im = np.zeros( (row_num*disp_len, col_num*disp_len) ) + float('nan')
    for i in range(disp_len):
        for j in range(disp_len):
            im[(i*row_num) : ((i+1)*row_num-1),  (j*col_num) : ((j+1)*col_num-1)] = vt[:,:, slide_count]
            slide_count += 1
            
            if (slide_count >= slide_num):
                break
            
        
        if (slide_count >= slide_num):
            break
   
    
    im_v = im[np.isfinite(im)]

    if im_v.max() > im_v.min(): 
        im = (im - im_v.min()) / (im_v.max() - im_v.min())

    return {'im':im, 'vt':vt}


# configuration
class Config:
    # candidateKs = [5]  ### candidate number of clusters to test, it is also possible to set just one large K that overpartites the data

    M = 4  ### number of iterations ###
    # lr = 1e-5  ### CNN learning rate ###
    label_smoothing_factor = 0.2  ### label smoothing factor ###
    reg_covar = 0.00001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':  

    print('Config.device= ',Config.device)
    parser = argparse.ArgumentParser()
    #dataset options
    parser.add_argument("--filtered_data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/data.h5')
    parser.add_argument("--data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/original')
    #stored path
    parser.add_argument("--saving_path", type=str, default='/data/zfr888/EMD_4603/Results_torch1008_success2')
    parser.add_argument("--algorithm_name", type=str, default='gmm')

    parser.add_argument("--sub_epoch", type=int,default=10)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--subtomo_num",type=int, default=8000)
    parser.add_argument("--M", type=int,default=1)
    parser.add_argument("--batch_size", type=int,default=16)
    parser.add_argument("--candidateKs", default=[8,9,10])
    parser.add_argument("--u_filter_rate", type=str, default=0.025)
    parser.add_argument("--alpha",type=float, default=1.0)
    parser.add_argument("--reg_covar", type=int, default=0.000001)
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--scanning_num", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=24)
    parser.add_argument("--factor", default=2)
    args = parser.parse_args()

    M = args.M   ### number of iterations ###
    lr = args.lr   ### CNN learning rate ###
    reg_covar = args.reg_covar
    scanning_num = args.scanning_num ### the number of scanning ###
    candidateKs = args.candidateKs
    input_size = args.input_size
    factor = args.factor ### the num of scanning division###
    
    # paths used to stored
    saving_path = args.saving_path
    algorithm_name = args.algorithm_name
    yopomodel_path = saving_path+'/models/deltamodel_%s_M_%s_subtomo_num_%s_epoch_%s_lr_%s_reg_%s.h5' \
        %(algorithm_name,str(M),str(args.subtomo_num),str(args.sub_epoch),str(lr),str(reg_covar))
    classification_model_path = saving_path+'/models/classificationmodel_%s_M_%s_subtomo_num_%s_epoch_%s_lr_%s_reg_%s.h5' \
        %(algorithm_name,str(M),str(args.subtomo_num),str(args.sub_epoch),str(lr),str(reg_covar))
    label_path = saving_path+'/results/labels_%s_M_%s_subtomo_num_%s_epoch_%s_lr_%s_reg_%s.pickle' \
        %(algorithm_name,str(M),str(args.subtomo_num),str(args.sub_epoch),str(lr),str(reg_covar))

    for creat_path in ['/models','/figures','/results']:
        creat_folder_path = saving_path+creat_path
        if not os.path.exists(creat_folder_path):
            os.makedirs(creat_folder_path)

    # data set
    filtered_data_path = args.filtered_data_path
    h5f = h5py.File(filtered_data_path,'r')                                                        
    total_subtomo = len(h5f['dataset_1'][:]) # only 'dataset_1'  [16265,24,24,24,24] 

    subtomo_num = args.subtomo_num
    filtered_data = h5f['dataset_1'][total_subtomo- subtomo_num:] # only 'dataset_1'  [16265,24,24,24,24]                            
    h5f.close()
    data_path = args.data_path
    infos = pickle_load(args.data_path+'/info.pickle')
    infos = infos[total_subtomo- subtomo_num:]       # info.shape = [16265]

    v = read_mrc_numpy_vol(args.data_path+'/emd_4603.map')
    algorithms = ['deltamodel']+args.algorithm_name.split('_')
    v = (v - np.mean(v))/np.std(v)
    s = 32//2

    figures_path = args.saving_path+'/figures/'
    figure_path = os.path.join(figures_path,'_'.join('no_train_YOPO'))
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)


    data = filtered_data

    data_array_normalized = []
    for i in range(data.shape[0]):
        x = data[i]
        data_array_normalized.append(x)
    

    filtered_data = np.array(data_array_normalized)
    data_array_normalized = np.array(data_array_normalized).reshape(data.shape[0],1,data.shape[1],data.shape[2], data.shape[3])
    print('data_array_normalized.shape=',data_array_normalized.shape) # [1000,1,24,24,24]
    
    x_train = torch.tensor(data_array_normalized, dtype=torch.float32)  ### load the x_train data, should be shape (n, 1, shape_1, shape_2, shape_3)







    for scanning_it in range(scanning_num):
        print('# Scanning:', scanning_it)


        gt = None  ### load or define label ground truth here, if for simulated data
        
        ### Generalized EM Process ###
        K = None
        labels = None
        DBI_best = float('inf')

        done = False
        i = 0

        total_loss = []

        while not done:
            print('Iteration:', i)
            
        # feature extraction
            if i == 0:
                model_feature = YOPOFeatureModel().to(Config.device)
            else:
                model_feature = nn.Sequential(*list(model.children())[:-1])[0]

            features = np.empty((0, 32))
            train_input_loader = DataLoader(x_train, batch_size = args.batch_size, shuffle = False)
            
            train_pbar = tqdm(train_input_loader, desc='Feature extraction')
            model_feature.eval()    
            with torch.no_grad():
                for batch in train_pbar:
                    batch = batch.to(Config.device)
                    temp_features = model_feature(batch).detach().cpu().numpy()
                    features = np.append(features, temp_features, axis=0) 
        # features=[100,32], max=3.57, min=-3.19
            labels_temp_proba, labels_temp, K, same_K, features_pca, gmm = statistical_fitting_torch_split_merge(features = np.squeeze(features), \
                    labels = labels, candidateKs = candidateKs,\
                    K = K, reg_covar = reg_covar, it = i,\
                    u_filter_rate=args.u_filter_rate, alpha = args.alpha)

            print('..............................Clustering.......................................')
            print('set(labels_temp)=',set(labels_temp))# set() 将集合中重复的元素去除
            print('## Cluster sizes:', [np.sum(labels_temp == k) for  k in set(labels_temp)])  # 返回每个标签的数量


            ############################################################
            ############################################################
            ############################################################
            # infos = infos[:,:]
            _info = []
            for k in range(len(infos)):
                _info.append(infos[k][2])

            for _i in tqdm(range(np.max(labels_temp) + 1)):
                locs = np.array(_info)[labels_temp == _i]         
                ##################################
                v_i = np.zeros_like(v)      # v_i=[928,928,464]
                for j in locs:              # j包含一个三维坐标, s=16, 
                    v_i[j[0] - s: j[0] + s, j[1] - s: j[1] + s, j[2] - s: j[2] + s] = v[j[0] - s: j[0] + s, j[1] - s: j[1] + s, j[2] - s: j[2] + s]
                    # 相当于对一个32x32x32的体素进行赋值
                save_png(cub_img(v_i[:,:,::15])['im'], os.path.join(figure_path, 'GMM'+str(_i) + 'NO_TRAIN' + '.png'))



        ### Matching Clusters by Hungarian Algorithm ###    
            labels_values = np.unique(labels)
            labels_temp_values = np.unique(labels_temp)

            if same_K:
                labels_temp, col_ind = align_cluster_index(labels, labels_temp)
                labels_temp_proba = labels_temp_proba[:,col_ind]
            i, labels, done = convergence_check(i=i, M=args.M, labels_temp=labels_temp, labels=labels, done=done)

        ### Validate Clustering by distortion-based DBI ###

            DBI = DDBI(features_pca, labels)
            if DBI < DBI_best:
                print("i=",i)
                if i > 2:
                    torch.save(model_feature.state_dict(),yopomodel_path)
                    torch.save(model_classification.state_dict(),classification_model_path)
                    pickle_dump(labels, label_path)
                    print('new model is saved!!!!!!!!!!!!!')

                labels_best = labels  ### save current labels if DDBI improves ###
                DBI_best = DBI

            print('DDBI:', DBI, '############################################')

        ## Permute Samples ###   
            print('Prepearing training data')
            # data_array_normalized=(1000,1,24,24,24)
            # data_array_normalized2 = data_array_normalized.transpose(0,2,3,4,1) # data_array_normalized2=[1000,24,24,24,1]
            # filtered_data=[1000,24,24,24,1]
            label_one_hot, x_train_permute, labels_permute = prepare_training_datav2(x_train = filtered_data, labels = labels, n = 1)
            # label_one_hot=(1000,8) 
            
            # x_train_permute[0]=(1000,24,24,24,1)
            # labels_permute[0]=(1000,8)

            print('Finished')
        ### Finetune new model with current estimated K ### 
            if not same_K: 
                print('..................train classifier.......................')
                print('Updating output layer')
                model_classification = YOPOClassification(num_labels=len(set(labels))).to(Config.device)
                optim = torch.optim.NAdam(model_classification.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)

                class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels) 
                class_weights=torch.tensor(class_weights,dtype=torch.float).to(Config.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                # criterion = nn.CrossEntropyLoss()

                # Convert features and label_one_hot to PyTorch tensors

                dataset = Subtomogram_Dataset(features, label_one_hot)

                # Create a DataLoader for batch processing
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

                model_loss = []
                
                for epoch in range(int(args.sub_epoch)):
                    model_classification.train()
                    train_total = 0.0
                    train_correct = 0.0
                    epoch_loss = 0.0
                    start_time = time.time()
                    pbar = tqdm(train_loader, desc = 'Iterating over train data, Epoch: {}/{}'.format(epoch + 1, 10))
                    for features, labels2 in pbar:
                        features = features.to(Config.device)
                        labels2 = labels2.to(Config.device)
                        pred = model_classification(features)
                        optim.zero_grad()

                        predicted = torch.argmax(pred, 1)
                        labels_1 = torch.argmax(labels2, 1)
                        loss = criterion(pred, labels_1)
                        epoch_loss += loss.item()

                        loss.backward()
                        optim.step()
                        
                        train_correct += (predicted == labels_1).sum().float().item()
                        train_total += labels2.size(0)

                    exec_time = time.time() - start_time
                    model_loss.append(epoch_loss)
                    
                    # calculate accuracy
                    accuracy = train_correct / train_total
                    

                    print('Epoch: {}/{} Loss: {:.4f} accuracy: {:.4f} In: {:.4f}s'.format(epoch + 1, 10, epoch_loss, accuracy, exec_time))
                
                ### New YOPO ### 
                model = YOPO_Final_Model(model_feature, model_classification)

                print('Output layer updated')
        #########################################################################################
        #########################################################################################
        #########################################################################################
        ### CNN Training ### 
            print('..................train whole model.......................')
            print('Start CNN training')
            lr *= 0.95
            optim = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)

            # learning rate decay
            scheduler  = StepLR(optim, step_size = 1, gamma = 0.95)


            x_train_permute[0] = x_train_permute[0].transpose(0,4,1,2,3)
            x_train_permute[1] = x_train_permute[1].transpose(0,4,1,2,3)
            x_train_permute[2] = x_train_permute[2].transpose(0,4,1,2,3)
            
            _x_train_permute0 = np.expand_dims(x_train_permute[0],1)
            _x_train_permute1 = np.expand_dims(x_train_permute[1],1)
            _x_train_permute2 = np.expand_dims(x_train_permute[2],1)
            _x_train_permute_total = np.concatenate((_x_train_permute0,_x_train_permute1,_x_train_permute2),1)
            ###################################################################################
            ###################################################################################
            ###################################################################################

            _labels_permute0 = np.expand_dims(labels_permute[0],1)
            _labels_permute1 = np.expand_dims(labels_permute[1],1)
            _labels_permute2 = np.expand_dims(labels_permute[2],1)
            _labels_permute_total = np.concatenate((_labels_permute0,_labels_permute1,_labels_permute2),1)

            
            
            # class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels_permute[0]), y = labels_permute[0]) 
            class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels) 
            class_weights=torch.tensor(class_weights,dtype=torch.float).to(Config.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            MSE_criterion = nn.MSELoss()
            
            dataset = Subtomogram_Dataset(_x_train_permute_total, _labels_permute_total)
            train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  

            model.train()  
            iteration_loss = 0.0
            train_correct = 0.0
            train_total = 0.0
            start_time = time.time()
            
            pbar = tqdm(train_loader, desc='Iterating over train data, Iteration: {}/{}'.format(i - 1, args.M))
            for train, label in pbar:

                # pass to device
                train = train.to(Config.device)
                label = label.to(Config.device)
                
                input_normal = train[:,0]
                y_pred_normal = model(input_normal)       # input_normal=[16,1,24,24,24],y_pred_normal=[16,9]
                label_normal = label[:,0]   # label_normal=[16,9]

                optim.zero_grad()
                label_1 = torch.argmax(label_normal, 1)
                loss = criterion(y_pred_normal, label_1)           # label_1=[8], y_pred_normal=[16,9]
                
                
                iteration_loss += loss.item()

                loss.backward()
                optim.step()

                predicted = torch.argmax(pred, 1)
                # train_correct += (predicted == label_1).sum().float().item()
                train_total += label.size(0)
            scheduler.step()
            total_loss.append(iteration_loss)

            # calculate accuracy
            # accuracy = train_correct / train_total

            exec_time = time.time() - start_time
            # print('Loss: {:.4f} Accuracy: {:.4f}  In: {:.4f}s'.format(iteration_loss, accuracy, exec_time))
            print('Loss: {:.4f}  In: {:.4f}s'.format(iteration_loss, exec_time))
            
            print("finally save the results!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            torch.save(model_feature.state_dict(),yopomodel_path)
            torch.save(model_classification.state_dict(),classification_model_path)




        ############################################################################################
        ############################################################################################
        ############################################################################################
        ############################################################################################
        ### scanning ###  
        # building the subtomo #   

        '''
        model_feature_load = torch.load(yopomodel_path)
        model_classification_load = torch.load(classification_model_path)

        model_feature.load_state_dict(model_feature_load)
        model_classification.load_state_dict(model_classification_load)

        scanning_model = YOPO_Final_Model(model_feature, model_classification)



        fi = 0    
        pp_indexs = []
        for f in sorted(os.listdir(data_path)):   
            if f.split("_")[0] != 'emd':
                continue 
            tom = read_mrc_numpy_vol(os.path.join(data_path,f))            
            tom = (tom - np.mean(tom))/np.std(tom)        
            tom[tom > 4.] = 4.    
            tom[tom < -4.] = -4.   
            adding_pre = math.floor(input_size/2)       # adding_pre=12
            adding_post = math.ceil(input_size/2)-1     # adding_post=11
            # tom.shape=[928,928,464], factor=10
            # x_interval_start, y_interval_start, z_interval_start = \
            #     [np.array(range(adding_pre, tom.shape[i]-adding_post, int(tom.shape[i]/factor))) for i in range(3)]                        

            # x_interval_end, y_interval_end, z_interval_end = \
            #     [np.array(list(range(adding_pre, tom.shape[i]-adding_post, int(tom.shape[i]/factor)))[1:] + [tom.shape[i]]) for i in range(3)]   
                
            # x_interval_start -= adding_pre
            # y_interval_start -= adding_pre
            # z_interval_start -= adding_pre
            # x_interval_end[:-1] += adding_post
            # y_interval_end[:-1] += adding_post
            # z_interval_end[:-1] += adding_post   

            # subvolumes = []        
            # #print('interval num: ', len(x_interval_start)) 

            # for i in range(factor): 
            #     for j in range(factor):           
            #         for k in range(factor):       
            #             subvolume = tom[x_interval_start[i]: x_interval_end[i], y_interval_start[j]: y_interval_end[j], \
            #                             z_interval_start[k]: z_interval_end[k]]    
            #             subvolumes.append(np.expand_dims(np.array(subvolume), [0,-1]))


            # x_interval_start = np.linspace(0, 888, num=38).astype(int)
            # x_interval_end = np.linspace(24, 912, num=38).astype(int)

            # y_interval_start = np.linspace(0, 888, num=38).astype(int)
            # y_interval_end = np.linspace(24, 912, num=38).astype(int)

            # z_interval_start = np.linspace(0, 432, num=19).astype(int)
            # z_interval_end = np.linspace(24, 456, num=19).astype(int)


            x_interval_start = np.linspace(0, 72, num=4).astype(int)
            x_interval_end = np.linspace(24, 96, num=4).astype(int)

            y_interval_start = np.linspace(0, 72, num=4).astype(int)
            y_interval_end = np.linspace(24, 96, num=4).astype(int)

            z_interval_start = np.linspace(0, 72, num=4).astype(int)
            z_interval_end = np.linspace(24, 96, num=4).astype(int)

            subvolumes = []
            for i in range(len(x_interval_start)): 
                for j in range(len(y_interval_start)):           
                    for k in range(len(z_interval_start)):       
                        subvolume = tom[x_interval_start[i]: x_interval_end[i], y_interval_start[j]: y_interval_end[j], \
                                        z_interval_start[k]: z_interval_end[k]]    
                        subvolumes.append(np.expand_dims(np.array(subvolume), [0,-1]))
            # subvolumes = subvolumes[0:20]
            # predict #
            subvolumes_label = []
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'constructing new data sets:')
            for subv_i in tqdm(range(len(subvolumes))):
                subt = subvolumes[subv_i]           # # subt=[1,115,115,69,1]
                subt = torch.tensor(subt)
                subt = subt.permute(0,4,1,2,3)
                subt_label = scanning_model(subt) # subt=[1,1,46,46,34],sub_label=[1,4]
                subt_label = torch.unsqueeze(subt_label, dim=-1)
                subt_label = torch.unsqueeze(subt_label, dim=-1)
                subt_label = torch.unsqueeze(subt_label, dim=-1)
                subt_label = torch.unsqueeze(subt_label, dim=-1)
                subt_label = torch.squeeze(subt_label,dim=0)
                subt_label = subt_label.detach().numpy()
                subt_label = np.max(subt_label,axis=0)
                
                subvolumes_label.append(subt_label)

            pp_map = np.zeros([tom.shape[0] - (input_size - 1), \
                            tom.shape[1] - (input_size - 1), \
                                tom.shape[2] - (input_size - 1), \
                                    subvolumes_label[0].shape[-1]]) # pp_map.shape=[905,905,441,4]
            m = 0                 
            for i in tqdm(range(len(x_interval_start))):             
                for j in range(len(y_interval_start)):             
                    for k in range(len(z_interval_start)):
                        pp_map[x_interval_start[i]: x_interval_start[i] + subvolumes_label[m].shape[1], \
                            y_interval_start[j]: y_interval_start[j] + subvolumes_label[m].shape[2], \
                                z_interval_start[k]: z_interval_start[k] + subvolumes_label[m].shape[3]] = subvolumes_label[m]   
                        m += 1


            #pp_map_filtered_labels = np.where(pp_map[:, :, :, 0]<0.5,np.argmax(pp_map, -1),0) # (l,w,h)
            pp_map_filtered_labels = np.argmax(pp_map, -1)


            particle_filtered = []
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'scanning:')
            
            x_con_start, y_con_start, z_con_start = [np.array(range(0, pp_map_filtered_labels.shape[i], \
                        int(pp_map_filtered_labels.shape[i]/factor)))[:-1] for i in range(3)]
            x_con_end, y_con_end, z_con_end = [np.array(list(range(0, pp_map_filtered_labels.shape[i], \
                                            int(pp_map_filtered_labels.shape[i]/factor)))[1:-1] \
                                + [pp_map_filtered_labels.shape[i]]) for i in range(3)]



            for i in tqdm(range(len(x_interval_start))):
                for j in range(len(y_interval_start)):
                    for k in range(len(z_interval_start)):
                        pp_subvolume = pp_map_filtered_labels[x_con_start[i]: x_con_end[i], \
                                    y_con_start[j]: y_con_end[j], \
                                        z_con_start[k]: z_con_end[k]]
                        if args.filtered_particle_saving_path is None:
                            particle_filtered.append(con_hist_filtering(pp_subvolume,\
                            scanning_bottom=args.scanning_bottom, scanning_upper=args.scanning_upper))
                        else:
                            if not os.path.exists(args.filtered_particle_saving_path):
                                os.makedirs(args.filtered_particle_saving_path)
                            particle_filtered.append(con_hist_filtering(pp_subvolume,\
                            scanning_bottom=args.scanning_bottom, scanning_upper=args.scanning_upper,\
                            saving_path = '%s/hist_%s_%s_%s.npy' %(args.filtered_particle_saving_path,str(i),str(j),str(k))))

            pp_index = np.concatenate(particle_filtered)
            #save_png(cub_img(pp_map_non_noise[:, :, ::20])['im'], '/local/scratch/v_yijian_bai/disca/deepgmmu/disca/v.png')

            pp_indexs.append(pp_index)
            
        '''