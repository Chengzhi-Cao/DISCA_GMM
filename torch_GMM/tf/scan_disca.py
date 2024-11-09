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

# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config_yaml', type=str, \
#         default='/code/DISCA_GMM/config/train.yaml', help='YAML config file')
# config_parser = parser.parse_args()
# args = parse_args_yaml(config_parser)



parser = argparse.ArgumentParser()
#dataset options
parser.add_argument("--filtered_data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/data.h5')
parser.add_argument("--data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/original')

#stored path
parser.add_argument("--saving_path", type=str, default='/data/zfr888/EMD_4603/Results_tf')
parser.add_argument("--algorithm_name", type=str, default='gmmu_cavi_llh_hist')
parser.add_argument("--filtered_particle_saving_path", type=str, default='/data/zfr888/EMD_4603/Results_tf/filtered_particle')

parser.add_argument("--image_size", type=int, default=24)
parser.add_argument("--input_size", type=int, default=24)
parser.add_argument("--candidateKs", default=[7,8,9,10,11])

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--factor", default=10)
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

parser.add_argument("--DIVIDE", type=int,default=10)
parser.add_argument("--scanning_num", type=int, default=2)
parser.add_argument("--M", type=int,default=2)
parser.add_argument("--sub_epoch", type=int,default=2)
parser.add_argument("--subtomo_num",type=int, default=1000)
parser.add_argument("--subtomo_num_test",type=int, default=100)
args = parser.parse_args()


data_path = args.data_path


# setting of YOPO and GMMU
image_size = args.image_size #None   ### subtomogram size ###
input_size = args.input_size
candidateKs = args.candidateKs   ### candidate number of clusters to test
        
batch_size = args.batch_size
scanning_num = args.scanning_num ### the number of scanning ###
factor = args.factor ### the num of scanning division###
M = args.M   ### number of iterations ###
lr = args.lr   ### CNN learning rate ###

reg_covar = args.reg_covar


# paths used to stored
saving_path = args.saving_path
algorithm_name = args.algorithm_name
model_path = saving_path+'/models/deltamodel_%s_M_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(M),str(lr),str(reg_covar))
classification_model_path = saving_path+'/models/classificationmodel_%s_M_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(M),str(lr),str(reg_covar))
gmmu_model_path = saving_path+'/models/gmmumodel_%s_M_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(M),str(lr),str(reg_covar))
model_path_last = saving_path+'/models/deltamodel_%s_M_%s_lr_%s_reg_%s_last.h5' \
    %(algorithm_name,str(M),str(lr),str(reg_covar))

label_path = saving_path+'/results/labels_%s_M_%s_lr_%s_reg_%s.pickle' \
    %(algorithm_name,str(M),str(lr),str(reg_covar))
label_path_last = saving_path+'/results/labels_%s_M_%s_lr_%s_reg_%s_last.pickle' \
    %(algorithm_name,str(M),str(lr),str(reg_covar))

for creat_path in ['/models','/figures','/results']:
    creat_folder_path = saving_path+creat_path
    if not os.path.exists(creat_folder_path):
        os.makedirs(creat_folder_path)
 

### Generalized EM Process ### 

strategy = tf.distribute.MirroredStrategy() 
pp_indexs = []


### Load data ###   
DBI_best = np.inf
K = None
lr = args.lr
labels = None 
it = 0
best_i = it
done = False
x_train = []    
fi = 0    


### loading the trained model ###
parallel_model_feature = tf.keras.models.load_model(model_path, \
            custom_objects={'CosineSimilarity': CosineSimilarity})
model_classification= tf.keras.models.load_model(classification_model_path, \
            custom_objects={'CosineSimilarity': CosineSimilarity,'SNN': SNN, 'NSNN': NSNN})

# Because this is only used for choosing new sample, we can just considering the main input
input_n = parallel_model_feature.layers[-1].input
part_model = tf.keras.Model(parallel_model_feature.layers[-1].input, \
    parallel_model_feature.layers[-1].output)
output_n = part_model(input_n)
part_class_model = tf.keras.Model(model_classification.layers[-3].input, \
    model_classification.layers[-3].output)
scanning_model = tf.keras.Model(input_n, \
                part_class_model(output_n))


### scanning ###  
# building the subtomo #   
fi = 0    
pp_indexs = []
for f in sorted(os.listdir(data_path)):   
    if f.split("_")[0] != 'emd':
        continue 
    tom = read_mrc_numpy_vol(os.path.join(data_path,f))      # tom.shape=[928,928,464]      
    tom = (tom - np.mean(tom))/np.std(tom)        
    tom[tom > 4.] = 4.    
    tom[tom < -4.] = -4.   
    adding_pre = math.floor(input_size/2)
    adding_post = math.ceil(input_size/2)-1

    x_interval_start, y_interval_start, z_interval_start = \
        [np.array(range(adding_pre, tom.shape[i]-adding_post, int(tom.shape[i]/factor))) for i in range(3)]                        

    x_interval_end, y_interval_end, z_interval_end = \
        [np.array(list(range(adding_pre, tom.shape[i]-adding_post, int(tom.shape[i]/factor)))[1:] \
                                                            + [tom.shape[i]]) for i in range(3)]   
        
    x_interval_start -= adding_pre
    y_interval_start -= adding_pre
    z_interval_start -= adding_pre
    x_interval_end[:-1] += adding_post
    y_interval_end[:-1] += adding_post
    z_interval_end[:-1] += adding_post   

    subvolumes = []        
    #print('interval num: ', len(x_interval_start)) 

    for i in range(factor):         # factor=40 
        for j in range(factor):           
            for k in range(factor):       
                subvolume = tom[x_interval_start[i]: x_interval_end[i], y_interval_start[j]: y_interval_end[j], \
                                z_interval_start[k]: z_interval_end[k]]    
                subvolumes.append(np.expand_dims(np.array(subvolume), [0,-1]))  # subvolumes=64000*[1,46,46,34,1]

    # predict #
    subvolumes_label = []
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'constructing new data sets:')
    for subv_i in tqdm(range(len(subvolumes))):
        subt = subvolumes[subv_i]
        subt_label = scanning_model.predict(subt, verbose=0) 
        subvolumes_label.append(subt_label)

    pp_map = np.zeros([tom.shape[0] - (input_size - 1), \
                    tom.shape[1] - (input_size - 1), \
                        tom.shape[2] - (input_size - 1), \
                            scanning_model.output_shape[-1]])
    m = 0                 
    for i in tqdm(range(factor)):             
        for j in range(factor):             
            for k in range(factor):
                # because we only need the identified tomo, the label can be ignored
                # When using mean-shift, we should modify this part.
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

    for i in tqdm(range(factor)):
        for j in range(factor):
            for k in range(factor):
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

    print('pp_index=',len(pp_index))
    print('pp_index=',pp_index)
