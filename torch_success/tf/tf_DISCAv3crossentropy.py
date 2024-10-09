# 2024/8/17
# without scanning
# Chengzhi Cao

# v3版本：训练整个模型的时候直接使用crossentropy loss，而不是之前的对比学习，看看结果如何

#v1 可以得到不错的效果

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
parser.add_argument("--saving_path", type=str, default='/data/zfr888/EMD_4603/Results_tf_v1')
parser.add_argument("--algorithm_name", type=str, default='gmmu_cavi_llh_hist')
parser.add_argument("--filtered_particle_saving_path", type=str, default='/data/zfr888/EMD_4603/Results_tf_v1/filtered_particle')

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
parser.add_argument("--scanning_num", type=int, default=1)
parser.add_argument("--M", type=int,default=4)
parser.add_argument("--sub_epoch", type=int,default=10)
parser.add_argument("--subtomo_num",type=int, default=1000)
parser.add_argument("--subtomo_num_test",type=int, default=100)

parser.add_argument("--filterdata_flag",type=bool, default=False)
# filtered_data
args = parser.parse_args()



# data set
filtered_data_path = args.filtered_data_path
h5f = h5py.File(filtered_data_path,'r')                                                        
# filtered_data = h5f['dataset_1'][:] # only 'dataset_1'  

total_subtomo = len(h5f['dataset_1'][:]) # only 'dataset_1'  [16265,24,24,24,24] 

_h5f = h5f['dataset_1']     # [16265,24,24,24,1]
subtomo_num = args.subtomo_num
filtered_data = h5f['dataset_1'][total_subtomo- subtomo_num:] # only 'dataset_1'  [16265,24,24,24,24]                            

h5f.close()
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
model_path = saving_path+'/models/deltamodel_%s_scan_%s_M_%s_epoch_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(scanning_num),str(M),str(args.sub_epoch),str(lr),str(reg_covar))
classification_model_path = saving_path+'/models/classificationmodel_%s_scan_%s_M_%s_epoch_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(scanning_num),str(M),str(args.sub_epoch),str(lr),str(reg_covar))
gmmu_model_path = saving_path+'/models/gmmumodel_%s_scan_%s_M_%s_epoch_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(scanning_num),str(M),str(args.sub_epoch),str(lr),str(reg_covar))
model_path_last = saving_path+'/models/deltamodel_%s_scan_%s_M_%s_epoch_%s_lr_%s_reg_%s_last.h5' \
    %(algorithm_name,str(scanning_num),str(M),str(args.sub_epoch),str(lr),str(reg_covar))

label_path = saving_path+'/results/labels_%s_scan_%s_M_%s_epoch_%s_lr_%s_reg_%s.pickle' \
    %(algorithm_name,str(scanning_num),str(M),str(args.sub_epoch),str(lr),str(reg_covar))
label_path_last = saving_path+'/results/labels_%s_scan_%s_M_%s_epoch_%s_lr_%s_reg_%s_last.pickle' \
    %(algorithm_name,str(scanning_num),str(M),str(args.sub_epoch),str(lr),str(reg_covar))

for creat_path in ['/models','/figures','/results']:
    creat_folder_path = saving_path+creat_path
    if not os.path.exists(creat_folder_path):
        os.makedirs(creat_folder_path)
 

### Generalized EM Process ### 

strategy = tf.distribute.MirroredStrategy() 
pp_indexs = []

for scanning_it in range(scanning_num):
    print('# Scanning:', scanning_it)

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
    for f in sorted(os.listdir(data_path)):   
        if f.split("_")[0] != 'emd':
            continue 
        # tom = read_mrc_numpy_vol(os.path.join(data_path,f))
        tom = np.random.random((100,100,100))           
        tom = (tom - np.mean(tom))/np.std(tom)      
        tom[tom > 4.] = 4.    
        tom[tom < -4.] = -4.
        if scanning_it < 1:                                    
            # n = np.array([[np.random.randint(input_size/2,tom.shape[0]-input_size/2),\
            #                 np.random.randint(input_size/2,tom.shape[1]-input_size/2),\
            #                 np.random.randint(input_size/2,tom.shape[2]-input_size/2)] for pi in range(20000)])

            n = np.array([[np.random.randint(input_size/2,tom.shape[0]-input_size/2),\
                            np.random.randint(input_size/2,tom.shape[1]-input_size/2),\
                            np.random.randint(input_size/2,tom.shape[2]-input_size/2)] for pi in range(20)])
        else:        
            # n = np.array([[np.random.randint(input_size/2,tom.shape[0]-input_size/2), \
            #                 np.random.randint(input_size/2,tom.shape[1]-input_size/2), \
            #                 np.random.randint(input_size/2,tom.shape[2]-input_size/2)] for pi in range(10000)]) 
                            
            n = np.array([[np.random.randint(input_size/2,tom.shape[0]-input_size/2), \
                            np.random.randint(input_size/2,tom.shape[1]-input_size/2), \
                            np.random.randint(input_size/2,tom.shape[2]-input_size/2)] for pi in range(10)]) 

            n = np.concatenate([n, pp_indexs[fi][np.random.randint(pp_indexs[fi].shape[0], size=50), :] + input_size/2])

        for j in range(len(n)): #random cutting from tomo, time: 0.43s
            v = cut_from_whole_map(tom, n[j], input_size)
            if v is not None:
                x_train.append(v)
            
        fi += 1
                    
    x_train = np.expand_dims(np.array(x_train), -1)  
    x_train_min = np.min(x_train, axis = (1,2,3,4))           
    x_train = x_train[x_train_min < np.median(x_train_min)] # x_train=[100,24,24,24,1]
    
    if args.filterdata_flag == True:
        x_train = np.concatenate([x_train, filtered_data])# x_train=[200,24,24,24,1]
    else:
        x_train = filtered_data     # x_train=[200,24,24,24,1]
    while not done:     # done=False, it will continue iteration 
        print('## Iteration:', it) 
        ### Feature Extraction ### 
        if it == 0: 
            if scanning_it == 0: #######
                with strategy.scope():            
                    parallel_model_feature = YOPO_feature(image_size)   ### create a new model
            else:
                with strategy.scope():
                    parallel_model_feature = tf.keras.models.load_model(model_path, \
                        custom_objects={'CosineSimilarity': CosineSimilarity})
        else:                
            with strategy.scope():           
                parallel_model_feature = tf.keras.Model(inputs=parallel_model.input, \
                    outputs=[parallel_model.layers[-2].get_output_at(0),parallel_model.layers[-2].get_output_at(1),\
                             parallel_model.layers[-2].get_output_at(2)]) 

        # parallel_model_feature.compile(loss=args.loss_function, optimizer=args.optimizer)     # this command will influence the model saving processure.
        
        features = parallel_model_feature.predict([x_train, x_train, x_train])[0] # the first one is the original input
        # the new feature extraction will out put same size, so it is (b,d1,d2,d3,c)
        # features=[1000,1,1,1,32], max=7.33, min=-4.60 K=None
        
        
        ### Feature Clustering ###                              
        labels_temp_proba, labels_temp, K, same_K, features_pca, gmm = \
            statistical_fitting_tf_split_merge(features = np.squeeze(features), \
                                               labels = labels, candidateKs = candidateKs,\
                                                    K = K, reg_covar = reg_covar, it = it,\
                                                    u_filter_rate=args.u_filter_rate, alpha = args.alpha)
        print('..............................Clustering.......................................')
        print('set(labels_temp)=',set(labels_temp))# set() 将集合中重复的元素去除
        print('## Cluster sizes:', [np.sum(labels_temp == k) for  k in set(labels_temp)])  # 返回每个标签的数量

        ### Matching Clusters by Hungarian Algorithm ### 
        if same_K:          # same_K= False
            labels_temp, col_ind = align_cluster_index(labels, labels_temp)
            labels_temp_proba = labels_temp_proba[:,col_ind]

        # i, the numb of iteration, is added 1 here
        it, labels, done = convergence_check(i = it, M = M, labels_temp = labels_temp, labels = labels, done = done) 

        print('..............................Clustering.......................................')
        print('set(labels)=',set(labels))# set() 将集合中重复的元素去除
        print('## Cluster sizes:', [np.sum(labels == k) for  k in set(labels)])  # 返回每个标签的数量

        ### Validate Clustering by distortion-based DBI ### 
        # depending the intialization, DBI could be Nan            
        DBI = DDBI_uniform(features_pca, labels) 
        if DBI < DBI_best: 
            if it > 1:             
                parallel_model_feature.save(model_path)   ### save model here ###  
                model_classification.save(classification_model_path)   ### save classification model here ###          
                labels_best = labels.copy()   ### save current labels if DDBI improves ###                 
                pickle_dump(labels_best, label_path)
                print('## new modele is saved') 
                best_i = it
            else: 
                parallel_model_feature.save(model_path)   ### save model here ###           # 第一次运行，是没有分类器的，只能保存YOPO
                labels_best = labels.copy()   ### save current labels if DDBI improves ###                 
                pickle_dump(labels_best, label_path)
                print('## new modele is saved') 
                best_i = it
                # parallel_model_feature = tf.keras.models.load_model(model_path, custom_objects={'CosineSimilarity': CosineSimilarity})
            DBI_best = DBI                                                                                                                   
        else:
            if it <= 2:
                parallel_model_feature.save(model_path)   ### save model here ###  
                model_classification.save(classification_model_path)   ### save classification model here ###          
                labels_best = labels.copy()   ### save current labels if DDBI improves ###                 
                pickle_dump(labels_best, label_path)
                print('## new modele is saved') 
                best_i = it
        
        print('## DDBI:', DBI)
        if np.isinf(DBI) and it>1:
            continue 

        ### Permute Samples ###             
        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'Preparing data sets')
        # prepare_training_data_simple doesn't use the parallel acceleration like prepare_training_data
        if np.any(np.isnan(labels_temp_proba)) or np.any(np.isnan(labels)):
            if np.any(np.isnan(labels_temp_proba)) and np.any(np.isnan(labels)):
                print('there is nan in labels_temp_proba and labels')
                continue
            if np.any(np.isnan(labels)):
                print('there is nan in labels')
                continue
            if np.any(np.isnan(labels_temp_proba)):
                print('there is nan in labels_temp_proba')
                continue


        label_one_hot, x_train_permute, labels_permute = prepare_training_data(x_train = x_train, labels_temp_proba = labels_temp_proba, \
                                        labels = labels, n = 1)     
        # label_one_hot=(1000,8) 
        
        # x_train_permute[0]=(1000,24,24,24,1)
        # x_train_permute[1]=(1000,24,24,24,1)
        # x_train_permute[2]=(1000,24,24,24,1)
        
        # labels_permute[0]=(1000,8)
        # labels_permute[1]=(1000,8)
        # labels_permute[2]=(1000,8)
        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'Preparing data sets ends')
        label_one_hot = np.expand_dims(label_one_hot, axis=(1,2,3))         # label_one_hot=(1000,1,1,1,9)
        labels_permute[0] = np.expand_dims(labels_permute[0], axis=(1,2,3)) # labels_permute[0]=(1000,1,1,1,9)
        labels_permute[1] = np.expand_dims(labels_permute[1], axis=(1,2,3))
        labels_permute[2] = np.expand_dims(labels_permute[2], axis=(1,2,3))
            
        ### Finetune new model with current estimated K ### 
        if not same_K:
            with strategy.scope():                             
                model_classification = YOPO_classification(num_labels=len(set(labels)), vector_size = 32) 
                model_classification._name = 'classifier' 
                optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)                  
                model_classification.compile(optimizer= optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', \
                                                                        'categorical_crossentropy', SNN, NSNN], \
                                                loss_weights = [1, 0, 0, 0, 0])                      
                model_classification.fit([features, features, features], \
                    [label_one_hot, label_one_hot, label_one_hot, np.zeros(features.shape[0]), np.zeros(features.shape[0])], \
                        epochs=args.sub_epoch, batch_size=batch_size, shuffle=True, verbose = 0) 

                ##############################################################################################
                _labels = model_classification.predict([features, features, features])[0]
                pred_labels = np.array([np.argmax(_labels[q, :]) for q in range(len(_labels))])
                print('..................train classifier.......................')
                # print('pred_labels=',pred_labels)
                print('set(labels)=',set(pred_labels))# set() 将集合中重复的元素去除
                print('## pred_labels Cluster sizes:', [np.sum(pred_labels == k) for  k in set(pred_labels)])  # 返回每个标签的数量
                ##############################################################################################
        
        ### New YOPO ###
                # same reason as squeeze before to use flatten
                parallel_model = tf.keras.Model(parallel_model_feature.input, \
                    model_classification([parallel_model_feature.output[0],\
                                        parallel_model_feature.output[1],\
                                        parallel_model_feature.output[2]]))
                #print(parallel_model.summary())
                
        ### CNN Training ###   
        # add scope?        
        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'Train new model')
        lr *= 0.95 
        # optimizer = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)          
        # parallel_model.compile(optimizer= optimizer, loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', mse, nmse], loss_weights = [0, 0, 0, 1, 1])
        
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)          
        parallel_model.compile(optimizer= optimizer, loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', mse, nmse], loss_weights = [1, 0, 0, 0, 0])

        from sklearn.utils import class_weight        
        print('..................train whole model.......................')
        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels) 
        parallel_model.fit(x_train_permute, [labels_permute[0], labels_permute[1], labels_permute[2], \
                                             np.zeros((x_train_permute[0].shape[0],)), \
                                            np.zeros((x_train_permute[0].shape[0],))],\
                            epochs=args.sub_epoch, batch_size=batch_size, shuffle=True,verbose = 1)       
        
        ##############################################################################################
        _labels = parallel_model.predict(x_train_permute)[0]
        pred_labels = np.array([np.argmax(_labels[q, :]) for q in range(len(_labels))])
        
        print('set(labels)=',set(pred_labels))# set() 将集合中重复的元素去除
        print('## pred_labels Cluster sizes:', [np.sum(pred_labels == k) for  k in set(pred_labels)])  # 返回每个标签的数量
        ##############################################################################################
        
        
        del x_train_permute
        gc.collect() 


    #DBI = DDBI_uniform(features_pca, labels)     
    #parallel_model_feature.save(model_path_last)   ### save model here ###                
    #pickle_dump(labels, label_path_last)
    #print('## latest modele is saved')                                                                                                                  
    #print('## DDBI:', DBI)
    print('## the best iteration is %s' % str(best_i-1))


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
    if scanning_it > 1:
    
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
            # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'constructing new data sets:')
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
            # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'scanning:')
            

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
