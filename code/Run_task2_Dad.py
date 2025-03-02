from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *
import pandas as pd

params: Parameters = Parameters(2,"dad")

params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 6  # dimensiunea celulei
params.cells_p_block = 2

params.overlap = 0.3
params.threshold = 0 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.use_hard_mining = True  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite


# params.dir_test_examples = os.path.join('..','evaluare','fake_test_20')
# params.path_annotations = os.path.join('..','validare','task2_dad_gt_validare20.txt')

params.dir_test_examples = os.path.join('..','evaluare','fake_test')
params.path_annotations = os.path.join('..','validare','task2_dad_gt_validare.txt')


# files = glob.glob(os.path.join(params.dir_pos_examples, "*.jpg"))
# num_images = len(files)
# params.number_positive_examples = num_images
# if params.use_flip_images:
#     params.number_positive_examples *= 2

# files = glob.glob(os.path.join(params.dir_neg_examples, "*.jpg"))
# num_images = len(files)
# params.number_negative_examples = num_images

# if params.use_hard_mining == True:
#     files = glob.glob(os.path.join(params.dir_hard_mining, "*.jpg"))
#     num_images = len(files)
#     params.number_hard_mining = num_images
# else:
#     params.number_hard_mining = 0

facial_detector: FacialDetector = FacialDetector(params)

params.number_positive_examples = 2572
params.number_negative_examples = 49054
params.number_hard_mining = 20863

# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente
positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                        str(facial_detector.params.number_positive_examples) + '.npy')
print("positive_features_path: ",positive_features_path)

if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features = facial_detector.get_positive_descriptors()
    # os.makedirs(positive_features_path, exist_ok= True)
    np.save(positive_features_path, positive_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)


# exemple negative
negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_' +
                        str(facial_detector.params.number_negative_examples) + '.npy')
print("negative_features_path: ", negative_features_path)

if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    # os.makedirs(negative_features_path, exist_ok= True)
    np.save(negative_features_path, negative_features)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

if params.use_hard_mining == True:
    hard_mining_path = os.path.join(params.dir_save_files, 'descriptoriHardMining_' + str(params.dim_hog_cell) + '_' +
                            str(facial_detector.params.number_hard_mining) + '.npy')
    print("hard_mining_path: ", hard_mining_path)

    if os.path.exists(hard_mining_path):
        hard_mining_features = np.load(hard_mining_path)
        print('Am incarcat descriptorii pentru exemplele hard mining')
    else:
        print('Construim descriptorii pentru exemplele hard mining:')
        hard_mining_features = facial_detector.get_hard_mining_descriptors()
        # os.makedirs(negative_features_path, exist_ok= True)
        np.save(hard_mining_path, hard_mining_features)
        print('Am salvat descriptorii pentru exemplele hard mining in fisierul %s' % hard_mining_path)

# Pasul 4. Invatam clasificatorul liniar
if params.use_hard_mining == True:
    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features), np.squeeze(hard_mining_features)), axis=0)

    hard_mining_labels = 0 * np.ones(hard_mining_features.shape[0])
    train_labels = np.concatenate((np.ones(facial_detector.params.number_positive_examples), np.zeros(facial_detector.params.number_negative_examples), hard_mining_labels))
    print("positives: ", facial_detector.params.number_positive_examples, "\nnegatives: ", facial_detector.params.number_negative_examples, "\nhard mining: ", facial_detector.params.number_hard_mining)


else:
        training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
        train_labels = np.concatenate((np.ones(facial_detector.params.number_positive_examples), np.zeros(facial_detector.params.number_negative_examples)))
        print("positives: ", facial_detector.params.number_positive_examples, "\nnegatives: ", facial_detector.params.number_negative_examples)


svm_file_name = os.path.join(params.dir_save_files, 'best_model_%d_%d_%d_%d' %
                                (params.dim_hog_cell, params.number_negative_examples,
                                params.number_positive_examples, params.number_hard_mining))



facial_detector.train_classifier(svm_file_name,training_examples, train_labels)

# Pasul 5. (optional) Antrenare cu exemple puternic negative (detectii cu scor >0 din cele 274 de imagini negative)
# Daca implementati acest pas ar trebui sa modificati functia FacialDetector.run()
# astfel incat sa va returneze descriptorii detectiilor cu scor > 0 din cele 274 imagini negative
# completati codul in continuare
# TODO:  (optional)  completeaza codul in continuare
#0.445
# ratios = [0.58, 0.66, 0.81, 0.87, 0.95, 1.03, 1.11, 1.21, 1.36, 1.6]

#0.461
ratios = np.linspace(0.7, 1.5, 10)

# resizing 20 - 0.45
nr_of_resizing = 10
detections, scores, file_names = facial_detector.run(ratios,nr_of_resizing)

if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)


np.save(facial_detector.params.sol_detections_path ,detections, allow_pickle=True)
np.save(facial_detector.params.sol_scores_path ,scores, allow_pickle=True)
np.save(facial_detector.params.sol_filenames_path , file_names, allow_pickle=True)


#########################hard_mining
# print("\n\nDAD\n\n")
# params.threshold = 2 
# params.dir_test_examples = os.path.join('..','antrenare','dad')
# params.path_annotations = os.path.join('..','antrenare','dad_annotations.txt')

# detections, scores, file_names = facial_detector.run(ratios,nr_of_resizing)

# data = {
#     "detections": [list(det) for det in detections],
#     "scores": scores,
#     "file_names": file_names,
# }
# df_dad = pd.DataFrame(data)

# output_csv_path = os.path.join('..','antrenare',"task2_dad_hard_mining","dad.csv")
# df_dad.to_csv(output_csv_path, index=False)

# ########################################
# print("\n\nMOM\n\n")
# params.dir_test_examples = os.path.join('..','antrenare','mom')
# params.path_annotations = os.path.join('..','antrenare','mom_annotations.txt')

# detections, scores, file_names = facial_detector.run(ratios,nr_of_resizing)

# data = {
#     "detections": [list(det) for det in detections],
#     "scores": scores,
#     "file_names": file_names,
# }
# df_mom = pd.DataFrame(data)

# output_csv_path = os.path.join('..','antrenare',"task2_dad_hard_mining","mom.csv")
# df_mom.to_csv(output_csv_path, index=False)

    

# ###################################
# print("\n\nDEXTER\n\n")


# params.dir_test_examples = os.path.join('..','antrenare','dexter')
# params.path_annotations = os.path.join('..','antrenare','dexter_annotations.txt')

# detections, scores, file_names = facial_detector.run(ratios,nr_of_resizing)

# data = {
#     "detections": [list(det) for det in detections],
#     "scores": scores,
#     "file_names": file_names,
# }
# df_dexter = pd.DataFrame(data)


# output_csv_path = os.path.join('..','antrenare',"task2_dad_hard_mining","dexter.csv")
# df_dexter.to_csv(output_csv_path, index=False)


# ######################
# print("\n\DEEDEE\n\n")
# params.threshold = 0
# params.dir_test_examples = os.path.join('..','antrenare','deedee')
# params.path_annotations = os.path.join('..','antrenare','deedee_annotations.txt')

# detections, scores, file_names = facial_detector.run(ratios,nr_of_resizing)

# data = {
#     "detections": [list(det) for det in detections],
#     "scores": scores,
#     "file_names": file_names,
# }
# df_deedee = pd.DataFrame(data)

# output_csv_path = os.path.join('..','antrenare',"task2_dad_hard_mining","deedee.csv")
# df_deedee.to_csv(output_csv_path, index=False)




