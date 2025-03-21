from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *
import pandas as pd

params: Parameters = Parameters()
params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 6  # dimensiunea celulei
params.cells_p_block = 2
# params.number_positive_examples = 1168  # numarul exemplelor pozitive
# params.number_negative_examples = 4000  # numarul exemplelor negative
params.overlap = 0.3
params.threshold = 2

params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite



files = glob.glob(os.path.join(params.dir_pos_examples, "*.jpg"))
num_images = len(files)
params.number_positive_examples = num_images
if params.use_flip_images:
    params.number_positive_examples *= 2

files = glob.glob(os.path.join(params.dir_neg_examples, "*.jpg"))
num_images = len(files)

params.number_negative_examples = num_images

if params.use_hard_mining == True:
    files = glob.glob(os.path.join(params.dir_hard_mining, "*.jpg"))
    num_images = len(files)
    params.number_hard_mining = num_images
else:
    params.number_hard_mining = 0

facial_detector: FacialDetector = FacialDetector(params)

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
    print(f"training examples: {training_examples.shape}")
    print(f"train labels: {train_labels.shape}")

else:
    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(facial_detector.params.number_positive_examples), np.zeros(facial_detector.params.number_negative_examples)))
    print("positives: ", facial_detector.params.number_positive_examples, "\nnegatives: ", facial_detector.params.number_negative_examples)
    print(f"training examples: {training_examples.shape}")
    print(f"train labels: {train_labels.shape}")

# Pasul 5. (optional) Antrenare cu exemple puternic negative (detectii cu scor >0 din cele 274 de imagini negative)
# Daca implementati acest pas ar trebui sa modificati functia FacialDetector.run()
# astfel incat sa va returneze descriptorii detectiilor cu scor > 0 din cele 274 imagini negative
# completati codul in continuare
# TODO:  (optional)  completeaza codul in continuare
facial_detector.train_classifier(training_examples, train_labels)

# print("\n\nDAD\n\n")
# params.threshold = 2
# params.dir_test_examples = os.path.join('..','antrenare','dad')
# params.path_annotations = os.path.join('..','antrenare','dad_annotations.txt')

# detections, scores, file_names = facial_detector.run()

# data = {
#     "detections": [list(det) for det in detections],
#     "scores": scores,
#     "file_names": file_names,
# }
# df = pd.DataFrame(data)


# output_csv_path = os.path.join('..','antrenare',"hard_mining_dad.csv")
# df.to_csv(output_csv_path, index=False)


#####################
print("\n\nMOM\n\n")
params.dir_test_examples = os.path.join('..','antrenare','mom')
params.path_annotations = os.path.join('..','antrenare','mom_annotations.txt')

detections, scores, file_names = facial_detector.run()

data = {
    "detections": [list(det) for det in detections],
    "scores": scores,
    "file_names": file_names,
}
df = pd.DataFrame(data)


output_csv_path = os.path.join('..','antrenare',"hard_mining_mom.csv")
df.to_csv(output_csv_path, index=False)
    

###################################
print("\n\DEXTER\n\n")


params.dir_test_examples = os.path.join('..','antrenare','dexter')
params.path_annotations = os.path.join('..','antrenare','dexter_annotations.txt')

detections, scores, file_names = facial_detector.run()

data = {
    "detections": [list(det) for det in detections],
    "scores": scores,
    "file_names": file_names,
}
df = pd.DataFrame(data)


output_csv_path = os.path.join('..','antrenare',"hard_mining_dexter.csv")
df.to_csv(output_csv_path, index=False)


######################
print("\n\DEEDEE\n\n")
params.threshold = 0
params.dir_test_examples = os.path.join('..','antrenare','deedee')
params.path_annotations = os.path.join('..','antrenare','deedee_annotations.txt')

detections, scores, file_names = facial_detector.run()

data = {
    "detections": [list(det) for det in detections],
    "scores": scores,
    "file_names": file_names,
}
df = pd.DataFrame(data)


output_csv_path = os.path.join('..','antrenare',"hard_mining_deedee.csv")
df.to_csv(output_csv_path, index=False)