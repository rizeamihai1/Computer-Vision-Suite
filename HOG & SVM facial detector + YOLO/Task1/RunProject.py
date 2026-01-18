from Parameters import *
from FacialDetector import *
import numpy as np
import os

params: Parameters = Parameters()
params.dim_window = 36
params.dim_hog_cell = 6
params.overlap = 0.3
params.number_positive_examples = 6547
params.number_negative_examples = 13000

params.threshold = 0.2
params.has_annotations = True

params.use_hard_mining = True
params.use_flip_images = True

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples) + '.npy')

if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive.')
else:
    print('Construim descriptorii pentru exemplele pozitive...')
    positive_features = facial_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features)
    print(f'Am salvat descriptorii pozitivi in: {positive_features_path}')

negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples) + '.npy')

if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative.')
else:
    print('Construim descriptorii pentru exemplele negative...')
    negative_features = facial_detector.get_negative_descriptors()
    np.save(negative_features_path, negative_features)
    print(f'Am salvat descriptorii negativi in: {negative_features_path}')

pos_feats = np.reshape(positive_features, (positive_features.shape[0], -1))
neg_feats = np.reshape(negative_features, (negative_features.shape[0], -1))

training_examples = np.concatenate((pos_feats, neg_feats), axis=0)

train_labels = np.concatenate((
    np.ones(pos_feats.shape[0]), 
    np.zeros(neg_feats.shape[0])
))

print(f"Antrenare initiala pe {training_examples.shape[0]} exemple...")
facial_detector.train_classifier(training_examples, train_labels)

if params.use_hard_mining:
    print("Incepem etapa de Hard Negative Mining...")
    hard_negative_features = facial_detector.get_hard_negative_descriptors()
    
    if len(hard_negative_features) > 0:
        hard_neg_feats = np.reshape(hard_negative_features, (hard_negative_features.shape[0], -1))
        
        training_examples = np.concatenate((training_examples, hard_neg_feats), axis=0)
        train_labels = np.concatenate((train_labels, np.zeros(hard_neg_feats.shape[0])))
        
        print(f"Re-antrenam cu {len(hard_neg_feats)} exemple puternic negative adaugate.")
        facial_detector.train_classifier(training_examples, train_labels)
    else:
        print("Nu au fost gasite exemple puternic negative noi.")


print("Rulam detectorul pe imaginile de test (multi-scale)...")
detections, scores, file_names = facial_detector.run()

solutie_path_task1 = os.path.join('..', 'fisiere_solutie', 'task1')

if not os.path.exists(solutie_path_task1):
    os.makedirs(solutie_path_task1, exist_ok=True)

np.save(os.path.join(solutie_path_task1, "detections_all_faces.npy"), detections)
np.save(os.path.join(solutie_path_task1, "scores_all_faces.npy"), scores)
np.save(os.path.join(solutie_path_task1, "file_names_all_faces.npy"), file_names)
print(f"Rezultatele au fost salvate in directorul: {solutie_path_task1}")
