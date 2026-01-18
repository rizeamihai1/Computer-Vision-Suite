from Parameters import *
from FacialDetector import *
import numpy as np
import os
import pickle

# 1. Inițializare parametri
params: Parameters = Parameters()
params.dim_window = 36 
params.dim_hog_cell = 6 
params.overlap = 0.3
params.threshold = 0.25
params.use_hard_mining = True 
params.use_flip_images = True 

facial_detector: FacialDetector = FacialDetector(params)


characters = ['shaggy', 'fred', 'velma', 'daphne']
base_faces_path = params.faces 
unknown_path = os.path.join(base_faces_path, 'unknown')

# --- PASUL 1: Antrenare One-vs-All cu Hard Negative Mining ---

for target in characters:
    print(f"\n Antrenare pentru personajul: {target.upper()}")
    
    # extragere descriptori pozitivi
    pos_dir = os.path.join(base_faces_path, target)
    pos_features = facial_detector.get_positive_descriptors(pos_dir)
    pos_feats = np.reshape(pos_features, (pos_features.shape[0], -1)) 

    # extragere descriptori negativi (si ia din fundal + celelalte 3 persoanje + toti cei din unknown)
    rival_folders = [os.path.join(base_faces_path, c) for c in characters if c != target] # extrag din folderul mare, folderele pentru fiecare personj (diferit de cel curent)
    rival_folders.append(unknown_path)
    neg_features = facial_detector.get_negative_descriptors(extra_folders=rival_folders)
    neg_feats = np.reshape(neg_features, (neg_features.shape[0], -1))

    # prima antrenare - inainte de hard negative mining
    x = np.concatenate((pos_feats, neg_feats), axis=0)
    y = np.concatenate((np.ones(pos_feats.shape[0]), np.zeros(neg_feats.shape[0])))
    
    print(f"Antrenare prima etapa ({target})...")
    model = facial_detector.train_classifier(x, y, target)
    facial_detector.best_model = model 

    # hard negative mining
    if params.use_hard_mining:
        print(f"Cautam Hard Negatives pentru {target}...")
        hard_negatives = facial_detector.get_hard_negative_descriptors()
        
        if len(hard_negatives) > 0:
            # extrag caracteristicile exemplelor negative (care aveau scor > 0) și re-antrenez
            hard_neg_feats = np.reshape(hard_negatives, (hard_negatives.shape[0], -1))
            x = np.concatenate((x, hard_neg_feats), axis=0)
            y = np.concatenate((y, np.zeros(hard_neg_feats.shape[0])))
            
            print(f"Re-antrenam {target} cu {len(hard_neg_feats)} exemple...")
            model = facial_detector.train_classifier(x, y, target)

print("\n Toate modelele au fost antrenate cu succes.")

# --- PASUL 2: Detectie si evaluare ---

print("\n Pornim detectia Task 2 pe imaginile de test...")
results_all = facial_detector.run_task2() 

task2_config = {
    'shaggy': params.path_annotationstask2_shaggy,
    'fred': params.path_annotationstask2_fred,
    'velma': params.path_annotationstask2_velma,
    'daphne': params.path_annotationstask2_daphne
}

for char_name, gt_path in task2_config.items():
    print(f"\n Rezultate finale pentru: {char_name.upper()}")
    
    dets = np.array(results_all[char_name]['dets'])
    scores = np.array(results_all[char_name]['scores'])
    files = np.array(results_all[char_name]['files'])
    
    # am comentat pentru a nu a mai sta la fiecare detectie sa fie afisate
    
    # if len(dets) > 0:
    #     facial_detector.eval_detections(dets, scores, files, gt_path, char_name)
        
    #     params.path_annotations = gt_path
    #     show_detections_with_ground_truth(dets, scores, files, params)
        
solutie_path = os.path.join('..', 'fisiere_solutie', 'task2')

if not os.path.exists(solutie_path):
    os.makedirs(solutie_path)
    print(f"Director creat: {solutie_path}")

for char in ['shaggy', 'fred', 'velma', 'daphne']:
    # Extragem datele din detectie
    dets = np.array(results_all[char]['dets'])
    scores = np.array(results_all[char]['scores'])
    files = np.array(results_all[char]['files'])
    
    # salvare fisiere npy pt scriptu de evaluare
    print(f"Salvam solutia pentru {char}...")
    np.save(os.path.join(solutie_path, f"detections_{char}.npy"), dets)
    np.save(os.path.join(solutie_path, f"scores_{char}.npy"), scores)
    np.save(os.path.join(solutie_path, f"file_names_{char}.npy"), files)

print(f" Toate cele 12 fisiere .npy au fost salvate in: {solutie_path}")