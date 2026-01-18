from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog

class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None

    def get_positive_descriptors(self, folder_path):
        """Extrage descriptorii HOG dintr-un folder specific."""
        files = glob.glob(os.path.join(folder_path, '*.jpg'))
        descriptors = []
        
        print(f"Extragem descriptori pozitivi din: {folder_path}")
        for f in files:
            img = cv.imread(f, cv.IMREAD_GRAYSCALE)
            if img is None: continue
            if img.shape[0] != 36 or img.shape[1] != 36:
                img = cv.resize(img, (36, 36))
            
            # HOG Standard
            descr = hog(img, pixels_per_cell=(6, 6), cells_per_block=(2, 2), feature_vector=True)
            descriptors.append(descr)
            
            # Data Augmentation: Oglindire orizontala.
            if self.params.use_flip_images:
                features_flipped = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                       cells_per_block=(2, 2), feature_vector=True)
                descriptors.append(features_flipped)
            
            # 2. Rotire patchuri
            center = (self.params.dim_window // 2, self.params.dim_window // 2)
            for angle in [-5, 5]:
                M = cv.getRotationMatrix2D(center, angle, 1.0)
                img_rotated = cv.warpAffine(img, M, (self.params.dim_window, self.params.dim_window))
                features_rot = hog(img_rotated, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                   cells_per_block=(2, 2), feature_vector=True)
                descriptors.append(features_rot)

            # 3. Ajustare Luminozitate
            for alpha in [0.7, 1.3]: # 0.7 = mai întunecat, 1.3 = mai luminos
                img_bright = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                features_bright = hog(img_bright, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                      cells_per_block=(2, 2), feature_vector=True)
                descriptors.append(features_bright)
                
            # 4. Ajustare Contrast
            for beta in [0.7, 1.3]: # 0.7 = contrast mai mic, 1.3 = contrast mai mare
                img_contrast = cv.convertScaleAbs(img, alpha=beta, beta=0)
                features_contrast = hog(img_contrast, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                        cells_per_block=(2, 2), feature_vector=True)
                descriptors.append(features_contrast)

        return np.array(descriptors)
    
    def get_negative_descriptors(self, extra_folders=None):
        """
        Extrage negative din fundal si din folderele celorlalte personaje.
        Daca o imagine este prea mica, o redimensionam la 36x36.
        """
        negative_descriptors = []
        dim = self.params.dim_window  # 36
        
        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        num_negative_per_image = self.params.number_negative_examples // num_images

        print(f'Procesare exemple negative din fundal ({num_images} imagini)...')
        for i in range(num_images):
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            if img is None: continue
            h, w = img.shape

            # Pentru imagini mai mici de 36x36, redirimensionam la 36x36
            if h < dim or w < dim:
                img = cv.resize(img, (dim, dim))
                h, w = img.shape

             # Daca e fix 36x36, extragem direct HOG
            if h == dim and w == dim:
                descr = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                            cells_per_block=(2, 2), feature_vector=True)
                negative_descriptors.append(descr)
            else:
                # Extragem patch-uri aleatorii din imaginile mai mari
                for _ in range(num_negative_per_image):
                    y = np.random.randint(low=0, high=h - dim + 1)
                    x = np.random.randint(low=0, high=w - dim + 1)
                    patch = img[y: y + dim, x: x + dim]
                    descr = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                cells_per_block=(2, 2), feature_vector=True)
                    negative_descriptors.append(descr)

        # Acum adaugam pentru fiecare personaj, celelate personaje ca si exemplu negativ
        if extra_folders:
            print(f"Adaugam negative din {len(extra_folders)} folderele celorlalte personaje...")
            for folder in extra_folders:
                extra_files = glob.glob(os.path.join(folder, '*.jpg'))
                for f in extra_files:
                    img = cv.imread(f, cv.IMREAD_GRAYSCALE)
                    if img is None: 
                        continue
                    
                    if img.shape[0] != dim or img.shape[1] != dim:
                        img = cv.resize(img, (dim, dim))
                        
                    descr = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                cells_per_block=(2, 2), feature_vector=True)
                    negative_descriptors.append(descr)

        return np.array(negative_descriptors)
    
    def train_classifier(self, training_examples, train_labels, character_name):
        """Antreneaz SVM-ul si il salvez cu numele personajului."""
        svm_file_name = os.path.join(self.params.dir_save_files, f'best_model_{character_name}.pkl')
        
        print(f'Antrenam clasificatorul pentru {character_name} pe {len(training_examples)} exemple...')
        model = LinearSVC(C=1.0, class_weight='balanced') # fata de task1, folosim class_weight pentru dezechilibru clase
        model.fit(training_examples, train_labels)
        
        pickle.dump(model, open(svm_file_name, 'wb'))
        acc = model.score(training_examples, train_labels)
        print(f'Acuratete {character_name}: {acc:.4f}')
        return model

    def get_hard_negative_descriptors(self):
        """
        Caut ferestrele din imaginile negative pe care SVM le clasifica gresit (scor > 0) - fals pozitive
        """
        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        hard_negatives = []
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        
        print("Cautam Hard Negatives...")
        for i in range(len(files)):
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            
            hog_descriptors = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                  cells_per_block=(2, 2), feature_vector=False)
            
            num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1
            for y in range(hog_descriptors.shape[0] - num_cell_in_template):
                for x in range(hog_descriptors.shape[1] - num_cell_in_template):
                    descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                    score = np.dot(descr, w)[0] + bias
                    
                    if score > 0: # le adaug manual inca o data in lista de negative
                        hard_negatives.append(descr)
            
                
        return np.array(hard_negatives)

    def intersection_over_union(self, bbox_a, bbox_b):
        """
        Calculeaza gradul de suprapunere intre doua ferestre.
        """
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        return inter_area / float(box_a_area + box_b_area - inter_area)

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Elimina detectiile redundante care se suprapun.
        """
        if len(image_detections) == 0:
            return image_detections, image_scores

        image_detections[:, [0, 2]] = np.clip(image_detections[:, [0, 2]], 0, image_size[1])
        image_detections[:, [1, 3]] = np.clip(image_detections[:, [1, 3]], 0, image_size[0])

        sorted_indices = np.argsort(image_scores)[::-1]
        sorted_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(sorted_detections), dtype=bool)
        for i in range(len(sorted_detections)):
            if is_maximal[i]:
                for j in range(i + 1, len(sorted_detections)):
                    if is_maximal[j]:
                        if self.intersection_over_union(sorted_detections[i], sorted_detections[j]) > 0.15:
                            is_maximal[j] = False
        
        return sorted_detections[is_maximal], sorted_scores[is_maximal]

    def run_task2(self):
        """
        Detectie multi-scale pentru Task 2 cu validare prin culoarea părului.
        """
        
        test_files = glob.glob(os.path.join(self.params.dir_test_examples, '*.jpg'))
        characters = ['shaggy', 'fred', 'velma', 'daphne']
        
        # Voi filtra detectiile finale folosind culoare parului pentru fiecare personaj
        color_profiles = {
            'shaggy': {'low': np.array([10, 40, 40]),   'high': np.array([25, 255, 180])},
            'fred':   {'low': np.array([20, 40, 100]),  'high': np.array([35, 255, 255])},
            'velma':  {'low': np.array([0, 50, 20]),    'high': np.array([20, 255, 100])},
            'daphne': {'low': np.array([5, 100, 100]), 'high': np.array([18, 255, 255])}
        }

        weights = {}
        biases = {}
        
        print("Incarcam modelele pentru Task 2...")
        for char in characters:
            model_path = os.path.join(self.params.dir_save_files, f'best_model_{char}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    weights[char] = model.coef_.T
                    biases[char] = model.intercept_[0]
            else:
                print(f"ATENTIE: Modelul pentru {char} nu a fost gasit la {model_path}!")
                # Daca lipseste un model, punem valori vide pentru a nu crapa bucla de dot product
                weights[char] = np.zeros((self.params.dim_descriptor_cell * 49, 1)) # Ajusteaza conform lungimii HOG
                biases[char] = -100 

        all_results = {char: {'dets': [], 'scores': [], 'files': []} for char in characters}

        # --- PASUL 3: PROCESAREA IMAGINILOR ---
        for i, path in enumerate(test_files):
            fname = ntpath.basename(path)
            print(f'Procesam imaginea {i+1}/{len(test_files)}: {fname}')
            
            img_orig = cv.imread(path, cv.IMREAD_GRAYSCALE)
            img_color = cv.imread(path) # ma folosesc si de imaginea color pt pasul de la final
            
            img_dets = {c: [] for c in characters}
            img_scores = {c: [] for c in characters}
            
            scale = 1.0
            while img_orig.shape[0] * scale >= 36 and img_orig.shape[1] * scale >= 36:
                img_scaled = cv.resize(img_orig, (0, 0), fx=scale, fy=scale)
                hog_descriptors = hog(img_scaled, pixels_per_cell=(6, 6), 
                                      cells_per_block=(2, 2), feature_vector=False)
                num_cell = 36 // 6 - 1
                
                for y in range(hog_descriptors.shape[0] - num_cell):
                    for x in range(hog_descriptors.shape[1] - num_cell):
                        descr = hog_descriptors[y:y + num_cell, x:x + num_cell].flatten()
                        
                        best_score = -np.inf
                        best_char = None
                        
                        # calculez scorul pentru fiecare personaj
                        for c in characters:
                            score = np.dot(descr, weights[c])[0] + biases[c]
                            if score > self.params.threshold and score > best_score:
                                best_score = score
                                best_char = c
                        
                        # daca am gasit un personaj cu scor peste prag, validam culoarea parului
                        if best_char:
                            x_min = int(x * 6 / scale)
                            y_min = int(y * 6 / scale)
                            x_max = int((x * 6 + 36) / scale)
                            y_max = int((y * 6 + 36) / scale)
                            
                            patch_color = img_color[y_min:y_max, x_min:x_max]
                            
                            if patch_color.size > 0:
                                hair_zone = patch_color[0:int(patch_color.shape[0] * 0.35), :]
                                hsv_hair = cv.cvtColor(hair_zone, cv.COLOR_BGR2HSV)
                                
                                mask = cv.inRange(hsv_hair, color_profiles[best_char]['low'], 
                                                            color_profiles[best_char]['high'])
                                
                                pixel_ratio = np.sum(mask > 0) / mask.size
                                
                                # daca nu are pixeli de culoarea parului, ignor detectia
                                if pixel_ratio > 0.08: 
                                    img_dets[best_char].append([x_min, y_min, x_max, y_max])
                                    img_scores[best_char].append(best_score)
                
                scale *= 0.9 # redimensionare pt sliding window

            # NMS per personaj
            for c in characters:
                if len(img_scores[c]) > 0:
                    d, s = self.non_maximal_suppression(np.array(img_dets[c]), np.array(img_scores[c]), img_orig.shape)
                    all_results[c]['dets'].extend(d)
                    all_results[c]['scores'].extend(s)
                    all_results[c]['files'].extend([fname] * len(s))
                    
        return all_results

    def eval_detections(self, detections, scores, file_names, gt_path, character_name):
        '''
        functie de vizualizare a detectiilor impreuna cu ground truth-urile
        '''
        ground_truth = np.loadtxt(gt_path, dtype='str')
        
        if ground_truth.ndim == 1:
            ground_truth = ground_truth.reshape(1, -1)
            
        gt_names = ground_truth[:, 0]
        gt_bboxes = ground_truth[:, 1:5].astype(float)
        
        num_gt = len(gt_bboxes)
        gt_detected = np.zeros(num_gt)

        idx = np.argsort(scores)[::-1]
        detections, scores, file_names = detections[idx], scores[idx], file_names[idx]

        tp, fp = np.zeros(len(detections)), np.zeros(len(detections))
        for i in range(len(detections)):
            relevant_gt_idx = np.where(gt_names == file_names[i])[0]
            max_ov, max_idx = -1, -1
            
            for g_idx in relevant_gt_idx:
                ov = self.intersection_over_union(detections[i], gt_bboxes[g_idx])
                if ov > max_ov:
                    max_ov, max_idx = ov, g_idx
            
            # Pragul de IoU 0.3
            if max_ov >= 0.3:
                if gt_detected[max_idx] == 0:
                    tp[i] = 1
                    gt_detected[max_idx] = 1
                else: fp[i] = 1
            else: fp[i] = 1

        rec = np.cumsum(tp) / num_gt
        prec = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
        
        ap = np.trapz(prec, rec)
        plt.figure()
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'PR {character_name.upper()}: AP = {ap:.3f}')
        plt.show()