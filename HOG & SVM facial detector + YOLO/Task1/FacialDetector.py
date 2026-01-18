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

    def get_positive_descriptors(self):
        """
        Extrage descriptorii HOG pentru exemplele pozitive (fete cropate 36x36).
        """
        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        
        print(f'Calculam descriptorii pt {num_images} imagini pozitive...')
        for i in range(num_images):
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            
            # Calculam HOG: fereastra 36x36, celula 6x6 => 5x5 blocuri de 2x2 celule.
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)
            positive_descriptors.append(features)
            
            # Data Augmentation: Oglindire orizontala.
            if self.params.use_flip_images:
                features_flipped = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                       cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features_flipped)
            
            # 2. Rotire patchuri
            center = (self.params.dim_window // 2, self.params.dim_window // 2)
            for angle in [-5, 5]:
                M = cv.getRotationMatrix2D(center, angle, 1.0)
                img_rotated = cv.warpAffine(img, M, (self.params.dim_window, self.params.dim_window))
                features_rot = hog(img_rotated, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                   cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features_rot)

            # 3. Ajustare Luminozitate
            for alpha in [0.7, 1.3]: # 0.7 = mai întunecat, 1.3 = mai luminos
                img_bright = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                features_bright = hog(img_bright, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                      cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features_bright)

        return np.array(positive_descriptors)

    def get_negative_descriptors(self):
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

    
        return np.array(negative_descriptors)
    
    def train_classifier(self, training_examples, train_labels):
        """
        Antreneaza un SVM liniar si il salveaza.
        """
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model.pkl')
        
        if os.path.exists(svm_file_name) and not self.params.use_hard_mining:
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        print(f'Antrenam clasificatorul pe {len(training_examples)} exemple...')
        model = LinearSVC(C=0.01)
        model.fit(training_examples, train_labels)
        
        self.best_model = model
        pickle.dump(model, open(svm_file_name, 'wb'))
        
        acc = model.score(training_examples, train_labels)
        print(f'Acuratete pe setul de antrenare: {acc:.4f}')

    def get_hard_negative_descriptors(self):
        """
        Etapa 5 (Optionala): Cauta ferestrele din imaginile negative pe care SVM le clasifica gresit (scor > 0).
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
                    
                    if score > 0:
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
                        if self.intersection_over_union(sorted_detections[i], sorted_detections[j]) > 0.10:
                            is_maximal[j] = False
        
        return sorted_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        """
        Implementare detecție prin piramida de imagini (multi-scale).
        """
        test_files = glob.glob(os.path.join(self.params.dir_test_examples, '*.jpg'))
        detections, scores, file_names = [], [], []
        
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        
        for i, path in enumerate(test_files):
            print(f'Procesam imaginea {i+1}/{len(test_files)}: {ntpath.basename(path)}')
            img_orig = cv.imread(path, cv.IMREAD_GRAYSCALE)
            
            current_image_detections = []
            current_image_scores = []
            
            scale = 1.0
            scaling_factor = 0.8
            
            while img_orig.shape[0] * scale >= self.params.dim_window and \
                  img_orig.shape[1] * scale >= self.params.dim_window:
                
                img_scaled = cv.resize(img_orig, (0, 0), fx=scale, fy=scale)
                
                # ATENTIE: Folosim cells_per_block=(2, 2) pentru a se potrivi cu modelul antrenat!
                hog_descriptors = hog(img_scaled, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                      cells_per_block=(2, 2), feature_vector=False)
                
                num_cell_template = self.params.dim_window // self.params.dim_hog_cell - 1
                
                for y in range(hog_descriptors.shape[0] - num_cell_template):
                    for x in range(hog_descriptors.shape[1] - num_cell_template):
                        descr = hog_descriptors[y:y + num_cell_template, x:x + num_cell_template].flatten()
                        score = np.dot(descr, w)[0] + bias
                        
                        if score > self.params.threshold:
                            x_min = int(x * self.params.dim_hog_cell / scale)
                            y_min = int(y * self.params.dim_hog_cell / scale)
                            x_max = int((x * self.params.dim_hog_cell + self.params.dim_window) / scale)
                            y_max = int((y * self.params.dim_hog_cell + self.params.dim_window) / scale)
                            
                            current_image_detections.append([x_min, y_min, x_max, y_max])
                            current_image_scores.append(score)
                
                scale *= scaling_factor

            if len(current_image_scores) > 0:
                det, sc = self.non_maximal_suppression(np.array(current_image_detections), 
                                                        np.array(current_image_scores), img_orig.shape)
                detections.extend(det)
                scores.extend(sc)
                file_names.extend([ntpath.basename(path)] * len(sc))

        return np.array(detections), np.array(scores), np.array(file_names)

    def eval_detections(self, detections, scores, file_names):
        """
        Evalueaza performanta folosind Precizie-Recall.
        """
        ground_truth = np.loadtxt(self.params.path_annotations, dtype='str')
        gt_names = ground_truth[:, 0]
        gt_bboxes = ground_truth[:, 1:].astype(int)
        
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
            
            if max_ov >= 0.3:
                if gt_detected[max_idx] == 0:
                    tp[i] = 1
                    gt_detected[max_idx] = 1
                else: fp[i] = 1
            else: fp[i] = 1

        rec = np.cumsum(tp) / num_gt
        prec = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
        
        ap = np.trapz(prec, rec)
        
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'Average Precision: {ap:.3f}')
        plt.show()