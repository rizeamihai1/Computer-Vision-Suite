import os
import cv2 as cv

path_adnotari = 'adnotari.txt'
path_imagini = 'data/images'
output_labels_dir = 'dataset/labels'
os.makedirs(output_labels_dir, exist_ok=True)

class_map = {
    'shaggy': 0,
    'fred': 1,
    'velma': 2,
    'daphne': 3,
    'unknown': 4
}

def convert_to_yolo():
    with open(path_adnotari, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6: continue
        
        img_name = parts[0]
        xmin, ymin, xmax, ymax = map(float, parts[1:5])
        character = parts[5].lower()
        
        if character not in class_map: continue
        class_id = class_map[character]

        img_path = os.path.join(path_imagini, img_name)
        img = cv.imread(img_path)
        if img is None:
            print(f"Eroare: Nu am gasit imaginea {img_path}")
            continue
        
        h, w, _ = img.shape

        x_center = (xmin + xmax) / (2.0 * w)
        y_center = (ymin + ymax) / (2.0 * h)
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        label_file = os.path.join(output_labels_dir, img_name.replace('.jpg', '.txt'))
        with open(label_file, 'a') as lf:
            lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

convert_to_yolo()