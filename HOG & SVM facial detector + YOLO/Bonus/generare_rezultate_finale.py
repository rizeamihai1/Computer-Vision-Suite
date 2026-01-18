import numpy as np
import os
import cv2 as cv
from ultralytics import YOLO

# --- CONFIGURARE CĂI ---
base_dir = '../data'
validare_dir = '../validare' # De modificat cu testare!!
MODEL_PATH = r'runs\detect\detectie_scooby2\weights\best.pt'
IMAGES_PATH = os.path.join(validare_dir, 'validare') 
base_dir_sol = '../fisiere_solutie'
SOLUTIE_BASE = os.path.join(base_dir_sol, 'bonus')

class_mapping = {
    0: 'shaggy',
    1: 'fred',
    2: 'velma',
    3: 'daphne'
}

def generate_npy_yolo():
    model = YOLO(MODEL_PATH)
    
    print(f"Maparea claselor din model: {model.names}")
    results = model(IMAGES_PATH, conf=0.01, device=0) 
    
    task1_data = {'dets': [], 'scores': [], 'names': []}
    
    task2_data = {char: {'dets': [], 'scores': [], 'names': []} for char in class_mapping.values()}

    for r in results:
        fname = os.path.basename(r.path)
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy() 
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            char_name = class_mapping.get(cls_id)
            if char_name is None:
                continue

            task1_data['dets'].append(coords)
            task1_data['scores'].append(conf)
            task1_data['names'].append(fname)

            task2_data[char_name]['dets'].append(coords)
            task2_data[char_name]['scores'].append(conf)
            task2_data[char_name]['names'].append(fname)

    def safe_save(folder, prefix, dets, scores, names):
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
        final_dets = np.array(dets) if len(dets) > 0 else np.zeros((0, 4))
        final_scores = np.array(scores)
        final_names = np.array(names)
        
        np.save(os.path.join(folder, f"detections_{prefix}.npy"), final_dets)
        np.save(os.path.join(folder, f"scores_{prefix}.npy"), final_scores)
        np.save(os.path.join(folder, f"file_names_{prefix}.npy"), final_names)

    print("Salvam fisierele pentru Task 1...")
    save_dir_t1 = os.path.join(SOLUTIE_BASE, 'task1')
    safe_save(save_dir_t1, 'all_faces', task1_data['dets'], task1_data['scores'], task1_data['names'])

    print("Salvam fisierele pentru Task 2...")
    save_dir_t2 = os.path.join(SOLUTIE_BASE, 'task2')
    for char in class_mapping.values():
        print(f" -> {char}: {len(task2_data[char]['dets'])} detectii")
        safe_save(save_dir_t2, char, task2_data[char]['dets'], task2_data[char]['scores'], task2_data[char]['names'])

    print(f"Fisierele .npy au fost salvate corect in {SOLUTIE_BASE}")

if __name__ == '__main__':
    generate_npy_yolo()