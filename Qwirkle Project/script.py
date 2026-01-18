import cv2
import numpy as np
import argparse
import sys
import os
import glob

"""
cv2 (OpenCV) version: 4.11.0 
NumPy version: 2.2.3 
Python version: 3.10.9

Pentru a rula (pentru fiecare folder cu jocuri):
python script.py -i <input_folder_name> -o <output_folder_name> -t <template_folder_name>
"""

# =============================================================================
#  SECTIUNEA 1: EXTRAGEREA CADRANULUI (PRE-PROCESARE TABLA)
# =============================================================================

def normalizeaza_iluminarea(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
    l_normalized = clahe.apply(l_channel) 
    normalized_lab_image = cv2.merge((l_normalized, a_channel, b_channel))
    normalized_bgr_image = cv2.cvtColor(normalized_lab_image, cv2.COLOR_LAB2BGR)
    return normalized_bgr_image

def gaseste_contur_tabla(normalized_image):
    # Gasesc masca pentru a izola tabla de restul imaginii (ma folosesc de culoarea verde cu intervalele HSV)
    hsv_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV) 
    lower_green = np.array([30, 25, 25]) 
    upper_green = np.array([90, 255, 255]) 
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    kernel_close = np.ones((35, 35), np.uint8) 
    mask_closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_close, iterations=4)
    
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return None
    board_contour = max(contours, key=cv2.contourArea)
    return board_contour

def ordoneaza_colturi(pts):
    # functie sa sortez punctele tablei
    puncte = pts.tolist()
    
    puncte.sort(key=lambda p: p[1])
    sus = puncte[:2]
    jos = puncte[2:]
    sus.sort(key=lambda p: p[0])
    jos.sort(key=lambda p: p[0])
    
    return np.array([sus[0], sus[1], jos[1], jos[0]], dtype="float32")

def gaseste_colturi_tabla(normalized_image, board_contour):
    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(board_contour)
    padding = 10
    y_start_roi = max(0, y_roi - padding)
    y_end_roi = min(normalized_image.shape[0], y_roi + h_roi + padding)
    x_start_roi = max(0, x_roi - padding)
    x_end_roi = min(normalized_image.shape[1], x_roi + w_roi + padding)
    
    roi_image = normalized_image[y_start_roi:y_end_roi, x_start_roi:x_end_roi]
    hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    
    lower_green_strict = np.array([35, 50, 50])
    upper_green_strict = np.array([85, 255, 255])
    clean_mask = cv2.inRange(hsv_roi, lower_green_strict, upper_green_strict)
    
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    clean_gray_roi = cv2.bitwise_and(gray_roi, gray_roi, mask=clean_mask)

    binary_grid = cv2.adaptiveThreshold(clean_gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -5)

    grid_line_contours, _ = cv2.findContours(binary_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not grid_line_contours:
        return None
    
    all_grid_points = np.concatenate(grid_line_contours)
    rotated_rect = cv2.minAreaRect(all_grid_points)
    grid_corners = cv2.boxPoints(rotated_rect)
    
    corners = np.array([
        [x_start_roi + pt[0], y_start_roi + pt[1]] for pt in grid_corners
    ], dtype="float32")
    
    return corners

def transforma_perspectiva_tabla(image, sorted_corners, output_size_wh=(800, 800)):
    output_width, output_height = output_size_wh
    destination_points = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(sorted_corners, destination_points)
    warped_image = cv2.warpPerspective(image, matrix, (output_width, output_height))
    return warped_image

def proceseaza_extragere_tabla(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    
    # 1. Normalizare
    normalized_image = normalizeaza_iluminarea(image)
    
    # 2. Contur General
    board_contour_roi = gaseste_contur_tabla(normalized_image)
    if board_contour_roi is None: return None
    
    # 3. Colturile tablei
    corners = gaseste_colturi_tabla(normalized_image, board_contour_roi)
    if corners is None: return None
    
    # 4. wrap si resize la 800x800
    sorted_corners = ordoneaza_colturi(corners)
    warped_board = transforma_perspectiva_tabla(normalized_image, sorted_corners)
    
    return warped_board


# =============================================================================
#  SECTIUNEA 2: PROCESAREA IMAGINII DE START (CONFIGURATIA INITIALA)
# =============================================================================

def determina_configuratia_initiala(warped_board):
    board_grid = np.zeros((16, 16), dtype=int) 
    quadrants = [(0, 0), (0, 8), (8, 0), (8, 8)]
    cell_size = 50
    padding = 15 

    for r_offset, c_offset in quadrants:
        val_diag_principala = [] # Diagonala (1,1) -> (6,6)
        val_diag_secundara = []  # Diagonala (1,6) -> (6,1)
        
        coords_principala = []
        coords_secundara = []

        for i in range(1, 7):
            r1, c1 = r_offset + i, c_offset + i
            y1, x1 = r1 * cell_size, c1 * cell_size
            roi1 = warped_board[y1+padding : y1+cell_size-padding, x1+padding : x1+cell_size-padding]
            
            if roi1.size > 0:
                hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
                mean_brightness1 = np.mean(hsv1[:, :, 2])
            else:
                mean_brightness1 = 255.0
            
            val_diag_principala.append(mean_brightness1)
            coords_principala.append((r1, c1))

            r2, c2 = r_offset + i, c_offset + (7 - i)
            y2, x2 = r2 * cell_size, c2 * cell_size
            roi2 = warped_board[y2+padding : y2+cell_size-padding, 
                                x2+padding : x2+cell_size-padding]
            
            if roi2.size > 0:
                hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
                mean_brightness2 = np.mean(hsv2[:, :, 2])
            else:
                mean_brightness2 = 255.0
                
            val_diag_secundara.append(mean_brightness2)
            coords_secundara.append((r2, c2))

        avg_b1 = np.mean(val_diag_principala)
        avg_b2 = np.mean(val_diag_secundara)

        if avg_b1 < avg_b2:
            winning_coords = coords_principala
        else:
            winning_coords = coords_secundara

        for (rr, cc) in winning_coords:
            board_grid[rr, cc] = 1 

    return board_grid

# =============================================================================
#  SECTIUNEA 3: GASIREA POZITIILOR PIESELOR
# =============================================================================

def detecteaza_schimbari_intre_imagini(current_warped, previous_warped):
    board_grid = np.zeros((16, 16), dtype=int)
    cell_size_px = 50
    
    HIGH_THRESHOLD = 70 # e direct piesa
    LOW_THRESHOLD = 60 # poate e piesa sau s-a miscat telefonu cand s-a facut poza si compar cu diferenta intre media patchurilor asa
    
    diff_image = cv2.absdiff(previous_warped, current_warped)
    diff_image = cv2.GaussianBlur(diff_image, (23,23), 0)
    
    for row in range(16):
        for col in range(16):
            y = row * cell_size_px
            x = col * cell_size_px
            
            
            diff_patch = diff_image[y:y+cell_size_px, x:x+cell_size_px]
            if diff_patch.size == 0: continue
            
            diff_val = np.mean(diff_patch)
            is_piece = False
            
            val_medie_curenta = 0
            val_medie_veche = 0
            
            if diff_val > HIGH_THRESHOLD:
                is_piece = True
            else:
                if diff_val > LOW_THRESHOLD:
                    val_medie_curenta = np.mean(previous_warped[y:y+cell_size_px, x:x+cell_size_px])
                    val_medie_veche = np.mean(current_warped[y:y+cell_size_px, x:x+cell_size_px])
                    #print(f"MUTARE la [{row}, {col}]: Vechi={val_medie_veche:.2f} | Nou={val_medie_curenta:.2f} | Diff={abs(val_medie_curenta - val_medie_veche):.2f}")
                    
                    # daca e sub 30 diferenta intre medii, e prea putin pentru a fii piesa.
                    if (abs(val_medie_curenta - val_medie_veche) < 30):
                        is_piece = False
                    else:
                        is_piece = True

            if is_piece:
                board_grid[row, col] = 1
    return board_grid, diff_image

def identifica_mutari_noi(current_diff_grid, previous_board_state):
    new_moves_coords = set()
    for r in range(16):
        for c in range(16):
            if current_diff_grid[r, c] == 1 and previous_board_state[r, c] != 1:
                new_moves_coords.add((r, c))
    return new_moves_coords

# =============================================================================
#  SECTIUNEA 4: DETECTIE CULOARE
# =============================================================================

def determina_culoarea_piesei(patch_bgr):
    h_img, w_img = patch_bgr.shape[:2]

    # verific in 5 zone media culorii din acel punct, pentru piesele puse extrem de prost tot gasesc culoarea prin acest 5 puncte
    # primul este centrul patcului, celelalte sunt colturi aproape de margini
    zones = [
        (h_img // 2, w_img // 2),  # centru
        (10, 10),                  # sus-stanga
        (10, 40),                  # sus-dreapta
        (40, 10),                  # jos-stanga
        (40, 40),                  # jos-dreapta
    ]

    for y, x in zones:
        y0, y1 = max(0, y-2), min(h_img, y+3)
        x0, x1 = max(0, x-2), min(w_img, x+3)
        roi = patch_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        
        mean_bgr = np.mean(roi, axis=(0, 1))
        pixel_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(pixel_hsv[0]), int(pixel_hsv[1]), int(pixel_hsv[2])

        if (s < 55) and (v > 110): 
            return "Alb"
        if (21 <= h <= 35) and (60 <= s <= 255) and (100 <= v <= 255): 
            return "Galben"
        if (3 <= h <= 20) and (65 <= s <= 255) and (100 <= v <= 255): 
            return "Portocaliu"
        if (80 <= h <= 135) and (100 <= s <= 255) and (80 <= v <= 255): 
            return "Albastru"
        if (28 <= h <= 79) and (30 <= s <= 255) and (40 <= v <= 220): 
            return "Verde"
        if ((0 <= h <= 10) or (160 <= h <= 180)) and (80 <= s <= 255) and (80 <= v <= 255): 
            return "Rosu"
    return None

# =============================================================================
#  SECTIUNEA 5: DETECTIE FORMA
# =============================================================================

def centreaza_imaginea(thresh_image):
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return thresh_image # Daca e toata imaginea neagra, nu am ce sa centrez
    
    # iau conturul maxim
    c = max(contours, key=cv2.contourArea)
    
    M = cv2.moments(c)
    
    if M["m00"] > 0:
        # Coordonatele centrului de greutate al piesei (CX, CY)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Dimensiunile imaginii
        h, w = thresh_image.shape
        
        # Coordonatele centrului imaginii (unde vrem sa fie plasat outputul)
        img_centerX = w // 2
        img_centerY = h // 2
        
        # calculez deplasarea fata de centru
        tX = img_centerX - cX
        tY = img_centerY - cY
        
        # creez matrice de translatie
        Translation_Matrix = np.float32([[1, 0, tX], [0, 1, tY]])
        
        centered_image = cv2.warpAffine(thresh_image, Translation_Matrix, (w, h), borderValue=0)
        
        return centered_image
    
    return thresh_image

def preproceseaza_pentru_forma(img, target_size=(64, 64)): 
    # Pasul 1: Fac upscale (4x) pentru a se pastra detaliile
    # Exemplu: 50x50 -> devine 200x200
    img_big = cv2.resize(img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    # Pasul 2: Scot V-ul din HSV (Grayscale based on brightness)
    if len(img_big.shape) == 3:
        hsv = cv2.cvtColor(img_big, cv2.COLOR_BGR2HSV)
        gray = hsv[:,:,2] 
    else:
        gray = img_big

    # Pasul 3: Aplic blur si threshold
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # acum am patchul alb negru de dimensiune 200x200, dar pe margini inca este zgomot, si l elimin direct taiind margini din patch
    P = 10
    ##cv2.imshow("Thresh Forma Inainte Crop", thresh) 
    # Obținem dimensiunile imaginii curente (H, W)
    h, w = thresh.shape
    
    # Facem slicing :
    # De la P pana la H-P pe verticala
    # De la P pana la W-P pe orizontala
    # pentru a scoate zgomotul de pe margini 
    
    thresh_centered = centreaza_imaginea(thresh)
    ##cv2.imshow("Thresh Forma Centrat", thresh_centered)
    thresh_cropped = thresh_centered[P : h-P, P : w-P]
    
    ##cv2.imshow("Thresh Forma Dupa Crop", thresh_cropped)
    # --- PASUL 5: Resize final la target_size (ex: 64x64) ---
    # pentru a putea compara cu template-urile, aducem rezultatul la dimensiunea standard
    result = cv2.resize(thresh_cropped, target_size, interpolation=cv2.INTER_AREA)
    return result

def incarca_sabloane_forme(folder_path):
    shapes = ["cerc", "patrat", "romb", "stea_4", "stea_8", "trifoi"]
    templates = {shape: [] for shape in shapes}
    extensie =  ".png"
    
    for shape in shapes:
        path = None
        p = os.path.join(folder_path, shape + extensie)
        if os.path.exists(p):
            img = cv2.imread(p)
            processed = preproceseaza_pentru_forma(img)
            templates[shape].append(processed)
    return templates

def roteste_imaginea(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def identifica_forma_piesei(patch_colorat, templates_dict):
    # 1. Preprocesare
    processed_patch = preproceseaza_pentru_forma(patch_colorat)
    
    if len(processed_patch.shape) == 3:
        processed_patch = cv2.cvtColor(processed_patch, cv2.COLOR_BGR2GRAY)
    processed_patch = processed_patch.astype(np.uint8)

    # aplic tempalteMatching cu SQDIFF, si caut valoarea minima -> (0 = perfect)
    best_score = 1.0 
    best_match = ""
    # rotesc pathcurile in cazul in care piesele au fost puse prost
    for angle in range(-3, 4, 1): 
        rotated_patch = roteste_imaginea(processed_patch, angle)
        
        for shape_name, template_list in templates_dict.items():
            for template in template_list:
                
                if rotated_patch.shape != template.shape:
                    template = cv2.resize(template, (rotated_patch.shape[1], rotated_patch.shape[0]))

                res = cv2.matchTemplate(rotated_patch, template, cv2.TM_SQDIFF_NORMED)
                min_val, _, _, _ = cv2.minMaxLoc(res)
                score = min_val 

                if score < best_score:
                    best_score = score
                    best_match = shape_name
                    
                    if best_score < 0.02: 
                        return best_match

    return best_match
# =============================================================================
#  SECTIUNEA 6: LOGICA DE JOC (BONUSURI SI SCOR)
# =============================================================================

def genereaza_matrice_bonus(board_config_initial):
    bonus_matrix = np.zeros((16, 16), dtype=int)
    quadrants = [(0, 0), (0, 8), (8, 0), (8, 8)]
    
    for r_offset, c_offset in quadrants:
        has_main_diag = False
        
        if board_config_initial[r_offset + 1, c_offset + 1] == 1:
            has_main_diag = True
            
        if has_main_diag:
            bonus_matrix[r_offset + 1, c_offset + 6] = 2
            bonus_matrix[r_offset + 6, c_offset + 1] = 2
            
            for i in range(1, 6):
                bonus_matrix[r_offset + i, c_offset + i + 1] = 1
                bonus_matrix[r_offset + i + 1, c_offset + i] = 1
                
        else:
            bonus_matrix[r_offset + 1, c_offset + 1] = 2
            bonus_matrix[r_offset + 6, c_offset + 6] = 2
            
            for i in range(1, 7): 
                r_local = i
                c_local = 7 - i
                if c_local - 1 >= 1:
                    bonus_matrix[r_offset + r_local, c_offset + c_local - 1] = 1
                
                if r_local + 1 <= 6:
                    bonus_matrix[r_offset + r_local + 1, c_offset + c_local] = 1

    return bonus_matrix

def calculeaza_scor_tura(mutari_noi_coords, board_state, bonus_matrix):
    scor_total = 0
    
    if not mutari_noi_coords:
        return 0, bonus_matrix

    for (r, c) in mutari_noi_coords:
        val_bonus = bonus_matrix[r, c]
        if val_bonus > 0:
            scor_total += val_bonus
            bonus_matrix[r, c] = -1

    rows = [m[0] for m in mutari_noi_coords]
    cols = [m[1] for m in mutari_noi_coords]
    
    is_horizontal = len(set(rows)) == 1
    is_vertical = len(set(cols)) == 1
    
    def get_line_length(start_r, start_c, dr, dc):
        length = 0
        curr_r, curr_c = start_r, start_c
        while 0 <= curr_r < 16 and 0 <= curr_c < 16 and board_state[curr_r, curr_c] == 1:
            length += 1
            curr_r += dr
            curr_c += dc
        
        curr_r, curr_c = start_r - dr, start_c - dc
        while 0 <= curr_r < 16 and 0 <= curr_c < 16 and board_state[curr_r, curr_c] == 1:
            length += 1
            curr_r -= dr
            curr_c -= dc
            
        return length

    if len(mutari_noi_coords) > 0:
        r0, c0 = mutari_noi_coords[0]
        if len(mutari_noi_coords) == 1:
            len_h = get_line_length(r0, c0, 0, 1)
            len_v = get_line_length(r0, c0, 1, 0)
            
            if len_h > 1: 
                scor_total += len_h
                if len_h == 6: 
                    scor_total += 6
                
            
            if len_v > 1:
                scor_total += len_v
                if len_v == 6: 
                    scor_total += 6
                
        else:
            if is_horizontal:
                len_main = get_line_length(r0, c0, 0, 1)
                if len_main > 1:
                    scor_total += len_main
                    if len_main == 6: 
                        scor_total += 6
                    
                for (r, c) in mutari_noi_coords:
                    len_sec = get_line_length(r, c, 1, 0)
                    if len_sec > 1:
                        scor_total += len_sec
                        if len_sec == 6: 
                            scor_total += 6

            elif is_vertical:
                len_main = get_line_length(r0, c0, 1, 0)
                if len_main > 1:
                    scor_total += len_main
                    if len_main == 6: 
                        scor_total += 6

                for (r, c) in mutari_noi_coords:
                    len_sec = get_line_length(r, c, 0, 1)
                    if len_sec > 1:
                        scor_total += len_sec
                        if len_sec == 6: 
                            scor_total += 6

    return scor_total, bonus_matrix

# =============================================================================
#  VIZUALIZARE SI MAIN
# =============================================================================

def deseneaza_analiza_pe_imagine(warped_board, board_state_grid, detected_pieces_info):
    analysis_image = warped_board.copy()
    cell_size_px = 50
    new_moves_coords = set([(d['r'], d['c']) for d in detected_pieces_info])
    
    for r in range(16):
        for c in range(16):
            y = r * cell_size_px
            x = c * cell_size_px
            state = board_state_grid[r, c]
            
            if state == 1:
                if (r, c) in new_moves_coords:
                    draw_color = (0, 255, 0) 
                    cv2.rectangle(analysis_image, (x+2, y+2), (x+cell_size_px-2, y+cell_size_px-2), draw_color, 2)
                    info = next((item for item in detected_pieces_info if item["r"] == r and item["c"] == c), None)
                    if info:
                        cv2.putText(analysis_image, f"{info['color']}", (x+2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
                        cv2.putText(analysis_image, f"{info['shape']}", (x+2, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
                else:
                    draw_color = (0, 0, 200) 
                    cv2.rectangle(analysis_image, (x+2, y+2), (x+cell_size_px-2, y+cell_size_px-2), draw_color, 1)
            elif state == 2: # DIAGONALA
                draw_color = (255, 255, 0) 
                cv2.circle(analysis_image, (x + 25, y + 25), 5, draw_color, -1)
    return analysis_image
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-t", "--templates_folder", required=True)
    
    args = vars(parser.parse_args())
    input_folder = args["input_folder"]
    output_folder = args["output_folder"]
    templates_folder = args["templates_folder"]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    COL_LETTERS = "ABCDEFGHIJKLMNOP" 
    SHAPE_CODES = {
        "cerc": "1", "trifoi": "2", "romb": "3", 
        "patrat": "4", "stea_4": "5", "stea_8": "6",
        "Necunoscut": "?"
    }
    COLOR_CODES = {
        "Rosu": "R", "Albastru": "B", "Verde": "G", 
        "Galben": "Y", "Portocaliu": "O", "Alb": "W",
        "Necunoscut": "?"
    }

    shape_templates = incarca_sabloane_forme(templates_folder)
    if not shape_templates:
        print("Eroare: Nu s-au putut incarca sabloanele.")
        sys.exit(1)

    all_files = glob.glob(os.path.join(input_folder, "*.[jp][pn]g"))
    if not all_files:
        print("Eroare: Folder input gol.")
        sys.exit(1)

    games = {}
    for filepath in all_files:
        filename = os.path.basename(filepath)
        try:
            game_id = int(filename.split('_')[0])
            if game_id not in games:
                games[game_id] = []
            games[game_id].append(filepath)
        except ValueError:
            continue

    sorted_game_ids = sorted(games.keys())

    for game_id in sorted_game_ids:
        print(f"--- Procesare JOCUL {game_id} ---")
        
        game_images = sorted(games[game_id], key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

        full_board_state = np.zeros((16, 16), dtype=int)
        bonus_matrix = np.zeros((16, 16), dtype=int)
        previous_warped_board = None 
        
        for image_path in game_images:
            filename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
            parts = filename_no_ext.split('_')
            idx_j = int(parts[1])

            output_txt_path = os.path.join(output_folder, f"{filename_no_ext}.txt")
            
            warped_board = proceseaza_extragere_tabla(image_path)
            if warped_board is None:
                if idx_j > 0:
                    with open(output_txt_path, "w") as f: f.write("EROARE_PROCESARE\n0")
                continue

            pieces_to_write = [] 
            current_turn_score = 0

            if idx_j == 0:
                print(f"  > Initializare tabla (j=0): {filename_no_ext}")
                full_board_state = determina_configuratia_initiala(warped_board)
                bonus_matrix = genereaza_matrice_bonus(full_board_state)
                previous_warped_board = warped_board
                continue

            else:
                if previous_warped_board is None:
                    print(f"Eroare: Lipseste imaginea de start (j=0) pentru jocul {game_id}")
                    continue

                print(f"  > Procesare mutare {idx_j}: {filename_no_ext}")
                
                current_diff_grid, _ = detecteaza_schimbari_intre_imagini(warped_board, previous_warped_board)
                mutari_noi_set = identifica_mutari_noi(current_diff_grid, full_board_state)
                mutari_noi_list = list(mutari_noi_set)
                
                for (r, c) in mutari_noi_list:
                    y, x = r*50, c*50
                    patch = warped_board[y:y+50, x:x+50]
                    shape_name = identifica_forma_piesei(patch, shape_templates)
                    color_name = determina_culoarea_piesei(patch)
                    pieces_to_write.append((r, c, shape_name, color_name))
                
                for (r, c) in mutari_noi_list:
                    full_board_state[r, c] = 1
                
                current_turn_score, bonus_matrix = calculeaza_scor_tura(mutari_noi_list, full_board_state, bonus_matrix)
                previous_warped_board = warped_board

                pieces_to_write.sort(key=lambda x: (x[0], x[1]))
                
                with open(output_txt_path, "w") as f:
                    for item in pieces_to_write:
                        r, c, s_name, c_name = item
                        row_str = str(r + 1)
                        col_str = COL_LETTERS[c]
                        shape_code = SHAPE_CODES.get(s_name, "?")
                        color_code = COLOR_CODES.get(c_name, "?")
                        f.write(f"{row_str}{col_str} {shape_code}{color_code}\n")
                    f.write(str(current_turn_score))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()