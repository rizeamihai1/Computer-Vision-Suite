import cv2
import numpy as np
import argparse
import sys
import os
import glob

# =============================================================================
#  FUNCTII UTILITARE (Display & Scor)
# =============================================================================

def show_debug(window_name, image, scale=0.3):
    if image is None: return
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    cv2.imshow(window_name, resized)

def calculeaza_scor_qwirkle(new_pieces_data, board_state):
    total_score = 0
    counted_lines_h = set()
    counted_lines_v = set()

    for (nx, ny) in new_pieces_data:
        min_x, max_x = nx, nx
        while (min_x - 1, ny) in board_state: min_x -= 1
        while (max_x + 1, ny) in board_state: max_x += 1
        
        line_len_h = max_x - min_x + 1
        line_id_h = (ny, min_x, max_x)
        
        if line_len_h > 1 and line_id_h not in counted_lines_h:
            pts = line_len_h
            if line_len_h == 6: pts += 6
            total_score += pts
            counted_lines_h.add(line_id_h)

        min_y, max_y = ny, ny
        while (nx, min_y - 1) in board_state: min_y -= 1
        while (nx, max_y + 1) in board_state: max_y += 1
        
        line_len_v = max_y - min_y + 1
        line_id_v = (nx, min_y, max_y)
        
        if line_len_v > 1 and line_id_v not in counted_lines_v:
            pts = line_len_v
            if line_len_v == 6: pts += 6
            total_score += pts
            counted_lines_v.add(line_id_v)

    return total_score

# =============================================================================
#  FUNCTIA 1: DETECTIE POZA INITIALA (THRESHOLD)
# =============================================================================
def detecteaza_piese_start(curr_img, tile_size, debug_img):
    """
    Gaseste originea si toate piesele din prima imagine folosind threshold simplu.
    Returneaza: (origin_pixel, list_of_pieces)
    """
    # transform imaginea in grey scale
    gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    # aplic putin blur sa scap de zgomot
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # aplic threshold invers: piesele sunt negre (sub 60) si devin albe, restul negru
    _, mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    
    # umplu golurile din interiorul pieselor (reflexii) ca sa fie blocuri solide
    contours_fill, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours_fill, -1, 255, thickness=cv2.FILLED)
    
    # gasesc contururile finale pe masca plina
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filtrez contururile prea mici (zgomot)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
    
    if not valid_contours:
        return None, []

    # 1. Sortam dupa X (coloana) apoi dupa Y ca sa gasim sigur piesa din stanga-sus (Originea)
    valid_contours.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    
    # iau primul contur valid ca referinta
    c0 = valid_contours[0]
    x0, y0, w0, h0 = cv2.boundingRect(c0)
    
    # setez originea in centrul primei sub-piese din acest contur
    origin_cx = int(x0 + tile_size / 2)
    origin_cy = int(y0 + tile_size / 2)
    origin_pixel = (origin_cx, origin_cy)
    
    # print(f" -> Origine setata: {origin_pixel}")
    cv2.circle(debug_img, origin_pixel, 10, (0, 0, 255), -1)
    
    start_pieces = []
    
    # 3. Procesam toate contururile gasite
    for c in valid_contours:
        # iau dreptunghiul minim care incadreaza conturul
        x, y, w, h = cv2.boundingRect(c)
        
        # impart dreptunghiul in cate piese ar putea fi in el (pe lungime si latime)
        cols = max(1, int(round(w / tile_size)))
        rows = max(1, int(round(h / tile_size)))
        
        # parcurg fiecare posibil loc de piesa din bloc
        for r in range(rows):
            for c_idx in range(cols):
                # calculez coordonatele teoretice ale sub-piesei
                sub_x = x + c_idx * tile_size
                sub_y = y + r * tile_size
                
                # calculez centrul patratului
                cx = int(sub_x + tile_size / 2)
                cy = int(sub_y + tile_size / 2)
                
                # distanta fata de origine in pixeli
                dx = cx - origin_pixel[0]
                dy = cy - origin_pixel[1]
                
                # transform in coordonate de grid
                grid_col = int(round(dx / tile_size))
                grid_row = int(round(dy / tile_size))
                
                ux = grid_col
                uy = -grid_row
                
                # verific sa nu fi adaugat deja piesa in lista
                if (ux, uy) not in start_pieces:
                    start_pieces.append((ux, uy))
                    
                    # desenez pentru debug
                    #cv2.rectangle(debug_img, (int(sub_x), int(sub_y)), (int(sub_x+tile_size), int(sub_y+tile_size)), (0, 255, 0), 2)
                    #cv2.putText(debug_img, f"{ux} {uy}", (int(sub_x), int(sub_y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    return origin_pixel, start_pieces
# =============================================================================
#  FUNCTIA 2: DETECTIE PIESE NOI (DIFERENTA)
# =============================================================================

def detecteaza_piese_noi(curr_img, prev_img, origin_pixel, tile_size, occupied_set, debug_img):
    """
    Gasim piesele noi facand diferenta intre imaginea curenta si anterioara
    iau o variabila prin care fac diferenta in abs
    transform imaginea diferenta in grey scale 
    aplic threshold pe imagine -> daca pixelul este sub 60 devine negru, altfel alb
    """
    new_pieces = []
    
    diff = cv2.absdiff(curr_img, prev_img)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh_diff = cv2.threshold(gray_diff, 60, 255, cv2.THRESH_BINARY)    
    
    # umplu golurile din imagine
    contours_fill, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh_diff, contours_fill, -1, 255, thickness=cv2.FILLED)
    
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # acum avem contururile zonelor noi + posibil zgomot care trebuie filtrat
    
    for c in contours:
        # filtrez zonele prea mici => zgomot
        if cv2.contourArea(c) < 500: 
            continue
        # acum iau pentru fiecare contur coordonatele dreptunghiului minim care il includ
        x, y, w, h = cv2.boundingRect(c)
        
        # impart fiecare dreptunghi in cate posibile piese ar putea fii in el
        # il impart pe lungime / latime la 165 px (cat e o piesa)
        cols = max(1, int(round(w / tile_size)))
        rows = max(1, int(round(h / tile_size)))
        
        for r in range(rows):
            for c_idx in range(cols):
                # calculez pentru fiecare patrat ce coordonate ar avea in imaginea initiala 
                sub_x = x + c_idx * tile_size
                sub_y = y + r * tile_size
                
                t_x0 = int(max(0, sub_x))
                t_y0 = int(max(0, sub_y))
                t_x1 = int(min(thresh_diff.shape[1], sub_x + tile_size))
                t_y1 = int(min(thresh_diff.shape[0], sub_y + tile_size))
                
                mask_slice = thresh_diff[t_y0:t_y1, t_x0:t_x1]
                if mask_slice.size == 0: 
                    continue
                # verific pentru patrat sa aibe cel putin 0.5 pixeli albi (ca sa fie piesa, nu pun mai mult de 50% din cauza ca la piese albe poate da negru zona din mijloc cand
                # se face diferenta fata de masa)
                if cv2.countNonZero(mask_slice) / mask_slice.size < 0.5:
                    continue
                
                # coordonatele centrului patratului
                cx = int(sub_x + tile_size / 2)
                cy = int(sub_y + tile_size / 2)
                
                # coordonatele in raport cu centrul (0,0)
                dx = cx - origin_pixel[0]
                dy = cy - origin_pixel[1]
                
                grid_col = int(round(dx / tile_size))
                grid_row = int(round(dy / tile_size))
                
                ux = grid_col
                uy = -grid_row
                
                # verific sa nu mai fie in setul de piese deja, daca nu e ii dau append
                if (ux, uy) not in occupied_set:
                    if (ux, uy) not in new_pieces:
                        new_pieces.append((ux, uy))
                        # cv2.rectangle(debug_img, (int(sub_x), int(sub_y)), (int(sub_x+tile_size), int(sub_y+tile_size)), (0, 255, 0), 2)
                        # cv2.putText(debug_img, f"{ux} {uy}", (int(sub_x), int(sub_y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return new_pieces

# =============================================================================
#  SECTIUNEA 3: DETECTIE CULOARE
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
#  SECTIUNEA 4: DETECTIE FORMA
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
#  MAIN LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-t", "--templates_folder", required=True) 
    args = parser.parse_args()

    templates = incarca_sabloane_forme(args.templates_folder)
    image_paths = sorted(glob.glob(os.path.join(args.input_folder, "*.[jp][pn]g"))) # sa accepte si png si jpg
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    #debug_dir = "debug_bonus"
    #if not os.path.exists(debug_dir): os.makedirs(debug_dir)

    TILE_SIZE = 165.0
    SHAPE_CODES = {"cerc": "1", "trifoi": "2", "romb": "3", "patrat": "4", "stea_4": "5", "stea_8": "6"}
    COLOR_CODES = {"Rosu": "R", "Albastru": "B", "Verde": "G", "Galben": "Y", "Portocaliu": "O", "Alb": "W"}

    ORIGIN_PIXEL = None 
    prev_image = None
    OCCUPIED_POSITIONS = set() 

    for idx, path in enumerate(image_paths):
        curr_img = cv2.imread(path)
        base_name = os.path.basename(path)
        txt_filename = os.path.splitext(base_name)[0] + ".txt"
        output_path = os.path.join(args.output_folder, txt_filename)
        
        print(f"--- Procesez: {base_name} ---")
        debug_img = curr_img.copy()
        current_turn_pieces = [] 

        # poza initiala 
        if idx == 0:
            ORIGIN_PIXEL, current_turn_pieces = detecteaza_piese_start(curr_img, TILE_SIZE, debug_img)
            for p in current_turn_pieces:
                OCCUPIED_POSITIONS.add(p)

        # acum de gasit piesele noi
        else:
            if ORIGIN_PIXEL is not None and prev_image is not None:
                current_turn_pieces = detecteaza_piese_noi(curr_img, prev_image, ORIGIN_PIXEL, TILE_SIZE, OCCUPIED_POSITIONS, debug_img)
                for p in current_turn_pieces:
                    OCCUPIED_POSITIONS.add(p)

        prev_image = curr_img
        
        turn_score = calculeaza_scor_qwirkle(current_turn_pieces, OCCUPIED_POSITIONS)
        current_turn_pieces.sort(key=lambda p: (p[0]**2 + p[1]**2, p[0], p[1]))
        
        with open(output_path, "w") as f_txt:
            if not current_turn_pieces:
                f_txt.write("0\n")
            else:
                for (ux, uy) in current_turn_pieces:
                    # cx, cy -> refac coordonatele din cele ale jocului in pixeli din imagine normala
                    # pentru a gasii tipul si culoarea piesei
                    cx = int(ux * TILE_SIZE + ORIGIN_PIXEL[0])
                    cy = int(-uy * TILE_SIZE + ORIGIN_PIXEL[1])
                    
                    half = int(TILE_SIZE // 2)
                    y0 = max(0, cy - half)
                    y1 = min(curr_img.shape[0], cy + half)
                    x0 = max(0, cx - half)
                    x1 = min(curr_img.shape[1], cx + half)
                    
                    roi = curr_img[y0:y1, x0:x1] # pathcul piesei
                    shape, color = "", ""
                    
                    if roi.size > 0:
                        shape = identifica_forma_piesei(roi, templates)
                        color = determina_culoarea_piesei(roi)
                        shape = SHAPE_CODES.get(shape)
                        color = COLOR_CODES.get(color)
                        
                        #cv2.putText(debug_img, f"{shape}{color}", (cx-20, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                    f_txt.write(f"{ux} {uy} {shape}{color}\n")
                    #print(f"   Detectat: {ux} {uy} -> {shape}{color}")
            
            f_txt.write(f"{turn_score}")
        
        #print(f"   SCOR: {turn_score}")
        #show_debug("Rezultat", debug_img)
        
        #print("Apasa tasta pt continuare, 'q' iesire.")
        #key = cv2.waitKey(0) & 0xFF
        #if key == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()