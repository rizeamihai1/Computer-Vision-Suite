import os

class Parameters:
    def __init__(self):
        self.base_dir = '../data'
        self.validare = '../validare'
        self.dir_pos_examples = os.path.join(self.base_dir, 'exemplePozitive')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exempleNegative')
        self.dir_test_examples = os.path.join(self.validare,'validare')  # !de modificat pentru folderul testare
        self.path_annotations = os.path.join(self.validare, 'task1_gt_validare.txt') # !de modificat pentru folderul testare
        # de exemplu:
        # self.test = '../testare' in loc de validare
        # ...
        
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        self.dim_window = 36  
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.number_positive_examples = 6547  # numarul exemplelor pozitive
        self.number_negative_examples = 13000  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0.2
