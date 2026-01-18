import os

class Parameters:
    def __init__(self):
        self.base_dir = '../data'
        self.validare = '../validare'
        self.dir_pos_examples = os.path.join(self.base_dir, 'exemplePozitive')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exempleNegative')
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        
        self.dir_test_examples = os.path.join(self.validare,'validare')
        self.base_dir_annotations_task_2 = os.path.join(self.validare)
        self.path_annotationstask2_daphne = os.path.join(self.base_dir_annotations_task_2, 'task2_daphne_gt_validare.txt')
        self.path_annotationstask2_fred = os.path.join(self.base_dir_annotations_task_2, 'task2_fred_gt_validare.txt')
        self.path_annotationstask2_shaggy = os.path.join(self.base_dir_annotations_task_2, 'task2_shaggy_gt_validare.txt')
        self.path_annotationstask2_velma = os.path.join(self.base_dir_annotations_task_2, 'task2_velma_gt_validare.txt')

        # pentru testare:
        # self.base_test = '../testare' in loc de validare
        # ...
        
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 36  
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.number_positive_examples = 5713  # numarul exemplelor pozitive
        self.number_negative_examples = 6000  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
        #task 2:
        self.unknown_faces = os.path.join(self.base_dir, 'unknown')
        self.faces = os.path.join(self.base_dir, 'personaje separate')
