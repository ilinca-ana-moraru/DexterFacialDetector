import os

class Parameters:
    def __init__(self,task,character = None):
        self.task = task
        self.character = character
        if task == 1:
            self.path_solutions = os.path.join("..", "evaluare", "fisiere_solutie", "463_Moraru_Ilinca", "task1")
            os.makedirs(self.path_solutions, exist_ok=True)
            self.sol_detections_path = os.path.join(self.path_solutions,"detections_all_faces.npy")
            self.sol_filenames_path = os.path.join(self.path_solutions,"file_names_all_faces.npy")
            self.sol_scores_path = os.path.join(self.path_solutions,"scores_all_faces.npy")


            self.base_dir = os.path.join('..','data','task1')
        if task == 2:
            self.path_solutions = os.path.join("..", "evaluare", "fisiere_solutie", "463_Moraru_Ilinca", "task2")
            os.makedirs(self.path_solutions, exist_ok=True)

            self.sol_detections_path = os.path.join(self.path_solutions,f"detections_{character}.npy")
            self.sol_filenames_path = os.path.join(self.path_solutions,f"file_names_{character}.npy")
            self.sol_scores_path = os.path.join(self.path_solutions,f"scores_{character}.npy")
            self.base_dir = os.path.join('..','data','task2',character)

        self.dir_pos_examples = os.path.join(self.base_dir, 'exemplePozitive')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exempleNegative')
        self.dir_hard_mining = os.path.join(self.base_dir, 'exempleHardMining')

        self.dir_test_examples = os.path.join('..','evaluare','fake_test')
        self.path_annotations = os.path.join('..','validare','task1_gt_validare.txt')


        # self.dir_test_examples = os.path.join('..','evaluare','fake_test_20')
        # self.path_annotations = os.path.join('..','validare','task1_gt_validare20.txt')

        # self.dir_test_examples = os.path.join('..','antrenare','test_train','dexter')
        # self.path_annotations = os.path.join('..','validare','test_train','dexter_annotations.txt')

        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 3  # dimensiunea celulei
        self.cells_p_block = 4
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.number_positive_examples = 10  # numarul exemplelor pozitive
        self.number_negative_examples = 40  # numarul exemplelor negative
        self.number_hard_mining = 20
        self.overlap = 0.1
        self.has_annotations = False
        self.threshold = 0

