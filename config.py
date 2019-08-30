import datetime

"""
The Config file is where we set all the experiment variables
This is the one-stop place to control the flow of the entire experiment
Please refer to the helpers package for any global helpers used throughout the process

"""
DEFAULT_CONFIG = {
    'verbose': True,
    'run_data_selector': False,
    'run_candidate_generator': True,
    'experiment_base_dir': '/gpfs/fs0/data/DeepLearning/sabedin/experiemnts/pe_active_learning',
    'experiment_id': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    'debug_dir': '/gpfs/fs0/data/DeepLearning/sabedin/experiemnts/pe_active_learning/debug',

}

CANDIDATE_GENERATOR_CONFIG = {
    'experiment_id': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),

    # Training
    'training_images_path': '/home/sabedin/Data/pe_dense_object_detection/',
    'model_name': 'UNETMULTISLICE',
    'batch_size': 32,
    'max_queue_size': 200,
    'workers': 60,
    'slab_size': (512, 512, 9),
    'nb_epoch': 200,
    # Pre-Processing
    'preprocessing': True,
    'use_windowing': False,
    'overwrite_training_images': True,
    'resample': True,
    'z_spacing': 1.0,
    'window_minimum': -300,
    'window_maximum': 450,
    'dilation_kernel_radius': 0.0,
    'raw_images_path': '/gpfs/fs0/data/MedicalSieve/repositories/Abnormalities/PulmonaryEmbolism/Annotations_2019-06-12',
    'training_images_save_path': '/home/sabedin/Data/pe_dense_object_detection/',
    # Prediction
    'prediction_images_path': '',
    'prediction_images_gt_path': '',
    'prediction_mask_save_path': '',
    'save_predicted_mask': True,
    'prediction_images_csv_path': '',
}


DATA_SELECTOR_CONFIG = {
    'training_set_file': '/home/sabedin/Data/pe_dense_object_detection/pe_training_sets/Init_Set_Tr_2019-07-31.csv',
    'validation_set_file': '/home/sabedin/Data/pe_dense_object_detection/pe_training_sets/Init_Set_Val_2019-07-31.csv',
}
