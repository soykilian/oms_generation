import os

# ROOT DIR
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# DATA DIRS
DATA_DIR = os.path.join(ROOT_DIR, 'VSim_dataset')
SIMULATED_DATA_DIR = os.path.join(ROOT_DIR, 'simulated_data')

# DATA PRODUCER DIRS
DATA_PRODUCER_PARAMS_DIR = os.path.join(ROOT_DIR, 'data_producer', 'data_producer_params.yaml')

# EVENT_DATA_PRODUCER DIRS
OMS_V1_DIR = os.path.join(ROOT_DIR, 'dvs_methods', 'oms.py')
DVS_V1_DIR = os.path.join(ROOT_DIR, 'dvs_methods', 'dvs.py')
DVS_V2E_DIR = os.path.join(ROOT_DIR, 'dvs_methods', 'dvs_v2e.py')
EVENTS_TO_FRAMES_DIR = os.path.join(ROOT_DIR, 'dvs_methods', 'event_encoding.py')
# OMS_TRUE_DIR = os.path.join(ROOT_DIR, 'dvs_methods', 'oms_true.py')

# PARAMETER_SEARCH DIRS
BAYES_OPT_PARAMS = os.path.join(ROOT_DIR, 'parameter_search', 'bayes_opt_params.yaml')

