from simulator import *
import json
import sys
from configs import PROJECT_DIR

def run_simulations(input_file):
    try:
        # Read the input file
        with open(input_file, 'r') as f:
            params = json.load(f)

        print('Running ', params)
        temp = Simulation(params, {})
        temp.run_simulation()
    
    except Exception as e:
        print("Error:", e)

def generate_an_example_simulation_params(fn = 'simulation_params_example.json'):
    import json
    config_dict = {
        'rand_seed': 0,
        'algorithm_name': 'oversubscription-oracle',
        'ds_name': '2021_burstable',
        'num_arrival_vms_per_time_idx': 2,
        'time_bound': 86400,
        'first_model': '0.5X',
        'prediction_type': 'est',
        'lb_name': 'worst-fit_usage',
        'number_of_servers': 10,
        'server_capacity': 48,
        'acceptable_violation': 0.01,
        'retreat_num_samples': 0,
        'drop': False,
        'steady_state_time': 2016
    }
    
    # File path to save the params
    param_file_path = PROJECT_DIR + 'simulation_param_files/'
    if not os.path.exists(param_file_path):
            os.makedirs(param_file_path)
    # Save the dictionary to the file
    with open(param_file_path + fn, 'w') as f:
        json.dump(config_dict, f)

    # Load the param dictionary from the file
    with open(param_file_path + fn, 'r') as f:
        loaded_param_dict = json.load(f)

if __name__ == "__main__":
    # # Check if the input file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)
    
    # Get the input file path from command-line arguments
    input_file = sys.argv[1]
    
    
    # input_file = 'simulation_param_files/simulation_params_example.json' # uncoment this if you want the example to run
    # Call the run_simulations function with the input file
    run_simulations(input_file)

    