# Audible Artifact for ASPLOS 2024

Welcome to the official repository for the Audible algorithm, presented in our ASPLOS 2024 paper, "AUDIBLE: A Convolution-Based Resource Allocator for Burstable Virtual Machines in Cloud Platforms." This repository provides all necessary source code, detailed instructions, and datasets to replicate the simulation results discussed in our research.

## Repository Structure

- **`data/`**: Holds the datasets. Processed datasets are also stored here post-processing.
- **`src/`**: Contains Python modules required to run the simulator.
- **`results/`**: Includes directories named after each algorithm, where simulation results are stored.
- **`simulation_param_files/`**: Stores JSON files needed for running simulation experiments with `src/main.py`.

## Hardware Requirements

- The simulation is designed to run on a single core and benefits from an SSD, although it's not mandatory.
- It is memory-intensive, retaining data in memory to enhance performance. Expect memory requirements to be between 8-16GB, depending on the chosen algorithm.

## Software Requirements

- Tested environments: macOS Sonoma 14.2, CentOS 7.9.2009, and Ubuntu 18.04 LTS. It may work on other Linux distributions but has not been tested.
- Compatible with Python 3.x, and tested with 3.6 and 3.9.
- Installation of Python modules listed in `requirements.txt` is necessary:
  ```bash
  pip install -r requirements.txt
  ```

## Preparing the Datasets
- Download the required datasets from the provided link.
- Please note that the datasets are stored in Parquet format, which can demand a significant amount of memory for loading in entirety. Our testing on a system equipped with 32GB of RAM indicated the feasibility of loading one dataset entirely at a time.
- Process 2021 burstable dataset `BurstableVMs_2021_Data.parquet` using the steps outlined in `src/Data_processing.ipynb`. This notebook details each step required to prepare the data for simulation. Processed files will be saved to the `data/` directory.

## Evaluation Setup
- Use `run_simulator.ipynb` to generate simulation parameter files through its GUI interface. These files are saved in the `simulation_param_files/` directory.
- The JSON simulation param file names reflect their content for easier identification.

## Running Simulations
To execute a simulation, provide the generated simulation parameter file to `main.py`:
```
./src/main.py simulation_params/[Name of the simulation parameter file]
```
- Examples for each algorithm are available in `run_simulator.ipynb`.
- Simulations either complete successfully, storing results in the `results/` directory, or terminate early if a VM must be rejected and the parameters are configured to halt under such conditions.

## Collecting Results
Output for each simulation is saved in a directory named after the algorithm in the parameter file.
Results include a `.feather`` file with CPU usage and carry-overs for each server and a `.npy` file containing a dictionary with the simulation parameters and the number of VMs rejected by the algorithm.
- Examples of reading and interpreting result files are provided in `read_simulation_result.ipynb`.

## Reproducing Paper Results
To produce the results presented in the paper, a comprehensive series of simulations was conducted. The parameter files required for these simulations are created in the section titled "Simulation Parameter Values for the ASPLOS24 Paper" within the `run_simulator.ipynb` file. Additionally, a summary of the simulation configuration parameters utilized in the paper is provided below:


| Configuration Key                          | Values                                            | Description                                                                                                   |
|--------------------------------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `rand_seed`                                | 0, 1, 2, 3, ...                                   | Specifies the seed value to shuffle the order of arriving VMs for simulation.                                 |
| `algorithm_name`                           | "audible", "oversubscription oracle", "CLT", "rc" | Specifies the name of the algorithm being used.                                                               |
| `ds_name`                                  | "2021_burstable" or "2021_regular"                | Specifies the dataset name.                                                                                   |
| `num_arrival_vms_per_time_idx`             | A positive value                                  | Specifies how many VMs are placed at each simulation time point, varies depending on the algorithm.           |
| `time_bound`                               | 86400 \*                                            | Specifies the number of simulation time points, simulating 5 minutes in reality. 86400 reflects 10 months.    |
| `first_model`                              | Varies by algorithm                               | Specifies the first model to be used for each algorithm upon VM arrival based on its type.                    |
| `prediction_type`                          | "Oracle" or "est"                                 | Specifies the prediction type.                                                                                |
| `lb_name`                                  | "worst-fit_usage" \*                                | Specifies the name of the load balancer. Fixed to Worst Fit algorithm according to the servers usage.                                          |
| `(number_of_servers, server_capacity)`     | (1008, 36), (756, 48), (567, 64)                   | Specifies the number of servers and the number of cores in each server. Fixed total core capacity for each of the setups.                                      |
| `acceptable_violation`                     | 0.005, 0.01, 0.025, and 0.05                      | Specifies the target violation that each algorithm tries to achieve.                                          |
| `retreat_num_samples`                      | 0 \*                                                | Specifies the number of simulation points to stop placing VMs on a server after a past violation.             |
| `drop`                                     | True  \*                                            | If set to "False", allows rejecting VMs for placement. Set to "true" to always accept VMs for placement.      |
| `steady_state_time`                        | 2016 \*                        | Specifies the steady state time to be used for reporting results, using the last week or 2016 simulation points.|

\* Constant across simulations


## Citing Our Work
Please reference our ASPLOS 2024 paper in your work. [Will update citation details here later].

Thank you for exploring the Audible artifact repository!
