#%%
import pandas as pd
import datetime
import numpy as np
import bisect
from configs import PROJECT_DIR
from server import Server
from tqdm import tqdm
from algorithm import SchedulerFactory
import os

class Simulation:
    def __init__(self, parameters, data_files):
        # Initialize simulation parameters
        self.params = parameters
        # to keep the random behavior reproducable
        np.random.seed(self.params['rand_seed'])
        self.data_files = data_files
        self.read_data()
        self.all_selected_vmids = np.random.choice(self.data_files['vmids'], self.params['num_arrival_vms_per_time_idx']*self.params['time_bound']).reshape(self.params['time_bound'], self.params['num_arrival_vms_per_time_idx'])
        self.dropped_vmids = []
        self.servers = []
        self.record_bools = {}
        # create the scheduler module
        self.scheduler = SchedulerFactory.create_scheduler(self.params['algorithm_name'], self.data_files, parameters)
        
    def read_data(self):
        # Read VM features and dataframes from files
        start_time = datetime.datetime.now()
        if 'vmids' not in self.data_files:
            self.data_files['vmids'] = np.load( PROJECT_DIR + 'data/test_arrival_vmids_' + self.params['ds_name'] + '.npy')
        if 'vmid_to_baseline' not in self.data_files:
            size_str = 'baseline' if 'burstable' in self.params['ds_name'] else 'corecount'
            self.data_files['vmid_to_baseline'] = np.load(PROJECT_DIR + 'data/vmid_to_{}_{}.npy'.format(size_str, self.params['ds_name']), allow_pickle = True).reshape(1, )[0]
        if 'vmid_to_trace' not in self.data_files:            
            self.data_files['vmid_to_trace'] = pd.read_feather(PROJECT_DIR + f"/data/test_arrival_vmid_to_trace_{self.params['ds_name']}.feather").set_index('vmid')['trace'].to_dict()
        print('Reading data files took {}'.format(datetime.datetime.now() - start_time)) # TODO: move these print statements into logging
        # self.data_files['vmid_to_avg'] = None
        # self.data_files['vmid_to_var'] = None
        
    def run_simulation(self):
        start_time = datetime.datetime.now()
        # read the offline data
        self.record_bools, self.lb_metric = self._gen_config_bools()
        
        sorted_server_load_idx = [(0, i) for i in range(self.params['number_of_servers'])]
        self._init_servers() 
        
        # main simulation loop
        for time_idx in tqdm(range(self.params['time_bound']), position=0, leave=True):    
            # Update CPU usage pdf of the server sequentially to be used for 'audible' based algorithms
            if 'audible' in self.params['algorithm_name']:
                for server in self.servers:
                    server.server['cpu_usage_pdf'][:] = server.server['cpu_usage_pdf_from_usage'] # to use the copy of the pdf that is not affected by the convolutions in the algorithm
                if time_idx > 0:
                    [server.update_cpu_usage_pdf(time_idx) for server in self.servers]
                for server in self.servers:
                    server.server['cpu_usage_pdf_from_usage'][:] = server.server['cpu_usage_pdf'] # to save a copy of the pdf that is not affected by the convolutions in the algorithm
                    
            # sort sorted_server_load_idx again at the begning of each time_idx
            sorted_server_load_idx = self._lb_sort_servers(time_idx)
            for vmid in self.all_selected_vmids[time_idx]:
                placed = False
                for idx_in_sorted_server_load_idx in range(int(0.1*self.params['number_of_servers'])): # Only checking 10% of the servers
                    laod, server_idx = sorted_server_load_idx[idx_in_sorted_server_load_idx]
                    balance = True
                    server = self.servers[server_idx]
                    # retreat
                    if (self.params['retreat_num_samples'] > 0 and np.any(server['usage'][time_idx - self.params['retreat_num_samples']:time_idx] >= self.params['server_capacity'])): # if this server is in retreat mode
                        balance = False
                    if balance: # check if the VM can be placed on this server or not
                        can_place_bool = self.scheduler.run_algorithm(time_idx, server, vmid)
                        if can_place_bool:
                            placed = True
                            server.update_server(vmid, time_idx, self.params, self.data_files['vmid_to_baseline'][vmid], self.record_bools, self.scheduler.data_files)
                            # update position of server in the (server_load, server_idx) list
                            self._lb_resort_servers(sorted_server_load_idx, idx_in_sorted_server_load_idx, time_idx)
                            break
                if not placed: # no server can host this vm
                    # Drop this vm
                    if self.params['drop']:
                        print(f'Rejecting a VM at time {time_idx}')
                        return 'failed' 
                    else:
                        self.dropped_vmids.append((time_idx, vmid))
            if self.params['algorithm_name'] == 'oversubscription-oracle' and time_idx>0 and time_idx%2016 == 0 and self._report_usage_violation(time_idx)[1] > 0:
                return 'failed'
                
        # Check for carry overs
        if self.record_bools['record_carry_over_bool']:
            for server in self.servers:
                server.run_carry_over(self.params['time_bound'] - (self.params['time_bound']+4), (self.params['time_bound']+4))# to account for violations that can cause cascading carry overs we added +4
        
        print('length of dropped_vmids for arrival ', self.params['num_arrival_vms_per_time_idx'], ' is ', len(self.dropped_vmids))
        avg_usage, num_servers_with_extreme_viol = self._report_usage_violation(time_idx, print_bool = True)
        
        if self.params['algorithm_name'] == 'oversubscription-oracle' and num_servers_with_extreme_viol > 0: # only for oversubscription-oracle to make it genie-aided (basically it says if oversubscription-oracle experience violation above the threshold, consider it failed)
            return 'failed'
        
        # save result
        fn = '_'.join([str(self.params[i]) for i in self.params]) 
        self._save_results(PROJECT_DIR + f"results/{self.params['algorithm_name']}/", fn, large = False)
        return 'succeed'
        
    def _report_usage_violation(self, time_idx, print_bool = False):
        steady_state_time = self.params['steady_state_time']
        avg_usage = np.mean([np.mean(server.server['usage'][time_idx-steady_state_time:time_idx]) for server in self.servers])
        if print_bool:
            print('Average utilization accross all servers:', avg_usage)
        temp = np.array([len(np.where(server.server['usage'][time_idx-steady_state_time:time_idx] >= server.server['capacity'])[0])/steady_state_time for server in self.servers])
        num_servers_with_severe_violation = len(np.where(temp>self.servers[0].server['acceptable_violation'])[0])
        if print_bool:
            print('Number of servers with violation more than {}% in the last week is {}'.format(100*self.servers[0].server['acceptable_violation'], num_servers_with_severe_violation) )
        return avg_usage, num_servers_with_severe_violation

    def _gen_first_model(self):
        if '-' not in self.params.first_model: # such as 2X
            self.data_files['vmid_to_firstmodel'] = {vmid: self.data_files['vmid_to_baseline'][vmid]*(float(self.params.first_model.strip('X'))) for vmid in self.data_files['vmid_to_baseline']}
        elif 'rc-' in self.params.first_model: # such as resouce central with 95 percentile
            pass
        else: # it's the percentile value
            for size in self.data_files['vmsize_to_avg']:
                np.quantile(self.data_files['vmsize_to_avg'], )
    
    def _gen_revisions(revisions_names):
        revision_name_to_revisions = {}
        for revisions_name  in revisions_names:
            if revisions_name.count('-') == 1:
                revisions = [int(revisions_name.split('-')[0]), 288*365] 
            else:
                ps = int(revisions_name.split('-')[2])
                revisions = [int(revisions_name.split('-')[0])] + list(range(ps,365*ps,ps))
            revision_name_to_revisions[revisions_name] = revisions
        return revision_name_to_revisions

    def _gen_config_bools(self):
        # specify load balancer parameters
        best_fit_bool, lb_metric = self.params['lb_name'].split('_', 1)
        best_fit_bool = True if 'best-fit' == best_fit_bool else False
        # create a set of bools to speed up the program and avoid recording redundant info
        record_mean_bool = True if (self.params['algorithm_name'] in ['oversubscription-oracle', 'rc', 'CLT', 'B.E.']) else False
        record_variance_bool = True if (self.params['algorithm_name'] in ['CLT', 'B.E.']) else False
        record_baselines_bool = True if ('baseline' in lb_metric) else False# or (self.params['algorithm_name'] in ['oversubscription-oracle']) else False
        record_third_mom_bool = True if self.params['algorithm_name'] == 'B.E.' else False
        #record_bools as dict
        record_bools = {'record_number_lr_bool': False, 'record_number_sr_bool': False, 'record_usage_bool': True,
                        'record_variance_bool': record_variance_bool, 'record_mean_bool': record_mean_bool,
                        'record_third_mom_bool': record_third_mom_bool,
                        'record_num_large_bool': False if not self.params.get('limit_num_large') else True,
                        'record_carry_over_bool': True, #False if params['retreat_num_samples'] == 0 else True, # it says if running carry_over is necessary for the algorithm
                        'record_baselines_bool': record_baselines_bool, 'record_peakcpus_bool': False, 'best_fit_bool': best_fit_bool}
        return record_bools, lb_metric
    
    def _init_servers(self):
        self.servers = [Server(self.params['time_bound'], self.params['server_capacity'], self.params['acceptable_violation'],
                                       self.params['algorithm_name'], self.record_bools) for i in range(self.params['number_of_servers'])]
    
    def _lb_sort_servers(self, time_idx):
        """
        Return the Sorted list of the servers based on the load-balancing algorithm in lb_name
        """
        if 'mean_variance' == self.lb_metric: # for CLT and audible
            sorted_server_load_idx = [(self.servers[server_idx].server['mean'][time_idx] + (self.servers[server_idx].server['variance'][time_idx]**0.5), server_idx) for server_idx in range(len(self.servers))]
        else:
            sorted_server_load_idx = [(self.servers[server_idx].server[self.lb_metric][time_idx], server_idx) for server_idx in range(len(self.servers))]
        sorted_server_load_idx.sort(reverse = (True if self.record_bools['best_fit_bool'] else False))
        return sorted_server_load_idx

    def _lb_resort_servers(self, sorted_server_load_idx, idx_in_sorted_server_load_idx, time_idx):
        """
        Update the sorted list of servers after a placement happened.
        """
        old_load, server_idx = sorted_server_load_idx.pop(idx_in_sorted_server_load_idx)
        if 'mean_variance' == self.lb_metric:
            new_load = self.servers[server_idx].serever['mean'][time_idx] + (self.servers[server_idx].server['variance'][time_idx]**0.5) 
        else:
            new_load = self.servers[server_idx].server[self.lb_metric][time_idx] 
        if self.record_bools['best_fit_bool']: 
            bisect.insort(sorted_server_load_idx, (new_load, server_idx), key=lambda x: -1 * x[0])
        else:
            bisect.insort(sorted_server_load_idx, (new_load, server_idx))

    def _save_results(self, location, fn, large = False): 
        if not os.path.exists(location):
            os.makedirs(location)
        
        if large:
            result = {'params':self.params, 'servers_dict': self.servers, 'vmids': self.all_selected_vmids, 'dropped_vmids': self.dropped_vmids}
            print('Saving ' + fn + '.npy', end = '...' )
            np.save(location + fn + '.npy', result ,allow_pickle = True)
        else:
            # smaller result with necessary information
            small_servers = []
            for server in self.servers:
                small_server = {}
                # f_to_save = ['usage', 'variance', 'mean', 'num_large', 'carry_over', 'baselines', 'peakcpus']
                f_to_save = [r[7:-5] for r in self.record_bools if self.record_bools[r]]
                for f in f_to_save:
                    small_server[f] = server.server[f][-(self.params['steady_state_time']):]
                small_server['deployed_times'] = server.server['deployed_times'] # it should be as is, because the lenght of the deployed_times can not be determined with index
                small_servers.append(small_server)
            
            # save the params and len of dropped vmids in a params file
            temp_df = pd.DataFrame(small_servers)
            temp_df.to_feather(location + 'small_' + fn + '.feather')
            # save the server dictionary in a feather table
            param_result = {'params':self.params, 'len_dropped_vmids': len(self.dropped_vmids)}
            np.save(location + 'small_' + fn + '_params.npy', param_result , allow_pickle = True)

# %%
