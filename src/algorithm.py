import configs
import numpy as np
from scipy.stats import norm
from configs import PROJECT_DIR
import pandas as pd

class SchedulerFactory:
    @staticmethod
    def create_scheduler(scheduler_type, data_files, params):
        if scheduler_type == 'CLT':
            return CLTScheduler(data_files, params)
        elif scheduler_type == 'rc':
            return ResourceCentralScheduler(data_files, params)
        elif scheduler_type == 'audible':
            return AudibleScheduler(data_files, params)
        elif scheduler_type == 'oversubscription-oracle':
            return OversubscriptionOracleScheduler(data_files, params)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class Scheduler:
    def __init__(self, data_files, params):
        self.data_files = data_files if data_files else {}
        self.params = params
        # structure of the data_files {'vmid_to_trace':{}, 'vmid_to_baseline': {}}

    def _gen_esential_data(self):
        raise NotImplementedError("Subclasses must implement run_algorithm method")
    
    def run_algorithm(self, time_idx, server, vmid):
        raise NotImplementedError("Subclasses must implement run_algorithm method")

    def modify_record_bools(self):
        raise NotImplementedError("Subclasses can implement modify_record_bools method")
        

class CLTScheduler(Scheduler):
    def __init__(self, data_files, params):
        super().__init__(data_files, params)
        self._gen_esential_data()

    def _gen_esential_data(self):
        # Initialize parameters specific to CLT algorithm
        self.NORMAL_THRESHOLDS = {self.params['acceptable_violation']: norm().ppf( 1 - self.params['acceptable_violation'])}
        # standard normal loolup table
        self.SN_LOOKUP_TABLE = np.load(PROJECT_DIR + 'data/' + 'standard_lookup_table_precision_{}.npy'.format(6), allow_pickle = True).reshape(1, )[0] # precision = 6
        temp = list(self.SN_LOOKUP_TABLE.values())
        self.max_sn_lookup_table, self.min_sn_lookup_table = max(temp), min(temp)
        # read avg/var predictions
        first_model = self.params['first_model']
        pred_type = self.params['prediction_type']
        temp = pd.read_feather(PROJECT_DIR + f"/data/{first_model}_first_model_{pred_type}_rest_{self.params['ds_name']}.feather")
        temp.set_index('vmid', inplace = True)
        self.data_files['vmid_to_avg'] = temp['trace_pred_avgs'].to_dict()
        self.data_files['vmid_to_var'] = temp['trace_pred_vars'].to_dict()
    
    def _get_SN_LOOKUP_TABLE(self, desired_cdf, precision):
        try:
            return self.SN_LOOKUP_TABLE[round(desired_cdf, precision)]
        except: # if the value is extremly large or small, it might be 0 or 1.
            return (self.max_sn_lookup_table if desired_cdf > 0.5 else self.min_sn_lookup_table)
    
    
    def run_algorithm(self, time_idx, server, vmid):
        # Implement CLT scheduling algorithm
        new_variance = server.server['variance'][time_idx] + self.data_files['vmid_to_avg'][vmid][0]
        new_mean = server.server['mean'][time_idx] + self.data_files['vmid_to_var'][vmid][0]
        new_std = max(0.0000000000001, new_variance**0.5)
        z = (server.server['capacity'] - new_mean) / new_std
        if z >= self.NORMAL_THRESHOLDS[server.server['acceptable_violation']]: #2.32635
            return True
        else:
            return False


class ResourceCentralScheduler(Scheduler):
    def __init__(self, data_files, params):
        super().__init__(data_files, params)
        self._gen_esential_data()
    
    def _gen_esential_data(self):
        # Initialize parameters specific to ResourceCentral algorithm
        self.prediction_quantile = self.params.get('first_model').strip('rc-')
        # generate it, or read it offline
        try:#
            self.allocated_cores_per_vmid = pd.read_feather(PROJECT_DIR + f"/data/first_models_quantile_{self.prediction_quantile}_{self.params['ds_name']}.feather").set_index('vmid')['first_rc_0.95'].to_dict()
            self.data_files['vmid_to_avg'] = {vmid: self.allocated_cores_per_vmid[vmid]*np.ones(len(self.data_files['vmid_to_trace'][vmid])) for vmid in self.data_files['vmid_to_trace']}
        except:
            self.BUCKET_RC_PER_BASELINE = self._get_buckets_per_baseline()
            self.allocated_cores_per_vmid = self._get_allocated_cores()
            self.data_files['vmid_to_avg'] = {vmid: self.allocated_cores_per_vmid[vmid]*np.ones(len(self.data_files['vmid_to_trace'][vmid])) for vmid in self.data_files['vmid_to_trace']}
            # save file
            temp = pd.DataFrame(data = list(self.allocated_cores_per_vmid.values()), index = list(self.allocated_cores_per_vmid.keys()), columns = [f'first_rc_{self.prediction_quantile}'])
            temp.index.name = 'vmid'
            temp.reset_index().to_feather(PROJECT_DIR + f"data/first_models_quantile_{self.prediction_quantile}_{self.params['ds_name']}.feather")
                
    def run_algorithm(self, time_idx, server, vmid):
        # Implement ResourceCentral scheduling algorithm
        return server.server['mean'][time_idx] + self.allocated_cores_per_vmid[vmid] <= server.server['capacity']
    
    def _get_buckets_per_baseline(self):
        # The buckets for predicting in the resource central method
        if 'burstable' in self.params['ds_name']:
            PEAK_CPU_PER_BASELINE = {3.37: 20, 2.7: 16, 2.02: 12, 1.35: 8, 0.9: 4, 0.6: 2, 0.4: 2, 0.2: 1, 0.1: 1, 0.05: 1}
        else: 
            unique_baselines = np.array(list(set(self.data_files['vmid_to_baseline'].values())))
            PEAK_CPU_PER_BASELINE = dict(zip(unique_baselines, unique_baselines))
        BUCKET_RC_PER_BASELINE = {}
        for baseline in PEAK_CPU_PER_BASELINE:
            values1 = np.linspace(0, baseline, num = 5, endpoint=False)
            values2 = np.linspace(baseline, PEAK_CPU_PER_BASELINE[baseline], num = 4, endpoint=True)
            BUCKET_RC_PER_BASELINE[baseline] = np.concatenate((values1, values2), axis=0)[1:]
        return BUCKET_RC_PER_BASELINE
    
    def _get_allocated_cores(self):
        # for all the traces, calculate the bucket value that bound their usage
        bucketized_predictions_per_vmid = {}
        for vmid in self.data_files['vmid_to_trace']:
            baseline, trace = self.data_files['vmid_to_baseline'][vmid], self.data_files['vmid_to_trace'][vmid]
            idx = np.argmin(self.BUCKET_RC_PER_BASELINE[baseline] <= np.quantile(trace, self.prediction_quantile))
            bucketized_predictions_per_vmid[vmid] = self.BUCKET_RC_PER_BASELINE[baseline][idx]
        return bucketized_predictions_per_vmid

class AudibleScheduler(Scheduler):
    def __init__(self, data_files, params):
        super().__init__(data_files, params)
        self._gen_esential_data()
    
    def _gen_esential_data(self):
        # Initialize parameters specific to Audible algorithm
        self.PROBABILITY_DENSITY_PER_BASELINE = self._get_pmf_per_baseline()
        if 'burstable' in self.params['ds_name']:
            self.PEAK_CPU_PER_BASELINE = {3.37: 20, 2.7: 16, 2.02: 12, 1.35: 8, 0.9: 4, 0.6: 2, 0.4: 2, 0.2: 1, 0.1: 1, 0.05: 1}
        else:
            self.PEAK_CPU_PER_BASELINE = dict(zip([1, 2, 4, 8, 16, 24, 32], [1, 2, 4, 8, 16, 24, 32]))

    
    
    def run_algorithm(self, time_idx, server, vmid):
        # Implement Audible algorithm
        baseline = self.data_files['vmid_to_baseline'][vmid]

        if server.server['cpu_usage_pdf'][-1] > server.server['acceptable_violation']:
            return False
        temp_v = server.server['usage'][(time_idx - 289 if time_idx> 288 else 0):time_idx]
        if len(temp_v) == 0:
                temp_v = 0
        else:
            temp_v = np.max(temp_v)
        if temp_v + sum([self.PEAK_CPU_PER_BASELINE[b] for b in server.server['cand_vm_baselines']]) + self.PEAK_CPU_PER_BASELINE[baseline] >= server.server['capacity']:
            while(server.server['cand_vm_baselines']): # compute the joint probability of VMs and update server's cpu_usage_pdf for the VMs that have been placed on this server at the current time. 
                # Since this is computationally expensive, only runs it if there is a chance that this can impact the result.
                cand_vm_baseline = server.server['cand_vm_baselines'].pop()
                # Update the convolution for the VMs that have been placed so far at this time point.
                conv_res = np.convolve(server.server['cpu_usage_pdf'][:-1], self.PROBABILITY_DENSITY_PER_BASELINE[cand_vm_baseline])
                r = sum(conv_res[server.server['capacity']*100:]) +  server.server['cpu_usage_pdf'][-1]
                server.server['cpu_usage_pdf'][:server.server['capacity']*100] = conv_res[:server.server['capacity']*100]
                server.server['cpu_usage_pdf'][-1] = r

            # Only calculate the convolution of the tail
            # if can_place_cython(server.server['cpu_usage_pdf'], PROBABILITY_DENSITY_PER_BASELINE[baseline], server.server['acceptable_violation']):
            if self.can_place(server.server['cpu_usage_pdf'], self.PROBABILITY_DENSITY_PER_BASELINE[baseline], server.server['acceptable_violation']):
                server.server['cand_vm_baselines'].append(baseline) # Add this VM's baseline to the list of candid VMs to be placed at this time
                return True
            else:
                return False
        else:
            server.server['cand_vm_baselines'].append(baseline) # Add this VM's baseline to the list of candid VMs to be placed at this time
            return True
    
    def can_place(self, cpu_usage_pdf, vm_pmf, acceptable_violation):
        if not any(cpu_usage_pdf[-len(vm_pmf)-1:]): # meaning the sum of convolution will be zero
            return True
        return np.sum(np.convolve(cpu_usage_pdf[-len(vm_pmf)-1:-1], vm_pmf)[-len(vm_pmf)+1:]) + cpu_usage_pdf[-1] <= acceptable_violation

    def _get_pmf_per_baseline(self):
        # reading PDF for every VMs
        size_str = 'baseline' if 'burstable' in self.params['ds_name'] else 'corecount'
        return np.load(PROJECT_DIR + 'data/probability_density_training_vms_per_{}_{}.npy'.format(size_str, self.params['ds_name']), allow_pickle = True).reshape(1, )[0]
    

class OversubscriptionOracleScheduler(Scheduler):
    def __init__(self, data_files, params):
        super().__init__(data_files, params)
        self._gen_esential_data()
    
    def _gen_esential_data(self):
        self.coef = float(self.params.get('first_model').strip('X'))
        self.allocated_cores_per_vmid = self._get_allocated_cores()
        self.data_files['vmid_to_avg'] = {vmid: self.allocated_cores_per_vmid[vmid]*np.ones(len(self.data_files['vmid_to_trace'][vmid])) for vmid in self.data_files['vmid_to_trace']}    
        
    def run_algorithm(self, time_idx, server, vmid):
        # Implement oversubscription-oracle scheduling algorithm
        return server.server['mean'][time_idx] + self.allocated_cores_per_vmid[vmid] <= server.server['capacity']
    
    def _get_allocated_cores(self):
        # for all the traces, calculate the coef-baseline
        return {vmid: self.data_files['vmid_to_baseline'][vmid] * self.coef for vmid in self.data_files['vmid_to_baseline']}
     
    def modify_record_bools(self):
        pass
        # Change the default value
        
