import numpy as np
from configs import LARGE_BVM_BASELINE_THRESHOLD

class Server:
    def __init__(self, time_bound, server_capacity, acceptable_violation, algo_name, record_bools):
        self.server = {}
        self.server['acceptable_violation'] = acceptable_violation
        self.server['capacity'] = server_capacity
        self.server['deployed_times'] = []

        # Initialize server features
        for server_feature in [k[7:-5] for k in record_bools if record_bools[k]]:
            self.server[server_feature] = np.zeros(time_bound)

        if 'audible' in algo_name:
            # Only for audible method
            self.BINS = np.concatenate((np.arange(0, server_capacity + 0.01 , 0.01), [2 * server_capacity]))
            self.server['cpu_usage_pdf_from_usage'] = np.zeros(len(self.BINS)-1)
            self.server['cpu_usage_pdf'] = np.zeros(len(self.BINS)-1)
            self.server['cand_vm_baselines'] = []


    
    def run_carry_over(self, time_idx, window):
        violation_relative_idxs = np.where(self.server['usage'][time_idx: time_idx + window] > self.server['capacity'])[0]

        if len(violation_relative_idxs) > 0: # if the usage goes above capacity with this update
            carry_over_relative_idx = -1 # an index to show up to what point carry overs have been proceeded
            for violation_relative_idx in violation_relative_idxs:
                if violation_relative_idx > carry_over_relative_idx: # if the carry over has been processed to an indx beyond this violation point, no need to process it again
                    carry_over_relative_idx = violation_relative_idx
                    step = 0
                    while( ( (violation_relative_idx + time_idx + step)<(len(self.server['usage'])-1) ) and self.server['usage'][violation_relative_idx + time_idx + step] > self.server['capacity']):
                        carry_over = self.server['usage'][violation_relative_idx + time_idx + step] - self.server['capacity']
                        self.server['carry_over'][violation_relative_idx + time_idx + step] += carry_over
                        self.server['usage'][violation_relative_idx + time_idx + step] = self.server['capacity'] # cap usage to server capacity
                        step += 1
                        carry_over_relative_idx = violation_relative_idx + step
                        self.server['usage'][violation_relative_idx + time_idx + step] += carry_over

    def update_cpu_usage_pdf(self, time_idx):
        """
        Update the cpu_usage_pdf for the given server according to what has been observed at the previous time.
        """
        # UPDATE SERVER CPU Usage HISTOGRAM and PDF
        server = self.server  # Reference server dictionary

        if (time_idx - (24*12) - 1) >= 0:  # TODO change the 24*12 or 288 to be a parameter of the algorithm
            upc1 = server['usage'][time_idx - (24*12) - 1]  # Aggregated usage at the exact previous 289 time samples
            upc1_idx = min(int(upc1*100), 100*server['capacity'])

            upc2 = server['usage'][time_idx - 1]  # Aggregated usage at the exact previous 289 time samples
            upc2_idx = min(int(upc2*100), 100*server['capacity'])

            server['cpu_usage_pdf'][upc1_idx] -= (1/288)
            server['cpu_usage_pdf'][upc2_idx] += (1/288)
        else:
            tot_sum = time_idx
            upc2 = server['usage'][time_idx - 1]  # Aggregated usage at the exact previous 289 time samples
            upc2_idx = min(int(upc2*100), 100*server['capacity'])

            server['cpu_usage_pdf'] = ((time_idx - 1)/time_idx)*server['cpu_usage_pdf']
            server['cpu_usage_pdf'][upc2_idx] += 1/time_idx

        server['cand_vm_baselines'] = []  # reset this to an empty list
    
    def _get_window(self, trace, time_idx, time_bound):
        return min(len(trace), time_bound - time_idx)

    def _get_active_vmids(self, violation_idx): # TODO: to record the active VMs at the violation point for analysis
        pass
    
    def _update_usage(self, time_idx, window, trace):
        self.server['usage'][time_idx: time_idx + window] += trace[:window]

    def _update_mean(self, time_idx, window, trace_pred_avgs):
        self.server['mean'][time_idx: time_idx + window] += trace_pred_avgs[:window]

    def _update_variance(self, time_idx, window, trace_pred_vars):
        self.server['variance'][time_idx: time_idx + window] += trace_pred_vars[:window]

    def _update_third_mom(self, time_idx, window, trace_pred_third_mom, baseline):
        self.server['third_mom'][time_idx: time_idx + window] += trace_pred_third_mom[:window]
        # TODO: Add logic for updating third_mom based on the condition you mentioned

    def _update_baselines(self, time_idx, window, baseline):
        self.server['baselines'][time_idx: time_idx + window] += baseline

    def _update_peakcpus(self, time_idx, window, baseline):
        # Assuming PEAK_CPU_PER_BASELINE is defined elsewhere
        self.server['peakcpus'][time_idx: time_idx + window] += PEAK_CPU_PER_BASELINE[baseline]

    def _update_num_large(self, time_idx, window, baseline):
        if baseline > LARGE_BVM_BASELINE_THRESHOLD:
            self.server['num_large'][time_idx: time_idx + window] += 1

    def _update_num_long_short_large(self, vmid, time_idx, window, trace, ds_name):
        if len(trace) > 288:
            self.server['number_lr'][time_idx: time_idx + window] += 1
        else:
            self.server['number_sr'][time_idx: time_idx + window] += 1

    def _update_deployed_times(self, vmid, time_idx, window):
        self.server['deployed_times'].append((time_idx, time_idx + window, vmid))
    
    def update_server(self, vmid, time_idx, params, baseline, record_bools, data_files):
        trace = data_files['vmid_to_trace'][vmid]
        window = self._get_window(trace, time_idx, params['time_bound'])
        last_week_bool = (time_idx + window) >= (params['time_bound'] - params['steady_state_time'])

        if record_bools['record_number_lr_bool'] or record_bools['record_number_sr_bool']:
            self._update_num_long_short_large(vmid, time_idx, window, trace, params)

        if record_bools['record_usage_bool']:
            self._update_usage(time_idx, window, trace)
        if record_bools['record_variance_bool']:
            self._update_variance(time_idx, window, data_files['vmid_to_var'][vmid])
        if record_bools['record_mean_bool']:
            self._update_mean(time_idx, window, data_files['vmid_to_avg'][vmid])
        if record_bools['record_num_large_bool']:
            self._update_num_large(time_idx, window, baseline)
        if record_bools['record_third_mom_bool']:
            self._update_third_mom(time_idx, window, third_mom_per_vmid[vmid], baseline)

        if last_week_bool and record_bools['record_baselines_bool']:
            self._update_baselines(time_idx, window, baseline)
        if last_week_bool and record_bools['record_peakcpus_bool']:
            self._update_peakcpus(time_idx, window, baseline)

        if last_week_bool:
            self._update_deployed_times(vmid, time_idx, window)