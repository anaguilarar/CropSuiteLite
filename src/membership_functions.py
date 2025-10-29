import copy
import os
import numpy as np

class CropSensitivity():
    """
    A class to read, handle, and write crop sensitivity parameters from .inf files.
    """
    def __init__(self, crop, parameters_path):
        """
        Initializes the CropSensitivity class.

        Args:
            crop (str): The name of the crop (e.g., 'beans').
            parameters_path (str): The path to the directory containing the parameter files.
        """
        self.crop = crop
        self.parameters_path = parameters_path
        self.params = {}

    def read_crop_configuration(self):
        """
        Reads the crop configuration .inf file and parses the parameters.

        The method reads the file line by line, splitting keys and values.
        It handles the crop name as a string, single values as floats,
        and parameter curves as lists of floats.

        Returns:
            dict: A dictionary containing the parsed crop parameters.
        """
        fn = os.path.join(self.parameters_path, f'{self.crop}.inf')
        print(fn)
        assert os.path.exists(fn), f'The file name path {fn} does not exist'

        config = {}
        with open(fn, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'name':
                        config[key] = value
                    else:
                        values = [v.strip() for v in value.split(',')]
                        try:
                            # Convert all values to float
                            numeric_values = [float(v) for v in values]
                            # If there's only one value, store it as a single number
                            if len(numeric_values) == 1:
                                config[key] = numeric_values[0]
                            else:
                                config[key] = numeric_values
                        except ValueError:
                            # If conversion to float fails, store as string(s)
                            if len(values) == 1:
                                config[key] = values[0]
                            else:
                                config[key] = values
        self.params = config
        return self.params

    def write_configuration(self, output_path=None):
        """
        Writes the current crop parameters to a .inf file.

        Args:
            output_path (str, optional): The path to write the file to.
                                        If not provided, it overwrites the original file.
        """
        if output_path is None:
            output_path = os.path.join(self.parameters_path, f'{self.crop}.inf')

        with open(output_path, 'w') as f:
            for key, value in self.params.items():
                f.write(f'{key} = \t\t')
                if isinstance(value, list):
                    f.write(','.join(map(str, value)) + '\n')
                else:
                    f.write(str(value) + '\n')

    def create_new_max_parameter_values(self, parameter: str, new_max: float, percentage = True) -> np.ndarray:
        """
        Shifts and stretches the parameter suitability value to a new maximum .

        Args:
            parameter (str): Parameter's name
            new_max (float): The new maximum parameter's value.

        Returns:
            np.array: The new array of values for the transformed range.
        """
        assert parameter + '_vals' in self.params, f'{parameter} is not in the parametes list {var.params}'
        parameter_values = copy.deepcopy(self.params[parameter + '_vals'])
        parameter_suit = copy.deepcopy(self.params[parameter + '_suit'])
        
        min_value = parameter_values[0]
        original_max_value = parameter_values[-1]
        if percentage:
            new_max = original_max_value * (1+(new_max/100))
        
        
        optimal_value_index = np.argmax(parameter_suit)
        optimal_value = parameter_values[optimal_value_index]

        original_range = original_max_value - min_value
        opt_ratio = 0 if original_range == 0 else (optimal_value - min_value) / original_range
        
        new_range = new_max - min_value
        new_optimal_value = min_value + (opt_ratio * new_range)

        new_param_vals = np.zeros(len(parameter_values)+1)
        for i in range(new_param_vals.shape[0]):
            if i <= optimal_value_index:
                val = parameter_values[i]
                new_param_vals[i] = val
            else:
                val = parameter_values[i-1]
                original_downhill_range = original_max_value - optimal_value
                if original_downhill_range == 0:
                    proportion = 0
                else:
                    proportion = (val - optimal_value) / original_downhill_range
                new_param_vals[i] = new_optimal_value + (proportion * (new_max - new_optimal_value))
                
        parameter_suit.insert(optimal_value_index, np.max(parameter_suit))
        return parameter_suit, new_param_vals
        
    
    def create_new_min_parameter_values(self, parameter: str, new_min: float) -> np.ndarray:
        """
        Shifts and stretches the parameter suitability value to a new maximum .

        Args:
            parameter (str): Parameter's name
            new_min (float): The new minimum parameter's value.

        Returns:
            np.array: The new array of values for the transformed range.
        """
        assert parameter + '_vals' in self.params, f'{parameter} is not in the parametes list {self.params}'
        parameter_values = self.params[parameter + '_vals']
        parameter_suit = self.params[parameter + '_suit']
        
        original_min_value = parameter_values[0]
        original_max_value = parameter_values[-1]
        optimal_value_index = np.argmax(parameter_suit)
        optimal_value = parameter_values[optimal_value_index]

        original_range = original_max_value - original_min_value
        opt_ratio = 0 if original_range == 0 else (optimal_value - original_min_value) / original_range
        
        new_range = original_max_value - new_min
        new_optimal_value = new_min + (opt_ratio * new_range)

        new_param_vals = np.zeros_like(parameter_values)
        for i, val in enumerate(parameter_values):
            if i <= optimal_value_index:
                original_uphill_range = optimal_value - original_min_value
                if original_uphill_range == 0:
                    proportion = 0
                else:
                    proportion = (val - original_min_value) / original_uphill_range
                new_param_vals[i] = new_min + (proportion * (new_optimal_value - new_min))
            else:
                # Rescale the "downhill" side (from optimal to max)
                original_downhill_range = original_max_value - optimal_value
                if original_downhill_range == 0:
                    proportion = 0
                else:
                    proportion = (val - optimal_value) / original_downhill_range
                
                new_param_vals[i] = new_optimal_value + (proportion * (original_max_value - new_optimal_value))
                
        return parameter_suit, new_param_vals
    
    def create_new_min_parameter_valuesv2(self, parameter: str, new_min: float) -> np.ndarray:
        """
        Shifts and stretches the parameter suitability value to a new maximum .

        Args:
            parameter (str): Parameter's name
            new_min (float): The new minimum parameter's value.

        Returns:
            np.array: The new array of values for the transformed range.
        """
        
        assert parameter + '_vals' in self.params, f'{parameter} is not in the parametes list {self.params}'
        parameter_values = copy.deepcopy(self.params[parameter + '_vals'])
        parameter_suit = copy.deepcopy(self.params[parameter + '_suit'])
        
        
        threshold_index = np.where(np.array(parameter_values) <= parameter_values[np.argmax(parameter_suit)])[0]
        threshold_index = threshold_index.tolist()
        #if np.argmax(parameter_suit) in threshold_index:
        #    threshold_index.drop(np.argmax(parameter_suit) )
        prec_vals_new = np.array(parameter_values).copy()
        prec_vals_new[threshold_index] = np.array(parameter_values)[threshold_index] * (1-new_min)
        prec_vals_new = prec_vals_new.tolist()
        #prec_vals_new.append(parameter_values[np.argmax(parameter_suit)])
        #parameter_suit.append(parameter_suit[np.argmax(parameter_suit)])
        
        prec_vals_new.insert(np.argmax(parameter_suit)+1, parameter_values[np.argmax(parameter_suit)])
        parameter_suit.insert(np.argmax(parameter_suit)+1, parameter_suit[np.argmax(parameter_suit)])
        return parameter_suit, prec_vals_new
    
    def multiply_suit_vals(self, parameter: str, perc_factor = 0):
        
        assert parameter + '_vals' in self.params, f'{parameter} is not in the parametes list {self.params}'
        
        parameter_suit = np.array(self.params[parameter + '_suit']).copy()
        factor_value = (perc_factor /100) if (perc_factor /100) > 1 else 1+(perc_factor /100)
        newvals = parameter_suit * factor_value
        
        newvals[newvals>1] = 1
        
        return newvals.tolist(), self.params[parameter + '_vals']
        

    def remove_crop_lethal_conditions(self):
        keys_names = list(self.params.keys())
        
        for k in keys_names:
            if k.startswith('lethal'):
                self.params.pop(k)
            elif k.endswith('_lim'):
                self.params.pop(k)
    
    def export_crop_params(self, scenarios,output_path:str, suit_vals = None, code = None, export_original = True):
        params_copy = self.params.copy()
        orig_nam = code if code else params_copy['name']
        
        if export_original:
            self.write_configuration(os.path.join(output_path, f'{orig_nam}_original.inf')) 
        
        for k,v in scenarios.items():
            var_name = k.split('_')[0]

            v = [round(float(i),2) for i in v]
            self.params[f'{var_name}_vals'] = v
            if suit_vals is not None: self.params[f'{var_name}_suit'] = suit_vals

            self.params['name'] = orig_nam + '_' + k
            nametoexport = self.params['name']
            self.write_configuration(os.path.join(output_path, f'{nametoexport}.inf'))    
            self.params = params_copy
        
    def plot_solutions_profiles(self, scenario_values, suit_values = None, output_fig_path:str = None):
        import matplotlib.pyplot as plt
        param_tochange = list(scenario_values.keys())[0].split('_')[0]
        
        limit = 'Min' if param_tochange == 'prec' else 'Max'
        unit = 'mm' if param_tochange == 'prec' else '°C'
        
        colors = ['green', 'purple', 'red']
        
        vals_orig, suit_orig = self.params[f'{param_tochange}_vals'], self.params[f'{param_tochange}_suit']
        ref_val = vals_orig[0] if param_tochange == 'prec' else vals_orig[::-1][0]
        
        opt_val = vals_orig[np.argmax(suit_orig)]
        
        template_label = '{name}: (Opt: {opt_val:.2f} {unit}, {limit}: {ref_val:.2f} {unit})'
        orig_label = template_label.format(name= 'Original', opt_val = opt_val, unit = unit, 
                                                                                    limit = limit, ref_val = ref_val)

        plt.figure(figsize=(10, 5.5))
        plt.plot(vals_orig, suit_orig, 'o-', label=orig_label, color='blue')
        for i, (k, v) in enumerate(scenario_values.items()):
            opt_val = v[np.argmax(suit_orig)]
            ref_val = v[0] if param_tochange == 'prec' else v[::-1][0]
            tmp_label = template_label.format(name = k, opt_val = opt_val, unit = unit, limit = limit, ref_val = ref_val)
            
            plt.plot(v, suit_values, '^--', label=tmp_label, color=colors[i])

        plt.xlabel('Precipitation (mm)' if param_tochange == 'prec' else 'Temperature (°C)')
        plt.ylabel('Suitability Score')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        if output_fig_path:
            plt.savefig(output_fig_path)
        else:
            plt.show()
