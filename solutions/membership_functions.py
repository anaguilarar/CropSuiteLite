import copy
import os
import yaml
from typing import List
        
import numpy as np

from .utils import find_solution_type

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
        assert parameter + '_vals' in self.params, f'{parameter} is not in the parametes list {self.params}'
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
        optimal_value_index = np.argmax(parameter_suit)
        if factor_value == 0: parameter_suit.tolist(), self.params[parameter + '_vals']
        
        newvals = parameter_suit[:optimal_value_index] * factor_value
        newvals[newvals>1] = 1
        newvals = newvals.tolist() + parameter_suit.tolist()[optimal_value_index:]

        return newvals, self.params[parameter + '_vals']
        
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
        
    def plot_solutions_profiles(self, scenario_values = None, variable = None, suit_values = None, output_fig_path:str = None, add_solution = True, labelsize = 15, figsize=(10, 5.5)):
        import matplotlib.pyplot as plt
        
        if variable is not None:
            param_tochange = variable
        elif variable is None and scenario_values is not None:
            param_tochange = list(scenario_values.keys())[0].split('_')[0]
        else:
            raise ValueError('you must provided either variable or scenario values')
        
        limit = 'Min' if param_tochange == 'prec' else 'Max'
        unit = 'mm' if param_tochange == 'prec' else '°C'
        
        colors = ['green', 'purple', 'red']
        
        vals_orig, suit_orig = self.params[f'{param_tochange}_vals'], self.params[f'{param_tochange}_suit']
        if suit_values is None: suit_values = suit_orig
            
        ref_val = vals_orig[0] if param_tochange == 'prec' else vals_orig[::-1][0]
        
        opt_val = vals_orig[np.argmax(suit_orig)]
        
        template_label = '{name}: (Opt: {opt_val:.2f} {unit}, {limit}: {ref_val:.2f} {unit})'
        orig_label = template_label.format(name= 'Original', opt_val = opt_val, unit = unit, 
                                                                                    limit = limit, ref_val = ref_val)
        plt.figure(figsize = figsize)
        plt.plot(vals_orig, suit_orig, 'o-', label=orig_label, color='blue')
        if add_solution and scenario_values is not None:
            print(param_tochange)
            for i, (k, v) in enumerate(scenario_values.items()):
                opt_val = v[np.argmax(suit_orig)]
                ref_val = v[0] if param_tochange == 'prec' else v[::-1][0]
                tmp_label = template_label.format(name = k, opt_val = opt_val, unit = unit, limit = limit, ref_val = ref_val)
                
                plt.plot(v, suit_values, '^--', label=tmp_label, color=colors[i])

        plt.xlabel('Precipitation (mm)' if param_tochange == 'prec' else 'Temperature (°C)', fontsize = labelsize, fontweight='bold')
        plt.ylabel('Suitability Score', fontsize = labelsize, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(self.crop.title(), fontsize = int(labelsize*1.3), fontweight='bold')
        plt.legend()
        plt.tick_params(axis='both', which='major', labelsize=int(labelsize*0.8))
        
        if output_fig_path:
            plt.savefig(output_fig_path)
        else:
            plt.show()

def change_otherst_parameters(thresholds: List, crops_params: CropSensitivity, v2 = True):
    
    new_params = None
    solution_parameter = {}
    
    i = 0 
    if thresholds[0] != 0:
        upperlimit_temp = thresholds[0]
        vals_orig = crops_params.params['temp_vals']
        maxtmp = vals_orig[::-1][0]
        new_maxtemp = maxtmp * (1+(upperlimit_temp/100))
        #new_maxtemp = maxtmp + upperlimit_temp
        suit_vals, new_params = crops_params.create_new_max_parameter_values(parameter='temp',new_max=new_maxtemp, percentage = False)
        solution_parameter[f'temp_{i+1}'] = new_params
        i +=1 
    
    if thresholds[1] != 0:
        lowerlimit_prec = thresholds[1]
        vals_orig = crops_params.params['prec_vals']

        print('--> lowerlimit_prec: ', lowerlimit_prec)
        suit_vals, new_params = crops_params.multiply_suit_vals('prec', lowerlimit_prec)
        #suit_vals, new_params = crops_params.create_new_min_parameter_valuesv2(parameter='prec',new_min=lowerlimit_prec/100)
        solution_parameter[f'prec_{i+1}'] = new_params
    
    return solution_parameter, suit_vals

def solution_dict_query(response_function_file: str, solution_code: str):
    assert os.path.exists(response_function_file), 'the file path does not exist'
    
    dict_query = {'crop':str  ,
                'solution_type':str,
                'solution': str,
                'thresholds': List,
                }
    
    with open(response_function_file, 'r') as file:
        sol_config_dict = yaml.safe_load(file)
    
    solution_type_code, solution_code, crop_code = solution_code.split('_')
    
    print(solution_type_code, solution_code, crop_code)
    
    solution_type = sol_config_dict['SOLUTIONS_TYPE'].get(solution_type_code, None)
    solution = sol_config_dict['SOLUTIONS_CODE'].get(solution_code, None)
    crop_str = sol_config_dict['CROPS'].get(crop_code, None)
    threshold = None if solution is None else sol_config_dict[solution_type_code][solution_code][crop_code]
    
    dict_query.update({'crop': crop_str,
                    'solution_type': solution_type,
                    'solution':  solution,
                    'crop': crop_str,
                    'thresholds':threshold})
    return  dict_query

def change_st1_parameter(solution: str, thresholds: List, crops_params: CropSensitivity, v2 = True):
    
    new_params = None
    solution_parameter = {}
    
    for i in range(len(thresholds)):
        if 'heat' in solution.lower():
            upperlimit_temp = thresholds[i]
            
            vals_orig = crops_params.params['temp_vals']
            maxtmp = vals_orig[::-1][0]
            new_maxtemp = maxtmp * (1+(upperlimit_temp/100))

            #new_maxtemp = maxtmp + upperlimit_temp
            suit_vals, new_params = crops_params.create_new_max_parameter_values(parameter='temp',new_max=new_maxtemp, percentage = False)
            
            solution_parameter[f'temp_{i+1}'] = new_params
            
        elif 'drought' in solution.lower():
            lowerlimit_prec = thresholds[i]
            vals_orig = crops_params.params['prec_vals']
            minprec = vals_orig[0]
            new_minprec = max(minprec - (minprec * lowerlimit_prec/100), 0)
            if v2: 
                print('--> lowerlimit_prec: ', lowerlimit_prec)
                suit_vals, new_params = crops_params.multiply_suit_vals('prec', lowerlimit_prec)
                #suit_vals, new_params = crops_params.create_new_min_parameter_valuesv2(parameter='prec',new_min=lowerlimit_prec/100)
            else:
                suit_vals, new_params = crops_params.create_new_min_parameter_values(parameter='prec',new_min=new_minprec)
            
            solution_parameter[f'prec_{i+1}'] = new_params
            
    return solution_parameter, suit_vals

def create_crop_parameters(config_dict):
    
    solution_codes = config_dict['SOLUTIONS'].get('solution_codes')
    crop_codes = config_dict['SOLUTIONS'].get('crop_codes')
    solutions_path = config_dict['SOLUTIONS'].get('solutions_path')
    plant_params_input = config_dict['GENERAL'].get('plant_param_dir', 'plant_params/available')
    plant_params_output = config_dict['GENERAL'].get('plant_params_output', 'plant_params')
    if not os.path.exists(plant_params_output): os.mkdir(plant_params_output)    
    if len(solution_codes):
        for crop_code in crop_codes:
            
            for i, sol_code in enumerate(solution_codes):
                solution_type = find_solution_type(sol_code)
                print('solution_type --> ',solution_type)

                if solution_type is None: continue
                
                solution_to_implement = f'{solution_type}_{sol_code}_{crop_code}'

                solution_implementation = solution_dict_query(solutions_path, solution_to_implement)
                print(solution_implementation)
                crops_params = CropSensitivity(crop = solution_implementation['crop'], 
                                            parameters_path = plant_params_input)
                crops_params.read_crop_configuration()
                crops_params.remove_crop_lethal_conditions()
                if len(solution_implementation['thresholds']):
                    if solution_type == 'ST1':
                        sv, suit_vals = change_st1_parameter(solution_implementation['solution'],solution_implementation['thresholds'],  crops_params, v2 = True)
                    else:
                        sv, suit_vals = change_otherst_parameters(solution_implementation['thresholds'],  crops_params)
                        
                    crops_params.plot_solutions_profiles(sv, suit_values = suit_vals, output_fig_path=os.path.join(plant_params_output,f'{solution_to_implement}.png'))
                    crops_params.export_crop_params(sv, plant_params_output, suit_vals = suit_vals, export_original = i == 0, code = solution_to_implement)
                    with open(os.path.join(plant_params_output, f'{solution_to_implement}.yaml'), 'w') as file:
                        yaml.dump(solution_implementation, file, default_flow_style=False)
    else:
        for crop_code in crop_codes:
            solution_to_implement = f'ST1_null_{crop_code}'
            solution_implementation = solution_dict_query(solutions_path, solution_to_implement)
            cname = solution_implementation['crop']
            crops_params = CropSensitivity(crop = cname, 
                                            parameters_path = plant_params_input)
            crops_params.read_crop_configuration()
            crops_params.remove_crop_lethal_conditions()
            #crops_params.export_crop_params(sv, plant_params_output, suit_vals = suit_vals, export_original = i == 0, code = solution_to_implement)
            crops_params.write_configuration(os.path.join(plant_params_output, f'{cname}.inf'))

        print(solution_implementation)
        


