import os
import warnings

from typing import Dict

def find_solution_type(crop_code):
    
    soltypes = {
            'ST1': ['s1', 's2'],
            'ST2': ['s3', 's4', 's5', 's6', 's21', 's22', 's25', 's26', 's28']
        }

    solution_type = [k for k,v in soltypes.items() if crop_code in v]
    if len(solution_type):
        return solution_type[0]
    else:
        return None


def modify_soil_parameters(cropsuite_configuration, parameters_to_modify:Dict):
    
    soil_parameters = {'base_saturation':'bsat', 'coarse_fragments':'cfvo', 'clay_content':'clay', 
                       'gypsum':'gyps', 'pH':'ph', 'salinity':'sal', 
                       'sand_content':'sand', 'soil_organic_carbon':'soc', 
                       'sodicity':'sod', 'soildepth':'soildepth'}
    
    for soil_param, val in soil_parameters.items():
        for k, v in parameters_to_modify.items():
            if k == 'data_directory':
                #if os.listdir(v):
                cropsuite_configuration[f'parameters.{soil_param}'][k] = os.path.join(v, val)
            else:
                cropsuite_configuration[f'parameters.{soil_param}'][k] = v

    return cropsuite_configuration

def modify_general_files(cropsuite_configuration, parameters_to_modify:Dict):
    general_config_list = list(cropsuite_configuration['files'].keys())

    for k, v in parameters_to_modify.items():
        #if not (k in general_config_list): continue
        if k in list(cropsuite_configuration['options'].keys()):
            cropsuite_configuration['options'][k] = v
        elif k in general_config_list:
            cropsuite_configuration['files'][k] = v
        else:
            warnings.warn(f"{k} is not in options either files")

    return cropsuite_configuration

def modify_extent(cropsuite_configuration, parameters_to_modify:Dict):
    extent_list = list(cropsuite_configuration['extent'].keys())

    for k, v in parameters_to_modify.items():
        #if not (k in general_config_list): continue
        if k in extent_list:
            cropsuite_configuration['extent'][k] = v
        
    return cropsuite_configuration