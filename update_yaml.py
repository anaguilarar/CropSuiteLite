import yaml
import sys

def main(scenario):
    config_path = 'tmp_configurations/general_config_solutions.yaml'
    
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['GENERAL']['climate_scenarios'] = [scenario]

    print(config_dict['GENERAL'])
    
    with open(config_path, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

if __name__ == '__main__':
    scenario = sys.argv[1]
    main(scenario)