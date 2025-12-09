# CropSuiteLite execution

This guide demonstrates how to execute **CropSuiteLite**, a lighter version of the CropSuite framework.
Key enhancements in this version include:

-   **No GUI**: Streamlined for server/HPC environments.
-   **Parallelization**: Multi-core processing for faster results.
-   **Custom Tiling**: User-defined memory management via tiling options.


## 1. Core Configuration (.ini)

CropSuiteLite maintains the original idea of using a configuration file, which defines all required paths, parameters, and modeling options.
Below is an example `.ini` configuration:

> example "View Example Configuration (`config_win_test.ini`)"

``` Ini
[files]
output_dir = results
climate_data_dir = ..\climate_data\mpi-esm1_ssp585_2021_2040
plant_param_dir = plant_params
fine_dem = data\srtm_1km_world_recalculated.tif
land_sea_mask = data\worldclim_land_sea_mask.tif
texture_classes = data\usda_texture_classification.dat

[options]
max_workers = 30
no_tiles = 15
use_scheduler = y
irrigation = 0
day_interval = 10
precipitation_downscaling_method = 1
temperature_downscaling_method = 1
output_format = geotiff
output_all_startdates = y
output_grow_cycle_as_doy = y
downscaling_window_size = 4
downscaling_use_temperature_gradient = y
downscaling_dryadiabatic_gradient = 0.00976
downscaling_saturation_adiabatic_gradient = 0.007
downscaling_temperature_bias_threshold = 0.0005
downscaling_precipitation_bias_threshold = 0.0001
downscaling_precipitation_per_day_threshold = 0.5
output_all_limiting_factors = y
remove_interim_results = y
output_soil_data = y
multiple_cropping_turnaround_time = 21
remove_downscaled_climate = n
rrpcf_interpolation_method = linear
consider_crop_rotation = y
simulate_calcification = 0
resolution = 4
xarray_process = False
debug = 1

[extent]
upper_left_x = -25.0			! Longitude of upper left corner, Format as decimal degree
upper_left_y = 39.0			! Latitude of upper left corner
lower_right_x = 55.0			! Longitude of lower right corner
lower_right_y = -36.0			! Latitude of lower right corner

[parameters.base_saturation]
data_directory = ..\soilgrids\bsat
weighting_method = 0
weighting_factors = 1.0,0.0,0.0,0.0,0.0,0.0
conversion_factor = 1.0
no_data = -128.0
interpolation_method = 0
rel_member_func = base_sat

[parameters.coarse_fragments]
data_directory = ..\soilgrids\cfvo
weighting_method = 2
weighting_factors = 2,1.5,1,0.75,0.5,0.25
conversion_factor = 10
interpolation_method = 0
rel_member_func = coarsefragments

[parameters.clay_content]
data_directory = ..\soilgrids\clay
weighting_method = 2
weighting_factors = 2,1.5,1,0.75,0.5,0.25
conversion_factor = 10
interpolation_method = 0
rel_member_func = texture

[parameters.gypsum]
data_directory = ..\soilgrids\gyps
weighting_method = 0
weighting_factors = 2,1.5,1,0.75,0.5,0.25
conversion_factor = 10
interpolation_method = 0
rel_member_func = gypsum

[parameters.pH]
data_directory = ..\soilgrids\ph
weighting_method = 2
weighting_factors = 2.0,1.5,1.0,0.75,0.5,0.25
conversion_factor = 10.0
interpolation_method = 0
rel_member_func = ph

[parameters.salinity]
data_directory = ..\soilgrids\sal
weighting_method = 0
weighting_factors = 2,1.5,1,0.75,0.5,0.25
conversion_factor = 1
interpolation_method = 0
rel_member_func = elco

[parameters.sand_content]
data_directory = ..\soilgrids\sand
weighting_method = 2
weighting_factors = 2,1.5,1,0.75,0.5,0.25
conversion_factor = 10
interpolation_method = 0
rel_member_func = texture

[parameters.soil_organic_carbon]
data_directory = ..\soilgrids\soc
weighting_method = 1
weighting_factors = 2,1.5,1,0.75,0.5,0.25
conversion_factor = 100
interpolation_method = 0
rel_member_func = organic_carbon

[parameters.sodicity]
data_directory = ..\soilgrids\sod
weighting_method = 0
weighting_factors = 2,1.5,1,0.75,0.5,0.25
conversion_factor = 1
interpolation_method = 0
rel_member_func = esp

[parameters.soildepth]
data_directory = ..\soilgrids\soildepth
weighting_method = 0
weighting_factors = 2,1.5,1,0.75,0.5,0.25
conversion_factor = 100
interpolation_method = 0
rel_member_func = soildepth

```

### Basic python execution
Once your `.ini` file is configured, you can run the model directly via Python:

``` Python
from CropSuite import CropSuiteLite

config_file_path = "ini_configurations/config_win_test.ini"
cs_lite = CropSuiteLite(config_file=config_file)
cs_lite.run()
```
## 2. Automated workflow

For complex workflows involving multiple scenarios or resolutions, it is recommended to control the model using a YAML configuration file. This file acts as a high-level wrapper that automatically updates the underlying `.ini` settings.

### YAML configuration

Modify `yaml_configurations/general_config_solutions.yaml` to centralize your run parameters.

``` Yaml
GENERAL:
  climate_scenarios: '..\nex-gddp-cmip6_AFRICA_cs_005\mpi-esm1-2-hr_ssp245_2021_2040'
  soil_grids_data: '..\soilgrids_005'
  srtm_path: 'data\srtm_005_world_recalculated.tif'
  landsea_path: 'data\worldclim_land_sea_mask_005.tif'
  output_path: 'results'
  config_path: 'ini_configurations/config_win_test.ini'
  plant_param_dir: 'plant_params/available'
  plant_params_output: 'results/plant_params'
  extent: [-25.0, 39.0, 55.0, -36.0	] ## xmin, ymax, xmax, ymin
  final_resolution: 4
  max_workers: 5
  no_tiles: 1
  day_interval: 10
  consider_crop_rotation: n
  xarray_process: False

```
### Python execution with YAML

The following script reads the YAML, generates a temporary `.ini` file with the updated paths, and executes the model:

``` Python
from CropSuite import CropSuiteLite
from src.utils import modify_initial_cropsuite_config
import yaml

# Load YAML configuration (assumed loaded into config_dict)

config_file_path = modify_initial_cropsuite_config(config_dict)
print(f'The new file is located at: '{config_file_path})

# Run the model
cs_lite = CropSuiteLite(config_file=config_file_path)
cs_lite.run()
```

## 3. Batch Processing

The automation module supports batch processing for multiple spatial resolutions or climate scenarios.

### Spatial Resolutions

To run the model across different resolutions, provide a list of integers in the YAML file. These integers correspond to WGS84 grid sizes:

- 0: 0.5°
- 1: 0.25°
- 2: 0.1°
- 3: 0.083333°
- 4: 0.041666°
- 5: 0.008333°


``` Yaml
GENERAL:
  final_resolution: 
    - 3
    - 4
    - 5
```

### Multiple climate scenarios

To process multiple time periods or models sequentially:

``` Yaml
GENERAL:
  climate_scenarios: 
    - '..\nex-gddp-cmip6_AFRICA_cs_005\mpi-esm1-2-hr_ssp245_2021_2040'
    - '..\nex-gddp-cmip6_AFRICA_cs_005\mpi-esm1-2-hr_ssp245_2041_2060'
    - '..\nex-gddp-cmip6_AFRICA_cs_005\mpi-esm1-2-hr_ssp245_2061_2080'
```

### Command-line execution

Once the YAML is configured, run the full pipeline from the command line:

``` bash

python run_cropsuitelite.py -config yaml_configurations/general_config_solutions.yaml

```


## Outputs 

The model generates several raster outputs:

* **`crop_suitability.tif`**: Final suitability index (0–100), using the limiting-factor approach (minimum of climate and soil suitability).
* **`crop_suitability_multi.tif`**: An alternative suitability index where soil and climate suitability are multiplied rather than minimized.
*   **`climate_suitability.tif`**: The suitability score (0–100) derived solely from climate conditions (temperature, precipitation, solar radiation) during the optimal growing period.
*   **`soil_suitability.tif`**: The suitability score (0–100) derived solely from soil and terrain properties (e.g., texture, pH, SOC, slope, elevation).

*   **`optimal_sowing_date.tif`**: The Day of Year (DOY, 1–365) when the growing cycle starts to achieve the maximum calculated suitability.
*   **`suitable_sowing_days.tif`**: The total count of days in the year on which a crop cycle could successfully start and achieve a suitability score above the defined threshold.

*   **`limiting_factor.tif`**: A categorical raster indicating the single most restrictive constraint (the "Liebig minimum") for each pixel. Values correspond to specific variables (e.g., precipitation deficit, temperature stress, or specific soil constraints).
*   **`all_climlim_factors.tif`**: A detailed breakdown of specific climate constraints. This file identifies which specific climate variable (e.g., drought, heat stress, vernalization requirements) reduced the suitability score during the growing cycle.
