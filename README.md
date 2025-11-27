# CropSuiteLite

**CropSuiteLite** is a lightweight implementation inspired by the open-source framework **CropSuite** developed by [Zabel et al. (2025)](https://gmd.copernicus.org/articles/18/1067/2025/).  
This repository provides a simplified and modular version of the original CropSuite framework.


> **Disclaimer:**  
> This repository is an independent implementation based on the concepts, workflow, and structure described in *Zabel et al. (2025)*, “CropSuite: An open modular framework for site-specific crop modeling at global scale.”  
> The official CropSuite framework is available from the [CropSuite Zenodo repository](https://zenodo.org/records/16759895).

## Key Features

- **Performance Optimized:** Utilizes parallel processing (`concurrent.futures`) to accelerate computationally intensive steps like climate data downscaling and suitability analysis.
- **Scalable with Xarray and Dask:** Offers an alternative processing backend using `xarray` and `dask`, enabling out-of-core computation that can handle datasets larger than available RAM.
- **Memory Management with Tiling:** Sets a manual option to divide large geographic areas into tiles.
- **Flexible Configuration:** Uses YAML files for run configuration, allowing for easy management of different scenarios, crop varieties (e.g., heat or drought-tolerant "solutions"), and climate inputs.

## Dependencies

The model requires Python 3 and the libraries listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `numpy`
- `rasterio`
- `xarray` & `netCDF4`
- `dask`
- `numba`
- `scipy`

## How to Run the Model

The model is executed from the command line using the `run_cropsuitelite.py` script. You must provide a path to a YAML configuration file.

```bash
python run_cropsuitelite.py -config yaml_configurations/general_config_solutions.yaml
```

## Configuration File (`general_config_solutions.yaml`)

This YAML file provides a high-level, user-friendly way to configure a `CropSuiteLite` model run. It is divided into two main sections: `GENERAL` and `SOLUTIONS`.

### `GENERAL` Section

This section defines the core parameters for the model run, including data paths, geographic area, and performance settings.

*   **Data Paths:**
    *   `climate_scenarios`, `soil_grids_data`, `srtm_path`, `landsea_path`: These keys specify the locations of all necessary input data, such as climate model outputs, soil property maps, the digital elevation model (for slope), and the land-sea mask.
    *   `output_path`: Defines the main directory where all results will be saved.
    *   `config_path`: Points to a base `.ini` file that serves as a template for the detailed model configuration.
    *   `plant_param_dir` and `plant_params_output`: Specify the input directory for base crop parameters and the output directory for any modified "solution" parameters.

*   **Geographic and Resolution Settings:**
    *   `extent`: Defines the precise geographic bounding box for the analysis, specified as `[xmin, ymax, xmax, ymin]`.
    *   `final_resolution`: An integer code that sets the spatial resolution of the output maps (e.g., `5` corresponds to approximately 1km resolution).

*   **Performance and Execution Settings:**
    *   `max_workers`: Sets the number of parallel CPU cores to use, allowing you to control the resource usage.
    *   `no_tiles`: Specifies the number of tiles to split the study area into. This is a key memory management feature for processing large areas.
    *   `day_interval`: Controls the time step for the climate analysis (e.g., `1` for daily calculations, `10` for every 10 days).
    *   `consider_crop_rotation`: A flag (`y`/`n`) to enable or disable the crop rotation analysis module.

### `SOLUTIONS` Section

This section is designed for scenario analysis, allowing you to test how different crop varieties or conditions might perform.

*   `solutions_path`: Points to another YAML file (`response_functions.yaml`) that contains the specific definitions for different "solutions" (e.g., parameters for a more drought-tolerant or heat-resistant version of a crop).
*   `solution_codes`: A list of codes specifying which solutions to apply. If left empty, the model runs a baseline scenario using the standard crop parameters.
*   `crop_codes`: A list of codes identifying which crops to include in the analysis.

## Data Download (CMIP6 Climate Scenarios)

CropSuiteLite includes a utility to download daily climate projection data from the CMIP6 NEX-GDDP dataset, which can then be used as input for the crop suitability model. This utility leverages parallel processing for efficient data acquisition.

### Usage Example

To download CMIP6 data, run the `download_data.py` script with its configuration file:

```bash
python datasets/download_data.py -config yaml_configurations/download_cmip6.yaml
```

## 📘 Reference
Zabel, F., Müller, C., Rezaei, E. E., Webber, H., Porwollik, V., Schaphoff, S., et al. (2025).  
**CropSuite: An open modular framework for site-specific crop modeling at global scale.**  
*Geoscientific Model Development*, 18, 1067–1094.  
[https://doi.org/10.5194/gmd-18-1067-2025](https://doi.org/10.5194/gmd-18-1067-2025)


