# CropSuiteLite

**CropSuiteLite** is a lightweight, optimized Python implementation of the crop suitability modeling framework presented in the [CropSuite paper by Zabel et al. (2025)](https://gmd.copernicus.org/articles/18/1067/2025/). It is designed to assess how suitable a geographic area is for growing different crops by analyzing climate and soil conditions.

This version focuses on performance, scalability, and flexibility, incorporating modern data science libraries and parallel processing to handle large datasets and complex scenarios efficiently.

> **Disclaimer:**  
> This repository is an independent implementation based on the concepts, workflow, and structure described in *Zabel et al. (2025)*, “CropSuite: An open modular framework for site-specific crop modeling at global scale.”  
> The official CropSuite framework is available from the [CropSuite Zenodo repository](https://zenodo.org/records/16759895).

## Key Features
There are a few implementations that were included into this version:

- **Parallel Processing:** Utilizes parallel processing (`concurrent.futures`) to accelerate computationally intensive steps such as climate data downscaling and daily climate suitability analysis.
- **Scalable with Xarray and Dask:** Offers an alternative processing backend using `xarray` and `dask`, enabling out-of-core computation that can handle datasets larger than available RAM. Only available for non-winter crops.
- **Memory management tiling:** Offers a manual option to divide large geographic areas into tiles for processing. Original version divides the tiles depending on the RAM size. This was done to make it feasible to run high-resolution analysis on standard hardware.
- **Flexible configuration:** Uses YAML files for run configuration, allowing for easy management of different scenarios and climate inputs.
- **Customizable time step:** Allows the user to adjust the time step used in climate analysis. In the original version, climate suitability is computed daily; in this implementation, the time step can be customized.

## Documentation Overview

This documentation covers:

- **Installation & Setup**: Environment configuration and dependencies.
- **Getting Started**: Downloading CMIP6 data and running your first model with CropSuiteLite.
- **Atlas Solutions**: Examples, data visualization, and supporting materials.
- **API Reference**: Detailed documentation of the code modules.

Select a topic from the top navigation bar to begin with.