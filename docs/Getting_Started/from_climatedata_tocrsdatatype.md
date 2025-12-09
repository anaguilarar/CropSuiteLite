# Preprocessing Input Data

Before running CropSuiteLite, raw climate and soil variables must be transformed into climatological daily raster products.

## Climate dataset transformation

The original NEX-GDDP-CMIP6 data is provided at 0.25° resolution. This module reprocesses it to **0.05°** to match CropSuiteLite's spatial requirements. While this example uses a 20-year climatological period, this duration is configurable.

This step generates daily climatological raster datasets for temperature and precipitation.

To perform the transformation, define the parameters in `yaml_configurations/crop_suite_datasets.yaml`. Key settings include:

*   **Reference Mask:** Defines the target resolution and coordinate grid (`reference_mask_layer_path`).
*   **Clip Extent:** The geographic bounding box.
*   **Periods:** The time windows for climatological aggregation.
*   **SSPs:** The emission scenarios to process.

Example configuration:
``` Yaml
GENERAL_INFO:
  process_climate: True
  clip_extent: [-26,-37, 56, 40] # AFRICA
  reference_mask_layer_path: 'data/africa_mask_005.tif' 

CLIMATE:
  input_path: 'climate/nex-gddp-cmip6_AFRICA/'
  output_path: 'climate/nex-gddp-cmip6_AFRICA_cs_005/'
  periods: 
    - [2021,2040]
    - [2041,2060]
    - [2061,2080]
    - [2081, 2100]
  ssps: ['ssp585']

```

Once the configuration file is ready, run the download script:


``` bash
python datasets/create_spatial_datasets.py -config ./yaml_configurations/crop_suite_datasets.yaml
```

After processing the climate data (e.g., for the access-esm1-5 model under ssp585), the output directory structure should look like this:

``` Text
CropSuiteLite/
│
├── climate/
│    ├── nex-gddp-cmip6_AFRICA_cs/
│    └── nex-gddp-cmip6_AFRICA_cs_005/
│         ├── access-esm1-5_ssp585_2021_2040/
│         │    ├── Prec_avg.tif
│         │    └── Temp_avg.tif
│         ├── access-esm1-5_ssp585_2041_2060/
│         │    ├── Prec_avg.tif
│         │    └── Temp_avg.tif
│         ├── access-esm1-5_ssp585_2061_2080/
│         │    ├── Prec_avg.tif
│         │    └── Temp_avg.tif
│         └── access-esm1-5_ssp585_2081_2100/
│              ├── Prec_avg.tif
│              └── Temp_avg.tif
```

## Other datasets
In addition to climate variables, CropSuiteLite requires soil data (from [SoilGrids](https://soilgrids.org/)) and terrain information (SRTM elevation).
To mask and resample these datasets, update the same configuration file (crop_suite_datasets.yaml) by enabling soil processing and specifying the directories.


``` Yaml
GENERAL_INFO:

  process_soil: True
  clip_extent: [-26,-37, 56, 40] # AFRICA
  reference_mask_layer_path: 'data/africa_mask_005.tif' 

SOIL:
  input_path: '../soilgrids/'
  output_path: '../soilgrids_005/'

```

Run the processing script:


``` bash
python datasets/create_spatial_datasets.py -config ./yaml_configurations/crop_suite_datasets.yaml
```