# Download daily climate scenarios

To conduct future suitability comparisons, it is necessary to obtain climate scenario data.
This example demonstrates how to download climate variables from the [NEX-GDDP-CMIP6 project](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6), which provides statistically downscaled daily climate projections for the CMIP6 experiment across “Tier 1” SSP emission scenarios.

??? "Dataset Version"
    The code supports **Version 2 (v2)** of the dataset. If a specific product is not yet available in v2, the tool automatically falls back to the previous version

## Python API Usage

You can download specific scenarios by defining SSPs, global climate models (GCMs), time periods, and meteorological variables directly in Python.

### 1. Define the Query
First, initialize the downloader and set your query parameters:

``` Python
from datasets.download_data import DownloadCMIP6Data

yearsofinterest = list(range(2021, 2100))
cmip_dowloader = DownloadCMIP6Data(url_root = config.CLIMATE.url_root)


cmip_dowloader.set_query(
    years=yearsofinterest,
    gcms=['ACCESS-ESM1-5', 'MPI-ESM1-2-HR'],
    ssps=['ssp585'],
    variables=['pr', 'rsds']
)

print(cmip_dowloader.products_to_download)

```

### 2. Download a Product

You can iterate through the ```products_to_download``` list to download specific files:

``` Python
from pathlib import Path

# Select a single product from the list (example: index 0)
id_product = 0
gcm, ssp, var, year = list(cmip_dowloader.products_to_download)[id_product]
url_dir, fn = cmip_dowloader.find_remote_file(gcm, ssp, var, year)

# Path where the data will be saved
output_path = Path(output_dir) / var / ssp / gcm
# Start the download
dwnd_path = cmip_dowloader.download_data(url_dir, fn, output_path)

```

### 3. Preprocessing and Clipping
Raw downloaded files cover the global domain. You can clip the dataset to a specific geographic extent and convert units using the ProcessTools class.

Meteorological variables also require unit conversion:

* Temperature variables (tas, tasmin, tasmax) are provided in Kelvin.

* Precipitation (pr) requires multiplication by 86,400 to convert from kg m⁻² s⁻¹ to mm/day.


``` Python
from datasets.download_data import ProcessTools
xrdata_processor = ProcessTools()

clip_extent = [-26, -37, 56, 40]  # min lon, min lat, max lon, max lat
xrdata = xrdata_processor.clip_spdata(
    clip_extent,
    xrdata_path=dwnd_path,
    rotate=True,
    chunks='auto'
)


# Preprocess (Unit conversion)
xrdata = xrdata_processor.preprocess_xrdata(xrdata)

# Export the processed dataset as a NetCDF file
xrdata_processor.save_asnc(xrdata, fn=dwnd_path)
```



## Command-Line Usage

Downloading can be fully automated using a YAML configuration file.


### 1. Configure the YAML

The file ```download_cmip6.yaml``` (located in ```yaml_configuration/```) allows users to specify:

``` Yaml
GENERAL_INFO:
  output_dir: '../nex-gddp-cmip6_AFRICA/'  # directory where data will be downloaded
  variables: ['pr', 'tasmin', 'tasmax', 'rsds']  # meteorological variables
  years: [2021, 2100]  # period of interest
  clip_extent: [-26, -37, 56, 40]  # Africa extent
  nworkers: 16  # maximum number of download workers

CLIMATE:
  gcms: ['ACCESS-ESM1-5', 'MPI-ESM1-2-HR', 'EC-Earth3', 'INM-CM5-0', 'MRI-ESM2-0'] # climate models
  ssps: ['ssp245', 'ssp126', 'ssp585']  # emission scenarios
  url_root: 'https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/NEX-GDDP-CMIP6'

```

### 2. Download the data

Once the configuration file is ready, run the download script:


``` bash
python datasets/download_data.py -config yaml_configuration/download_cmip6.yaml
```

