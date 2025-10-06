import os
import glob

import numpy as np

import xarray
import rioxarray as rio
from rasterio.enums import Resampling


def compute_averaged_temp(
    tmin_path: str,
    tmax_path: str,
    year: int,
    suffix: str = "_v2.0.nc"
    ) -> xarray.DataArray:
    """
    Compute the daily average temperature from minimum and maximum temperatures.

    Parameters
    ----------
    tmin_path : str
        Directory path containing the minimum temperature files.
    tmax_path : str
        Directory path containing the maximum temperature files.
    year : int
        Year of the files to process.
    suffix : str, optional
        Filename suffix to match (default is "_v2.0.nc").

    Returns
    -------
    xr.DataArray
        A DataArray containing the average daily temperature for the given year,
        with dimensions (time, lat, lon).
    """
    
    tmin_file =  glob.glob(tmin_path + f'/*_{year}' + suffix)
    tmax_file = glob.glob(tmax_path + f'/*_{year}' + suffix)
        
    if not (len(tmin_file)==1 and len(tmax_file)==1):
        raise ValueError(f"No unique tmin/tmax files found for year {year}")
    
    xrdata = xarray.open_mfdataset([tmin_file[0], tmax_file[0]],  combine='nested') 
    
    tasavg = xarray.concat([xrdata["tasmax"], xrdata["tasmin"]], dim="var").mean(dim="var", skipna=True)
    tasavg.name = "tasavg"
    tasavg.attrs = xrdata.attrs
    return tasavg


class BaseProcessor():
    def __init__(self, input_pathdir: str = None) -> None:
        """
        Initialize the preprocessing class.

        """
        if input_pathdir:
            assert os.path.exists(input_pathdir), f'The input path does not exist {input_pathdir}'
        self.input_path = input_pathdir

    @staticmethod
    def directories_list(path):
        return [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]

    @staticmethod
    def mask_data(xrdata: xarray.DataArray , mask_layer: xarray.DataArray, method: str = "bilinear"
    ) -> xarray.DataArray:
        """
        Mask and resample data to match a reference layer.

        Parameters
        ----------
        xrdata : xarray.DataArray
            Input soil data.
        mask_layer : xarray.DataArray
            Mask layer defining spatial extent and resolution.
        method : {'bilinear', 'nearest'}, default='bilinear'
            Resampling method.

        Returns
        -------
        xarray.DataArray
            Resampled and masked soil property layer.
        """
        
        
        if method == 'bilinear':
            interp_fun = Resampling.bilinear
        elif method == 'nearest':
            interp_fun = Resampling.nearest
        
        if xrdata.rio.height != mask_layer.rio.height or xrdata.rio.width != mask_layer.rio.width:
            x_min, y_min, x_max, y_max = float(min(mask_layer.x).values), float(min(mask_layer.y).values), float(max(mask_layer.x).values), float(max(mask_layer.y).values)
            
            xrdata_r = xrdata.sel(x = slice(x_min,x_max), y = slice(y_min, y_max))
            
            if xrdata_r.sizes.get("y", 0) == 0:
                # Handle reversed y-axis
                soil_prop_red = xrdata.sel(x = slice(x_min,x_max), y = slice(y_max, y_min))
                
                
            xrdata_r = xrdata_r.rio.reproject_match(mask_layer, resampling = interp_fun)

        return xrdata_r.where(mask_layer, np.nan)
    

class Resample_SoilGridsData(BaseProcessor):
    """
    Utility class to handle resampling and masking of SoilGrids data.

    Parameters
    ----------
    input_pathdir : str
        Root directory containing SoilGrids GeoTIFF files organized
        by soil property variable.
    """
    def __init__(self, input_pathdir: str) -> None:
        """
        Initialize the preprocessing class.

        Parameters
        ----------
        input_pathdir : str
            Root directory containing climate model outputs.
        """
        
        super().__init__(input_pathdir)
        self._files_path_dict = None
        
    
    @property
    def files_path_dict(self):
        """
        Dictionary mapping soil variables to available GeoTIFF file paths.

        Returns
        -------
        dict of str -> list of str
            Keys are variable IDs (directory names),
            values are lists of corresponding `.tif` file paths.
        """
        if self._files_path_dict is None:
            self._files_path_dict  = {i:glob.glob(os.path.join(self.input_path, i) + f'/*.tif') for i in self.directories_list(self.input_path)}
            
        return self._files_path_dict
    
    
    
    def read_soil_property(self, variable_id: str, depth: int = None, mask_layer = None) -> xarray.DataArray:
        """
        Read a soil property layer.

        Parameters
        ----------
        variable_id : str
            Soil property identifier (e.g., 'sand', 'clay').
        depth : int, optional
            Depth index to select (default is None, meaning first file).
        mask_layer : xarray.DataArray, optional
            If provided, the layer will be masked to this extent.

        Returns
        -------
        xarray.DataArray
            Soil property raster layer as an xarray object.
        """
        fn_path = self.files_path_dict[variable_id][0] if depth is None else self.files_path_dict[variable_id][depth]
            
        soil_da = rio.open_rasterio(fn_path, masked=True).squeeze()

        if mask_layer is not None:
            soil_da = self.mask_data(soil_da, mask_layer)
            
        return soil_da
    

class NexGenPreProcessing(BaseProcessor): 
    """
    Class to preprocess CMIP6 climate data organized in a directory structure:

        variable/
            scenario/
                model/

    Example
    -------
    Directory structure:

    pr/ssp126/ACCESS-ESM1-5/
    """
    
    @property
    def variable_names(self):
        return {'tmin': 'tasmin',
         'tmax': 'tasmax',
         'prec': 'pr'}
        
    ## assumning that there is the same scenarios and models across the variables
    @property
    def variables(self):
        if self._variables is None:
            self._variables = self.directories_list(self.input_path)
        return self._variables
    
    @property
    def scenarios(self):
        if self._scenarios is None:
            self._scenarios = self.directories_list(os.path.join(self.input_path, self.variables[0]))
        return self._scenarios

    @property
    def models(self):
        if self._models is None:
            self._models = self.directories_list(os.path.join(self.input_path, self.variables[0], self.scenarios[0]))
        return self._models
    
    
    def __init__(self, input_pathdir):
        """
        Initialize the preprocessing class.

        Parameters
        ----------
        input_pathdir : str
            Root directory containing climate model outputs.
        """
        
        super().__init__(input_pathdir)
        
        self._variables = None
        self._scenarios = None
        self._models = None
        
    def create_path(self, variable, sceneratio, model):
        return os.path.join(self.input_path, variable, sceneratio, model)


    @staticmethod    
    def _compute_mean_over_year_list(xrdata_list, year_list):
        stacked = xarray.concat(xrdata_list, dim="year")
        stacked = stacked.assign_coords(year=("year", year_list))
        
        stacked = stacked.mean(dim="year", skipna= True)
        stacked.attrs = xrdata_list[0].attrs
        
        return stacked
    
        
    def calculate_climatological_precipitation_daily_mean(
        self, starting_year: int, ending_year: int, scenario_id: str, model_id: str
    ) -> xarray.DataArray:
        """
        Calculate the climatological daily mean precipitation.

        Parameters
        ----------
        starting_year : int
            Start year.
        ending_year : int
            End year (inclusive).
        scenario_id : str
            Scenario identifier (must exist in `self.scenarios`).
        model_id : str
            Model identifier (must exist in `self.models`).

        Returns
        -------
        xr.DataArray
            Daily climatological precipitation (time=day of year).
        """
        assert scenario_id in self.scenarios
        assert model_id in self.models
        
        prec_list = []
        year_list = []
        suffix = '_v2.0.nc'
        
        for year in range(starting_year,ending_year+1):
            
            prec_path = self.create_path( self.variable_names['prec'], scenario_id, model_id)
            prec_path =  glob.glob(prec_path + f'/*_{year}' + suffix)
            
            da = xarray.open_dataset(prec_path[0])
            
            da = da.assign_coords(time=da["time"].dt.dayofyear)
            prec_list.append(da)
            year_list.append(year)
        
        return self._compute_mean_over_year_list(prec_list, year_list)
        
    
    def calculate_climatological_temperature_daily_mean(self, starting_year, ending_year, scenario_id, model_id) -> xarray.DataArray:
        """
        Calculate the climatological daily mean temperature.

        Parameters
        ----------
        starting_year : int
            Start year.
        ending_year : int
            End year (inclusive).
        scenario_id : str
            Scenario identifier (must exist in `self.scenarios`).
        model_id : str
            Model identifier (must exist in `self.models`).

        Returns
        -------
        xarray.DataArray
            Daily climatological mean temperature (tasavg).
        """
        assert scenario_id in self.scenarios
        assert model_id in self.models
        
        xtempavg_list = []
        year_list = []
        for year in range(starting_year,ending_year+1):
            da = compute_averaged_temp(self.create_path( self.variable_names['tmin'], scenario_id, model_id), 
                            self.create_path( self.variable_names['tmax'], scenario_id, model_id), year)
            da = da.assign_coords(time=da["time"].dt.dayofyear)
            
            xtempavg_list.append(da)
            year_list.append(year)
            
        return self._compute_mean_over_year_list(xtempavg_list, year_list)
    

def resample_soilgrids_data(soilgrid_input_path, outputpath, masklayer_path = None):
    #soilgrid_input_path = 'data/soilgrids'
    if masklayer_path:
        mask_layer = rio.open_rasterio(
                        masklayer_path,
                        masked=True,
                    ).squeeze()
    else:
        mask_layer = None

    soilgrids = Resample_SoilGridsData(soilgrid_input_path)

    soil_properties = list(soilgrids.files_path_dict.keys())

    for j in range(len(soil_properties)):
        soil_variable = soil_properties[j]
        var_paths = soilgrids.files_path_dict[soil_variable]

        nfiles = len(var_paths)
        output_var_dir = os.path.join(outputpath, soil_variable)
        if not os.path.exists(output_var_dir):  os.mkdir(output_var_dir)
        
        for i in range(nfiles):
            fn = os.path.basename(var_paths[i])
            fn_path = os.path.join(output_var_dir, fn)
            
            if os.path.exists(fn_path): continue
            soildata = soilgrids.read_soil_property(soil_variable, depth=i)
            if mask_layer is None:
                soildata.rio.to_raster(fn_path, compress="LZW", driver= "GTiff")
            else:

                masked_data = soilgrids.mask_data(soildata, mask_layer == 2)
                masked_data.rio.to_raster(fn_path, compress="LZW", driver= "GTiff")
            print(f'data saved in {fn_path} ')
    

def create_climatology_from_nexgen(climate_input_path, outputpath, masklayer_path):
    #masklayer = 'data/africa_mask_005.tif'
    
    if not os.path.exists(outputpath): os.mkdir(outputpath)


    data_processor = NexGenPreProcessing(climate_input_path)

    periods = [[2021,2040],
            [2041,2060],
            [2061,2080]]

    for z in range(len(periods)):
        for j in range(len(data_processor.scenarios)):
            for m in range(len(data_processor.models)):
                poi = periods[z]
                scen = data_processor.scenarios[j]
                mod = data_processor.models[m]

                outputpath_product = '{}_{}_{}_{}'.format(mod, scen, *poi).lower()
                outputdir = os.path.join(outputpath, outputpath_product)
                
                print(f'Files will be saved in {outputdir}')
                if not os.path.exists(outputdir): os.mkdir(outputdir)

                ouput_fn_temp = os.path.join(outputdir, 'Temp_avg.tif')
                ouput_fn_prec = os.path.join(outputdir, 'Prec_avg.tif')

                xds = rio.open_rasterio(
                    masklayer_path,
                    masked=True,
                ).squeeze()

                print('data read from {} {} {}'.format(poi, scen, mod))
                
                try:

                    if not os.path.exists(ouput_fn_temp):
                       
                        tmp_avg = data_processor.calculate_climatological_temperature_daily_mean(poi[0], poi[1], scen, mod)
                        tmp_avg = tmp_avg.rename({'lat':'y', 'lon': 'x'})
                        tmp_avg.rio.write_crs(xds.rio.crs, inplace=True)
                        maskedtmp_data = data_processor.mask_data(tmp_avg, xds == 2, method= 'nearest')
                        del tmp_avg
                        maskedtmp_data.compute().astype(np.float32).rio.to_raster(ouput_fn_temp, compress="LZW", driver= "GTiff")
                        print(f'data saved in {ouput_fn_temp} ')
                        del maskedtmp_data
                        
                    if not os.path.exists(ouput_fn_prec):
                        prec_avg = data_processor.calculate_climatological_precipitation_daily_mean(poi[0], poi[1], scen, mod)
                        prec_avg = prec_avg.rename({'lat':'y', 'lon': 'x'})
                        prec_avg.rio.write_crs(xds.rio.crs, inplace=True)
                        maskedprec_data = data_processor.mask_data(prec_avg.squeeze(), xds == 2, method= 'nearest').pr
                        maskedprec_data.attrs = prec_avg.attrs
                        del prec_avg
                        maskedprec_data.compute().astype(np.float32).rio.to_raster(ouput_fn_prec, compress="LZW", driver= "GTiff")
                        print(f'data saved in {ouput_fn_prec} ')
                        del maskedprec_data
                except:
                    print('it was not possible to process {ouput_fn_temp} and {ouput_fn_prec}')



def main():
    climate_output_data = '../nex-gddp-cmip6_AFRICA_cs_005'
    climate_input_data = 'E:/Worspace2024/aaguilar/spatial_data/raster/nex-gddp-cmip6_AFRICA'
    mask_layer = 'data/africa_mask_005.tif'
    create_climatology_from_nexgen(climate_input_data, climate_output_data, masklayer_path=mask_layer)

    soilgrid_input_path = '../soilgrids'
    soilgrid_outputpath = '../soilgrids_005'
    #resample_soilgrids_data(soilgrid_input_path, soilgrid_outputpath, masklayer_path = mask_layer)

    ## srtm and land sea mask
    # 
    processing_fuctions = BaseProcessor()    
    
    srtm_path = 'data/srtm_1km_world_recalculated.tif'
    masklayer = 'data/africa_mask_005.tif'
    output_fn = 'data/srtm_005_world_recalculated.tif'
        
    xsrtm_re = processing_fuctions.mask_data(rio.open_rasterio(
                                                srtm_path,
                                                masked=True,
                                            ).squeeze(), rio.open_rasterio(
                                                        masklayer,
                                                    masked=True,
                                                ).squeeze())

    xsrtm_re.where(xsrtm_re !=0, np.nan).rio.to_raster(output_fn, compress="LZW", driver= "GTiff")

    landseapath = 'data/worldclim_land_sea_mask.tif'
    output_fn = 'data/worldclim_land_sea_mask_005.tif'

    xlandsea_re = processing_fuctions.mask_data(rio.open_rasterio(
                                                    landseapath,
                                                    masked=True,
                                                ).squeeze(), rio.open_rasterio(
                                                    masklayer,
                                                    masked=True,
                                                ).squeeze())
    
    xlandsea_re.rio.to_raster(output_fn, compress="LZW", driver= "GTiff")


if __name__ == '__main__':
    main()

