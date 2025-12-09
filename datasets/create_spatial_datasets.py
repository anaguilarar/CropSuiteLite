import os
import sys
import yaml

import numpy as np
import xarray
import rioxarray as rio

from utils import Resample_SoilGridsData, NexGenPreProcessing, BaseProcessor


def resample_soilgrids_data(soilgrid_input_path, outputpath, masklayer_path = None):

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
    

def create_climatology_from_nexgen(climate_input_path, outputpath, masklayer_path, periods, ssps = None):

    if not os.path.exists(outputpath): os.mkdir(outputpath)
    
    data_processor = NexGenPreProcessing(climate_input_path)
    data_processor._scenarios = ssps if ssps is not None else data_processor.scenarios
    print(data_processor.scenarios)
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


def mask_single_layer(layer_path, mask_reference, tool_processor):
    xr_masked=tool_processor.mask_data(rio.open_rasterio(
                                                    layer_path,
                                                    masked=True,
                                                ).squeeze(), rio.open_rasterio(
                                                    mask_reference,
                                                    masked=True,
                                                ).squeeze())
    
    return xr_masked
    

def main(config_path):
    assert os.path.exists(config_path), "the path does not exist"
    

    print(f'-------> Starting: ', config_path)
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    reference_mask_layer = config_dict['GENERAL_INFO']['reference_mask_layer_path']
    processing_fuctions = BaseProcessor() 

    if config_dict['GENERAL_INFO']['process_climate']:
        climate_input_data = config_dict['CLIMATE']['input_path']
        climate_output_data = config_dict['CLIMATE']['output_path']
        periods =  config_dict['CLIMATE']['periods']
        ssps = config_dict['CLIMATE'].get('ssps', None)
        create_climatology_from_nexgen(climate_input_data, climate_output_data, masklayer_path=reference_mask_layer, periods=periods, ssps = ssps)

    if config_dict['GENERAL_INFO']['process_soil']:

        soilgrid_input_path = config_dict['SOIL']['input_path']
        soilgrid_outputpath = config_dict['SOIL']['output_path'] 
        resample_soilgrids_data(soilgrid_input_path, soilgrid_outputpath, masklayer_path = reference_mask_layer)

    if config_dict['GENERAL_INFO']['process_dem']:
        srtm_path = config_dict['DEM']['input_path']
        output_fn = config_dict['DEM']['output_path'] 
        xsrtm_re = mask_single_layer(srtm_path, reference_mask_layer, processing_fuctions)
        xsrtm_re.where(xsrtm_re !=0, np.nan).rio.to_raster(output_fn, compress="LZW", driver= "GTiff")

    if config_dict['GENERAL_INFO']['process_land_sea']:
        landseapath = config_dict['LANDSEA']['input_path']
        output_fn = config_dict['LANDSEA']['output_path']
        xlandsea_re = mask_single_layer(landseapath, reference_mask_layer, processing_fuctions)
        xlandsea_re.rio.to_raster(output_fn, compress="LZW", driver= "GTiff")


if __name__ == '__main__':
    print('''\
      
            ======================================
            |                                    |
            |         CROPSUITE DATASETS         |    
            |                                    |
            ======================================      
      ''')

    args = sys.argv[1:]
    config = args[args.index("-config") + 1] if "-config" in args and len(args) > args.index("-config") + 1 else None
    print(config)    
    main(config)

        