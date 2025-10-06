import os
import sys
import requests
import gc
from pathlib import Path
import concurrent.futures
from itertools import product

from omegaconf import OmegaConf
import pandas as pd
import xarray
from tqdm import tqdm


def set_encoding(xrdata, compress_method="zlib"):
    return {k: {compress_method: True} for k in list(xrdata.data_vars.keys())}


class ProcessTools():
    @staticmethod    
    def preprocess_xrdata(xrdata, variable = None, replace = True):
        
        if variable is None:
            variable = list(xrdata.data_vars.keys())[0]
            
        if variable == 'pr':
            return xrdata.load() * 86400
        if variable == 'tasmax':
            return xrdata.load() - 273.15
        if variable == 'tasmin':
            return xrdata.load() - 273.15
        if variable == 'rsds':
            return xrdata.load() * 86400 / 1000000
        
    
    @staticmethod    
    def clip_spdata(extent, xrdata = None, xrdata_path = None, rotate = False, chunks = None):
        """_summary_

        Args:
            extent (_type_): xyxy extent option
            xrdata (_type_, optional): _description_. Defaults to None.
            xrdata_path (_type_, optional): _description_. Defaults to None.
            rotate (bool, optional): _description_. Defaults to False.
        """
        min_lon, min_lat, max_lon, max_lat = extent
        if xrdata_path:
            with xarray.open_dataset(xrdata_path, chunks = chunks) as src:
                if rotate:
                    src = src.assign_coords(lon=(((src.lon + 180) % 360) - 180))
                    src = src.sortby('lon')
                src = src.sel(lon = slice(min_lon, max_lon), lat = slice(min_lat,max_lat))
            return src
        else:
            src = xrdata.copy()
            if rotate:
                src = src.assign_coords(lon=(((src.lon + 180) % 360) - 180))
                src = src.sortby('lon')
            src = src.sel(lon = slice(min_lon, max_lon), lat = slice(min_lat,max_lat))
            return src
        
    @staticmethod
    def save_asnc(xrdata: xarray.Dataset, fn: str) -> None:
        """
        Save a dataset to a NetCDF file with appropriate encoding.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The dataset to save.
        fn : str
            Output file name.
        """
        dcengine = 'netcdf4' 
        encoding = set_encoding(xrdata)
        xrdata.to_netcdf(fn, encoding = encoding, engine = dcengine)
    
    
    
class DownloadCMIP6Data():
    
    @property
    def products_to_download(self):
        return product(self.gcms, self.ssp, self.var, self.years)
    
    def __init__(self, **kwargs):
        self._products_to_download = None
        self._gcm, self._ssp, self._var = None, None, None
        self.url_root = kwargs.get('url_root', None)
        self.file_names = [
            ['r1i1p1f1_gn', 'r1i1p1f1_gr','r1i1p1f1_gr1'],
            ['v2.0.nc', 'v1.2.nc', 'v1.1.nc', '.nc']
        ]    
    
    def set_query(self,gcms, ssps, variables, years):
        self.gcms = gcms
        self.ssp = ssps
        self.var = variables
        self.years = years
        
    def reset_vars(self):
        self._gcm, self._ssp, self._var = None, None, None

    def find_remote_file(self,gcm, ssp, var, year):
        self._gcm, self._ssp, self._var = gcm, ssp, var
        
        base_path = f"{gcm}/{ssp}/r1i1p1f1/{var}"
        
        for fn1, fn2 in product(self.file_names[0],self.file_names[1]):
            fname = f"{var}_day_{gcm}_{ssp}_{fn1}_{year}_{fn2}"
            url = f"{self.url_root}/{base_path}/{fname}"
            try:
                response = requests.head(url, timeout=10)
                if response.status_code == 200:
                    # Return the parent directory URL and the found filename
                    return (f"{self.url_root}/{base_path}", fname)
            except requests.RequestException:
                # Ignore connection errors and try the next file
                continue
        
        return (None, None) 
    @staticmethod
    def download_data(url, fn, output_path):
        
        if not output_path.exists(): output_path.mkdir(parents = True)
        output_fn = output_path / fn
        
        if output_fn.exists() and output_fn.stat().st_size > 1e5:
            print(f"'{fn}' already downloaded!")
            return None
        
        # Construct the full URL and download the file
        url = f"{url}/{fn}"
        print(f"Downloading: {url}")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status() # Will raise an exception for 4xx/5xx errors
                with open(output_fn, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return output_fn
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {fn}: {e}")        
    
    
def get_individual_file(cmip_dowloader, idx, output_dir, clip_extent = None):
    xrdata_processor = ProcessTools()
    gcm, ssp, var, year = list(cmip_dowloader.products_to_download)[idx]
    
    url_dir, fn = cmip_dowloader.find_remote_file(gcm, ssp, var, year)
    output_path = Path(output_dir) / var / ssp / gcm
    dwnd_path = cmip_dowloader.download_data(url_dir, fn, output_path)
    if dwnd_path is not None and clip_extent is not None:
        xrdata = xrdata_processor.clip_spdata(clip_extent, xrdata_path= dwnd_path, rotate= True, chunks= 'auto')
        xrdata = xrdata_processor.preprocess_xrdata(xrdata)
        os.remove(dwnd_path)
        
        xrdata_processor.save_asnc(xrdata, fn = dwnd_path)
        
        print(f'Data Downloaded in {dwnd_path}')
        del xrdata
    dwnd_path, output_path = None, None
    #cmip_dowloader.reset_vars()
    gc.collect()
    
        
def main(config_path):
    config = OmegaConf.load(config_path)

    years = list(range(config.GENERAL_INFO.years[0], config.GENERAL_INFO.years[1]))
    clip_extent = config.GENERAL_INFO.clip_extent
    nworkers = config.GENERAL_INFO.nworkers
    output_dir = config.GENERAL_INFO.output_dir
    
    cmip_dowloader = DownloadCMIP6Data(url_root = config.CLIMATE.url_root)
    cmip_dowloader.set_query(years = years, gcms = config.CLIMATE.gcms, ssps = config.CLIMATE.ssps, variables = config.GENERAL_INFO.variables)
    if nworkers == 0:
        for idpx in tqdm(range(len(list(cmip_dowloader.products_to_download)))):
            get_individual_file(cmip_dowloader,idpx, output_dir, clip_extent)
    else: 
        with tqdm(total=len(list(cmip_dowloader.products_to_download))) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
                future_to_tr ={executor.submit(get_individual_file, cmip_dowloader,idpx, output_dir, clip_extent
                                                ): (idpx) for idpx in range(len(list(cmip_dowloader.products_to_download)))}
                
                for future in concurrent.futures.as_completed(future_to_tr):
                    idpx = future_to_tr[future]
                    try:
                        future.result()
                    except Exception as exc:
                            print(f"Request for treatment {idpx} generated an exception: {exc}")
                    pbar.update(1)

        
if __name__ == '__main__':
    args = sys.argv[1:]
    config = args[args.index("-config") + 1] if "-config" in args and len(args) > args.index("-config") + 1 else None
    print(config)    
    main(config)
