# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:34:51 2024

@author: GnegyE
"""

import rdata
import xarray as xr
import glob, os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime
import pymannkendall as mk
#%%
data_type = 'CanGridP' # choose from CanGridP, CanKrig, or CanGRD

data_var = 'p' # p or tmin,tmean,tmax -- THIS IS ONLY FOR CanGRD (selection doesnt matter for CanGridP or CanKrig, both are precip)
    
#inclusive years
startyear = 1948

if data_type == 'CanGRD':
    if data_var == 'p':
        endyear = 2012
        longname = 'precipitation'
    elif data_var.startswith('t'):
        endyear = 2023
        longname = 'temperature'
    
else:
    endyear = 2018 #CanGridP, CanKrig
    data_var=''
    longname = 'precipitation'

# these are the years used for CanGRD data
startyear_base = 1961
endyear_base = 1990

savepath = 'C:/Users/GnegyE/Desktop/trends/nc_files/'
#%% read from RData file

if data_type == "CanKrig":
    file = '/Users/GnegyE/Desktop/CanKrig_189912_201909.RData'
    parsed = rdata.parser.parse_file(file)
    converted = rdata.conversion.convert(parsed)
    
    lats = converted['lats'].values
    lons = converted['lons'].values
    times = converted['yms']
    precip = converted['CanKrig']
    
    times_dt = pd.to_datetime(times, format='%Y%m')
    
    precip_da = xr.DataArray(precip,
                      dims=('lats', 'lons','time'),
                      coords={'lats':  (('lats', 'lons'), lats), 'lons':  (('lats', 'lons'), lons), 'time': times_dt})

elif data_type == 'CanGridP':
    data_filepath = "/Users/GnegyE/Downloads/CanGridP_mlyV1/" 
      
    df_sample = pd.read_csv(data_filepath + 'CanGridP_mlyV1_201901.txt', header=None,names=['lat','lon','pr'])

    x_len=470
    y_len=621

    lats = df_sample['lat'].values.reshape((x_len, y_len),order='F')
    lons = df_sample['lon'].values.reshape((x_len, y_len),order='F')

    #lil function to read cangrid data 
    def read_cangridp(file):

        df = pd.read_csv(data_filepath + 'CanGridP_mlyV1_' + file + '.txt', header=None,names=['lat','lon','pr'])
        pr = df['pr'].to_numpy(dtype=float)
        data = pr.reshape((x_len, y_len),order='F').astype(float)
        return data


    time, time_str = [], []
    for year in range(startyear-1, endyear + 1):
        for month in range(1, 13):  # Months 1-12
            time.append(datetime.datetime(year, month, 1))
            time_str.append(f'{year}{month:02d}')


    all_data = np.empty((x_len, y_len, len(time_str)))

        
    for i in range(len(time_str)):
        file = time_str[i]
        print(file)
        all_data[:, :, i] = read_cangridp(file) 

    precip_da = xr.DataArray(all_data,
                      dims=('lats', 'lons','time'),
                      coords={'lats':  (('lats', 'lons'), lats), 'lons':  (('lats', 'lons'), lons), 'time': time})

if data_type in ['CanKrig','CanGridP']:
    #% annual 
    precip_select = precip_da.sel(time=slice(str(startyear),str(endyear)))
    precip_base = precip_da.sel(time=slice(str(startyear_base),str(endyear_base)))
    
    #TODO: fix sum of NaNs being 0
    precip_yr = precip_select.resample(time="AS").sum() #annual sums
    
    #same but for base period
    precip_yr_base = precip_base.resample(time="AS").sum()
    precip_mean_base = precip_yr_base.mean(dim='time')
    
    #puts cankrig data in same format as cangrd
    da_ann = ((precip_yr - precip_mean_base)/precip_mean_base) * 100
    
    #% same but for seasonal 
    
    start_date = pd.Timestamp(str(startyear-1)+'-12-01') #get dec of previous year for first DJF
    end_date = pd.Timestamp(str(endyear)+'-11-01') #stop at nov of final year 
    
    start_date_base = pd.Timestamp(str(startyear_base-1)+'-12-01')
    end_date_base = pd.Timestamp(str(endyear_base)+'-11-01')
    
    precip_select_seas = precip_da.sel(time=slice(start_date,end_date))
    precip_base_seas = precip_da.sel(time=slice(start_date_base,end_date_base))
    
    #TODO: fix sum of NaNs being 0
    precip_seas = precip_select_seas.resample(time="QS-DEC").sum()
    precip_seas_base = precip_base_seas.resample(time="QS-DEC").sum()
    
    precip_seas_gr = precip_seas.groupby('time.season')
    precip_seas_means_base = precip_seas_base.groupby('time.season').mean('time')
    
    da_djf = (precip_seas_gr['DJF']-precip_seas_means_base.sel(season='DJF'))/precip_seas_means_base.sel(season='DJF') * 100
    da_mam = (precip_seas_gr['MAM']-precip_seas_means_base.sel(season='MAM'))/precip_seas_means_base.sel(season='MAM') * 100
    da_jja = (precip_seas_gr['JJA']-precip_seas_means_base.sel(season='JJA'))/precip_seas_means_base.sel(season='JJA') * 100
    da_son = (precip_seas_gr['SON']-precip_seas_means_base.sel(season='SON'))/precip_seas_means_base.sel(season='SON') * 100

#% read in cangrd data

if data_type == 'CanGRD':
    points_file = "C:/Users/GnegyE/Desktop/trends/raw_data/CANGRD_points_LL.txt"
    
    if data_var.startswith('t'):
        data_filepath = "C:/Users/GnegyE/Desktop/trends/raw_data/cangrd_t/"+data_var+"/"
    
    #elif data_var == "p":
    #    data_filepath = #path to "precip_anomalies" folder on shared drive 
      
    x_len = 95
    y_len = 125
    
    df = pd.read_csv(points_file, sep=",", header=None, names=["id1", "id2", "lat", "lon"])
    
    lats = df['lat'].values.reshape((x_len, y_len))
    lons = df['lon'].values.reshape((x_len, y_len))
    
    #lil function to read cangrid data 
    def read_cangrd(file):
        #file in format YYYYDD (DD is month or 13 for ann, 14-17 for seas - see cangrd readme)
        #open the GRD file
        df = pd.read_csv(data_filepath + data_var[0] + file + '.grd', header=None,skiprows=[0,1,2,3,4],names=[data_var])

        #fix spacing issues 
        df = df[data_var].str.split(expand=True).stack().values
        data = df.reshape((x_len, y_len)).astype(float)
        data[data > 10e30] = np.nan

    
    years = [datetime.datetime(year, 1, 1) for year in range(startyear-1, endyear + 1)]
    
    def load_cangrd(seas_ID):
        cangrd_data = np.empty((x_len, y_len, len(years)))
        
        for i in range(len(years)):
            year = years[i].year
            print(year)
            file = str(year) + seas_ID #13 for "annual" - see readme for cangrd
            cangrd_data[:, :, i] = read_cangrd(file) 
    
        cangrd_da = xr.DataArray(cangrd_data,
                          dims=('lats', 'lons','time'),
                          coords={'lats':  (('lats', 'lons'), lats), 'lons':  (('lats', 'lons'), lons), 'time': years})
    
    
        return(cangrd_da)
    
    da_ann = load_cangrd("13")
    da_djf = load_cangrd("14")
    da_mam = load_cangrd("15")
    da_jja = load_cangrd("16")
    da_son = load_cangrd("17")


#%%

def get_trendline(var):
    if np.all(np.isnan(var)) or np.all(var==0):
        trend = np.nan
        pval = np.nan
    else:
        mk_samp = mk.yue_wang_modification_test(var,lag=1)
    
        trend = mk_samp.slope * len(var) 

        pval = mk_samp.p
        
    return np.array([trend, pval])   

        
   
#%% testing apply_ufunc

# pr_da should already be in yearly (seasonal) format (lat,lon,years)
def get_trends(pr_da):
    trend = xr.apply_ufunc(get_trendline,pr_da,input_core_dims=[["time"]],output_core_dims = [['metric']], output_sizes = {'metric': 2},output_dtypes = [float], dask='parallelized',vectorize=True)
    return xr.Dataset({'trend': trend.isel(metric = 0), 'pval': trend.isel(metric = 1)})
                       

#%% anomolies - precipitation given in percentage, temp in deg C

trend_ann = get_trends(da_ann)
trend_djf = get_trends(da_djf)
trend_mam = get_trends(da_mam)
trend_jja = get_trends(da_jja)
trend_son = get_trends(da_son)


#%% save it as a netcdf file


file = f"{savepath}trends_{startyear}-{endyear}_{data_type}{data_var}_anomalies.nc"

with Dataset(file,'w') as nc_file:

    lat_dim = nc_file.createDimension('lat',np.shape(lats)[0])
    lon_dim = nc_file.createDimension('lon',np.shape(lons)[1])

    ann_trend_var = nc_file.createVariable('ANN','f8',('lat','lon'))
    djf_trend_var = nc_file.createVariable('DJF','f8',('lat','lon'))
    mam_trend_var = nc_file.createVariable('MAM','f8',('lat','lon'))
    jja_trend_var = nc_file.createVariable('JJA','f8',('lat','lon'))
    son_trend_var = nc_file.createVariable('SON','f8',('lat','lon'))

    ann_pval_var = nc_file.createVariable('ANN_pval','f8',('lat','lon'))
    djf_pval_var = nc_file.createVariable('DJF_pval','f8',('lat','lon'))
    mam_pval_var = nc_file.createVariable('MAM_pval','f8',('lat','lon'))
    jja_pval_var = nc_file.createVariable('JJA_pval','f8',('lat','lon'))
    son_pval_var = nc_file.createVariable('SON_pval','f8',('lat','lon'))
    
    lat_var = nc_file.createVariable('lat','f8',('lat','lon'))
    lon_var = nc_file.createVariable('lon','f8',('lat','lon'))
    
    ann_trend_var.Description = f"{startyear}-{endyear} {data_type} annual {longname} trend (relative to {startyear_base}-{endyear_base})"
    djf_trend_var.Description = f"{startyear}-{endyear} {data_type} DJF {longname} trend (relative to {startyear_base}-{endyear_base})"
    mam_trend_var.Description = f"{startyear}-{endyear} {data_type} MAM {longname} trend (relative to {startyear_base}-{endyear_base})"
    jja_trend_var.Description = f"{startyear}-{endyear} {data_type} JJA {longname} trend (relative to {startyear_base}-{endyear_base})"
    son_trend_var.Description = f"{startyear}-{endyear} {data_type} SON {longname} trend (relative to {startyear_base}-{endyear_base})"
    
    ann_trend_var[:,:] = trend_ann['trend'].values[:,:]
    djf_trend_var[:,:] = trend_djf['trend'].values[:,:]
    mam_trend_var[:,:] = trend_mam['trend'].values[:,:]
    jja_trend_var[:,:] = trend_jja['trend'].values[:,:]
    son_trend_var[:,:] = trend_son['trend'].values[:,:]
    
    ann_pval_var[:,:] = trend_ann['pval'].values[:,:]
    djf_pval_var[:,:] = trend_djf['pval'].values[:,:]
    mam_pval_var[:,:] = trend_mam['pval'].values[:,:]
    jja_pval_var[:,:] = trend_jja['pval'].values[:,:]
    son_pval_var[:,:] = trend_son['pval'].values[:,:]
    
    lat_var[:,:] = lats[:,:]
    lon_var[:,:] = lons[:,:]



