
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:34:51 2024

@author: GnegyE
"""
import matplotlib.pyplot as plt
import rdata
import xarray as xr
import glob, os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime
import pymannkendall as mk

data_type = 'CanGridP' # choose from CanGridP, CanKrig, or CanGRD
data_var = 'p' # p or tmin,tmean,tmax -- THIS IS ONLY FOR CanGRD (selection doesnt matter for CanGridP or CanKrig, both are precip)
    
#note this isnt set up to plot CanGRD precip or CanKrig (so only CanGridP and CanGRD temperatures)
#%%

#inclusive years
startyear = 1948

if data_type == 'CanGRD':
    if data_var == 'p':
        endyear = 2012
        longname = 'precipitation'
        unit = '%'
    elif data_var.startswith('t'):
        endyear = 2023
        longname = 'temperature'
        unit = '°C'
    
else:
    endyear = 2018 #CanGridP, CanKrig
    longname = 'precipitation'
    unit = '%'

# these are the years used for CanGRD data
startyear_base = 1961
endyear_base = 1990

savepath = 'C:/Users/GnegyE/Desktop/trends/figures/timeseries/' + data_type

t_names = {
    'tmean': ' Mean ',
    'tmax': ' Max. ',
    'tmin': ' Min. ',
    'p': '',
}

seasons = {
    'DJF': 'Winter',
    'MAM': 'Spring',
    'JJA': 'Summer',
    'SON': 'Fall',
    'Annual': 'Annual',
}
#%%

province_codes = {
    "NU": 1,
    "YT": 2,
    "QC": 3,
    "AB": 4,
    "SK": 5,
    "MB": 6,
    "NL": 7,
    "BC": 8,
    "PE": 9,
    "NB": 10,
    "NS": 11,
    "ON": 12,
    "NT": 13
}

province_names = {
    "NU": "Nunavut",
    "YT": "Yukon",
    "QC": "Quebec",
    "AB": "Alberta",
    "SK": "Saskatchewan",
    "MB": "Manitoba",
    "NL": "Newfoundland and Labrador",
    "BC": "British Columbia",
    "PE": "Prince Edward Island",
    "NB": "New Brunswick",
    "NS": "Nova Scotia",
    "ON": "Ontario",
    "NT": "Northwest Territories",
    "Canada": "Canada"
}

if data_type in ['CanKrig','CanGridP']:
    prov_mask = "C:/Users/GnegyE/Desktop/trends/CanGridP_provinces.nc"
elif data_type == 'CanGRD':
    prov_mask = "C:/Users/GnegyE/Desktop/trends/CanGRD_provinces.nc"

prov_mask = xr.open_dataset(prov_mask)['province']

prov_mask = prov_mask.rename({"lat": "lats"})
prov_mask = prov_mask.rename({"lon": "lons"})


#%%

if data_type == 'CanGridP':

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
      
    time,time_str = [],[]
    for year in range(startyear-1, endyear + 1): #-1 to get D of previous year
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

    years = [datetime.datetime(year, 1, 1) for year in range(startyear, endyear + 1)]
    
    
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

elif data_type == 'CanGRD':

    points_file = "C:/Users/GnegyE/Desktop/trends/raw_data/CANGRD_points_LL.txt"
    data_filepath = "C:/Users/GnegyE/Desktop/trends/raw_data/cangrd_t/"+data_var+"/"
      
    x_len = 95
    y_len = 125

    df = pd.read_csv(points_file, delim_whitespace=True, header=None, names=["id1", "id2", "lat", "lon"])

    lats = df['lat'].values.reshape((x_len, y_len))
    lons = df['lon'].values.reshape((x_len, y_len))

    #lil function to read cangrid data 
    def read_cangrd(file):
        #file in format YYYYDD (DD is month or 13 for ann, 14-17 for seas - see cangrd readme)
        #open the GRD file
        df = pd.read_csv(data_filepath + 't' + file + '.grd', header=None,skiprows=[0,1,2,3,4],names=['t'])
        #fix spacing issues 
        df = df['t'].str.split(expand=True).stack().values
        data = df.reshape((x_len, y_len)).astype(float)
        data[data > 10e30] = np.nan
        return data

    years = [datetime.datetime(year, 1, 1) for year in range(startyear, endyear + 1)]

    def load_cangrd(seas_ID):
        all_data = np.empty((x_len, y_len, len(years)))
        
        for i in range(len(years)):
            year = years[i].year
            print(year)
            file = str(year) + seas_ID #13 for "annual" - see readme for cangrd
            all_data[:, :, i] = read_cangrd(file) 

        temperature_da = xr.DataArray(all_data,
                          dims=('lats', 'lons','time'),
                          coords={'lats':  (('lats', 'lons'), lats), 'lons':  (('lats', 'lons'), lons), 'time': years})

        return(temperature_da)

    da_ann = load_cangrd("13")
    da_djf = load_cangrd("14")
    da_mam = load_cangrd("15")
    da_jja = load_cangrd("16")
    da_son = load_cangrd("17")
    
#%%
def spatial_mean(data,code):
    if code == "Canada":
        all_codes = list(province_codes.values())  
        ds_mask = data.where(prov_mask.isin(all_codes))
    else:
        ds_mask = data.where(prov_mask==code)
    
    area_weights = np.cos(np.deg2rad(ds_mask.lats))
    data_weighted = ds_mask.weighted(area_weights).mean(dim=("lons","lats"))

    return(data_weighted)

#%%
for prov in ['Canada',"NU","YT","QC","AB","SK","MB","NL","BC","PE","NB","NS","ON","NT"]:
    print(prov)
    
    if prov != "Canada":
        code = province_codes[prov]
    else:
        code = "Canada"
        
    ann_mean = spatial_mean(da_ann,code)
    djf_mean = spatial_mean(da_djf,code)
    mam_mean = spatial_mean(da_mam,code)
    jja_mean = spatial_mean(da_jja,code)
    son_mean = spatial_mean(da_son,code)
    
    
    def plot_scatter(data,seas):
        plt.figure(figsize=(10,6),dpi=200)
        
        mk_samp = mk.yue_wang_modification_test(data,lag=1)
        trendline = mk_samp.slope * np.arange(0,len(data)) + mk_samp.intercept
    
        plt.scatter(years,data,s=100,marker='o') 
        plt.plot(years, data, linestyle='-', linewidth=1.5)
        
        plt.plot(years,trendline,linewidth=3,color='r',linestyle='--')     
        
        plt.ylabel(f"Change in {t_names[data_var]}{longname.capitalize()} ({unit})\n(relative to {startyear_base}-{endyear_base})", fontsize=18)    
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.axhline(0, linestyle='-', color='grey')
        
        plt.grid()
        
        if longname == "precipitation":
            change = int(trendline[-1] - trendline[0])
        else:
            change = round(trendline[-1] - trendline[0], 1)
            
        sign = "+" if change > 0 else ""
            
        plt.title(province_names[prov] + " (" + sign + str(change) + "%) - " + seasons[seas],fontsize=22)

        if longname == "temperature":
            plt.savefig(f"{savepath}/{data_var}/{seas}_{prov}.png", bbox_inches='tight', dpi=500)
        else:
            plt.savefig(f"{savepath}/{seas}_{prov}.png", bbox_inches='tight', dpi=500)
        plt.close()
        
    plot_scatter(ann_mean,'Annual')
    plot_scatter(djf_mean,'DJF')
    plot_scatter(mam_mean,'MAM')
    plot_scatter(jja_mean,'JJA')
    plot_scatter(son_mean,'SON')
    
