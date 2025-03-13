# -*- coding: utf-8 -*-
"""
Created on Sat May 11 09:12:27 2024

@author: GnegyE
"""


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import cartopy.feature as cf
import matplotlib as mpl
import matplotlib.colors as pltcol
import pylab as pl
import matplotlib.colorbar as cb
import geopandas as gpd


prov_shapefile = "C:/Users/GnegyE/Downloads/lpr_000b16a_e"
gdf = gpd.read_file(prov_shapefile)

#%% read from netcdf files

data_type = 'CanGRD' # choose from CanGridP, CanKrig, or CanGRD
data_var = 'tmax' # p or tmin,tmean,tmax -- THIS IS ONLY FOR CanGRD (selection doesnt matter for CanGridP or CanKrig, both are precip)
    
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

savepath = 'C:/Users/GnegyE/Desktop/trends/figures/maps/' + data_type

t_names = {
    'tmean': 'Mean ',
    'tmax': 'Max ',
    'tmin': 'Min ',
    'p': '',
}

#%%

if data_type == 'CanGRD':
    file = f'C:/Users/GnegyE/Desktop/trends/nc_files/aaa{data_var}_trends_{startyear}-{endyear}_{data_type}_anomalies.nc'
else:
    file = f'C:/Users/GnegyE/Desktop/trends/nc_files/trends_{startyear}-{endyear}_{data_type}_anomalies.nc'


nc = Dataset(file,'r')

ann_trend = nc.variables['ANN'][:]
djf_trend = nc.variables['DJF'][:]
mam_trend = nc.variables['MAM'][:]
jja_trend = nc.variables['JJA'][:]
son_trend = nc.variables['SON'][:]

ann_pval = nc.variables['ANN_pval'][:]
djf_pval = nc.variables['DJF_pval'][:]
mam_pval = nc.variables['MAM_pval'][:]
jja_pval = nc.variables['JJA_pval'][:]
son_pval = nc.variables['SON_pval'][:]

lats = nc.variables['lat']
lons = nc.variables['lon']

#%%

if longname == 'precipitation':
#made colorbar from image on this page: 
#https://www.researchgate.net/figure/Change-in-average-precipitation-based-on-multi-model-mean-projections-for-2081-2100_fig1_342989239    
    color = ['#a05323', '#b46b28', '#ce853e' , '#f6cd84', '#f4e19e', '#fdf7ba', \
              '#d0e8c6', '#a3d4aa', '#50bfa0', '#4b86a3', '#255e7f', '#123859']
    color_count = len(color)-2
    lim = [-75,75] #data limits 
    ticks = np.linspace(lim[0],lim[1],color_count+1) #location where I want ticks

elif longname == 'temperature': #from CCCS color guide
    color = ['#74add1','#C0E6F0',\
                '#ffffbf','#fee090','#fdae61','#f46d43','#d73027','#a50026','#800000']
    color_count = len(color)-2
    lim = [-1.5,5.5] #data limits 
    ticks = np.linspace(lim[0]+0.5,lim[1]-0.5,color_count) #location where I want ticks

ticklabels = [str(int(x)) if x == int(x) else str(x) for x in ticks]



# makes the colorbar
# NOTE: if you wanted the colorbar to be a gradient, you would change N to be really high (e.g. 1000) and remove [1:-1] indexing
cmap = pltcol.LinearSegmentedColormap.from_list("custom", color[1:-1],N=color_count) #[1:-1] is to not include the min/max arrow colors
#uncomment this line to see a gradient instead of step
cmap.set_over(color[-1]) #add the max arrow color
cmap.set_under(color[0]) #add the min arrow color

# this is a function that plots a colorbar, where the input is a colormap (cmap)
def plot_cbar(cmap,lim,label,norm=1,ticklabels=1,ticks=1):
    
    pl.figure(figsize=(20, 1.5),dpi=250)
    pl.gca().set_visible(False)
    cax = pl.axes([0.1, 0.2, 0.8, 0.6])
    
    if norm==1:
        norm = pltcol.Normalize(vmin=lim[0], vmax=lim[1])
    if isinstance(ticks,int):
        ticks=None
    else:
        ticks=ticks
    cb.ColorbarBase(cax,cmap=cmap,norm=norm,orientation="horizontal", extend='both',ticks=ticks)
    
    if ticklabels!=1:
        cax.set_xticks(ticks,ticklabels)
    
    cax.xaxis.set_tick_params(size=12,width=4)
    pl.xticks(fontsize=38)
    pl.xlabel(label,fontsize=38)

    plt.show()

#see if you like the colorbar
plot_cbar(cmap,lim,f"Change in {t_names[data_var]}{longname.capitalize()} ({unit})",ticks=ticks,ticklabels=ticklabels)

#%%
#plot the data 
extend='both'
vmin=lim[0]
vmax=lim[1]

def make_plots(data, pval, seas):
    proj = ccrs.RotatedPole(pole_latitude=42.5,pole_longitude=83)
    fig = plt.figure(figsize=(10,10),dpi=200)
    ax = fig.add_subplot(1,1,1, projection=proj)
    
    
    plt.pcolormesh(lons,lats,data,transform=ccrs.PlateCarree(),cmap=cmap,vmin=vmin,vmax=vmax)

    masked_grid = pval.copy()
    masked_grid[masked_grid<0.05] = np.nan #only put stippling where there ISNT confidence
    plt.pcolor(lons, lats, masked_grid, transform=ccrs.PlateCarree(), hatch='...', alpha=0,vmin=vmin,vmax=vmax)
    mpl.rcParams['hatch.linewidth'] = 0.8
    
    
    gdf_reprojected = gdf.to_crs(proj) #convert from the shapefile's EPSG to the projection i am plotting
    gdf_reprojected.plot(ax=ax,facecolor='none', edgecolor='black', linewidth=0.25, zorder=1)

    ax.set_extent([-131,-55,40,85],crs=ccrs.PlateCarree())
    
    cbar_ax = fig.add_axes([0.2,0.13,0.62,0.025])
    
    fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),cax=cbar_ax,orientation='horizontal',extend=extend,ticks=ticks)
    
    cbar_ax.tick_params(labelsize=16)
    cbar_ax.set_xlabel(f"Change in {t_names[data_var]}{longname.capitalize()} ({unit})",size=18)

    fig.suptitle(f"{data_type} {startyear}-{endyear}: {seas}\n(relative to {startyear_base}-{endyear_base} mean)", fontsize=20, y=0.9)   
    
    if longname == "temperature":
        plt.savefig(f"{savepath}/{data_var}/{seas}.png", bbox_inches='tight', dpi=500)
    else:
        plt.savefig(f"{savepath}/{seas}.png", bbox_inches='tight', dpi=500)

    plt.close()
    
#%%
make_plots(ann_trend, ann_pval, "Annual")
make_plots(djf_trend, djf_pval, "DJF")
make_plots(mam_trend, mam_pval, "MAM")
make_plots(jja_trend, jja_pval, "JJA")
make_plots(son_trend, son_pval, "SON")
