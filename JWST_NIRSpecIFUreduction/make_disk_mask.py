
# https://github.com/ChenXie-astro/jwstIFURDI/, written by C. Xie 2024
import numpy as np
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.patches as patches


# the function I used for drawing regions to measure the stellar jet. 
def jet_region(reg_file):
#>>>    load the source_region file  and compute the region of polygon  <<< 
    polygon = np.loadtxt(reg_file, usecols=(0,), dtype=str,unpack=True,skiprows=3, )
    polygon = str(polygon)
    polygon = polygon[8:-1]
    polygon = polygon.split()
    polygon = polygon[0]
    polygon = polygon.replace(',+',' ')
    polygon = polygon.split(',')     # now, you have each coordinate for each region point of ploygon

    #print polygon
    for i in range(len(polygon)//2):
        if i == 0:#path_data = [(mplPath.Path.MOVETO, (c.ra.degree, c.dec.degree))]
            path_data = [(mplPath.Path.MOVETO, (polygon[2*i],polygon[2*i+1]))]
            # path_data = [(mplPath.Path.MOVETO, (polygon[2*i+1],polygon[2*i]))]
        elif i != 0:
            path_data.append((mplPath.Path.LINETO, (polygon[2*i],polygon[2*i+1])))
            # path_data.append((mplPath.Path.LINETO,  (polygon[2*i+1],polygon[2*i])))		

    #xlimit = floor(c.ra.degree)
    #ylimit = floor(c.dec.degree) 
    codes, verts = zip(*path_data)
    # print(codes)
    # print(verts)
    # print(tuple([tuple(row) for row in verts_array]) )

    # to match the regions shown in ds9
    verts_array = (np.array(verts).astype(float)-1).astype(str)
    verts = tuple([tuple(row) for row in verts_array])  

    path = mplPath.Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='r', alpha=0.5)

    return path



def run_make_mask(dir_name, sci_file_name, plot=False):
    mask_path = Path(dir_name, 'mask')
    sci_cube = fits.getdata(Path(dir_name, f'centering/{sci_file_name}'))
    nz, ny, nx = sci_cube.shape

    mask_0_1 = np.ones((1, ny, nx))
    mask_0_1_PSFconvo_test = np.ones((1, ny, nx))
    path_outer = jet_region(Path(mask_path, 'IFU_align_disk_outer.reg'))
    for x in range(nx):
        for y in range(ny):
            if path_outer.contains_point((x,y)):
                mask_0_1[:, y, x] = np.nan
                mask_0_1_PSFconvo_test[:, y, x] = 1
    mask_0_1_PSFconvo_test[mask_0_1_PSFconvo_test==0]=np.nan
    if plot:
        plt.figure()
        plt.imshow(mask_0_1[0,:,:])
        plt.title('outer disk mask')
        plt.show()
    fits.writeto(Path(mask_path, 'disk_mask_0_1_2D.fits'), mask_0_1, overwrite=True)
    fits.writeto(Path(mask_path, 'disk_mask_0_1_2D_for_PSF_convolution_Test.fits'), mask_0_1, overwrite=True)

    mask_0_1 = np.zeros((1, ny, nx))
    path_FOV = jet_region(Path(mask_path, '/betaPic_IFU_align_FoV.reg'))
    path_bleed = jet_region(Path(mask_path, '/betaPic_IFU_align_bleeding.reg'))
    for x in range(nx):
        for y in range(ny):
            if path_FOV.contains_point((x,y)):
                mask_0_1[:, y, x] = 1
            if path_bleed.contains_point((x,y)):
                mask_0_1[:, y, x] = np.nan
    mask_0_1[mask_0_1==0] = np.nan
    if plot:
        plt.figure()
        plt.imshow(mask_0_1[0,:,:])
        plt.title('IFU FoV mask')
        plt.show()

    fits.writeto(Path(mask_path, 'IFU_align_FoV_extra_spike.fits'), mask_0_1, overwrite=True)



