# written by C. Xie 2024
# updated by S. Betti 2024

import time

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy import optimize

from astropy.io import fits
from astropy.stats import sigma_clip

from pathlib import Path


def single_frame_sub(sci_cube, ref_cube, mask_cube, ):
    
    """
    Returns the residual images by using the single ref frame RDI approch.
    Used for processing JWST/NIRSpec IFU data cube. 

    Args:
        sci_img: science images
        ref_img: reference images with the same shape as the sci_img
        mask: mask

    Returns:
        res_img: the residual images
        nu: the scaling factors 
        cost: minimum cost

    Written: Chen Xie, 2023-10.
    Updated: Sarah Betti, 2026-06
    """
    
    def cost_function_subtraction(nu,):
        """
        Returns the vaule of the cost function used for the single ref frame RDI approch.

        Args:
            nu: scaling factor 
        Returns:
            cost

        Written: Chen Xie, 2023-10.

        Note that: 'sci_image' (sci image), 'ref_img' (ref), 'mask_img' (mask) are global variables in this nested function that will be updated in each interation.
        """
        return np.log(np.nansum( ((nu * ref_img  - sci_img))**2  , axis=(0,1)))

    nz, ny, nx = sci_cube.shape
    scaling_factor = np.zeros((nz))
    cost = np.zeros((nz))
    if mask_cube is None:
        mask_cube = np.ones(sci_cube.shape)
        print('*******   Note: using the entire FoV in scaling the reference image in RDI   *******')

    for z in range(nz):
        nu_0 = np.nansum(sci_cube[z] * mask_cube[z], axis=(0,1)) / np.nansum(ref_cube[z] * mask_cube[z], axis=(0,1)) 

        mask_img = mask_cube[z]
        sci_img = sigma_clip(sci_cube[z]*mask_img, sigma=3.6, maxiters=15, masked=False, copy=True, axis=(0,1))
        ref_img = sigma_clip(ref_cube[z]*mask_img, sigma=3.6, maxiters=15, masked=False, copy=True, axis=(0,1))

        minimum = optimize.fmin(cost_function_subtraction, nu_0, disp=False)

        scaling_factor[z] = minimum[0]
        cost[z] = cost_function_subtraction(minimum[0]) 

    constant_scaling_factor = np.nanmean(scaling_factor[5:-5])
    print('constant scaling factor: ',constant_scaling_factor)
    res_cube = sci_cube - constant_scaling_factor * ref_cube

    return res_cube, scaling_factor, cost

def plot_scaling_factor(ydata, header, plotname, savepath):
    nz = ydata.shape[0]
    z_wave = np.arange(header['CRVAL3'], header['CRVAL3']+ (nz)*header['CDELT3'], header['CDELT3'])

    plt.figure(figsize=(10,5))
    plt.plot(z_wave[5:-5], ydata[5:-5], lw=2, alpha=1, color = '#EA8379', label= r'res = sci - $f \times$ref')
    plt.axhline(np.nanmean(ydata[5:-5]), label=r'$\langle f \rangle = $' + str(round(np.nanmean(ydata[5:-5]), 3)) + r'$\pm$' +  str(round(np.std((ydata[5:-5])),2)))
    plt.axhline(np.nanmean(ydata[5:-5])-(np.std((ydata[5:-5]))))
    plt.axhline(np.nanmean(ydata[5:-5])+(np.std((ydata[5:-5]))))
    
    plt.legend(loc='best', fontsize=15,frameon=True)
    plt.xlabel(r'Wavelength ($\rm \mu m$)', fontsize=16)
    plt.ylabel(r'Scaling factor ($f$ )', fontsize=16)
    plt.ylim(0.45, 0.65)

    plt.tick_params(which='both', direction='in', labelsize=16, right=True, top=True)
    plt.minorticks_on()

    pdfname = savepath + f'/{plotname}.pdf'
    print('saving scaling factor figure to: ', pdfname)
    plt.savefig(pdfname, dpi=300)
    plt.show()

def plot_spectrum(ydata, header, plotname, savepath, legend):
    nz = ydata.shape[0]
    z_wave = np.arange(header['CRVAL3'], header['CRVAL3']+ (nz)*header['CDELT3'], header['CDELT3'])

    plt.figure(figsize=(10,5))
    plt.plot(z_wave[1:-2], ydata[1:-2], lw=2, alpha=1, color = '#0099ff', label=legend)
    plt.legend(loc='upper right', fontsize=15,frameon=True)
    plt.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
    plt.minorticks_on()
    plt.xlabel(r'Wavelength ($\rm \mu m$)', fontsize=16)
    plt.ylabel(r'Flux', fontsize=16)
    
    pdfname = savepath + f'/{plotname}.pdf'
    print('saving spectrum factor figure to: ', pdfname)
    plt.savefig(pdfname, dpi=300)
    plt.show()

def extract_stellar_spec(input_cube, input_disk_mask, center_mask_radius):
    """
    Returns the 1D stellar spectrum, summing over the FoV of IFU

    Args:
        input_cube : input science cube (3D)
        input_disk_mask : disk mask (3D)
        center_mask_radius : central circular mask for masking the saturated data points around the star center.

    Returns:
        1D stellar spectrum

    Written: Chen Xie, 2023-10.
    """
    nz, ny, nx = input_cube.shape
    x_center = nx//2 
    y_center = ny//2 
    center_mask_2D = np.ones((1,ny,nx))
    for y in range(ny):
        for x in range(nx):
            if (abs(x - x_center)) ** 2 + (abs(y - y_center)) ** 2 < center_mask_radius ** 2:
                center_mask_2D[:,y,x] = np.nan
    stellar_cube = input_cube * input_disk_mask * center_mask_2D 
    # output_stellar_spec_1D = np.nansum(stellar_cube, axis=(1,2))
    output_stellar_spec_1D = np.nanmedian(stellar_cube, axis=(1,2))
    return output_stellar_spec_1D


def reduce_data(sci_filename, ref_filename, disk_mask_cube, disk_FoV_mask, savepath, savepath_intermediate_products, x_center, y_center, inner_mask_radius, outer_mask_radius):

    sci_cube, header_sci = fits.getdata(sci_filename, header=True)
    ref_cube, header_ref = fits.getdata(ref_filename, header=True)

    # creating spike mask 
    # There are two parameters here that have help to adjust the size of spike mask, the median filter size and threshold
    # to adjust the size and the threshold, we can check the output median filtered image and spike mask
    print('starting make spike mask')
    if Path(savepath_intermediate_products, 'spike_mask.fits').exists():
        spike_mask = fits.getdata(savepath_intermediate_products + '/spike_mask.fits')
    else:
        median_filtered_cube, med_filtered_combined_img, spike_mask = make_spike_mask(sci_cube, filter_size=40, threshold=100) 
        fits.writeto(savepath_intermediate_products + '/sci_cube_median_filtered.fits', median_filtered_cube , overwrite=True )
        fits.writeto(savepath_intermediate_products + '/sci_cube_median_filtered_combined.fits', med_filtered_combined_img , overwrite=True )
        fits.writeto(savepath_intermediate_products + '/spike_mask.fits', spike_mask, overwrite=True )
 
    if Path(savepath_intermediate_products, 'center_mask_cube.fits').exists():
        mask_cube = fits.getdata(savepath_intermediate_products + '/mask_cube.fits')
        center_mask_2D = fits.getdata(savepath_intermediate_products + '/center_mask_cube.fits')
    else:
        # create a mask
        print('creating make cube')
        nz, ny, nx = sci_cube.shape

        mask = np.ones((nz,ny,nx))
        center_mask_2D = np.ones((1,ny,nx))
        for y in range(ny):
            for x in range(nx):
                if (abs(x - x_center)) ** 2 + (abs(y - y_center)) ** 2 < inner_mask_radius ** 2:
                    mask[:,y,x] = np.nan
                if (abs(x - x_center)) ** 2 + (abs(y - y_center)) ** 2 > outer_mask_radius ** 2:
                    mask[:,y,x] = np.nan
                ## CENTER area
                if (abs(x - x_center)) ** 2 + (abs(y - y_center)) ** 2 < 6 ** 2:
                    center_mask_2D[:,y,x] = np.nan

        mask_cube = mask
        disk_mask_cube[disk_mask_cube==0] = np.nan 
        mask_cube = mask_cube * disk_mask_cube * spike_mask *disk_FoV_mask 
        fits.writeto(savepath_intermediate_products + '/mask_cube.fits', mask_cube, overwrite=True )
        fits.writeto(savepath_intermediate_products + '/center_mask_cube.fits', center_mask_2D, overwrite=True )

    print('starting stellar spectrum extraction')
    ref_stellar_spec_1D = extract_stellar_spec(ref_cube, disk_FoV_mask, center_mask_radius=6) 
    fits.writeto(savepath_intermediate_products + '/ref_stellar_spectrum_1D.fits', ref_stellar_spec_1D, overwrite=True )

    plot_spectrum(ref_stellar_spec_1D, header_sci, 'ref_stellar_spectrum', savepath_intermediate_products, 'reference stellar spectrum')

    res_cube, scaling_factor, cost = single_frame_sub(sci_cube, ref_cube, mask_cube )
    
    fits.writeto(savepath_intermediate_products + '/RDI_median.fits', np.nanmedian(res_cube, axis=0)*center_mask_2D, header_sci, overwrite=True )
    fits.writeto(savepath_intermediate_products + '/RDI_mean.fits', np.nanmean(res_cube, axis=0)*center_mask_2D, header_sci, overwrite=True )
    fits.writeto(savepath_intermediate_products + '/RDI_mean_spike_masked.fits', np.nanmean(res_cube, axis=0)*center_mask_2D*spike_mask, header_sci, overwrite=True )
    fits.writeto(savepath_intermediate_products + '/scaling_factor.fits', scaling_factor, overwrite=True )
    print('STD f_RDI = ', np.nanstd(scaling_factor))
    fits.writeto(savepath_intermediate_products + '/cost.fits', cost, overwrite=True )

    plot_scaling_factor(scaling_factor, header_sci, 'scaling_factor', savepath_intermediate_products)

    sci_stellar_spec_1D = extract_stellar_spec(res_cube, disk_FoV_mask, center_mask_radius=6) 
    plot_spectrum(sci_stellar_spec_1D, header_sci, 'sci_stellar_spectrum', savepath_intermediate_products, 'science stellar spectrum')

    fits.writeto(savepath_intermediate_products + '/RDI_sci_JWST_NIRSpec_wrongshape.fits', res_cube, header_sci, overwrite=True )

    return res_cube

def make_spike_mask(input_cube, filter_size, threshold=100):
    nz, ny, nx = input_cube.shape
    median_filtered_cube = np.zeros(input_cube.shape)
    input_cube[np.isnan(input_cube)] = 0
    for z in range(nz):
        median_filtered_cube[z,:,:] = input_cube[z,:,:] - ndimage.median_filter(input_cube[z,:,:], filter_size, mode='nearest')
    med_filtered_combined_img = np.nanmedian(median_filtered_cube, axis=0)  
    spike_mask = np.ones((ny,nx))

    spike_mask[med_filtered_combined_img > threshold] = 0  
    spike_mask[spike_mask==0] = np.nan 
    return median_filtered_cube, med_filtered_combined_img, spike_mask

def reshape(RDI_subtracted_cube, x_center, y_center, dir_name, sci_file_name, savepath):
    orig_data = fits.open(dir_name + '/cube_images/' + sci_file_name)
    orig_length = orig_data[1].header['NAXIS1'] 
    if (orig_length // 2) * 2 != orig_length:
        r_side = orig_length // 2
        l_side = (orig_length // 2) + 1
        
    orig_height = orig_data[1].header['NAXIS2'] 
    if (orig_height // 2) * 2 != orig_height:
        t_side = (orig_height // 2) + 1
        b_side = (orig_height // 2) 

    if (orig_height == RDI_subtracted_cube.shape[1]) & (orig_length == RDI_subtracted_cube.shape[2]):
        print('no resizing needed')
        orig_data[1].data = RDI_subtracted_cube
    elif (orig_height != RDI_subtracted_cube.shape[1]) & (orig_length == RDI_subtracted_cube.shape[2]):
        print('resizing height')
        new_data_reshaped = RDI_subtracted_cube[:, y_center-b_side:y_center+t_side, :]
    elif (orig_height == RDI_subtracted_cube.shape[1]) & (orig_length != RDI_subtracted_cube.shape[2]):
        print('resizing length')
        new_data_reshaped = RDI_subtracted_cube[:, :, x_center-l_side:x_center+r_side]
    else:
        print('resizing height and length')
        new_data_reshaped = RDI_subtracted_cube[:, y_center-b_side:y_center+t_side, x_center-l_side:x_center+r_side]
    
    orig_data[1].data = new_data_reshaped
    orig_data.writeto(savepath + '/RDI_sci_JWST_NIRSpec.fits', overwrite=True)

def run_PSFsubtraction(dir_name, inner_mask_radius, outer_mask_radius, x_center, y_center, sci_file_name):
    mask = Path(dir_name, 'masks/disk_mask_0_1_2D_2.fits')
    mask_fov = Path(dir_name, 'masks/IFU_align_FoV_extra_spike.fits')

    disk_mask_cube = fits.getdata(mask)
    disk_FoV_mask = fits.getdata(mask_fov) 
    
    sci_file = Path(dir_name, 'centering/sci_centered.fits')
    ref_file = Path(dir_name, 'centering/ref_centered.fits')

    savepath = Path(dir_name, 'psf_subtraction')
    savepath.mkdir(parents=True, exist_ok=True)

    savepath_intermediate_products = Path(dir_name, 'psf_subtraction/intermediate_products')
    savepath_intermediate_products.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    RDI_subtracted_cube = reduce_data(sci_file, ref_file, disk_mask_cube, disk_FoV_mask, str(savepath), str(savepath_intermediate_products), x_center, y_center, inner_mask_radius, outer_mask_radius )

    reshape(RDI_subtracted_cube, x_center, y_center, dir_name, sci_file_name, str(savepath))
    t2 = time.time()
    totalTime = t2-t1
    print('-- Total Processing time: ', round(totalTime,2), ' s')