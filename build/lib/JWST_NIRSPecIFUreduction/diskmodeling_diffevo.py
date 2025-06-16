# written by C. Xie and S. Betti 2024
# ulimit -n 4096

# import multiprocessing
# multiprocessing.set_start_method("fork")
# with python >3.8, mac multiprocessing switched to spawn so global variables do not work.  switch to fork instead.  

import h5py
import time

import numpy as np  
from pathlib import Path

import matplotlib.pyplot as plt

import astropy.io.fits as fits
from astropy.stats import sigma_clip

from scipy import ndimage
from scipy import optimize
from scipy.optimize import differential_evolution
from scipy.signal import convolve2d

from packaging import version
import vip_hci as vip
vvip = vip.__version__
print("VIP version: ", vvip)
if version.parse(vvip) < version.parse("1.0.0"):
    msg = "Please upgrade your version of VIP"
    msg+= "It should be 1.0.0 or above to run this notebook."
    raise ValueError(msg)
elif version.parse(vvip) <= version.parse("1.0.3"):
    from vip_hci.metrics import ScatteredLightDisk
else:
    from vip_hci.fm import ScatteredLightDisk


############# ############# ############# ############# ############# 
def make_NIRSpec_PSF(dirname):
    nirspec_psf_hdu=fits.open(Path(dirname, 'masks/NRS_IFU_Prism_PSF.fits'))
    Nirspec_PSF_cube=nirspec_psf_hdu[1].data
    Nirspec_PSF_cube[:5,:,:]=Nirspec_PSF_cube[6,:,:]
    Nirspec_PSF_cube[:-5,:,:]=Nirspec_PSF_cube[-6,:,:]
    PSF_slice=Nirspec_PSF_cube[0]
    for i in range(len(Nirspec_PSF_cube)):
        total_PSF=np.nansum(Nirspec_PSF_cube[i])
        scale_fact=1/total_PSF
        Nirspec_PSF_cube[i]=Nirspec_PSF_cube[i]*scale_fact

    return Nirspec_PSF_cube
############# ############# ############# ############# ############# 

def make_spike_mask(input_cube, filter_size, threshold):
    nz, ny, nx = input_cube.shape
    median_filtered_cube = np.zeros(input_cube.shape)
    input_cube[np.isnan(input_cube)] = 0
    for z in range(nz):
        median_filtered_cube[z,:,:] = input_cube[z,:,:] - ndimage.median_filter(input_cube[z,:,:], filter_size, mode='nearest')
    med_filtered_combined_img = np.nanmean(median_filtered_cube, axis=0) 
    spike_mask = np.ones((ny,nx))

    spike_mask[med_filtered_combined_img>threshold] = 0 
    spike_mask[spike_mask==0] = np.nan 

    return median_filtered_cube, med_filtered_combined_img, spike_mask


def single_frame_sub(sci_frame, ref_frame, mask_frame, mask_chi2, len_params, constant_f=None):

    def cost_function_subtraction(nu,):
        return np.log(np.nansum( ((nu * ref_img  - sci_img))**2  , axis=(0,1)))

    if mask_frame is None:
        mask_frame = np.ones(sci_frame.shape)
        print('*******   Note: using the entire FoV in scaling the reference image in RDI   *******')
    nu_0 = np.nansum(sci_frame * mask_frame, axis=(0,1)) / np.nansum(sci_frame * mask_frame, axis=(0,1)) 

    sci_img = sigma_clip(sci_frame*mask_frame, sigma=5, maxiters=10, masked=False, copy=True, axis=(0,1))
    ref_img = sigma_clip(ref_frame*mask_frame, sigma=5, maxiters=10, masked=False, copy=True, axis=(0,1))

    
    if constant_f is None:
        minimum = optimize.fmin(cost_function_subtraction, nu_0, disp=False, full_output=True)
        scaling_factor = minimum[0][0]
    else:
        scaling_factor = constant_f
    res_frame = sci_frame - (scaling_factor * ref_frame)

    mask_chi2_v2 = np.zeros_like(mask_chi2)
    mask_chi2_v2[123-5:123+5, 75-5:75+5] = 1
    mask_chi2_v2[mask_chi2_v2 ==0] = np.nan

    data1 = sci_frame * mask_chi2_v2
    res1 = res_frame * mask_chi2_v2

    chi2_kinda = np.nansum(res1**2. / data1**2.) / (np.count_nonzero(~np.isnan(data1)) - len_params)
    return chi2_kinda, scaling_factor, res_frame


def plot_parameters(filename, savepath):
    f = h5py.File(filename) # using the larger mask for whole reduction 
    r = f['residuals']
    bm = f['model_disk']
    p = f['best_params']

    res = np.copy(r)
    best_model =  np.copy(bm)
    param = np.copy(p)

    f.close()
    fig, axes = plt.subplots(nrows=param.shape[1], ncols=1, sharex=True, dpi=150)
    names = ['amplitude',  'g', 'f_RDI', 'chi2']
    for i, ax in enumerate(axes):
        ax.plot(param[:,i], color='k')
        ax.text(0.1, 0.2,names[i], transform=ax.transAxes)

        if i==2:
            IND = np.where(param[:,i] == 0)
            if not IND[0].size:
                IND = -1
            else:
                IND = IND[0][0]

            med = np.nanmean(param[:,i][0:IND])
            std = np.nanstd(param[:,i][0:IND]) 
            ax.axhline(med, color='goldenrod')
            ax.axhline(med-std, color='goldenrod')
            ax.axhline(med+std, color='goldenrod')
            ax.axvline(IND, color='olive')
            
            ax.text(0.7, 0.2,'f_RDI='+str(round(med,3)) + '+/-' + str(round(std, 3)), transform=ax.transAxes)

    for ax in axes:
        axT = ax.twiny()
        axT.set_xlim(0.6025, 5.3025)
        ax.set_xlim(0,len(param[:,0]))
        if ax == axes[0]:
            axT.set_xlabel('wavelength (μm)')
        else:
            axT.set_xticklabels([])
        axT.minorticks_on()
        ax.tick_params(which='both', direction='in', right=True)
        axT.tick_params(which='both', direction='in')

    axes[2].set_ylim(0.1, 0.5)
    axes[-1].set_xlabel('frame #')
    axes[-1].minorticks_on()
    axes[-1].set_ylim(0, .4)
    if std == 0:
        plt.savefig(Path(savepath, 'parameters_constant_fRDI.pdf'))
    else:
        plt.savefig(Path(savepath, 'parameters_variable_fRDI.pdf'))
    plt.show()

############# ############# ############# ############# #############  

def reshape(data, x_center, y_center, dir_name, sci_file_name):
    orig_data = fits.open(dir_name + '/cube_images/' + sci_file_name)
    orig_length = orig_data[1].header['NAXIS1'] 
    if (orig_length // 2) * 2 != orig_length:
        r_side = orig_length // 2
        l_side = (orig_length // 2) + 1
        
    orig_height = orig_data[1].header['NAXIS2'] 
    if (orig_height // 2) * 2 != orig_height:
        t_side = (orig_height // 2) + 1
        b_side = (orig_height // 2) 

    if (orig_height == data.shape[1]) & (orig_length == data.shape[2]):
        
        new_data_reshaped = data
    elif (orig_height != data.shape[1]) & (orig_length == data.shape[2]):
       
        new_data_reshaped = data[:, y_center-b_side:y_center+t_side, :]
    elif (orig_height == data.shape[1]) & (orig_length != data.shape[2]):
        
        new_data_reshaped = data[:, :, x_center-l_side:x_center+r_side]
    else:
       
        new_data_reshaped = data[:, y_center-b_side:y_center+t_side, x_center-l_side:x_center+r_side]
    
    return new_data_reshaped

def reshape2D(data, x_center, y_center, dir_name, sci_file_name):
    orig_data = fits.open(dir_name + '/cube_images/' + sci_file_name)
    orig_length = orig_data[1].header['NAXIS1'] 
    if (orig_length // 2) * 2 != orig_length:
        r_side = orig_length // 2
        l_side = (orig_length // 2) + 1
        
    orig_height = orig_data[1].header['NAXIS2'] 
    if (orig_height // 2) * 2 != orig_height:
        t_side = (orig_height // 2) + 1
        b_side = (orig_height // 2) 

    if (orig_height == data.shape[0]) & (orig_length == data.shape[1]):
        
        new_data_reshaped = data
    elif (orig_height != data.shape[0]) & (orig_length == data.shape[1]):
        
        new_data_reshaped = data[y_center-b_side:y_center+t_side, :]
    elif (orig_height == data.shape[0]) & (orig_length != data.shape[1]):
        
        new_data_reshaped = data[:, x_center-l_side:x_center+r_side]
    else:
        
        new_data_reshaped = data[y_center-b_side:y_center+t_side, x_center-l_side:x_center+r_side]
    
    return new_data_reshaped

def make_diskmodel(dir_name, savepath, sci_cube, ref_cube, cube_sci_filename, RDI_mask, chi2_mask, Nirspec_PSF_cube, x_center, y_center, bounds, 
                     dstar, itilt, pixel_scale, posang, a, ain, aout, ksi0, gamma, beta, constant_f = None):
    nz, ny, nx = sci_cube.shape
    final_image = np.zeros_like(reshape(sci_cube, x_center, y_center, dir_name, cube_sci_filename))

    if constant_f is None:
        filename = Path(savepath, 'bestfits_cube.h5')
        fitsfilename = Path(savepath, 'bestfits_cube.fits')
    else:
        filename = Path(savepath, f'bestfits_cube_f{round(constant_f,3)}.h5')
        fitsfilename = Path(savepath, f'bestfits_cube_f{round(constant_f,3)}.fits')

    if filename.is_file():
        with h5py.File(str(filename), 'r') as f:
            dset = f['residuals']
            start = dset.attrs['last_index']
    else:
        start = 0

    if start == nz:
        f = h5py.File(filename) # using the larger mask for whole reduction 
        p = f['best_params']
        return filename, fits.getdata(str(fitsfilename)), p[:,2]

    for i in np.arange(start, nz, 1):
        print('starting frame ', i)
        def disk_model_subtract(arguments):
            scale, g=arguments[0],arguments[1]
            bpic_model = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                        itilt=itilt, omega=0, pxInArcsec=pixel_scale, pa=posang,
                                        density_dico={'name':'2PowerLaws', 'ain':ain, 'aout':aout,
                                                    'a':a, 'e':0.0, 'ksi0':ksi0, 'gamma':gamma, 'beta':beta},
                                        spf_dico={'name':'HG', 'g':g, 'polar':False},
                                        flux_max=1.)
            bpic_model_map = bpic_model.compute_scattered_light() 
            convolved_image=convolve2d(scale*bpic_model_map,PSF,mode='same')
            chi2, scaling_factor, res_frame = single_frame_sub(DATA-convolved_image, REF, MASK_RDI, MASK_CHI2 , len(arguments), constant_f=constant_f)
            return chi2

        DATA = sci_cube[i,:,:]
        REF = ref_cube[i,:,:]
        MASK_RDI = RDI_mask[i,:,:]
        MASK_CHI2 = chi2_mask
        PSF = Nirspec_PSF_cube[i,:,:]
  
        result = differential_evolution(disk_model_subtract,bounds=bounds,popsize=10, 
                            recombination=0.7, mutation=(0.5, 1.0), 
                            seed=5, polish=False) #, updating='deferred',workers=-1
        solution = result['x']
        
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])


        scale,g=solution
        bpic_model = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                        itilt=itilt, omega=0, pxInArcsec=pixel_scale, pa=posang,
                                        density_dico={'name':'2PowerLaws', 'ain':ain, 'aout':aout,
                                                    'a':a, 'e':0.0, 'ksi0':ksi0, 'gamma':gamma, 'beta':beta},
                                        spf_dico={'name':'HG', 'g':g, 'polar':False},
                                        flux_max=1.)
        bpic_model_map = bpic_model.compute_scattered_light()
        convolved_image=convolve2d(scale* bpic_model_map,PSF,mode='same')  

        chi2, scaling_factor, res_frame = single_frame_sub(DATA-convolved_image, REF, MASK_RDI, MASK_CHI2, len(solution), constant_f=constant_f)
        print('solution: ', solution, ' chi2: ', chi2)
        solution = np.append(solution, scaling_factor)
        solution = np.append(solution, chi2)
        if i == 0:
            with h5py.File(str(filename), 'w') as f:
                # create empty data set
                sci_cube_reshape = reshape(sci_cube, x_center, y_center, dir_name, cube_sci_filename)
                sci_cube_shape = sci_cube_reshape.shape
                res = f.create_dataset('residuals', shape=sci_cube_shape,
                                        chunks=(1, sci_cube_shape[1], sci_cube_shape[2]),
                                        dtype=np.float32)
                
                moddisk = f.create_dataset('model_disk', shape=sci_cube_shape,
                                        chunks=(1, sci_cube_shape[1], sci_cube_shape[2]),
                                        dtype=np.float32)
                
                best_params = f.create_dataset('best_params', shape=(sci_cube_shape[0], len(solution)))
      
                res[0, :, :] = reshape2D(res_frame, x_center, y_center, dir_name, cube_sci_filename)
                moddisk[0,:,:] = reshape2D(convolved_image, x_center, y_center, dir_name, cube_sci_filename)
                # Create attribute with last_index value
                res.attrs['last_index']=1
                best_params[0,:] = solution
        else:
            # add more data
            with h5py.File(str(filename), 'a') as f: # USE APPEND MODE
                res = f['residuals']
                moddisk = f['model_disk']
                start = res.attrs['last_index']
                # add chunk of rows
                res[start:start+1, :, :] = reshape2D(res_frame, x_center, y_center, dir_name, cube_sci_filename)
                moddisk[start:start+1, :, :] = reshape2D(convolved_image, x_center, y_center, dir_name, cube_sci_filename)
                # Create attribute with last_index value
                res.attrs['last_index']=start+1
                best_params = f['best_params']
                best_params[start:start+1] = solution

        final_image[i,:,:] = reshape2D(res_frame, x_center, y_center, dir_name, cube_sci_filename)

    fits.writeto(str(fitsfilename), final_image, overwrite=True)
    return filename, final_image, scaling_factor

############# ############# ############# ############# ############# 

def run_diskmodeling(dir_name, sci_filename, ref_filename, cube_sci_filename, mask_spike, mask_2D, mask_cube,mask_disk_FoV, 
                     x_center, y_center, bounds, 
                     dstar, itilt, pixel_scale, posang, a, ain, aout, ksi0, gamma, beta):
    t1 = time.time()

    sci_cube, header_sci = fits.getdata(Path(dir_name, f'centering/{sci_filename}'), header=True)
    ref_cube, header_ref = fits.getdata(Path(dir_name, f'centering/{ref_filename}'), header=True)
    
    spike_mask = fits.getdata(Path(dir_name, f'psf_subtraction/intermediate_products/{mask_spike}'))
    disk_region = fits.getdata(Path(dir_name, f'masks/{mask_2D}'))
    mask_cube = fits.getdata(Path(dir_name, f'psf_subtraction/intermediate_products/{mask_cube}'))
    disk_FoV_mask = fits.getdata(Path(dir_name, f'masks/{mask_disk_FoV}')) 
    disk_mask = disk_region * spike_mask * disk_FoV_mask
    RDI_mask = mask_cube
    chi2_mask = disk_mask[0,:,:]
    chi2_mask[85:94, 73:78] = np.nan
    chi2_mask[35:43, 55:60] =np.nan

    savepath = Path(dir_name, 'disk_modeling')
    savepath.mkdir(parents=True, exist_ok=True)

    Nirspec_PSF_cube = make_NIRSpec_PSF(dir_name)
    
    print('starting variable f_RDI')
    filename, final_image, scaling_factor = make_diskmodel(dir_name, savepath, sci_cube, ref_cube, cube_sci_filename, RDI_mask, chi2_mask, Nirspec_PSF_cube, x_center, y_center, bounds, 
                     dstar, itilt, pixel_scale, posang, a, ain, aout, ksi0, gamma, beta, constant_f = None)

    plot_parameters(filename, savepath)
    constant_f = np.nanmean(scaling_factor)

    print('starting constant f_RDI = ', constant_f)
    filename, final_image, scaling_factor = make_diskmodel(dir_name, savepath, sci_cube, ref_cube, cube_sci_filename, RDI_mask, chi2_mask, Nirspec_PSF_cube, x_center, y_center, bounds, 
                     dstar, itilt, pixel_scale, posang, a, ain, aout, ksi0, gamma, beta, constant_f = constant_f)
    plot_parameters(filename, savepath)

    t2 = time.time()
    totalTime = t2-t1
    print('-- Total Processing time: ', round(totalTime,2), ' s')
