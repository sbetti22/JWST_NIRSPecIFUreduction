# functions taken from https://github.com/ChenXie-astro/jwstIFURDI/blob/main/jwstIFURDI/centering.py, written by C. Xie 2024

import time
import numpy as np
from astropy.io import fits
from pathlib import Path

from skimage.transform import rotate

from scipy import optimize
from scipy.interpolate import interp2d
from scipy.ndimage import fourier_shift, median_filter
from scipy.ndimage import shift as spline_shift

def samplingRegion(size_window, theta = [45, 135], m = 0.2, M = 0.8, step = 1, decimals = 2, ray = False):
    """This function returns all the coordinates of the sampling region, the center of the region is (0,0)
    When applying to matrices, don't forget to SHIFT THE CENTER!
    Input:
        size_window: the radius of the sampling region. The whole region should thus have a length of 2*size_window+1.
        theta: the angle range of the sampling region, default: [45, 135] for the anti-diagonal and diagonal directions.
        m: the minimum fraction of size_window, default: 0.2 (i.e., 20%). In this way, the saturated region can be excluded.
        M: the maximum fraction of size_window, default: 0.8 (i.e., 80%). Just in case if there's some star along the diagonals.
        step: the seperation between sampling dots (units: pixel), default value is 1pix.
        decimals: the precisoin of the sampling dots (units: pixel), default value is 0.01pix.
        ray: only half of the line?
    Output: (xs, ys)
        xs: x indecies, flattend.
        ys: y indecies, flattend.
    Example:
        1. If you call "xs, ys = samplingRegion(5)", you will get:
        xs: array([-2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83,  2.83, 2.12,  1.41,  0.71, -0.71, -1.41, -2.12, -2.83]
        ys: array([-2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83, -2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83]))
        2. For "radonCenter.samplingRegion(5, ray=True)", you will get:
        xs: array([ 0.71,  1.41,  2.12,  2.83, -0.71, -1.41, -2.12, -2.83])
        ys: array([ 0.71,  1.41,  2.12,  2.83,  0.71,  1.41,  2.12,  2.83])
    """
    
    if np.asarray(theta).shape == ():
        theta = [theta]
    #When there is only one angle
        
    theta = np.array(theta)
    if ray:
        zeroDegXs = np.arange(int(size_window*m), int(size_window*M) + 0.1 * step, step)
    else:
        zeroDegXs = np.append(np.arange(-int(size_window*M), -int(size_window*m) + 0.1 * step, step), np.arange(int(size_window*m), int(size_window*M) + 0.1 * step, step))
    #create the column indecies if theta = 0
    zeroDegYs = np.zeros(zeroDegXs.size)
    
    xs = np.zeros((np.size(theta), np.size(zeroDegXs)))
    ys = np.zeros((np.size(theta), np.size(zeroDegXs)))
    
    for i, angle in enumerate(theta):
        degRad = np.deg2rad(angle)
        angleDegXs = np.round(zeroDegXs * np.cos(degRad), decimals = decimals)
        angleDegYs = np.round(zeroDegXs * np.sin(degRad), decimals = decimals)
        xs[i, ] = angleDegXs
        ys[i, ] = angleDegYs
    
    xs = xs.flatten()
    ys = ys.flatten()

    return xs, ys


def smoothCostFunction(costFunction, halfWidth = 0):
    """
    smoothCostFunction will smooth the function within +/- halfWidth, i.e., to replace the value with the average within +/- halfWidth pixel.
    This function can be genrally used to smooth any 2D matrix.
    Input:
        costFunction: original cost function, a matrix.
        halfWdith: the half width of the smoothing region, default = 0 pix.
    Output:
        newFunction: smoothed cost function.
    """
    if halfWidth == 0:
        return costFunction
    else:
        newFunction = np.zeros(costFunction.shape)
        rowRange = np.arange(costFunction.shape[0], dtype=int)
        colRange = np.arange(costFunction.shape[1], dtype=int)
        rangeShift = np.arange(-halfWidth, halfWidth + 0.1, dtype=int)
        for i in rowRange:
            for j in colRange:
                if np.isnan(costFunction[i, j]):
                    newFunction[i, j] = np.nan
                else:
                    surrondingNumber = (2 * halfWidth + 1) ** 2
                    avg = 0
                    for ii in (i + rangeShift):
                        for jj in (j + rangeShift):
                            if (not (ii in rowRange)) or (not (jj in colRange)) or (np.isnan(costFunction[ii, jj])):
                                surrondingNumber -= 1
                            else:
                                avg += costFunction[ii, jj]
                    newFunction[i, j] = avg * 1.0 / surrondingNumber
    return newFunction
    
 
def searchCenter(image, x_ctr_assign, y_ctr_assign, size_window, m = 0.2, M = 0.8, size_cost = 5, theta = [45, 135], ray = False, smooth = 2, decimals = 2):
    """
    This function searches the center in a grid, 
    calculate the cost function of Radon Transform (Pueyo et al., 2015), 
    then interpolate the cost function, 
    get the center which corresponds to the maximum value in the cost function.
    
    Input:
        image: 2d array.
        x_ctr_assign: the assigned x-center, or starting x-position; for STIS, the "CRPIX1" header is suggested.
        x_ctr_assign: the assigned y-center, or starting y-position; for STIS, the "CRPIX2" header is suggested.
        size_window: half width of the sampling region; size_window = image.shape[0]/2 is suggested.
            m & M:  The sampling region will be (-M*size_window, -m*size_window)U(m*size_window, M*size_window).
        size_cost: search the center within +/- size_cost pixels, i.e., a square region.
        theta: the angle range of the sampling region; default: [45, 135] for the anti-diagonal and diagonal directions.
        ray: is the theta a line or a ray? Default: line.
        smooth: smooth the cost function, for one pixel, replace it by the average of its +/- smooth neighbours; defualt = 2.
        decimals: the precision of the centers; default = 2 for a precision of 0.01.
    Output:
        x_cen, y_cen
    """
    (y_len, x_len) = image.shape

    x_range = np.arange(x_len)
    y_range = np.arange(y_len)

    image_interp = interp2d(x_range, y_range, image, kind = 'cubic')
    #interpolate the image
    
    
    precision = 1
    x_centers = np.round(np.arange(x_ctr_assign - size_cost, x_ctr_assign + size_cost + precision/10.0, precision), decimals=1)
    y_centers = np.round(np.arange(y_ctr_assign - size_cost, y_ctr_assign + size_cost + precision/10.0, precision), decimals=1)
    costFunction = np.zeros((x_centers.shape[0], y_centers.shape[0]))
    #The above 3 lines create the centers of the search region
    #The cost function stores the sum of all the values in the sampling region
    
    size_window = size_window - size_cost
    (xs, ys) = samplingRegion(size_window, theta, m = m, M = M, ray = ray)
    #the center of the sampling region is (0,0), don't forget to shift the center!

    for j, x0 in enumerate(x_centers):
        for i, y0 in enumerate(y_centers):
            value = 0
            
            for x1, y1 in zip(xs, ys):
                x = x0 + x1    #Shifting the center, this now is the coordinate of the RAW IMAGE
                y = y0 + y1
            
                value += image_interp(x, y)
        
            costFunction[i, j] = value  #Create the cost function

    costFunction = smoothCostFunction(costFunction, halfWidth = smooth)
    #Smooth the cost function
    
    interp_costfunction = interp2d(x_centers, y_centers, costFunction, kind='cubic')
    
    
    for decimal in range(1, decimals+1):
        precision = 10**(-decimal)
        if decimal >= 2:
            size_cost = 10*precision
        x_centers_new = np.round(np.arange(x_ctr_assign - size_cost, x_ctr_assign + size_cost + precision/10.0, precision), decimals=decimal)
        y_centers_new = np.round(np.arange(y_ctr_assign - size_cost, y_ctr_assign + size_cost + precision/10.0, precision), decimals=decimal)
    
        x_cen = 0
        y_cen = 0
        maxcostfunction = 0
        value = np.zeros((y_centers_new.shape[0], x_centers_new.shape[0]))
    
        for j, x in enumerate(x_centers_new):
            for i, y in enumerate(y_centers_new):
                value[i, j] = interp_costfunction(x, y)
        
        idx = np.where(value == np.max(value))
        #Just in case when there are multile maxima, then use the average of them. 
        x_cen = np.mean(x_centers_new[idx[1]])
        y_cen = np.mean(y_centers_new[idx[0]])
        
        x_ctr_assign = x_cen
        y_ctr_assign = y_cen    
       
    x_cen = round(x_cen, decimals)
    y_cen = round(y_cen, decimals)
    return x_cen, y_cen

# imshift was copied from spaceKLIP
def imshift(image,
            shift,
            pad=True,
            cval=0.,
            method='fourier',
            kwargs={}):
    """
    Shift an image.
    
    Parameters
    ----------
    image : 2D-array
        Input image to be shifted.
    shift : 1D-array
        X- and y-shift to be applied.
    pad : bool, optional
        Pad the image before shifting it? Otherwise, it will wrap around
        the edges. The default is True.
    cval : float, optional
        Fill value for the padded pixels. The default is 0.
    method : 'fourier' or 'spline' (not recommended), optional
        Method for shifting the frames. The default is 'fourier'.
    kwargs : dict, optional
        Keyword arguments for the scipy.ndimage.shift routine. The default
        is {}.
    
    Returns
    -------
    imsft : 2D-array
        The shifted image.
    
    """
    
    if pad:
        
        # Pad image.
        sy, sx = image.shape
        xshift, yshift = shift
        padx = np.abs(int(xshift)) + 5
        pady = np.abs(int(yshift)) + 5
        impad = np.pad(image, ((pady, pady), (padx, padx)), mode='constant', constant_values=cval)
        
        # Shift image.
        if method == 'fourier':
            imsft = np.fft.ifftn(fourier_shift(np.fft.fftn(impad), shift[::-1])).real
        elif method == 'spline':
            imsft = spline_shift(impad, shift[::-1], order=5, **kwargs)
        else:
            raise UserWarning('Image shift method "' + method + '" is not known')
        
        # Crop image to original size.
        return imsft[pady:pady + sy, padx:padx + sx]
    else:
        if method == 'fourier':
            return np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift[::-1])).real
        elif method == 'spline':
            return spline_shift(image, shift[::-1],order=5, **kwargs)
        else:
            raise UserWarning('Image shift method "' + method + '" is not known')


def making_weight_map_2D(input_img, x_center, y_center, function_index):
    # center is the image center
    #############################
    #    weight_function = r**a
    #############################
    # input 2D array
    a = function_index
    ny, nx = input_img.shape 

    pixel_size = 0.1
    weight_map = np.ones(input_img.shape)
    for x in range(nx):
        for y in range(ny):
            r = np.sqrt( (x-x_center)**2 + (y-y_center)**2 ) * pixel_size
            if r == 0:
                weight_map[y,x] = 0.001
            else:
                weight_map[y,x] = r ** a 

    return weight_map

def frame_rotate(array, angle, rot_center=None, interp_order=4, border_mode='constant'):
    """ Rotates a frame or 2D array.   
    Parameters
    ----------
    array : Input image, 2d array.
    angle : Rotation angle.
    rot_center : Coordinates X,Y  of the point with respect to which the rotation will be 
                performed. By default the rotation is done with respect to the center 
                of the frame; central pixel if frame has odd size.
    interp_order: Interpolation order for the rotation. See skimage rotate function.
    border_mode : Pixel extrapolation method for handling the borders. 
                See skimage rotate function.
        
    Returns
    -------
    array_out : Resulting frame.      
    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')

    ny,nx = array.shape 
    x_center = nx//2  # There is a systematic offset from our frame registration.
    y_center = ny//2 
    rot_center = (x_center, y_center) 

    min_val = np.nanmin(array)
    im_temp = array - min_val
    max_val = np.nanmax(im_temp)
    im_temp /= max_val
    array_out = rotate(im_temp, angle, order=interp_order, center=rot_center, cval=-10000,
                       mode=border_mode)

    array_out *= max_val
    array_out += min_val
    IND = np.where(array_out == -10000)
    array_out[IND] = 0
    # array_out = np.nan_to_num(array_out)
             
    return array_out


def derotate_and_combine(cube, angles, method):
    """ Derotates a cube of images then mean-combine them.   
    Parameters
    ----------
    cube : the input cube, 3D array.
    angles : the list of parallactic angles corresponding to the cube.
        
    Returns
    -------
    image_out : the mean-combined image
    cube_out : the cube of derotated frames.   
    """
    if cube.ndim != 3:
        raise TypeError('Input cube is not a cube or 3d array')
    if angles.ndim != 1:
        raise TypeError('Input angles must be a 1D array')
    if len(cube) != len(angles):
        raise TypeError('Input cube and input angle list must have the same length')
        
    shape = cube.shape
    cube_out = np.zeros(shape)
    for im in range(shape[0]):
        cube_out[im] = frame_rotate(cube[im], -angles[im])
    
    if method == 'mean':
        image_out = np.nanmean(cube_out, axis=0)
    elif method == 'median':
        image_out = np.nanmedian(cube_out, axis=0)
 
    return image_out, cube_out



def output_information(title, nx, ny, offset_x_sci, offset_y_sci, offset_x_ref, offset_y_ref):
    print('------  {0}  ------'.format(title))
    print('x_cen_sci, y_cen_sci :', nx//2 - offset_x_sci, ny//2 - offset_y_sci)
    print('x_cen_ref, y_cen_ref :', nx//2 - offset_x_ref, ny//2 - offset_y_ref)

    print('offset_x_sci, offset_y_sci :', offset_x_sci, offset_y_sci)
    print('offset_x_ref, offset_y_ref :', offset_x_ref, offset_y_ref)

def expand_img_IFS(input_cube, cen_x, cen_y, original_img_size_x, original_img_size_y, new_img_size_x, new_img_size_y):
    # expanding the size of IFS cube for centering and rotating (if needed)
    # input_cube should be a 2D or 3D array
    box_x = original_img_size_x//2 
    box_y = original_img_size_y//2 
    if np.ndim(input_cube) == 2: 
        output_cube = np.zeros((new_img_size_y, new_img_size_x))
        # ny,nx = input_cube.shape
        # cen_y = ny//2 
        # cen_x = nx//2
        # input_cube = input_cube[65 : 250-65, 65 : 250-65]
        output_cube[cen_y - box_y : cen_y + box_y, cen_x - box_x : cen_x + box_x] =  input_cube

    elif np.ndim(input_cube) == 3:
        output_cube = np.zeros((input_cube.shape[0], new_img_size_y, new_img_size_x))
        # ny = input_cube.shape[1]
        # nx = input_cube.shape[2]
        # cen_y = ny//2 
        # cen_x = nx//2 
        # input_cube = input_cube[:, cen_y - box_y : cen_y + box_y, cen_x - box_x : cen_x + box_x]
        output_cube[:, cen_y - box_y : cen_y + box_y+1, cen_x - box_x : cen_x + box_x+1] = input_cube

    elif np.ndim(input_cube) == 4: 
        output_cube = np.zeros((input_cube.shape[0], input_cube.shape[1], new_img_size_y, new_img_size_x))
        # ny = input_cube.shape[2]
        # nx = input_cube.shape[3]
        # cen_y = ny//2 
        # cen_x = nx//2 
        output_cube[:, :, cen_y - box_y : cen_y + box_y, cen_x - box_x : cen_x + box_x] = input_cube
    output_cube[output_cube==0] = np.nan    
    return output_cube
    
def spike_img_weighted(input_cube, x_center, y_center, filter_size):
    # create spike images by using median filter and r2 weight
    nz, ny, nx = input_cube.shape       # 3D cube
    median_img = np.nanmedian(input_cube, axis=0)
    median_img[np.isnan(median_img)] = 0
    median_filtered = (median_img - median_filter(median_img, filter_size, mode='nearest'))
    weight_map = median_filtered * making_weight_map_2D(median_img , x_center, y_center, function_index=2) 
    weight_map  = weight_map.reshape(ny,nx) 

    return  weight_map 
 
def find_center(sci_cube, ref_cube, sci_header, ref_header, filter_size, x_center, y_center, savepath):
    nz, ny, nx = sci_cube.shape

    weight_map_sci = spike_img_weighted(sci_cube, x_center, y_center, filter_size)
    weight_map_ref = spike_img_weighted(ref_cube, x_center, y_center, filter_size)

    # e.g., theta_ifu = [45, 135]  # theta zeros point: A ray comes in from the axis, makes an angle at the origin (measured counter-clockwise from that axis), and departs from the origin.
    # theta_ifu = [v3PA-30, v3PA, v3PA+30]
    v3PA = sci_header['PA_V3']
    v3PA = 131 # if IFU_align
    theta_ifu_sci = [v3PA-30, v3PA, v3PA+30, v3PA+90 ] # HD 181327
    # theta_ifu_sci = [v3PA-30, v3PA,  v3PA+90 ] # edge on disk; i.e., beta pic 
    v3PA = ref_header['PA_V3']
    v3PA = 131 # if IFU_align
    theta_ifu_ref = [v3PA-30, v3PA, v3PA+30, v3PA+90] # HD 181327
    # theta_ifu_ref = [v3PA-30, v3PA,  v3PA+90] # edge on disk; i.e., beta pic
    (x_cen_sci, y_cen_sci) = searchCenter(weight_map_sci, x_ctr_assign=x_center, y_ctr_assign=y_center, size_window = weight_map_sci.shape[0]/2, theta = theta_ifu_sci)
    (x_cen_ref, y_cen_ref) = searchCenter(weight_map_ref, x_ctr_assign=x_center, y_ctr_assign=y_center, size_window = weight_map_ref.shape[0]/2, theta = theta_ifu_ref)
    
    weight_map_sci = spike_img_weighted(sci_cube, x_cen_sci, y_cen_sci, filter_size)
    weight_map_ref = spike_img_weighted(ref_cube, x_cen_ref, y_cen_ref, filter_size)
    fits.writeto(savepath + '/intermediate_products/sci_median_filtered_r2.fits',  weight_map_sci, overwrite=True )
    fits.writeto(savepath + '/intermediate_products/ref_median_filtered_r2.fits',  weight_map_ref, overwrite=True )

    (x_cen_sci, y_cen_sci) = searchCenter(weight_map_sci, x_ctr_assign=x_cen_sci, y_ctr_assign=y_cen_sci, size_window = weight_map_sci.shape[0]/2, theta = theta_ifu_sci)
    (x_cen_ref, y_cen_ref) = searchCenter(weight_map_ref, x_ctr_assign=x_cen_ref, y_ctr_assign=y_cen_ref, size_window = weight_map_ref.shape[0]/2, theta = theta_ifu_ref)

    offset_x_sci = nx//2 - x_cen_sci 
    offset_y_sci = ny//2 - y_cen_sci 

    offset_x_ref = nx//2 - x_cen_ref 
    offset_y_ref = ny//2 - y_cen_ref 
    return  offset_x_sci, offset_y_sci, offset_x_ref, offset_y_ref 


def IFU_centering(sci_filename, ref_filename,   savepath,savepath_intermediate_products, x_center, y_center, new_img_size_x, new_img_size_y, filter_size, channel_longest):
    print('-------- start processing  -------- ')
    sci_cube, sci_header = fits.getdata(sci_filename, header=True)
    ref_cube, ref_header = fits.getdata(ref_filename, header=True)

    # expend IFS cube

    sci_cube = expand_img_IFS(sci_cube, (new_img_size_x//2 + sci_cube.shape[2]//2-int(sci_header['CRPIX1'])), (new_img_size_y//2 + sci_cube.shape[1]//2-int(sci_header['CRPIX2'] -1)), sci_cube.shape[2], sci_cube.shape[1], new_img_size_x, new_img_size_y)
    ref_cube = expand_img_IFS(ref_cube, (new_img_size_x//2 + ref_cube.shape[2]//2-int(ref_header['CRPIX1'])), (new_img_size_y//2 + ref_cube.shape[1]//2-int(ref_header['CRPIX2'] -1)), ref_cube.shape[2], ref_cube.shape[1], new_img_size_x, new_img_size_y)

    fits.writeto(savepath_intermediate_products + '/sci_cube_expand.fits', sci_cube, sci_header, overwrite=True )
    fits.writeto(savepath_intermediate_products + '/ref_cube_expand.fits', ref_cube, ref_header, overwrite=True )
    fits.writeto(savepath_intermediate_products + '/sci_cube_median.fits', np.nanmedian(sci_cube, axis=0), sci_header, overwrite=True )


    # disk_region = fits.getdata(disk_mask_filename)
    disk_region = np.ones(sci_cube.shape)
    ref_cube_mask = np.ones(ref_cube.shape)
    ref_cube_mask[np.isnan(ref_cube)] = 0
    ref_cube_mask_shifted = np.zeros((ref_cube.shape))

    sci_cube_mask = np.ones(sci_cube.shape)
    sci_cube_mask[np.isnan(sci_cube)] = 0
    sci_cube_mask_shifted = np.zeros((sci_cube.shape))

    nz, ny, nx = sci_cube.shape

    offset_x_sci, offset_y_sci, offset_x_ref, offset_y_ref = find_center(sci_cube[0:channel_longest], ref_cube[0:channel_longest], sci_header, ref_header, filter_size, x_center, y_center, savepath)
    output_information('before first alignment (checking offsets)', nx, ny, offset_x_sci, offset_y_sci, offset_x_ref, offset_y_ref)


    aligned_sci_cube = np.zeros((sci_cube.shape))
    aligned_ref_cube = np.zeros((ref_cube.shape))
    # aligned_ref_cube_test = np.zeros((ref_cube.shape))
    ref_cube[np.isnan(ref_cube)] = 0
    sci_cube[np.isnan(sci_cube)] = 0
    for z in range(nz):
        # aligned_sci_cube[z, :, :] = imutils.shift(sci_cube[z, :,:].astype(float), (offset_x_sci, offset_y_sci), method='fft',)
        # aligned_ref_cube[z, :, :] = imutils.shift(ref_cube[z, :,:].astype(float), (offset_x_ref, offset_y_ref), method='fft',)
        aligned_sci_cube[z, :, :] = imshift(sci_cube[z, :,:].astype(float), [offset_x_sci, offset_y_sci], method = 'spline', )
        aligned_ref_cube[z, :, :] = imshift(ref_cube[z, :,:].astype(float), [offset_x_ref, offset_y_ref], method = 'spline', )
    
    residual_offset_x_sci, residual_offset_y_sci, residual_offset_x_ref, residual_offset_y_ref = find_center(aligned_sci_cube[0:channel_longest], aligned_ref_cube[0:channel_longest], sci_header, ref_header, filter_size, x_center, y_center, savepath)
    output_information('first alignment residual', nx, ny, residual_offset_x_sci, residual_offset_y_sci, residual_offset_x_ref, residual_offset_y_ref)



    new_aligned_sci_cube = np.zeros((aligned_sci_cube.shape))
    new_aligned_ref_cube = np.zeros((aligned_ref_cube.shape))
    for z in range(nz):
        new_aligned_sci_cube[z, :, :] = imshift(sci_cube[z, :,:].astype(float), [offset_x_sci + residual_offset_x_sci, offset_y_sci + residual_offset_y_sci], method = 'spline', )
        new_aligned_ref_cube[z, :, :] = imshift(ref_cube[z, :,:].astype(float), [offset_x_ref + residual_offset_x_ref, offset_y_ref + residual_offset_y_ref], method = 'spline', )

        # shifting mask
        sci_cube_mask_shifted[z, :, :] = imshift(sci_cube_mask[z, :,:].astype(float), (offset_x_sci + residual_offset_x_sci, offset_y_sci + residual_offset_y_sci), method='fourier',)
        ref_cube_mask_shifted[z, :, :] = imshift(ref_cube_mask[z, :,:].astype(float), (offset_x_ref + residual_offset_x_ref, offset_y_ref + residual_offset_y_ref), method='fourier',) 


    sci_cube_mask_shifted[sci_cube_mask_shifted<0.1] = np.nan
    sci_cube_mask_shifted[~np.isnan(sci_cube_mask_shifted)] = 1
    ref_cube_mask_shifted[ref_cube_mask_shifted <0.1] = 0
    ref_cube_mask_shifted[ref_cube_mask_shifted != 0] = 1


    # search for the PA offsets between sci and ref osbervations
    offset_x_sci, offset_y_sci, offset_x_ref, offset_y_ref = find_center(new_aligned_sci_cube[0:channel_longest], new_aligned_ref_cube[0:channel_longest], sci_header, ref_header, filter_size, x_center, y_center, savepath)
    output_information('final alignment residual', nx, ny, offset_x_sci, offset_y_sci, offset_x_ref, offset_y_ref)

    sci_header['CRPIX1'] = x_center 
    sci_header['CRPIX2'] = y_center  
    ref_header['CRPIX1'] = x_center  
    ref_header['CRPIX2'] = y_center 

    fits.writeto(savepath_intermediate_products + '/sci_cube_expand_shifted.fits',  new_aligned_sci_cube, sci_header, overwrite=True )
    fits.writeto(savepath_intermediate_products + '/ref_cube_expand_shifted.fits',  new_aligned_ref_cube, ref_header, overwrite=True )



    # search for the PA offsets between sci and ref osbervations
    def cost_function_theta(theta,):
        """
        Returns the vaule of the cost function used for the single ref frame RDI approch.

        Args:
            nu: scaling factor 
        Returns:
            cost

        Written: Chen Xie, 2023-10.

        Note that: 'sci_image' (sci image), 'ref_img' (ref), 'mask_img' (mask) are global variables in this nested function that will be updated in each interation.
        """
        theta = np.ones((1)) * theta
        tmp, ref_rotated = derotate_and_combine(weight_map_ref_shifted.reshape((1,ny,nx)), theta, method='median')
        return np.log(np.nansum( ((nu_0 * ref_rotated.reshape(ny,nx)  - sci_img) * mask_img)**2  , axis=(0,1)))


    new_aligned_sci_cube[np.isnan(new_aligned_sci_cube)] = 0
    new_aligned_ref_cube[np.isnan(new_aligned_ref_cube)] = 0

    median_img_sci = np.nanmedian(new_aligned_sci_cube, axis=0)
    median_img_ref = np.nanmedian(new_aligned_ref_cube, axis=0)
    median_img_sci[np.isnan(median_img_sci)] = 0
    median_img_ref[np.isnan(median_img_ref)] = 0
    weight_map_sci_shifted = (median_img_sci - median_filter(median_img_sci, filter_size, mode='nearest')) * making_weight_map_2D(median_img_sci, nx//2, ny//2, function_index=2) 
    weight_map_ref_shifted = (median_img_ref - median_filter(median_img_ref, filter_size, mode='nearest')) * making_weight_map_2D(median_img_ref, nx//2, ny//2, function_index=2)

    mask = np.ones(sci_cube.shape)
    nu_0 = np.nansum(weight_map_sci_shifted * mask[0], axis=(0,1)) / np.nansum(weight_map_ref_shifted * mask[0], axis=(0,1)) 
    ref_img = weight_map_ref_shifted
    sci_img = weight_map_sci_shifted
    mask_img = mask[0]

    offset_v3PA =  sci_header['PA_V3'] - ref_header['PA_V3']
    minimum = optimize.fmin(cost_function_theta, offset_v3PA, disp=False)

    print('nu_0', nu_0)
    print(minimum[0])
    print(offset_v3PA)
    print('PA difference in header: ', minimum[0] - offset_v3PA)
    if minimum[0] >= 0.01: 
        tmp, ref_rotated = derotate_and_combine(weight_map_ref_shifted.reshape((1,ny,nx)), np.ones(1)*minimum[0], method='median')
        check_residual = (nu_0 * ref_rotated.reshape(ny,nx)  - sci_img) * mask_img
        fits.writeto(savepath_intermediate_products + '/check_residual_aligned.fits',  check_residual,  overwrite=True)
        # return 

        aligned_ref_cube[np.isnan(aligned_ref_cube)] = 0
        tmp, aligned_ref_cube_rotated = derotate_and_combine(new_aligned_ref_cube, np.ones((nz))*minimum[0], method='median')
        # tmp, sci_rotated_mask = derotate_and_combine(sci_cube_mask, np.ones((nz))*minimum[0], method='median')
        tmp, ref_rotated_mask = derotate_and_combine(ref_cube_mask_shifted, np.ones((nz))*minimum[0], method='median')
        ref_rotated_mask[ref_rotated_mask<0.1] = np.nan
        ref_rotated_mask[~np.isnan(ref_rotated_mask)] = 1

        ref_header['PA_V3'] = minimum[0] + ref_header['PA_V3'] 

    else:
        print('angular offset too small (<0.01 deg); skiping the rotation step')
        ref_rotated = weight_map_ref_shifted
        check_residual = (nu_0 * ref_rotated.reshape(ny,nx)  - sci_img) * mask_img
        aligned_ref_cube_rotated = new_aligned_ref_cube
        ref_rotated_mask =  ref_cube_mask_shifted
        ref_rotated_mask[ref_rotated_mask<0.1] = np.nan
        ref_rotated_mask[~np.isnan(ref_rotated_mask)] = 1
        ref_rotated_mask[:,-1,:] = np.nan # betapic IFU_align cube
        ref_header['PA_V3'] = minimum[0] + ref_header['PA_V3'] 


    fits.writeto(savepath_intermediate_products + '/check_residual_aligned.fits',  check_residual,  overwrite=True) 
    fits.writeto(savepath_intermediate_products + '/sci_mask_aligned.fits',  sci_cube_mask_shifted, sci_header, overwrite=True )
    fits.writeto(savepath_intermediate_products + '/ref_mask_shifted.fits',  ref_cube_mask_shifted, ref_header, overwrite=True )
    fits.writeto(savepath_intermediate_products + '/ref_mask_aligned.fits',  ref_rotated_mask, ref_header, overwrite=True )


    # FINAL output sci and ref cube after centering
    fits.writeto(savepath + '/sci_centered.fits',  new_aligned_sci_cube*sci_cube_mask_shifted, sci_header, overwrite=True )
    fits.writeto(savepath + '/ref_centered.fits',  aligned_ref_cube_rotated*ref_rotated_mask, ref_header, overwrite=True )

    # ################### check residual #############
    offset_x_sci, offset_y_sci, offset_x_ref, offset_y_ref = find_center(new_aligned_sci_cube[0:channel_longest], aligned_ref_cube_rotated[0:channel_longest], sci_header, ref_header, filter_size, x_center, y_center, savepath)
    output_information('after rotation (checking residual)', nx, ny, offset_x_sci, offset_y_sci, offset_x_ref, offset_y_ref)
   
    return new_aligned_sci_cube*sci_cube_mask_shifted, aligned_ref_cube_rotated*ref_rotated_mask


def run_centering(dir_name, sci_file_name, ref_file_name, 
                  x_center, y_center, 
                  new_img_size_x, new_img_size_y, 
                  filter_size, channel_longest):
    
    path = Path(dir_name, 'cube_images')
    savepath = Path(dir_name, 'centering')
    print('saving products to: ', savepath)
    savepath.mkdir(parents=True, exist_ok=True)

    savepath_intermediate_products = Path(dir_name, 'centering/intermediate_products')
    savepath_intermediate_products.mkdir(parents=True, exist_ok=True)

    sci_file = Path(path, sci_file_name)
    ref_file = Path(path, ref_file_name)

    t1 = time.time()
    aligned_sci_cube, aligned_ref_cube_rotated = IFU_centering(str(sci_file), str(ref_file),  str(savepath), str(savepath_intermediate_products), x_center, y_center,new_img_size_x, new_img_size_y, filter_size, channel_longest )
    t2 = time.time()
    totalTime = t2-t1
    print('-- Total Processing time: ', round(totalTime,2), ' s')
    return aligned_sci_cube, aligned_ref_cube_rotated


