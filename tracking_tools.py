"""
Author: Esteban Wright
Created on 05/10/2023
- Updated on 07/09/2025
Collection of functions to help facilitate extracting data from dataframes produced by the python
tracking package Trackpy.
These functions include tools to extract data from tracked dataFrames, convert to physical units, shift data to system origin and/or start times. Also included are plotting tools to display the data.
"""

import pims as pims
from pims import pipeline
import trackpy as tp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.interpolate import griddata
import pandas as pd

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rc('axes', labelsize=14)


#-----------------------------------------------------------------------------------------------------
def make_pimsFrame(frames, crop_top=None, crop_bottom=None):
    '''
    Convert image data (type ndarray) to pims.Frame objects for tracking with trackpy
    
    Parameters
    ----------
    frames : ndarray
        Collection of video frames with type ndarray
    crop_top : integer 
        Value to crop images from the top of the frame
    crop_bottom : integer
        Value to crop images from the bottom of the frame

    Returns
    -------
    img_Frames : list
        List of pims.Frames objects corresponding to each image in data set frames
    '''
    img_Frames = [] # to store converted pims Frame objects
    for jj,img in enumerate(frames):
        single_Frame = pims.Frame(img[crop_top:crop_bottom]) 
        single_Frame.frame_no = jj 
        img_Frames.append(single_Frame) 
    
    return img_Frames



#-----------------------------------------------------------------------------------------------------
def make_median(frames,f0,diff):
    """
    Takes in a series of images and computes the median of the series.
    Note: Frames are converted to ndarrays at two places in this function. 
          May not be nescessary but I found removing the conversion did not change results.
          However, I kept the conversion just in case it helps with future compatiblity and/or object types.
    
    Parameters:
    -----------
    frames : ndarray
        Series of images from which median image is computed.
    f0 : int
        Index of first frame to include in median image.
    diff : int
        Number of frames to skip between each included frame in the image stack used to compute median image. 
    
    Return:
    -------
    medianim : ndarray
        Computed median from given image series.
    """
    ## Prepare array used for storing every diff'th frame (i.e. imarr below) 
    ff = np.array(frames[0]) # single frame from image series
    nf = len(frames) # number of frames
    ns = int(nf/diff) + 1 # number of frame samples to include in image stack
    imarr = np.zeros([ns,ff.shape[0],ff.shape[1]]) # storage for stacking every diff'th frame;
                                                   # has shape of ns and frame data

    ## Make a stack of every diff'th image
    jj=0; # counter for indexing into imarr
    for ii in range(f0,nf,diff): # loop through every diff'th frame, starting at frame f0
        ff = np.array(frames[ii]) # convert to ndarray
        imarr[jj,:,:] = ff # assign current frame to imarr
        jj += 1 # update counter for indexing into imarr 
    
    ## Take the median of the stack 
    medianim = np.median(imarr,axis=0)  # make a median image from the stack
    
    return medianim



#-----------------------------------------------------------------------------------------------------
@pipeline # From pims package; allows for interating through a various pims object types 
          # (e.g. ImageSequence, slicerators) 
def subtract_median(frames,medianim):
    """
    Subtracts median image from all images in frame sequence. 
    
    Parameters:
    -----------
    frames : ndarray, pims ImageSequence, or other pims data type
        Collection of images to be processed.
    medianim : ndarray
        Median image of frames to be subtracted.
        
    Returns:
    --------
    median_subtracted_frames : ndarray, pims ImageSequence, or other pims data type
        Median subtracted image set.
    """
    median_subtracted_frames = frames - medianim  # subtracts the median image
    
    return median_subtracted_frames



#-----------------------------------------------------------------------------------------------------
def track_predict(frames, f0, size, minmass, maxdist, memory):
    '''
    Tracking function that uses trackpy's prediction module
    Function follows procedure found in trackpy's turtorial (http://soft-matter.github.io/trackpy
    /dev/tutorial/walkthrough.html) 
    and this blog on using trackpy's prediction module (http://soft-matter.github.io/trackpy/v0.6.1
    /tutorial/prediction.html) 

    Parameters
    ----------
    frames : PIMS Frame object; or PIMS Slicerator object
        Collection of video frames with features to be tracked
    f0 : int
        Frame number when tracking starts
    size : odd integer 
        Spatial extent of features (diameter in trackpy documentation)
    minmass : integer
        Minimum brightness of feature
    maxdist : integer
        Max distance between frames for linking (search_range in trackpy documentation) 
    memory : integer
        The maximum number of frames during which a feature can vanish, then reppear nearby, and be
        considered the same particle

    Returns
    -------
    tr_pred : DataFrame
        DataFrame with tracked trajectories
    '''
    # supress printing from tp.batch(), which is slow (documentation says this too) 
    tp.quiet() 
    
    # batch frames with identified features
    batched_frames = tp.batch(frames[f0:], size, minmass=minmass, invert=False) 
    
    # link features w/ tp.predict
    pred = tp.predict.NearestVelocityPredict() # prediction method for trackpy tracking 
                                               # (can be changed depending on uses)
    traj_pred = pred.link_df(batched_frames, maxdist, memory=memory) # link features into trajectories
                                                                     # with prediction pred
    
    return traj_pred



#-----------------------------------------------------------------------------------------------------
def extract_data(df):
    '''
    Function to extract x,y-components, positions, and corresponding times from dataFrame of
    trajectories found by trackpy.
    
    Parameters 
    ----------
    df : pandas dataFrame
        DataFrame with tracked trajectories from trackpy

    Returns 
    -------
    x_comp : ndarray
        Horizontal component of trajectories
    y_comp : ndarray
        Vertical component of trajectories
    origin_dist : ndarray
        Euclidean distance from the system's origin (e.g. image data origin is top-left corner) 
    disp : ndarray
        Displacements from trajectory start position
    time_frames : ndarray
        Corresponding frame number (i.e. time) of trajectory
    part_ids : ndarray
        Array of unique particle ids in trajectory dataFrame
    '''
    # get unique list of particle ids (i.e. unique entries in 'particle' column in df) 
    part_ser = df.particle
    part_ids = part_ser.unique()
    
    npart = len(part_ids)
    traj_lengths = [len(df[part_ser == pid]) for pid in part_ids]
    max_traj_len = np.max(traj_lengths)
    
    # initialize arrays
    x_comp = np.zeros([npart, max_traj_len])*np.nan
    y_comp = np.zeros([npart, max_traj_len])*np.nan
    time_frames = np.zeros([npart, max_traj_len])*np.nan
    
    # extract spatial components (x,y) and times (in frames) 
    for kk, pid in enumerate(part_ids):
        # print(kk, pid)
        x_comp[kk, :traj_lengths[kk]] = df[part_ser == pid].x.to_numpy()
        y_comp[kk, :traj_lengths[kk]] = df[part_ser == pid].y.to_numpy()
        time_frames[kk, :traj_lengths[kk]] = df[part_ser == pid].frame.to_numpy()
    
    # calculate Euclidean distance from image coordinate origin (top-left corner of image)
    # (this will need to be recalculated once coordinates are shifted) 
    origin_dist = np.sqrt(x_comp**2 + y_comp**2) 
    
    # displacements from initial position
    disp = origin_dist - origin_dist[:,[0]] 

    return x_comp, y_comp, origin_dist, disp, time_frames, part_ids



#-----------------------------------------------------------------------------------------------------
def shift_origin(time, xarr, yarr, t0, x0, y0):
    '''
    Shift position and times of trajectories to new origin.
    All inputs should be in frames and pixels.
    
    Parameters: 
    -----------
    time : ndarray
        Temporal data for all trajectories
    xarr : ndarray
        Horizontal position component
    yarr : ndarray
        Vertical position component
    t0 : int
        Time (in frames) to be set as start time (i.e t=0) 
    x0 : int
        Horizontal location of new origin (in pixels) 
    y0 : int
        Vertical location of new origin (in pixels) 
    
    Returns: 
    -----------
    time_shifted : ndarray 
        Trajectory timing shifted to start time t0
    x_shifted : ndarray 
        Horizontal position component shifted to new origin 
    y_shifted : ndarray 
        Vertical position component shifted to new origin 
    radial_shifted : ndarray 
        Euclidean distance mesured relative to new origin  
    '''
    
    ## Shift time, x, y-position data, and get euclidean distance relative to new origin 
    time_shifted = time - t0
    x_shifted = xarr - x0 
    y_shifted =  yarr - y0
    radial_shifted = np.sqrt(x_shifted**2 + y_shifted**2)
    
    return time_shifted, x_shifted, y_shifted, radial_shifted



#-----------------------------------------------------------------------------------------------------
def convert_units(time_frames, x_pixels, y_pixels, framerate, pxscale):
    '''
    Convert trajectory data from frames and pixels to real/physical units (e.g. seconds, cm) 
    
    Parameters: 
    -----------
    time_frames : ndarray
        Temporal data for all trajectories (in frames) 
    x_pixels : ndarray
        Horizontal position component for all trajectories (in pixels) 
    y_pixels : ndarray
        Vertical position component for all trajectories (in pixels) 
    framerate : float
        Frame rate of video being processed. Has units of [frame/temporal_unit] 
    pxscale : float
        Length scale in video frame. Has units of [pixel/spatial_unit] 
    
    Returns: 
    -----------
    time_units : ndarray
        Trajectory timing in real units (i.e. your temporal_unit) 
    x_units, y_units : ndarray
        Spatial components in real units (i.e. your spatial_unit)
    origin_dist_units : ndarray
        Euclidean distance in real units (i.e. your spatial_unit) 
    disp_units : ndarray
        Final displacement from initial position in real units (i.e. your spatial_unit) 
    '''
    ## Convert time, x, y-position data for all trajectories to real/physical units 
    time_units = time_frames/framerate 
    x_units = x_pixels/pxscale 
    y_units = -y_pixels/pxscale # negative sign is for positive to be up 
                                # (as opposed to down for image data)
    origin_dist_units = np.sqrt( x_units**2 + y_units**2 ) 
    disp_units = origin_dist_units - origin_dist_units[:,[0]] 
    
    return time_units, x_units, y_units, origin_dist_units, disp_units



#-----------------------------------------------------------------------------------------------------
def display_traj_plot(x, y, part_id, hlabel='n/a', vlabel='n/a', coloring=None,\
                 legend_flag=False, save=False, outname=None):
    '''
    Plot components of trajectories found using trackpy. 
    Shape of input lists should be Nx1 where N is the number of trajectories tracked. 
    Each list has nested lists of variable size, one for each trajectory/particle  
    
    Parameters: 
    -----------
    x,y : list
        Positional components for all trajectories
    part_id : list
        List of unique particle ids assigned by trackpy during tracking
    hlabel : string
        String input for the label on the horizontal axis 
    vlabel : string
        String input for the label on the vertical axis 
    coloring : int
        Pick the way to color the trajectories (more can be added) 
        Default is None which is trackpy's default of giving each particle trajectory a seperate color
        Options - coloring=1 is color particle trajectories each with a seperate color 
                             given by a colormap (default: viridis)
    legend_flag : boolean
        Flag to decide if legend should be plotted; used to reduce clutter on the fiugure is 
        dealing with a large number of particles. Default is False.
    
    Returns: 
    -------
    None
    '''
    nn = len(x)

    ## Color particle trajectories by ...
    # trackpy defalut coloring of particles (random); coloring=None
    if coloring==None:
        colors = [None for ii in range(nn)] # list of None; needs something to be passed below
    # coloring of particle trajectories with colormap; coloring=1
    elif coloring==1:
        colors = plt.cm.viridis(np.linspace(0,1,nn)) 
    # when user inputs option that is not available
    else:
        raise ValueError('Coloring does not accept "{!r}" as a valid option. Currently, only None (default) or 1 are acceptable values. Please see documentation for currently acceptable values'.format(coloring)) 

    ## Make plots based on arugment of plot keyword
    fig,ax = plt.subplots(1,1,figsize=(9,4),dpi=200)
    for jj in range(nn):
        ax.plot(x[jj], y[jj], label=part_id[jj], lw=0.5, color=colors[jj]) 
        ax.set_xlabel('%s' %hlabel) # this is for units to be input
        ax.set_ylabel('%s' %vlabel) 

    ## Plot legend off to side of figure
    if legend_flag: # if True
        plt.legend(bbox_to_anchor = (1.1, 1.2) , ncol=6) 
        
    if save: # if True
        plt.savefig(outname,dpi=200,bbox_inches='tight',)



#-----------------------------------------------------------------------------------------------------
def display_traj_scat(x, y, part_id=None, hlabel='n/a', vlabel='n/a', clabel='n/a', time_arr=None, \
                      coloring=None, legend_flag=False, dark_mode=False, shrink_val=1.0, pad_val=0.15, axes=None):
    '''
    Plot components of trajectories found using trackpy. 
    Shape of input lists should be Nx1 where N is the number of trajectories tracked. 
    Each list has nested lists of variable size, one for each trajectory/particle  
    
    Parameters: 
    -----------
    x : ndarray
        Data to be plotted on horizontal axis
    y : ndarray
        Data to be plotted on vertical axis
    part_id : ndarray
        List of unique particle ids assigned by trackpy during tracking; 
        used for plot legend if displayed
    hlabel : string
        String input for the label on the horizontal axis 
    vlabel : string
        String input for the label on the vertical axis 
    clabel : string
        String input for the label on the colorbar, when used 
    time_arr : ndarray
        Temporal data for all trajectories, used to color data with time (see coloring) 
        Default is None
    coloring : int
        Pick the way to color the trajectories (more can be added) 
        Default is None, which uses trackpy's default of giving each color a seperate color
        Options - coloring=1 is color particle trajectories by their times
    legend_flag : boolean
        Flag to decide if legend should be plotted; 
        used to reduce clutter on the fiugure is dealing with a large number of particles. 
        Default is False.
    dark_mode : boolean
        Set figure to be black with white figure elements 
        Default is False.
    shrink_val : float
        Value passed to shrink keyword in colorbar()
    pad_val : float
        Padding for colorbar; default 0.15  
    axes : matplotlib axes object
        Axes passed to plot trajectories on, in case figure customization is needed outside function 
        
    Returns: 
    -------
    None
    '''
    nn = len(x)

    ## Color particle trajectories by ...
    # trackpy defalut coloring of particles (random); coloring=None
    if coloring==None:
        colors = [None for ii in range(nn)] # list of None; needs something to be passed below
        vminn = None; vmaxx = None # max of mappable color space
        cbar_flag = False # flag for not making colorbar
        
    # coloring of particle trajectories by time_arr; coloring=1
    elif coloring==1:
        # cmap = mpl.cm.viridis
        vminn = np.nanmin(time_arr) # min value of all time lists
        vmaxx = np.nanmax(time_arr) # max value of all time lists
        colors = time_arr # set colors list to be times list
        cbar_flag = True # flag for making colorbar
        
    # when user inputs option that is not available
    else:
        raise ValueError('Coloring does not accept "{!r}" as a valid option. Currently, only None (default) or 1 are acceptable values. Please see documentation for more details'.format(coloring)) 

    ## Make a plot from passed data 
    if axes is not None:
        ax = axes
    else:
        fig,ax = plt.subplots(1,1,figsize=(9,4),dpi=200)

    for jj in range(nn):
        sc = ax.scatter(x[jj], y[jj], label=part_id[jj], s=0.1, c=colors[jj], vmin=vminn, vmax=vmaxx)  
        
    ax.set_xlabel('%s' %hlabel) # this is for units to be input
    ax.set_ylabel('%s' %vlabel) 

    ## Plot colorbar, when used 
    if cbar_flag: # if True
        cbar = plt.colorbar(sc, shrink=shrink_val, pad=pad_val) 
        cbar.set_label('%s' %clabel)

        if dark_mode:
            cbar.set_label('%s' %clabel, color='w') 
            cbar.ax.yaxis.set_tick_params(colors='w')
            # fig = plt.gcf()
            fig.set_facecolor('k')
            ax.set_facecolor('k')
            ax.tick_params(colors='w', which='both') 
            ax.set_xlabel('%s' %hlabel, color='w')
            ax.set_ylabel('%s' %vlabel, color='w')
        
        
    ## Plot legend off to side of figure, when used 
    if legend_flag: # if True
        plt.legend(bbox_to_anchor = (1.1, 1.2) , ncol=6) 



#-----------------------------------------------------------------------------------------------------
def do_LoG(frame, guass_xkernel_size, guass_ykernel_size, laplace_kernerl_size):
    '''
    Apply a Laplacian of Guassian (LoG) filter from opencv library 
    
    Parameters: 
    -----------
    frame : ndarray
        Array of video frames (i.e. an array of image arrays) 
    guass_xkernel_size : int
        Horizontal kernel size for GaussianBlur (from opencv library) 
    guass_ykernel_size : int
        Vertical kernel size for GaussianBlur (from opencv library) 
    laplace_kernerl_size : int
        Kernel size for apply a Laplacian to the blurred frame 
    
    Returns:
    --------
    LoG_image : ndarray
        Frames with a Laplacian of Gaussian filter applied to them 
    '''
    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(frame, (guass_xkernel_size, guass_ykernel_size), 0)
    # Apply Laplace function
    ddepth = -1 # -1 is same depth as source
    LoG_image = cv2.Laplacian(src, ddepth, ksize=laplace_kernerl_size) 
    return -LoG_image



#-----------------------------------------------------------------------------------------------------
@pipeline
def extract_single_color(frames,channel):
    """
    Extracts the frames from a single color channel of the ImageSequence
    
    Parameters:
    -----------
    frames : PIMS ImageSequence
        Sequence of frames from input video to extract a color channel.
    channel : int
        Index of the color channel user wants to extract.
    
    Returns:
    --------
    mono_frame : PIMS ImageSequence
        Single channel frames of chosen channel.
    """
    mono_frame = frames[:,:,channel]

    return mono_frame



#-----------------------------------------------------------------------------------------------------
@pipeline
def crop_frames(frame, tcrop=None, bcrop=None, lcrop=None, rcrop=None):
    '''
    Crop the frames in a PIMS ImageSequence or slicerator object.

    Parameters:
    -----------
    frames : PIMS ImageSequence or slicerator
        Frames to be cropped
    tcrop, bcrop, lcrop, rcrop : int
        Crop frame (top, bottom, left, right) by value; default None

    Returns:
    --------
    crop_frame : PIMS slicerator 
        Cropped frames
    
    '''
    crop_frame = frame[tcrop:bcrop, lcrop:rcrop, :]
    return crop_frame



#-----------------------------------------------------------------------------------------------------
def get_pixel_scale(frame, coordinate1, coordinate2, extent, length_scale, orientation, units): 
    """
    Measures the distance, in pixels, of some chosen length scale in the image (e.g. ruler) 
    and returns a conversion factor from pixels to physical units.
    Note: Pixel scaling is returned as 'px/units', so to convert take values in pixels and 
          divide them by the return conversion factor
    
    Parameters:
    -----------
    frame : ndarray OR pims Frame object
        One frame from the image series.
    coordinate1 : int
        Coordinate to draw first measurment line.
    coordinate2 : int
        Coordinate to draw second measurment line.
    extent : int
        Full extent of image so that measurement lines are drawn across entire image.
    length_scale : float
        Some length scale (e.g. ruler) in the image to convert pixels to physical units
    orientation : string
        Orientation of chosen length_scale in image. 
        Acceptable values are "horizontal", "h", "vertical", or "v".
    units : string
        The units chosen length_scale is measured. 
        This is used to report the pixel scaling as "px/units".
    
    Returns:
    --------
    px_unit_conversion : float
        The conversion factor to go from pixels to physical units.
    """
    ## Creat figure object for plotting 
    fig,ax = plt.subplots(1,1,dpi=100) 
    
    ## Pixel scale for above viewpoint 
    if orientation == 'horizontal' or orientation == 'h': 
        ax.plot([0,extent],[coordinate1,coordinate1],'r',lw=.75) # plot a line
        ax.plot([0,extent],[coordinate2,coordinate2],'r',lw=.75) # plot a 2nd line
        ax.imshow(frame) # 

        px_unit_conversion = (coordinate2-coordinate1)/length_scale
        print('pixel scaling: %.2f px/%s' %(px_unit_conversion,units))
    
    ## Pixel scale for side viewpoint 
    elif orientation == 'vertical' or orientation == 'v':
        ax.plot([coordinate1,coordinate1],[0,extent],'r',lw=.75) # plot a vertical line
        ax.plot([coordinate2,coordinate2],[0,extent],'r',lw=.75) 
        ax.imshow(frame) # 

        px_unit_conversion = (coordinate2-coordinate1)/length_scale
        print('pixel scaling: %.2f px/%s' %(px_unit_conversion,units))
    
    else:
        plt.close() # close created figure object that was created but not used in this instance 
        raise ValueError('Orientation does not accept {!r} as a valid option. Only "horizontal", "h", "vertical", and "v" are acceptable values.'.format(orientation)) 
    
    return px_unit_conversion



#-----------------------------------------------------------------------------------------------------
def smooth_traj(time, x, y, medfilt_window):#, fac):
    """
    Resamples the trajectories using scipy.interpolate griddata, then smooths trajectories with
    scipy.ndimage median_filter.
    NOTE: As of 9/25/23 the resampling of the input arrays (time,x,y) has been disabled.
    
    Parameters:
    -----------
    time : list
        Arrays of times corresponding to trajectories
    x : list
        Arrays of horizontal position of the trajectories
    y : list
        Arrays of vertical position of the trajectories
    medfilt_window : int
        Window size for the median filter
    
    Returns:
    --------
    tsmo : list
        Evenly gridded times for each tracjectory
    xsmo : list
        Smoothed horizontal component of trajectories
    ysmo : list
        Smoothed vertical component of trajectories
    """
    
    tsmo = [] # evenly gridded times
    xsmo = [] # smoothed horizonal positions
    ysmo = [] # smoothed vertical positions

    for ii in range(len(time)):
        n = len(time[ii]) # number of data points in trajectory
        nsamp = int(n/1.2) 
        tarr_even  = np.linspace(np.min(time[ii]), np.max(time[ii]), nsamp) # new evenly spaced time vector
        
        # resample tracks; griddata did not like pandas series, use numpy arrays
        xarr_even = griddata(time[ii], x[ii], tarr_even)
        yarr_even = griddata(time[ii], y[ii], tarr_even)

        # median filter x,y
        xarr_smo = median_filter(xarr_even, medfilt_window, mode='nearest')    
        yarr_smo = median_filter(yarr_even, medfilt_window, mode='nearest')

        # append arrays
        tsmo.append(tarr_even)
        xsmo.append(xarr_smo)
        ysmo.append(yarr_smo)

    return tsmo, xsmo, ysmo#, nsamp, n



#-----------------------------------------------------------------------------------------------------
def get_vel_acc(time,xarr,yarr,vlen,alen):
    """
    Takes the derivative of trajectory positions and returns the velocities and accelerations 
    using scipy.signal savgol_filter().
    
    Parameters:
    -----------
    time : np array
        Arrays of times corresponding to trajectories
    xarr : np array
        Arrays of horizontal position of the trajectories
    yarr : np array
        Arrays of vertical position of the trajectories
    vlen : int
        Window size in the savgol_filter for velocities
    alen : int
        Window size in the savgol_filter for accelerations
    
    Returns:
    --------
    vx : np array
        Horizontal velocity component of trajectories
    vy : np array
        Vertical velocity component of trajectories
    ax : np array
        Horizontal acceleration component of trajectories
    ay : np array
        Vertical acceleration component of trajectories
    """
    
    ## time steps b/w time arrays for each trajectory
    dt = np.nanmedian(np.gradient(time, axis=1)) # should be the same as 1/framerate

    ## Compute vx, vy
    vx = savgol_filter(xarr, vlen, polyorder=2, delta=dt, mode='nearest', deriv=1) 
    vy = savgol_filter(yarr, vlen, polyorder=2, delta=dt, mode='nearest', deriv=1)  
        
    ## Compute ax,ay
    ax = savgol_filter(xarr, alen, polyorder=2, delta=dt, mode='nearest',deriv=2)
    ay = savgol_filter(yarr, alen, polyorder=2, delta=dt, mode='nearest', deriv=2)
        
    return vx, vy, ax, ay 



#-----------------------------------------------------------------------------------------------------
def sum_sequence(frames, di):
    """
    Sums a sequence of images to produce an image that mimics a long exposure image.
    
    Parameters:
    -----------
    frames : pims ImageSequence or Pipeline
        Sequence of frames to be summed
    di : int
        Spacing between frames that are used in the sum
    
    Returns:
    --------
    sumim : array 
        Image of the summed frames spaced by di
    """
    sumim = frames[0]*0 # initialize array of zeros
    ni = len(range(0, len(frames), di)) # number of total frames used in summing
    for i in range(0, len(frames), di):
        fac = 1.0/ni # opacity factor
        sumim = sumim + np.array(frames[i])*fac # add current frame to running summed image
    return np.array(sumim)



#-----------------------------------------------------------------------------------------------------
def increase_brightness(img, cutoff):
    """
    Brightens the input image img.
    Function from: https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv

    Parameters:
    -----------
    img : ndarray
        Image that will be brightened.
    cutoff : int
        Cutoff value for pixels to be brightened.
        Deafault 30
    
    Returns:
    --------
    img_bright : array 
        Brightened image from the original input image img
    """
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Converts img from BGR to HSV
    h, s, v = cv2.split(hsv) # Seperates the color channels; here hue, saturation, value (HSV) 

    limit = 255 - cutoff # Every pixel below the value of limit gets brightened
    v[v > limit] = 255 # Set pixel values above limit to white  
    v[v <= limit] += cutoff # Sets pixel value below limit to orginal value plus cutoff

    final_hsv = cv2.merge((h, s, v)) # Merge the three channels back into a single image
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR) # Convert HSV to BGR
    return img_bright



#-----------------------------------------------------------------------------------------------------
def make_dataframe(dict_keys, *args):
    '''
    Makes a pandas dataframe from the supplied args for one trajectory

    Parameters:
    -----------
    dict_keys : dict
        Dictionary keys to used as columns in a dataframe
    args : lists
        Lists of arrays to be saved in a dataframe
    
    Returns:
    --------
    df1 : DataFrame
        DataFrame filled with values in args
    '''
    dicc = {key: value for key, value in zip(dict_keys, args)}
    df1 = pd.DataFrame(data=dicc) 

    return df1



#-----------------------------------------------------------------------------------------------------
def make_timeseries_df(dict_keys, *args):
    '''
    Makes a dataframe for all trajectory time series data and measurements.
    Function is written to keep particle idxs and original trajcetory dataframe
    indices consistent.

    Parameters:
    -----------
    dict_keys : dict
        Dictionary keys to used as columns in a dataframe
    args : lists
        Lists of values to store dataframe
    
    Returns:
    --------
    saveddf : DataFrame
        DataFrame with all trajectory measured quantities
    '''
    saveddf = pd.DataFrame([])

    for ii in range(len(args[0])):

        current_arrs = [arg[ii] for arg in args]   

        df1 = make_dataframe(dict_keys, *current_arrs)
        saveddf = pd.concat([saveddf, df1]) 

    saveddf = saveddf.sort_values(by=[dict_keys[0]]) 
    saveddf = saveddf.set_index(dict_keys[0])

    return saveddf



#-----------------------------------------------------------------------------------------------------
def make_static_df(dict_keys, *args):
    '''
    Makes a dataframe for all quatinties related to the entire trajectory. 

    Parameters:
    -----------
    dict_keys : dict
        Dictionary keys to used as columns in a dataframe
    args : lists
        Lists of values to store dataframe
    
    Returns:
    --------
    saveddf : DataFrame
        DataFrame with all trajectory measured quantities
    '''
    saveddf = pd.DataFrame([])

    for ii in range(len(args[0])):

        current_arrs = [arg[ii] for arg in args]  
        dicc = {key: value for key, value in zip(dict_keys, current_arrs)}
        df1 = pd.DataFrame(data=dicc, index=[ii]) 
        
        saveddf = pd.concat([saveddf, df1]) 

    return saveddf



#-----------------------------------------------------------------------------------------------------
def get_displacements(xarr,yarr):
    '''
    Finds the displacements (i.e. changes is position) between each point of a trajectory.
    
    Parameters:
    -----------
    xarr : list
        List of arrays of trajectory x-components 
    yarr : list
        List of arrays of trajectory y-components 

    Returns:
    --------
    dxlst : list
        List of arrays of values for the change in x-component (dx) of trajectory 
    dylst : list
        List of arrays of values for the change in y-component (dy) of trajectory 
    drlst : list
        List of arrays of magnitudes of change in position vector 
    '''
    
    ## Initialize lists to store data arrays
    dxlst=[] # x-component of displacements 
    dylst=[] # y-component of displacements 
    drlst=[] # magnitude of displacements 

    ## Loop through trajectories
    for xx,yy in zip(xarr,yarr):
        dx_ = np.array([]); dy_ = np.array([]); dr_ = np.array([]) # intialize arrays for trajectory   

        ## Loop through points in trajectory 
        for ii in range(0,len(xx)-1): 
            dx = xx[ii+1] - xx[ii] # change in x-components 
            dy = yy[ii+1] - yy[ii] # change in y-components 
            dr = np.sqrt(dx**2+dy**2) # magnitude of displacments 

            dx_ = np.append(dx_,dx); dy_ = np.append(dy_,dy); dr_ = np.append(dr_,dr)
            # note these arrays have length one less than input trajectories xx/yy 
        
        ## Arrays must be length equal to len(xx) to save with other quantities; 
        ## (using 0.0 to prevent plotting of this final value) 
        dx_ = np.append(dx_, 0.0 ); dy_ = np.append(dy_, 0.0 ); dr_ = np.append(dr_, 0.0 )
        
        ## Append lists with data arrays
        dxlst.append(dx_); dylst.append(dy_); drlst.append(dr_)
    
    return dxlst, dylst, drlst



#-----------------------------------------------------------------------------------------------------
def get_final_displacements(xarr,yarr):
    '''
    Get the displacement from particle's intial position to final position.
    This function finds the changes in x/y-components and the magnitude of the displacement.
    
    Parameters:
    -----------
    xarr : list
        List of arrays of trajectory x-components 
    yarr : list
        List of arrays of trajectory y-components 
    
    Returns:
    --------
    dXlst : list
        List of values for the total change in x-component (dX) of trajectory 
    dYlst : list
        List of values for the total change in y-component (dY) of trajectory 
    dRlst : list
        List of magnitudes of change in position vector 
    '''
    
    dXlst = []; dYlst = []; dRlst = []; 
    
    for xx, yy in zip(xarr,yarr):
        dX = xx[-1]-xx[0] # change in x 
        dY = yy[-1]-yy[0] # change in y 
        dR = np.sqrt(dX**2+dY**2) # magnitude of displacement 
        
        dXlst.append(dX); dYlst.append(dY); dRlst.append(dR)
    
    return dXlst, dYlst, dRlst



#-----------------------------------------------------------------------------------------------------
def filter_trajectory_displacements(df_timeser, df_static, dr_min, dR_thresh, stublength, dr_max=1000.0):
    '''
    Filters trajectories by displacements in time and total displacement.
    
    Notes: The filtering removes parts of a trajectory that have small dr displacments. This can leave short trajectories (need to filter these out)
           
           dr (and probably dR) calculations need to be recomputed after filtering...but that will assign 0.0 to last trajectory point???
           dr was calculated as particle displacement from position early in time to later in time, and value was assigned to earlier position dataframe index.
           So, when filtering small displacements the later in time data point may be filtered out if this point has a dr<dr_thresh with the next data point in time.
    
    Parameters:
    -----------
    df_timeser : DataFrame
        DataFrame of time series data for trajectories, must contain dr values
    df_static : 
        DataFrame with final displacement dR information for each trajectory
    dr_min : float
        Minimum distance (in pixels) a trajectory is allowed to move between time steps 
    dR_thresh : float
        Minimum distance (in pixels) a trajectory is allowed to move for all time
    stublength : int
        Tracks shorter than stublength are removed
    dr_max : float
        Maximum distance (in pixels) a trajectory is allowed to move between time steps; default 1000.0
    
    Returns:
    --------
    df_traj_filt : DataFrame
        Dataframe of filtered time series trajectory information based on conditions
    df_dRfilt : DataFrame
        Dataframe containing statice trajectory information filtered by some final displacement dR_thresh
        (does this need to be returned? It's only a temporarly used object right?)
    '''
    print('Original DataFrame length: ', len(df_timeser))
    print('Original # of trajectories: ', len(np.unique(df_timeser.particle)))
    # print('\n')
    
    ## Filter trajectories by dr > dr_min
    cond1 = df_timeser.dr > dr_min
    cond2 = df_timeser.dr == 0.0 # need this condition otherwise filtering will remove last data point in any given trajectory 
    df_drfilt = df_timeser[cond1 | cond2] 
    print('Small dr filter DataFrame length: ', len(df_drfilt))
    print('Trajectories after dr filtering: ', len(np.unique(df_drfilt.particle)))

    ## Filter trajectories by dr < dr_max
    df_drfilt = df_drfilt[df_drfilt.dr < dr_max]
    print('Large dr filter DataFrame length: ', len(df_drfilt))
    print('Trajectories after dr filtering: ', len(np.unique(df_drfilt.particle)))

    ## Filter trajectories by dR > dR_thresh 
    df_dRfilt = df_static[df_static.dr_final>=dR_thresh] 
    static_pid = df_dRfilt.particle # particle ids that surviveed filter 
    
    df_traj_filt = pd.DataFrame([]) # dataframe that will store all trajectories that passed filter
    for pid in static_pid: # loop through particle ids
        df = df_drfilt[df_drfilt.particle == pid] # dataframe with a particle that equals pid
        df_traj_filt = pd.concat([df_traj_filt,df],axis=0) # append particle dataframe to trajectory storage dataframe
    
    df_traj_filt = df_traj_filt.sort_index() # sort dataframe by index for consistency 
    print('dR filter DataFrame length: ', len(df_traj_filt))
    print('Trajectories after dR filtering: ', len(np.unique(df_traj_filt.particle)))
    
    
    ## Remove short tracks using trackpy filter_stubs (from trackpy.filtering)
    df_traj_filt = tp.filter_stubs(df_traj_filt, stublength) # filter short tracks 
    df_traj_filt = df_traj_filt.reset_index(drop=True) # reindex traj_filt; 
                                                # tp.filter_stubs() uses 'frame' column as DataFrame index   
    print('Stub filter DataFrame length: ', len(df_traj_filt))
    print('Trajectories after stub filtering: ', len(np.unique(df_traj_filt.particle))) 

    return df_traj_filt, df_dRfilt



#------------------------------------------------
def filter_trajectories(df_traj, dr_min, dR_thresh, stublength, dr_max=1000.0):
    '''
    Calculates time series displacements and final displacements of the original tracked trajectories.
    Then filters trajectories by displacements in time and total displacement using filter_trajectory_displacements() above.
    Written this way to avoid unnescessarily having to do several calculations on trajectories later (e.g. velocities, accelerations, region identifying, &c.)

    Parameters:
    -----------
    df_traj : DataFrame
        DataFrame of time series data for trajectories, must contain dr values
    dr_min : float
        Minimum distance (in pixels) a trajectory is allowed to move between time steps 
    dR_thresh : float
        Minimum distance (in pixels) a trajectory is allowed to move for all time
    stublength : int
        Tracks shorter than stublength are removed
    dr_max : float
        Maximum distance (in pixels) a trajectory is allowed to move between time steps; default 1000.0

    Returns:
    --------
    df_filt : DataFrame
        Dataframe of filtered time series trajectory information based on conditions
    '''
    time, xarr, yarr, pid, dfidxs = extract_data(df_traj) 
    dx111, dy111, dr111 = get_displacements(xarr,yarr)
    test_dr = make_timeseries_df(['idx','dr'], dfidxs, dr111)
    df_timeser = pd.concat([df_traj, test_dr], axis=1)

    dXlst111, dYlst111, dRlst111 = get_final_displacements(xarr,yarr)
    df_dRfilt = make_static_df(['dr_final','particle'], dRlst111, pid)

    df_traj_filt, df_static_filt = filter_trajectory_displacements(df_timeser, df_dRfilt, dr_min, dR_thresh, stublength, dr_max)
    df_filt = df_traj_filt.iloc[:,:10] # get rid of last column containing dr

    return df_filt



#-----------------------------------------------------------------------------------------------------
def plot_hists_in_time(time, data, nbins, xaxis_label):
    '''
    Make histograms (stairs) of data colored by time.

    Parameters:
    -----------
    time : ndarray
        Times to color histograms
    data : ndarray
        Data to make histograms
    nbins : int
        Number of bins in histograms
    xaxis_label : str
        Label for x-axis 

    Returns:
    --------
    
    '''
    fig,ax=plt.subplots(1,1)
    
    cmap = mpl.cm.jet
    tmin=np.nanmin(time)
    tmax=np.nanmax(time)
    norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    ttt = np.linspace(tmin, tmax, np.shape(data)[0])
    print(len(ttt))

    data_max = np.nanmax(data) # same as for vmagsb
    data_min = np.nanmin(data) 
    print(data_max, data_min)

    for ss in range(np.shape(data)[0]):
        color = cmap(norm(ttt[ss]))
        data_ = data[ss][~np.isnan(data[ss])]
        
        vals, edges = np.histogram(data_, bins=nbins, range=(data_min, data_max)) # make histogram of combined dataset
        ax.stairs(vals, edges, color=color, alpha=0.5) # total distribution of large material

    ax.set_xlabel(xaxis_label)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # only needed for matplotlib < 3.1
    plt.colorbar(sm, ax=ax, label='time [s]')



#-----------------------------------------------------------------------------------------------------
def make_hists_time_3D(time, data, nbins, yaxis_label, color_map, output_path='', filename='', save=False):
    '''
    Make a 3D histogram plot of data v time, colored by distribution frequency.

    Parameters:
    -----------
    time : ndarray
        Times to color histograms
    data : ndarray
        Data to make histograms
    nbins : int
        Number of bins in histograms
    yaxis_label : str
        Label for y-axis 
    color_map : str
        Color map choice for plotting 
    output_path : str
        Directory to save figure if save is passed as True
    filename : str
        Named of saved file 
    save : boolean
        Flag to save figure at output_path; default False

    Returns:
    --------
    ax : matplotlib axes object
        Axes object of figure for furthur editing/customizing outside of this function 
    
    '''
    data_max = np.nanmax(data) # 
    data_min = np.nanmin(data) 
    print('data max: ', data_max)
    print('data min: ', data_min)

    tmin=np.nanmin(time)
    tmax=np.nanmax(time)
    norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    ttt = np.linspace(tmin, tmax, np.shape(data)[0])

    hist_img = np.zeros([nbins,len(ttt)]) # intialize image

    for ss in range(np.shape(data)[0]):

        data_ = data[ss][~np.isnan(data[ss])]
        vals, edges = np.histogram(data_, bins=nbins, range=(data_min, data_max)) # make histogram of combined dataset
        hist_img[:,ss] = vals 
        
    print('bin width: ', edges[1]-edges[0], ' [y-axis units]')
    
    ## Plot it!
    fig,ax = plt.subplots(1,1)

    cc = ax.imshow(hist_img, origin='lower', extent=[0., tmax, data_min, data_max], cmap=color_map)
    ax.set_aspect('auto')

    ax.set_ylabel(yaxis_label)
    ax.set_xlabel('time [s]')
    plt.colorbar(cc, label='counts')

    plt.savefig(output_path+filename+'.png', dpi=500, bbox_inches='tight')
    plt.savefig(output_path+filename+'.svg')#, dpi=500, bbox_inches='tight') 

    return ax



#-----------------------------------------------------------------------------------------------------
def reshape_arr_time(nr, nc, data_arr, time_arr): 
    '''
    Reformats array from (time, particle trajectory) to (particle trajectory, time) but with the times synced 
    (i.e. trajectory data at time t are all in the same column).

    Parameters:
    ------------
    nr,nc : int
        Shape of new array (row,column) 
    data_arr : ndarray
        Array with data to be reshaped
    time_arr : ndarray
        Array with times associated with values in data_arr (needs to have same shape as data_arr)

    Returns:
    --------
    reshaped_arr : ndarray
        Reshaped data array with time in rows and data aligned with correct time points; 
        has shape (nr,nc) and contains nans
    '''
    
    reshpaed_arr = np.empty(shape=(nr,nc))*np.nan 
    
    unique_times = np.unique(time_arr[~np.isnan(time_arr)])
    
    for tt in range(np.shape(reshpaed_arr)[0]):
        mask = time_arr==unique_times[tt]
        vals = data_arr[mask] 
    
        reshpaed_arr[tt,:len(vals)] = vals

    return reshpaed_arr



#-----------------------------------------------------------------------------------------------------
def plot_hist_w_edges(data, color):
    '''
    Make a histogram with bin edges drawn in black.

    Parameters:
    -----------
    data : ndarray
        Data to plot in histogram 
    color : str
        Color of histogram bars 
    '''
    fig, ax = plt.subplots(1,1)
    
    vals, edges = np.histogram(data, bins=20) # make histogram of combined dataset
    bin_width = edges[1]-edges[0] # equal bin width
    ax.stairs(vals, edges, fill=True, color=color, alpha=0.5) # total distribution of large material
    
    ## plot lines on bin edges
    ax.plot([edges[0], edges[0]], [0, vals[0]-0.005], 'k-', lw=0.5, alpha=0.5, zorder=10) # left-most edge
    for cc, bb in zip(vals, edges):
        ax.plot([bb, bb], [0, cc-0.005], 'k-', lw=0.5, alpha=0.5, zorder=10)
    ax.plot([bb+bin_width, bb+bin_width], [0, cc-0.005], 'k-', lw=0.5, alpha=0.5, zorder=10) # right-most dege

    return ax



#-----------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------




