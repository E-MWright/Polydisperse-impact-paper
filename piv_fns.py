"""
Author: Esteban Wright
Created on 07/16/2023
Collection of functions to help facilitate doing a particle image velocimetry (PIV) analysis of movie frames.
Uses the piv tools in the openpiv package
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from openpiv import tools, pyprocess, validation, filters, scaling
import json

#-----------------------------------------------------------------------------------------------------
def plot_multi_hist_ebins(data1, data2, labels, colors, zorders, alpha, nbins, save=False, savepath=None):
    '''
    Plot histograms of two datasets with equal bins. 

    Parameters:
    -----------
    data1, data2 : list or array-like
        Datasets to plot histograms 
    labels : list 
        List of figure legend labels to use (e.g. ['data1', 'data2'])
    colors : list 
        List of colors to use for each histogram (e.g. ['red', 'blue'])
    zorders : list 
        List of zorders to control which histogram is in figure foreground 
    alpha : list
        List of alpha values for histograms
    nbins : int
        Number of bins to use in histograms 
    save : bool
        Flag to save plot; default False
    savepath : str
        Directory to save velocity field frames; default is None

    Returns:
    --------
    ax : matplotlib axes object
        Axes object of the figure (for later use if user wishes) 
    '''
    fig,ax=plt.subplots(1,1,dpi=100)
    vals,edges = np.histogram(np.hstack((data1,data2)), bins=nbins) # make histogram of combined dataset
    bin_width = edges[1]-edges[0] # equal bin width
    # ax.stairs(vals,edges) # total distribution of large material
    
    na, abins, apatches = ax.hist(data1, edges, alpha=alpha[0], color=colors[0], zorder=zorders[0])
    nb, bbins, bpatches = ax.hist(data2, edges, alpha=alpha[1], color=colors[1], zorder=zorders[1])
    nvals = [a if a>b else b for a,b in zip(na,nb)] # 

    # plot lines on bin edges
    ax.plot([edges[0],edges[0]],[0,nvals[0]-0.005],'k-',lw=0.5, alpha=0.5, zorder=10) # left-most edge
    for cc,bb in zip(nvals,edges):
        ax.plot([bb,bb],[0,cc-0.005],'k-',lw=0.5, alpha=0.5, zorder=10)
    ax.plot([bb+bin_width,bb+bin_width],[0,cc-0.005],'k-',lw=0.5, alpha=0.5, zorder=10) # right-most dege

    # figure parameters
    ax.set_ylabel('counts', fontsize=14)
    ax.set_ylim(0,)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(handles=[apatches,bpatches], labels=labels, loc=0, fontsize=14) 

    # save figure (optional)
    if save: 
        plt.savefig(savepath+'traj_components.svg', dpi=200, bbox_inches='tight')
    
    # plt.show(block=False)

    return ax



#-----------------------------------------------------------------------------------------------------
def piv_one_step(frame1, frame2, winsize, overlap, dt, searchsize, new_origin, pxscale, qnoise, qvmag):
    '''
    Perform PIV analysis between frames.
    See https://openpiv.readthedocs.io/en/latest/src/tutorial1.html for turtorial and original documentation (and descriptions) for some inputs.

    Parameters:
    -----------
    frame1, frame2 : pims Frame object
        Frames to calculate velocity field between
    winsize : int
        Interrogation window size in frame A; in pixels
    overlap : int
        Overlap between frames, think of as percentage of winsize; in pixels
    dt : float
        Time step between frames 
    searchsize : int
        Search area size in frame B; in pixels 
    new_origin : list
        Coordinates of user defined orign with length 2 (i.e. [xc, yc]); x-coordinate is first element, y-coordinate is second element
    pxscale : float
        Pixel-to-real unit conversion factor
    qnoise : int
        Value of the q-th percentile to filter signal2noise ratio by 
    qvmag : int
        Value of the q-th percentile to filter velocity magnitudes by 
    
    Returns:
    --------
    xx3 : ndarray
        Horizontal position of velocity vectors, in units of pxscale 
    yy3 : ndarray
        Vertical position of velocity vectors, in units of pxscale 
    uu3 : ndarray
        Horizontal component of velocity vectors, in units of cm/sec 
    vv3 : ndarray
        Vertical component of velocity vectors, in units of cm/sec 
    vejecta_mag2 : ndarray
        Velocity magnitudes between two frames, in units of cm/sec 
    sig2noise_per_val : int 
        Value of the signal-to-noise threshold used to filter out unreliable velocities
    vmag_per_val_low :  int
        Value of the velocity magnitude threshold used to filter out small velocities 
    vmag_per_val_upp :  int
        Value of the velocity magnitude threshold used to filter out large velocities 
    '''
    
    ## Do PIV analysis 
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        frame1.astype(np.int32),
        frame2.astype(np.int32),
        window_size = winsize,
        overlap = overlap,
        dt = dt,
        search_area_size = searchsize,
        sig2noise_method = 'peak2peak',
    )
    
    ## Get coordinates of velocity vector positions
    xx0, yy0 = pyprocess.get_coordinates(
        image_size=frame1.shape,
        search_area_size=searchsize,
        overlap=overlap,
    )
    
    ## Choose percent quartile to filter signal-to-noise ratios 
    sig2noise_per_val = np.percentile(sig2noise, qnoise) # percentile filtering 
    # print('sig2noise threshold: ', sig2noise_per_val)
    sig2noise_bool = sig2noise>sig2noise_per_val
    sig2noise_percent = sig2noise[sig2noise_bool]
    
    ## Identify vectors below some signal-to-noise ratio 
    invalid_mask = validation.sig2noise_val(u0,v0,
        sig2noise,
        threshold = sig2noise_per_val, # 
    )
    
    ## Replace velocity vector components that do not meet masking threshold 
    u1, v1 = filters.replace_outliers(
        u0, v0,
        invalid_mask,
        method='localmean',
        max_iter=3,
        kernel_size=3 # could make variables to be passed...
    )
    
    ## Convert positions and velocities to physical units
    # convert x,y to cm
    # convert u,v to cm/sec
    xx1, yy1, u2, v2 = scaling.uniform(
        xx0, yy0, u1, v1,
        scaling_factor = pxscale
    )
    
    ## Shift origin of coordinate system to be bottom left corner
    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    xx2, yy2, u3, v3 = tools.transform_coordinates(xx1, yy1, u2, v2)

    ## Shift origin to be at user defined origin 
    xx2 = xx2 - new_origin[0]/pxscale
    yy2 = yy2 - new_origin[1]/pxscale
    
    ## Calculate velocity vectors 
    vejecta_mag = np.sqrt(u3**2 + v3**2)

    ## Filter velocities by magnitudes 
    vmag_per_val_low = np.percentile(vejecta_mag, 80) # lower bound
    vmag_per_val_upp = np.percentile(vejecta_mag, 99.7) # upper bound 
    # print('vmag (lower) percentile threshold: ', vmag_per_val_low)
    # print('vmag (upper) percentile threshold: ', vmag_per_val_upp)
    
    filt_cond = (vejecta_mag > vmag_per_val_low) & (vejecta_mag < vmag_per_val_upp)
    ind = np.where(filt_cond) # indices of surviving velocities 
    
    vejecta_mag2 = np.zeros_like(vejecta_mag) 
    vejecta_mag2[ind] = vejecta_mag[ind]
    
    xx3 = np.zeros_like(xx2)
    xx3[ind] = xx2[ind]
    
    yy3 = np.zeros_like(yy2)
    yy3[ind] = yy2[ind]
    
    uu3 = np.zeros_like(u3)
    uu3[ind] = u3[ind]
    
    vv3 = np.zeros_like(v3)
    vv3[ind] = v3[ind]

    return xx3, yy3, uu3, vv3, vejecta_mag2, sig2noise_per_val, vmag_per_val_low, vmag_per_val_upp



#-----------------------------------------------------------------------------------------------------
def piv_analysis(frames, t0_frame, new_origin, piv_params_dict):
    '''
    Perform PIV analysis on a movie.

    Parameters:
    -----------
    frames : pims ImageSequence
        Video frames to perform piv analysis
    t0_frame : int
        Frame index to use for time=0
    new_origin : list
        Coordinates of user defined orign with length 2 (i.e. [xc, yc]); x-coordinate is first element, y-coordinate is second element
    piv_params_dict : dict
        ...

    Returns:
    --------
    x_pos : list
        Horizontal postition of velocity vectors for entire video, in units of pxscale 
    y_pos : list
        Vertical postition of velocity vectors for entire video, in units of pxscale 
    u_vel : list
        Horizontal component of velocity vectors for entire video, in units of cm/sec 
    v_vel : list
        Vertical component of velocity vectors for entire video, in units of cm/sec 
    vel_mag : list
        Magnitudes of velocity vectors for entire video, in units of cm/sec 
    vmag_thresh_low : list
        Value of the velocity magnitude threshold in time used to filter out small velocities
    vmag_thresh_upp : list
        Value of the velocity magnitude threshold in time used to filter out large velocities
    sig2noise_thresh : list
        Value of the signal-to-noise threshold in time used to filter out unreliable velocities
    '''
    ## Constants
    NN = len(frames[:-1]) 

    ## Initialize lists
    x_pos = []
    y_pos = []
    u_vel = []
    v_vel = []
    vel_mag = []
    vmag_thresh_low = []
    vmag_thresh_upp = []
    sig2noise_thresh = [] 

    winsize = piv_params_dict['winsize']
    searchsize = piv_params_dict['searchsize']
    overlap = piv_params_dict['overlap']
    dt = piv_params_dict['dt']
    sig2noise_percentile = piv_params_dict['sig2noise_percentile']
    vmag_percentile = piv_params_dict['vmag_percentile']
    pxscale = piv_params_dict['pxscale']

    ## Start progress bar 
    pbar = tqdm(total = NN)
    
    for ii in range(NN):
        frame_curr = frames[ii] # current frame 
        frame_next = frames[ii+1] # next frame 
        xx_fn, yy_fn, uu_fn, vv_fn, vejecta_mag_fn, sigthresh, vmagthresh_low, vmagthresh_upp = piv_one_step(frame_curr, frame_next, winsize, overlap, dt, searchsize, new_origin, pxscale, sig2noise_percentile, vmag_percentile)
        vmag = np.sqrt(uu_fn**2 + vv_fn**2) # velocity magnitudes 
        
        x_pos.append(xx_fn) 
        y_pos.append(yy_fn) 
        u_vel.append(uu_fn) 
        v_vel.append(vv_fn) 
        vel_mag.append(vmag) 
        vmag_thresh_low.append(vmagthresh_low) 
        vmag_thresh_upp.append(vmagthresh_upp)
        sig2noise_thresh.append(sigthresh) 

        pbar.update(1) # update progress bar     

    return x_pos, y_pos, u_vel, v_vel, vel_mag, vmag_thresh_low, vmag_thresh_upp, sig2noise_thresh



#-----------------------------------------------------------------------------------------------------
def plot_vel_field(frame_curr, xx, yy, uu, vv, frame_idx, time_idx, framerate, new_origin, img_extent, save=False, savepath=None):
    '''
    Plots the velocity field found using PIV between two frames.
    
    Parameters:
    -----------
    frame_curr : pims Frame object
        Current frame that PIV is performed 
    xx : ndarray
        Horizontal position of velocity vectors 
    yy : ndarray
        Vertical position of velocity vectors 
    uu : ndarray
        Horizontal component of velocity vectors 
    vv : ndarray
        Vertical component of velocity vectors 
    frame_idx : int
        Index of frame number to use in save file; this way the file name matches the frame number in the original video 
    time_idx : int 
        Time index denoting time=0; used to mark the current time in the figures/frames 
    framerate : float 
        Frame rate of the original video, in 1/seconds
    new_origin : list
        Coordinates of user defined orign with length 2 (i.e. [xc, yc]); x-coordinate is first element, y-coordinate is second element
    img_extent : list
        Values to pass to imshow's extent option; video frame boundaries in real units 
    save : bool
        Flag to save plot; default False
    savepath : str
        Directory to save velocity field frames; default None

    Returns:
    --------
    None
    '''
    xc = new_origin[0]
    yc = new_origin[1]
    
    ## Plot final velocities overlaying original frame 
    fig,ax = plt.subplots(1,1,dpi=100) 
    ax.imshow(frame_curr, extent=img_extent, cmap='gray', vmin=0, vmax=255)
    # ax.quiver(xx, yy, uu, vv, color='y')#, scale=600, width=0.0025) 

    Q = ax.quiver(xx, yy, uu, vv, color='y', angles='xy', width=0.0015, scale=500) #
    arrow_size = 50
    qk = ax.quiverkey(Q, 15, -7.5, arrow_size, '%s cm/s' %arrow_size, labelcolor='y', color='y', coordinates='data', labelpos='N', labelsep=0.05, fontproperties={'size':8}) #

    ## Plot streamlines over velocity field 
    # XX, YY = np.meshgrid(np.linspace(0,nx/pxscale,90), np.linspace(0, ny/pxscale,50))
    # aa = ax.streamplot(XX, YY, uu_fn, -vv_fn, color='m', cmap='viridis', density=[.5, .5], linewidth=1.)
    
    ax.set_xlabel('x [cm]') 
    ax.set_ylabel('y [cm]') 
    # ax.set_title('Ejecta velocity field\nt = %.2f ms' %(time_idx/framerate*1000)) # title time in ms 
    ax.set_title('t = %.2f ms' %(time_idx/framerate*1000)) # title time in ms 

    if save:
        plt.savefig(savepath+'v_field_'+str(frame_idx)+'.png', bbox_inches='tight', dpi=200)
        
    plt.close() 



#-----------------------------------------------------------------------------------------------------
def make_vfield_frames(frames, xx_all, yy_all, uu_all, vv_all, framerate, t0_frame, new_origin, img_extent, save_dir):
    '''
    Generates ejecta velocity field quiver plot figures to be used as frames in movie.

    Parameter:
    ----------
    frames : pims ImageSequence
        Video frames to perform piv analysis
    xx_all : list
        Horizontal postition of velocity vectors for entire video 
    yy_all : list
        Vertical postition of velocity vectors for entire video 
    uu_all : list
        Horizontal component of velocity vectors for entire video
    vv_all : list
        Vertical component of velocity vectors for entire video
    framerate : float 
        Frame rate of the video, in 1/seconds 
    t0_frame : int
        Video frame number to take as time=0 
    new_origin : list
        Coordinates of user defined orign with length 2 (i.e. [xc, yc]); x-coordinate is first element, y-coordinate is second element
    img_extent : list
        Values to pass to imshow's extent option; video frame boundaries in real units 
    save_dir : str
        Directory name to save data files

    Returns:
    --------
    None 
    '''

    ## Start progress bar 
    pbar = tqdm(total = len(frames[:-1]))
    
    for kk, (frame,xx,yy,uu,vv) in enumerate(zip(frames,xx_all,yy_all,uu_all,vv_all)):
        plot_vel_field(frame, xx, yy, uu, vv, kk+t0_frame, kk, framerate, new_origin, img_extent, save=True, savepath=save_dir) 

        pbar.update(1) # update progress bar 



#-----------------------------------------------------------------------------------------------------
def save_piv_data(x_pos, y_pos, u_vel, v_vel, save_dir):
    '''
    Saves ejecta velocity field data to JSON files.
    
    Parameters:
    -----------
    x_pos : list
        Horizontal postition of velocity vectors for entire video 
    y_pos : list
        Vertical postition of velocity vectors for entire video 
    u_vel : list
        Horizontal component of velocity vectors for entire video 
    v_vel : list
        Vertical component of velocity vectors for entire video 
    save_dir : str 
        Directory name to save data files 
    
    Returns:
    --------
    None 
    '''
    ## Create the save directory
    # try:
    #     os.mkdir(save_dir)
    #     print(f"Directory '{save_dir}' created successfully.")
    # except FileExistsError:
    #     print(f"Directory '{save_dir}' already exists.")

    x_pos2 = [arr.tolist() for arr in x_pos]
    with open(save_dir+'vfield_x_positions.json', 'w') as f:
        json.dump(x_pos2, f)

    y_pos2 = [arr.tolist() for arr in y_pos]
    with open(save_dir+'vfield_y_positions.json', 'w') as f:
        json.dump(y_pos2, f)

    u_vel2 = [arr.tolist() for arr in u_vel]
    with open(save_dir+'vfield_u_components.json', 'w') as f:
        json.dump(u_vel2, f)

    v_vel2 = [arr.tolist() for arr in v_vel]
    with open(save_dir+'vfield_v_components.json', 'w') as f:
        json.dump(v_vel2, f)



#-----------------------------------------------------------------------------------------------------
def make_vmag_hists(data, t0_frame, framerate, plot_title, save_dir):
    '''
    Generate histograms of the ejecta velocity magnitudes

    Parameters:
    -----------
    data : ndarray
        Data in time to make histogram
    t0_frame : int
        Frame number corresponding to time=0 
    framerate : float 
        Frame rate of the original video, in 1/seconds
    plot_title : str
        Title of figure, depends on nature of data 
    save_dir : str
        Directory to save histogram figures

    Returns:
    --------
    None 
    '''
    # ## Create the save directory
    # try:
    #     os.mkdir(save_dir)
    #     print(f"Directory '{save_dir}' created successfully.")
    # except FileExistsError:
    #     print(f"Directory '{save_dir}' already exists.")
    
    
    for jj, arr in enumerate(data):
    
        fig,ax = plt.subplots(1,1,dpi=100)
        ax.hist(arr[arr>0].flatten(), bins=15, range=(np.min(data),np.max(data)))
        ax.set_title(plot_title+'\nt = %.1f ms' %(jj/framerate*1000)) # title time in ms 
        ax.set_ylabel('counts')
        ax.set_xlabel('|v| [cm/s]')
        ax.set_ylim(0,1000)
        ax.set_xlim(0,)
    
        plt.savefig(save_dir+'piv_hist_'+str(t0_frame+jj)+'.png', bbox_inches='tight', dpi=200)
    
        plt.close()



#-----------------------------------------------------------------------------------------------------
def get_angles_from_vel_components(u_vel, v_vel, vel_comp_cutoff):
    '''
    Calculate the angles of given velocity vectors that survive filtering.
    (This function may need to be updated to also filter velocity vector locations)

    Parameters:
    -----------
    u_vel : list
        Horizontal velocity component
    v_vel : list
        Vertical velocity component
    vel_comp_cutoff : float
        Threshold that filters out velocity components below this value

    Returns:
    --------
    angles_deg : numpy array
        Angles of velocity vectors, in degrees, that survived filtering
    '''

    u_vel_flat = np.array(u_vel).flatten()
    v_vel_flat = np.array(v_vel).flatten()
    u_vel_bool = np.abs(u_vel_flat) > vel_comp_cutoff
    v_vel_bool = np.abs(v_vel_flat) > vel_comp_cutoff
    idxs = np.where([a or b for a,b in zip(u_vel_bool, v_vel_bool)])[0]
    u_vel_flat_filt = u_vel_flat[idxs]
    v_vel_flat_filt = v_vel_flat[idxs]
    angles_rad = np.arctan2(v_vel_flat_filt, u_vel_flat_filt)
    angles_deg = angles_rad*180/np.pi

    return angles_deg 


#-----------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------