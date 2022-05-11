import py4DSTEM
import numpy as np
import scipy
from scipy.optimize import leastsq, least_squares
from scipy.ndimage.filters import gaussian_filter
from py4DSTEM.process.utils import get_CoM, tqdmnd 
from py4DSTEM.process.utils.elliptical_coords import radial_integral, radial_elliptical_integral, cartesian_to_polarelliptical_transform
from py4DSTEM.io.datastructure import PointList, PointListArray

def make_beamstop_mask(Q_Nx, 
                       Q_Ny,
                       x0,
                       y0,
                       rotation,
                       spanned_angle,
                       width,
                       xc,
                       yc,
                       circlemask_radius):
    """
    Makes beamstop mask
    """
    
    yy,xx = np.meshgrid(np.arange(Q_Ny),np.arange(Q_Nx))
    
    #Beamstop
    xx -= x0
    yy -= y0
    rr = np.hypot(xx,yy)
    tt = np.arctan2(yy,xx) - np.radians(rotation)
    phi = np.radians(spanned_angle)/2
    bs_mask = (tt>-phi)*(tt<=phi)
    bs_mask = np.logical_or(scipy.ndimage.binary_dilation(bs_mask,iterations=int(width//2)), (rr<width/2))
    bs_mask = bs_mask == False 
    
    #Circle
    rr_c = np.hypot(xx - xc,yy - yc)
    circle_mask = rr_c > circlemask_radius

    bs_mask = bs_mask * circle_mask
    
    return bs_mask == True

def universal_threshold(pointlistarray,  minIntensity, metric = 'manual', coords = None, minPeakSpacing=None,
                                                    maxNumPeaks=None, mask = None):
    """
    Takes a PointListArray of detected Bragg peaks and applies universal thresholding,
    returning the thresholded PointListArray. To skip a threshold, set that parameter to False.
    Accepts:
        pointlistarray        (PointListArray) The Bragg peaks. Must have
                              coords=('qx','qy','intensity')
        minIntensity          (float) the minimum allowed peak intensity, relative to the
                              selected metric (0-1), except in the case of 'manual' metric,
                              in which the threshold value based on the minimum intensity
                              that you want thresholder out should be set.
        metric                (string) the metric used to compare intensities. 'average'
                              compares peak intensity relative to the average of the maximum
                              intensity in each diffraction pattern. 'max' compares peak
                              intensity relative to the maximum intensity value out of all
                              the diffraction patterns.  'median' compares peak intensity relative
                              to the median of the maximum intensity peaks in each diffraction
                              pattern. 'manual' Allows the user to threshold based on a
                              predetermined intensity value manually determined.
        minPeakSpacing        (int) the minimum allowed spacing between adjacent peaks -
                              optional, default is false
        maxNumPeaks           (int) maximum number of allowed peaks per diffraction pattern -
                              optional, default is false
        mask                  (nd.array) 2D mask object- usually beam stop mask
    Returns:
       pointlistarray        (PointListArray) Bragg peaks thresholded by intensity.
    """
    assert all([item in pointlistarray.dtype.fields for item in ['qx','qy','intensity']]), (
                "pointlistarray must include the coordinates 'qx', 'qy', and 'intensity'.")
    if coords is not None:
        qx, qy = coords.get_origin()
    
    if metric != 'manual':
        HI_array = np.zeros( (pointlistarray.shape[0], pointlistarray.shape[1]) )
        for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1]):
                pointlist = pointlistarray.get_pointlist(Rx,Ry)
                pointlist.sort(coordinate='intensity', order='descending')
                if pointlist.data.shape[0] == 0:
                    top_value = np.nan
                else:
                    top_value = pointlist.data[0][2]
                    HI_array[Rx, Ry] = top_value

        mean_intensity = np.nanmean(HI_array)
        max_intensity = np.max(HI_array)
        median_intensity = np.median(HI_array)

    for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1]):
            pointlist = pointlistarray.get_pointlist(Rx,Ry)

            # Remove peaks below minRelativeIntensity threshold
            if minIntensity is not False:
                if metric == 'average':
                    deletemask = pointlist.data['intensity']/mean_intensity < minIntensity
                    pointlist.remove_points(deletemask)
                if metric == 'maximum':
                    deletemask = pointlist.data['intensity'] / max_intensity < minIntensity
                    pointlist.remove_points(deletemask)
                if metric == 'median':
                    deletemask = pointlist.data['intensity'] / median_intensity < minIntensity
                    pointlist.remove_points(deletemask)
                if metric == 'manual':
                    deletemask = pointlist.data['intensity'] < minIntensity
                    pointlist.remove_points(deletemask)

            # Remove peaks that are too close together
            if maxNumPeaks is not None:
                r2 = minPeakSpacing**2
                deletemask = np.zeros(pointlist.length, dtype=bool)
                for i in range(pointlist.length):
                    if deletemask[i] == False:
                        tooClose = ( (pointlist.data['qx']-pointlist.data['qx'][i])**2 + \
                                     (pointlist.data['qy']-pointlist.data['qy'][i])**2 ) < r2
                        tooClose[:i+1] = False
                        deletemask[tooClose] = True
                pointlist.remove_points(deletemask)

            # Keep only up to maxNumPeaks
            if maxNumPeaks is not None:
                if maxNumPeaks < pointlist.length:
                    deletemask = np.zeros(pointlist.length, dtype=bool)
                    deletemask[maxNumPeaks:] = True
                    pointlist.remove_points(deletemask)
            
            if mask is not None:
                if coords is not None:
                    deletemask = np.zeros(pointlist.length, dtype=bool)
                    for i in range(pointlist.length):
                        deletemask_ceil = np.where((mask[ np.ceil(pointlist.data['qx'] + qx[Rx, Ry]).astype(int),
                            np.ceil(pointlist.data['qy'] + qy[Rx,Ry]).astype(int) ] == False), True, False) 
                        pointlist.remove_points(deletemask_ceil)
                        deletemask_floor = np.where((mask[ np.floor(pointlist.data['qx'] + qx[Rx, Ry]).astype(int),
                            np.floor(pointlist.data['qy'] + qy[Rx,Ry]).astype(int) ] == False), True, False)
                        pointlist.remove_points(deletemask_floor)
                if coords is None:
                    for i in range(pointlist.length):
                        deletemask_ceil = np.where((mask[ np.ceil(pointlist.data['qx']).astype(int),
                            np.ceil(pointlist.data['qy']).astype(int) ] == False), True, False) 
                        pointlist.remove_points(deletemask_ceil)
                        deletemask_floor = np.where((mask[ np.floor(pointlist.data['qx']).astype(int),
                            np.floor(pointlist.data['qy']).astype(int) ] == False), True, False)
                        pointlist.remove_points(deletemask_floor)
    return pointlistarray

def polar_elliptical_transform_datacube(
    datacube,
    r_range,
    dr = 1,
    dphi = np.radians(2),
    mask = None,
    coords = None,
    p_ellipse = None
    ):
    """
    Performs the polar elliptical transformation of every diffraction pattern in a datacube based either the center positions
    and ellipse parameters set in the coordinates object or the p_ellipse parameters. Returns an ndarray of the polar elliptical
    data.
    
    Args:
    datacube (datacube)
    dr (int)
    dphi (int)
    r_range (list)
    mask (bool)
    coords (coordinate object)
    p_ellipse (tuple)
    
    Returns:
    transform_stack (ndarray)
    """
    ## enter assert statement saying that both coords and pellipse can't be None
    ## enter assert statement saying use polar transform datacube if coords a, b, theta are not there
    if coords is not None:
        qx0, qy0 = coords.get_origin()
        p_ellipse = (qx0[0,0], qy0[0,0], coords.a, coords.b, coords.theta)
    PET, rr, tt = cartesian_to_polarelliptical_transform(datacube.data[0, 0], 
                                                p_ellipse=p_ellipse,
                                                dr=dr,
                                                r_range=r_range, 
                                                dphi=dphi,
                                                mask=mask)
    transform_stack = np.zeros([coords.R_Nx, coords.R_Ny, PET.data.shape[0], PET.data.shape[1]])
    for (Rx,Ry) in tqdmnd(datacube.R_Nx,datacube.R_Ny,desc='Transforming',unit='DP',unit_scale=True):
        DP = datacube.data[Rx,Ry,:,:]
        if coords is not None:
            params = (qx0[Rx, Ry], qy0[Rx, Ry], coords.a, coords.b, coords.theta)
            PET, rr, tt = cartesian_to_polarelliptical_transform(
                datacube.data[Rx, Ry], 
                p_ellipse=params,
                dr=dr,
                r_range=r_range,
                dphi=dphi,
                mask=mask)
        else:
            PET, rr, tt = cartesian_to_polarelliptical_transform(
                datacube.data[Rx, Ry],
                p_ellipse=p_ellipse,
                dr=dr,
                r_range=r_range,
                dphi=dphi,
                mask=mask)
        transform_stack[Rx,Ry,:,:] = PET
    return transform_stack