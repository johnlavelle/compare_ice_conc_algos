
"""
    Collection of ice concentation algorithms. 

    Main entrance into this module is the calc_osi_conc and calc_nasa_conc functions. 

"""
from __future__ import division

import logging
import numpy as np
from collections import Iterable
import xarray as xa

LOG = logging.getLogger(__name__)


def read_apply_write(func):
    return func


def nasa(tb18v, tb18h, tb37v, area='nh'):
    """
        NASA-Team ice concetration algorithm

        Static tiepoint algorithm. 

        :param tb18v: Brightness temperatues 18 Ghz V polarized
            :type tb18v: numpy array
        :param tb18h: Brightness temperatues 18 Ghz H polarized
            :type tb18h: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
            :type tb37v: numpy array
        :param area: Area, Northern 'nh' or Southern 'sh' hemisphere
            :type area: string
        :returns: Ice concentation

    """
    for k, v in dict(tb18v=tb18v, tb18h=tb18h, tb37v=tb37v).items():
        assert isinstance(v, (np.ndarray, xa.core.dataarray.DataArray)), 'Variable %s must be of type np.ndarray' % k
    assert area in ('nh','sh'), "Variable area must be either 'nh' or 'sh'"

    LOG.info("Calculating NASA Team Ice conc for area %s" % area)
    
    if area.lower() == 'nh':
        tiepts = (177.10, 100.80, 201.7, 258.20, 242.80, 252.80, 223.20, 203.90, 186.30)
    elif area.lower() == 'sh':
        tiepts = (176.60, 100.30, 200.5, 249.80, 237.80, 243.3, 221.60, 193.70, 190.30)
    else:
        raise ValueError('NASA Team undefined area: %s', area)

    (tb18v_ow, tb18h_ow, tb37v_ow, tb18v_fy, tb18h_fy, 
     tb37v_fy, tb18v_my, tb18h_my, tb37v_my) = tiepts
       
    a0 = - tb18v_ow + tb18h_ow
    a1 =   tb18v_ow + tb18h_ow
    a2 =   tb18v_my - tb18h_my - tb18v_ow + tb18h_ow
    a3 = - tb18v_my - tb18h_my + tb18v_ow + tb18h_ow
    a4 =   tb18v_fy - tb18h_fy - tb18v_ow + tb18h_ow
    a5 = - tb18v_fy - tb18h_fy + tb18v_ow + tb18h_ow

    b0 = - tb37v_ow + tb18v_ow
    b1 =   tb37v_ow + tb18v_ow
    b2 =   tb37v_my - tb18v_my - tb37v_ow + tb18v_ow
    b3 = - tb37v_my - tb18v_my + tb37v_ow + tb18v_ow
    b4 =   tb37v_fy - tb18v_fy - tb37v_ow + tb18v_ow
    b5 = - tb37v_fy - tb18v_fy + tb37v_ow + tb18v_ow

    gr = (tb37v - tb18v) / (tb37v + tb18v)
    pr = (tb18v - tb18h) / (tb18v + tb18h)

    d0 = (-a2 * b4) + (a4 * b2)
    d1 = (-a3 * b4) + (a5 * b2)
    d2 = (-a2 * b5) + (a4 * b3)
    d3 = (-a3 * b5) + (a5 * b3)

    dd = d0 + (d1 * pr) + (d2 * gr) + (d3 * pr * gr)
    
    f0 = (a0 * b2) - (a2 * b0)
    f1 = (a1 * b2) - (a3 * b0)
    f2 = (a0 * b3) - (a2 * b1)
    f3 = (a1 * b3) - (a3 * b1)
    m0 = (-a0 * b4) + (a4 * b0)
    m1 = (-a1 * b4) + (a5 * b0)
    m2 = (-a0 * b5) + (a4 * b1)
    m3 = (-a1 * b5) + (a5 * b1)

    cf = (f0 + (f1 * pr) + (f2 * gr) + (f3 * pr * gr)) / dd
    cm = (m0 + (m1 * pr) + (m2 * gr) + (m3 * pr * gr)) / dd

    cf = cf
    cm = cm
    ct = cm + cf
    return ct


@read_apply_write
def calc_nasa_conc(tb19v, tb19h, tb37v, lats):
    """
        Calculates the ice concentation using the NASA Team algorithm
        (see https://nsidc.org/data/docs/daac/nasateam/).

        :param lats: The lalitudes of the pixels
            :type lats: numpy array
        :param tb18v: Brightness temperatues 18 Ghz V polarized
            :type tb18v: numpy array
        :param tb18h: Brightness temperatues 18 Ghz H polarized
            :type tb18h: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
            :type tb37v: numpy array
        :returns: The ice concentration as a numpy array

    """
    for k, v in dict(tb19v=tb19v, tb19h=tb19h, tb37v=tb37v, lats=lats).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray, not %s' % (k, type(v))

    nh_idx = (lats > 0)
    sh_idx = (lats <= 0)
    nasa_nh = nasa(tb19v[nh_idx], tb19h[nh_idx], tb37v[nh_idx], area='nh')
    nasa_sh = nasa(tb19v[sh_idx], tb19h[sh_idx], tb37v[sh_idx], area='sh')
    conc = np.zeros(lats.size)
    conc[nh_idx] = nasa_nh
    conc[sh_idx] = nasa_sh
    return conc


# def fcomiso_dynamic(tb19v, tb37v, tb19v_ow, tb37v_ow, slope_ice, offset_ice):
#     """
#         Calculates the ice concentration using the Fcomiso algorithm.
#
#         Uses dynamical tiepoints.
#
#         :param tb19v: Brightness temperatues 19 Ghz V polarized
#         :type tb19v: numpy array
#         :param tb37v: Brightness temperatues 37 Ghz V polarized
#         :type tb37v: numpy array
#         :param tb19v_ow: Brightness temperatues 19 Ghz V polarized, open water points
#         :type tb19v_ow: numpy array
#         :param tb37v_ow: Brightness temperatues 37 Ghz V polarized, open water points
#         :type tb37v_ow: numpy array
#         :param slope_ice: The slope of the principal component vector
#         :type slope_ice: float
#         :param offset_ice: The x-axis intersection of the line with slope_ice
#             that intersects with the mean tb values for both open water and ice
#             pixels
#         :returns: The ice concentration as a numpy array
#     """
#     # NOT SURE ABOUT: tb19v_ow, tb37v_ow: np.ndarray or float?
#     for k,v in dict(tb19v=tb19v, tb37v=tb37v).items():
#         assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray' % k
#     for k,v in dict(slope_ice=slope_ice, offset_ice=offset_ice).items():
#         assert isinstance(v, (int,float,long)), 'Variable %s must be of type (int,float,long)' % k
#
#     qf = (tb37v - tb37v_ow) / (tb19v - tb19v_ow)
#     wf = tb37v_ow - (qf * tb19v_ow)
#
#     ti19vf = (offset_ice - wf) / (qf - slope_ice)
#     cf = (tb19v - tb19v_ow) / (ti19vf - tb19v_ow)
#     return cf


def fcomiso_dynamic(tb19v, tb37v, tb19v_ow, tb37v_ow, slope_ice, offset_ice):
    """
        Calculates the ice concentration using the Fcomiso algorithm.

        Uses dynamical tiepoints.

        :param tb19v: Brightness temperatues 19 Ghz V polarized
            :type tb19v: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
            :type tb37v: numpy array
        :param tb19v_ow: Brightness temperatues 19 Ghz V polarized, open water points
            :type tb19v_ow: numpy array
        :param tb37v_ow: Brightness temperatues 37 Ghz V polarized, open water points
            :type tb37v_ow: numpy array
        :param slope_ice: The slope of the principal component vector
            :type slope_ice: float
        :param offset_ice: The x-axis intersection of the line with slope_ice
            that intersects with the mean tb values for both open water and ice
            pixels
        :returns: The ice concentration as a numpy array

    """
    # netcdf variables are 32-bit floats so explicitly set cf the same.
    # Will lead to casting TypeError if 64-bit vars.
    cf = np.zeros(tb19v.shape).astype(np.float32)
    # Find where tb19v and tb19v_ow are within numpy 32-bit float
    # precision (1.1920929e-07) of each other.
    bools = np.abs(tb19v - tb19v_ow) < np.finfo(np.float32).eps

    if ( bools.sum() != 0 ):
        # Concentration for indices where tb19v = tb19v_ow.
        # Needs to use conc = (tb37v - tb37v_ow) / (tb37v_intersect - tb37v_ow)
        cf[bools] = (tb37v[bools] - tb37v_ow) / ((offset_ice + slope_ice * tb19v[bools]) - tb37v_ow)

        # Concentration for indices where tb19v != tb19v_ow.
        # Incline of line through (tb19v_ow,tb37v_ow) & (tb19v,tb37v) points.
        qf = (tb37v[~bools] - tb37v_ow) / (tb19v[~bools] - tb19v_ow)
        # tb37v where tb19v = 0 in qf line, ie y offset.
        wf = tb37v_ow - (qf * tb19v_ow)
        """ What's to be done when qf = slope_ice? """
        # tb19v where qf line intersects with ice line defined by offset_ice & slope_ice.
        ti19vf = (offset_ice - wf) / (qf - slope_ice)
        # Calculate tb37v_intersect
        ti37vf = offset_ice + slope_ice * ti19vf

        # Check which is largest of (ti19vf - tb19v_ow) and (ti37vf - tb37v_ow) to
        # use different ice concentration formula accordingly.
        subset = np.abs(ti19vf - tb19v_ow) > np.abs(ti37vf - tb37v_ow)
        # Create temporary array same size as ~bools.
        cf_bools = np.zeros((~bools).shape).astype(np.float32)
        # Ice concentration as defined by conc = (tb19v - tb19v_ow) / (tb19v_intersect - tb19v_ow)
        cf_bools[subset] = (tb19v[~bools][subset] - tb19v_ow) / (ti19vf[subset] - tb19v_ow)
        # Ice concentration as defined by conc = (tb37v - tb37v_ow) / (tb37v_intersect - tb37v_ow)
        cf_bools[~subset] = (tb37v[~bools][~subset] - tb37v_ow) / (ti37vf[~subset] - tb37v_ow)
        # Add cf_bools values to ~bools indeices of cf
        cf[~bools] = cf_bools

    else:
        # bools array is false, so safe to use old algorithm.
        qf = (tb37v - tb37v_ow) / (tb19v - tb19v_ow)
        wf = tb37v_ow - (qf * tb19v_ow)
        ti19vf = (offset_ice - wf) / (qf - slope_ice)
        cf = (tb19v - tb19v_ow) / (ti19vf - tb19v_ow)

    # Previous version.
    #qf = (tb37v - tb37v_ow) / (tb19v - tb19v_ow)
    #wf = tb37v_ow - (qf * tb19v_ow)
    #ti19vf = (offset_ice - wf) / (qf - slope_ice)
    #cf = (tb19v - tb19v_ow) / (ti19vf - tb19v_ow)
    return cf


def bristol_dynamic(tb19v, tb37v, tb37h, ow_x, ow_y, slope, offset):
    """
        Calculates the ice concentration using the Bristol algorithm.

        Uses dynamical tiepoints.

        :param tb19v: Brightness temperatues 19 Ghz V polarized
            :type tb19v: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
            :type tb37v: numpy array
        :param tb37_h: Brightness temperatues 37 Ghz H polarized
            :type tb37_h: numpy array
        :param ow_x: Open water x values in Bristol space
            :type ow_x: numpy array
        :param ow_y: Open water y values in Bristol space
            :type ow_y: numpy array
        :param slope: The slope of the principal component vector in bristol space
            :type slope: float
        :param offset: The x-axis intersection of the line with slope
            that intersects with the mean tb values for both open water and ice
            pixels
            :type offset: float
        :returns: The ice concentration as a numpy array
    """
    # NOT SURE ABOUT ow_x, ow_y: np.ndarray or float?
    for k, v in dict(tb19v=tb19v, tb37v=tb37v,  tb37h=tb37h).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray' % k
    for k, v in dict(slope=slope, offset=offset).items():
        assert isinstance(v, (int, float, long)), 'Variable %s must be of type (int,float,long)' % k

    xt = tb37v + (1.045 * tb37h) + (0.525 * tb19v)
    yt = (0.9164 * tb19v) - tb37v + (0.4965 * tb37h)
    
    a_ht = (yt - ow_y) / (xt - ow_x)
    b_ht = ow_y - (a_ht * ow_x)
    
    xi = (offset - b_ht) / (a_ht - slope)
    cf = (xt - ow_x) / (xi - ow_x)
    c = cf
    return c


def blend(fcomiso_conc, bristol_conc, threshold):
    """
        Blends the ouput from the bristol and fcomiso algorithms using a linear
        relation. The bending the only applied beneath a threshold.

        :param fcomiso_conc: The ice concentation from the fcomiso algorithm
            :type fcomiso_conc: numpy array
        :param bristol_conc: The ice concentation from the bristol algorithm
            :type bristol_conc: numpy array
        :param threshold: The threshold concentration, between 0 and 1
            :type threshold: float
        :returns: The blended concentrations
    """
    for k, v in dict(fcomiso_conc=fcomiso_conc, bristol_conc=bristol_conc).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray' % k
    assert isinstance(threshold, (int, float, long)), 'Variable threshold must be of type (int,float,long)'

    LOG.info("Blending Bristol and FComiso algorithms using threshold %s" % threshold)

    blend = (np.abs(threshold - fcomiso_conc) + threshold - fcomiso_conc) / \
            (2 * threshold)
            
    below_mask = (fcomiso_conc < 0)

    bf = (blend * (1 - below_mask)) + below_mask

    conc = ((1 - bf) * bristol_conc) + (bf * fcomiso_conc)
    return conc


def osisaf_hybrid_dynamic(tb19v, tb37v, tb37h, fcomiso_ow_x, fcomiso_ow_y,
                          fcomiso_slope, fcomiso_offset, bristol_ow_x, 
                          bristol_ow_y, bristol_slope, bristol_offset, threshold=0.4):

    """
        The OSI SAF hybrid algorithm. The algorithm mixes Fcomiso and Bristol
        base on a ice concentration threshold value.


        Uses dynamical tiepoints.

        :param tb19v: Brightness temperatues 19 Ghz V polarized
            :type tb19v: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
            :type tb37v: numpy array
        :param tb37_h: Brightness temperatues 37 Ghz H polarized
            :type tb37_h: numpy array
        :param fcomiso_ow_x: Open water x values in fcomiso space
            :type fcomiso_ow_x: numpy array                               # IS THIS CORRECT? calc_osi_conc() uses a float
        :param fcomiso_ow_y: Open water y values in fcomiso space
            :type fcomiso_ow_y: numpy array                               # IS THIS CORRECT? calc_osi_conc() uses a float
        :param fcomiso_slope: The slope of the principal component vector in fcomiso space
            :type fcomiso_slope: float
        :param fcomiso_offset: The x-axis intersection of the line with slope
            that intersects with the mean tb values for both open water and ice
            pixels
            :type bristol_offset: float
        :param bristol_ow_x: Open water x values in bristol space
            :type bristol_ow_x: numpy array
        :param bristol_ow_y: Open water y values in bristol space
            :type bristol_ow_y: numpy array
        :param bristol_slope: The slope of the principal component vector in bristol space
            :type bristol_slope: float
        :param bristol_offset: The x-axis intersection of the line with slope
            that intersects with the mean tb values for both open water and ice
            pixels
            :type bristol_offset: float
        :param threshold: Ice concentration threshold value the algorithm will
            switch from bristol to fcomiso below this value
            :type threshold: float
        :returns: The ice concentration as a numpy array
    """
    # NOT SURE ABOUT fcomiso_ow_x, fcomiso_ow_y: np.ndarray or float?
    for k, v in dict(tb19v=tb19v, tb37v=tb37v).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray' % k
    for k, v in dict(fcomiso_slope=fcomiso_slope, fcomiso_offset=fcomiso_offset, bristol_slope=bristol_slope,
                    bristol_offset=bristol_offset, threshold=threshold).items():
        assert isinstance(v, (int, float, long)), 'Variable %s must be of type (int,float,long)' % k
    
    conc_comiso = fcomiso_dynamic(tb19v, tb37v, fcomiso_ow_x, fcomiso_ow_y, 
                                  fcomiso_slope, fcomiso_offset)

    conc_bristol = bristol_dynamic(tb19v, tb37v, tb37h, bristol_ow_x, bristol_ow_y, 
                                   bristol_slope, bristol_offset)
    
    return blend(conc_comiso, conc_bristol, threshold=threshold)


@read_apply_write
def calc_osi_conc(tb19v, tb37v, tb37h, lats, dyn_tps):
    """
        Calculates the OSI SAF ice concentration. Main entrance to the ice
        algorithms module.

        :param lats: pixel latitudes
            :type lats: float
        :param tb19v: Brightness temperatues 19 Ghz V polarized
            :type tb19v: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
            :type tb37v: numpy array
        :param tb37_h: Brightness temperatues 37 Ghz H polarized
            :type tb37_h: numpy array
        :param dyn_tps: Dynamical tiepoints
            :type dyn_tps: dict
        :returns: Ice concentration as a numpy array 

    """
    for k, v in dict(tb19v=tb19v, tb37v=tb37v, tb37h=tb37h, lats=lats).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray' % k
    assert isinstance(dyn_tps, dict), 'Variable dyn_tps must be of type dict'

    LOG.info("Calculation OSI SAF Ice conc using tiepoints: %s" % dyn_tps)

    nh_idx = (lats > 0)
    sh_idx = (lats <= 0)
    #invalid_idx = (tb19v < 150) | (tb19v > 295) | (tb37v < 150) | (tb37v > 295) | (tb37h < 150) | (tb37h > 295)

    nh_conc = osisaf_hybrid_dynamic(tb19v[nh_idx],
                                    tb37v[nh_idx],
                                    tb37h[nh_idx],
                                    dyn_tps['nh']['algorithm']['fcomiso']['mean_ow']['x'],  # dyn_tps['nh']['fcomiso_mean_ow']['x'],
                                    dyn_tps['nh']['algorithm']['fcomiso']['mean_ow']['y'],  # dyn_tps['nh']['fcomiso_mean_ow']['y'],
                                    dyn_tps['nh']['algorithm']['fcomiso']['slope'],  #  dyn_tps['nh']['fcomiso_slope'],
                                    dyn_tps['nh']['algorithm']['fcomiso']['offset'],  #  dyn_tps['nh']['fcomiso_offset'],
                                    dyn_tps['nh']['algorithm']['bristol']['mean_ow']['x'], #  dyn_tps['nh']['bristol_mean_ow']['x'],
                                    dyn_tps['nh']['algorithm']['bristol']['mean_ow']['y'], #  dyn_tps['nh']['bristol_mean_ow']['y'],
                                    dyn_tps['nh']['algorithm']['bristol']['slope'],  #  dyn_tps['nh']['bristol_slope'],
                                    dyn_tps['nh']['algorithm']['bristol']['offset'])  #  dyn_tps['nh']['bristol_offset'])
    sh_conc = osisaf_hybrid_dynamic(tb19v[sh_idx],
                                    tb37v[sh_idx],
                                    tb37h[sh_idx],
                                    dyn_tps['sh']['algorithm']['fcomiso']['mean_ow']['x'],  # dyn_tps['sh']['fcomiso_mean_ow']['x'],
                                    dyn_tps['sh']['algorithm']['fcomiso']['mean_ow']['y'],  # dyn_tps['sh']['fcomiso_mean_ow']['y'],
                                    dyn_tps['sh']['algorithm']['fcomiso']['slope'],  #  dyn_tps['sh']['fcomiso_slope'],
                                    dyn_tps['sh']['algorithm']['fcomiso']['offset'],  #  dyn_tps['sh']['fcomiso_offset'],
                                    dyn_tps['sh']['algorithm']['bristol']['mean_ow']['x'], #  dyn_tps['sh']['bristol_mean_ow']['x'],
                                    dyn_tps['sh']['algorithm']['bristol']['mean_ow']['y'], #  dyn_tps['sh']['bristol_mean_ow']['y'],
                                    dyn_tps['sh']['algorithm']['bristol']['slope'],  #  dyn_tps['sh']['bristol_slope'],
                                    dyn_tps['sh']['algorithm']['bristol']['offset'])  #  dyn_tps['sh']['bristol_offset'])
    conc = np.zeros(lats.size)  
    conc[nh_idx] = nh_conc
    conc[sh_idx] = sh_conc
    #conc[invalid_idx] = fillvalue
    return conc


@read_apply_write
def near90_linear(tb85v, tb85h):
    for k, v in dict(tb85v=tb85v, tb85h=tb85h).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray not %s' % (k, type(v))

    ct = 1.22673 - 0.02652*(tb85v - tb85h)
    return 100*ct


def n90_linear_dynamic(tb90v, tb90h, tb90v_ow, tb90h_ow, slope_ice, offset_ice):
    """
    Calculate the ice concentration using the Fcomiso algorithm.
    Uses implementation from the fcomiso_dynamic

    """
    for k, v in dict(tb90v=tb90v, tb90h=tb90h).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray' % k
    for k, v in dict(tb90v_ow=tb90v_ow, tb90h_ow=tb90h_ow, slope_ice=slope_ice, offset_ice=offset_ice).items():
        assert isinstance(v, (int, float, long)), 'Variable %s must be of type (int,float,long)' % k

    concentration = fcomiso_dynamic(tb90v, tb90h, tb90v_ow, tb90h_ow, slope_ice, offset_ice)
    return concentration


@read_apply_write
def calc_n90_conc(tb90v, tb90h, lats, dyn_tps):
    """
        Calculates ice concentration with a Near90 algorithm. Main entrance to
        the ice high-frequency algorithms module.

        :param lats: pixel latitudes
            :type lats: float
        :param tb90v: Brightness temperatues at near 90 Ghz V polarized
            :type tb90v: numpy array
        :param tb90h: Brightness temperatues at near 90 Ghz H polarized
            :type tb90h: numpy array
        :param dyn_tps: Dynamical tiepoints
            :type dyn_tps: dict
        :returns: Ice concentration as a numpy array

    """
    for k, v in dict(tb90v=tb90v, tb90h=tb90h, lats=lats).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray, not %s' % (k,type(v))
    assert isinstance(dyn_tps, dict), 'Variable dyn_tps must be a dictionary'

    LOG.info("Calculation near-90 SIC using tiepoints: %s" % dyn_tps)

    nh_idx = (lats > 0)
    sh_idx = (lats <= 0)

    nh_conc = n90_linear_dynamic(tb90v[nh_idx],
                                 tb90h[nh_idx],
                                 dyn_tps['nh']['algorithm']['n90lin']['mean_ow']['x'],  # dyn_tps['nh']['n90lin_mean_ow']['x'],
                                 dyn_tps['nh']['algorithm']['n90lin']['mean_ow']['y'],  # dyn_tps['nh']['n90lin_mean_ow']['y'],
                                 dyn_tps['nh']['algorithm']['n90lin']['slope'],  # dyn_tps['nh']['n90lin_slope'],
                                 dyn_tps['nh']['algorithm']['n90lin']['offset'])  # dyn_tps['nh']['n90lin_offset'])
    sh_conc = n90_linear_dynamic(tb90v[sh_idx],
                                 tb90h[sh_idx],
                                 dyn_tps['sh']['algorithm']['n90lin']['mean_ow']['x'],  # dyn_tps['sh']['n90lin_mean_ow']['x'],
                                 dyn_tps['sh']['algorithm']['n90lin']['mean_ow']['y'],  # dyn_tps['sh']['n90lin_mean_ow']['y'],
                                 dyn_tps['sh']['algorithm']['n90lin']['slope'],  # dyn_tps['sh']['n90lin_slope'],
                                 dyn_tps['sh']['algorithm']['n90lin']['offset'])  # dyn_tps['sh']['n90lin_offset'])
    conc = np.zeros(tb90v.shape)

    conc[nh_idx] = nh_conc
    conc[sh_idx] = sh_conc

    return conc


@read_apply_write
def calc_fcomiso_dynamic(tb19v, tb37v, lats, dyn_tps):

    for k, v in dict(tb19v=tb19v, tb37v=tb37v, lats=lats).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray' % k
    assert isinstance(dyn_tps, dict), 'Variable dyn_tps must be of type dict'

    nh_idx = (lats > 0)
    sh_idx = (lats <= 0)

    cf_nh = fcomiso_dynamic(tb19v[nh_idx],
                            tb37v[nh_idx],
                            dyn_tps['nh']['algorithm']['fcomiso']['mean_ow']['x'], # dyn_tps['nh']['fcomiso_mean_ow']['x'],
                            dyn_tps['nh']['algorithm']['fcomiso']['mean_ow']['y'], # dyn_tps['nh']['fcomiso_mean_ow']['y'],
                            dyn_tps['nh']['algorithm']['fcomiso']['slope'], # dyn_tps['nh']['fcomiso_slope'],
                            dyn_tps['nh']['algorithm']['fcomiso']['offset'])# dyn_tps['nh']['fcomiso_offset'])
    cf_sh = fcomiso_dynamic(tb19v[sh_idx],
                            tb37v[sh_idx],
                            dyn_tps['sh']['algorithm']['fcomiso']['mean_ow']['x'], # dyn_tps['sh']['fcomiso_mean_ow']['x'],
                            dyn_tps['sh']['algorithm']['fcomiso']['mean_ow']['y'], # dyn_tps['sh']['fcomiso_mean_ow']['y'],
                            dyn_tps['sh']['algorithm']['fcomiso']['slope'], # dyn_tps['sh']['fcomiso_slope'],
                            dyn_tps['sh']['algorithm']['fcomiso']['offset']) # dyn_tps['sh']['fcomiso_offset'])
    cf = np.zeros(lats.size)
    cf[nh_idx] = cf_nh
    cf[sh_idx] = cf_sh

    return cf


@read_apply_write
def tud_dyn(tb19v, tb37v, tb85v, tb85h, lats, dyn_tps):

    for k, v in dict(tb19v=tb19v, tb37v=tb37v, tb85v=tb85v, tb85h=tb85h, lats=lats).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray' % k
    assert isinstance(dyn_tps, dict), 'Variable dyn_tps must be of type dict'

    # check lats are correct: using 1D for 2D.
    cf = calc_fcomiso_dynamic(tb19v, tb37v, lats, dyn_tps)
    c90_dyn = calc_n90_conc(tb85v, tb85h, lats, dyn_tps).mean(axis=1)

    # if len(conc_near90.shape) > 1 and len(conc_comiso.shape) == 1:
    #     conc_comiso = conc_comiso[:,np.newaxis]

    conc = np.where(np.logical_and(c90_dyn > 0, cf > 10),  np.sqrt(c90_dyn * cf), cf)
    return conc


@read_apply_write
def tud_sta(tb19v, tb37v, tb85v, tb85h, lats, dyn_tps):

    for k, v in dict(tb19v=tb19v, tb37v=tb37v, tb85v=tb85v, tb85h=tb85h).items():
        assert isinstance(v, np.ndarray), 'Variable %s must be of type np.ndarray' % k
    assert isinstance(dyn_tps, dict), 'Variable dyn_tps must be of type dict'

    cf = calc_fcomiso_dynamic(tb19v, tb37v, lats, dyn_tps)
    c90_sta = near90_linear(tb85v, tb85h).mean(axis=1)

    # if len(conc_near90.shape) > 1 and len(conc_comiso.shape) == 1:
    #     conc_comiso = conc_comiso[:,np.newaxis]

    conc = np.where(np.logical_and(c90_sta > 0, cf > 10),  np.sqrt(c90_sta * cf), cf)
    return conc


