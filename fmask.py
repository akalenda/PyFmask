import numpy
import pandas
import theano
import theano.tensor as tt

# ######################## References #################################
# Zhu, Zhe, and Curtis E. Woodcock.
# "Object-based cloud and cloud shadow detection in Landsat imagery."
# Remote Sensing of Environment 118 (2012): 83-94.
#
# Zhu, Zhe, Shixiong Wang, and Curtis E. Woodcock.
# "Improvement and expansion of the Fmask algorithm: cloud, cloud shadow,
#  and snow detection for Landsats 4–7, 8, and Sentinel 2 images."
# Remote Sensing of Environment 159 (2015): 269-277.
# #####################################################################
#
# If performance becomes an issue, it's worth checking out TensorFlow as an alternative to Theano, or even
# look at https://github.com/Microsoft/CNTK ; these are not as mature and easy-to-use as Theano however.
# Whatever is used to perform the calculations, care should be taken to see that your solution mitigates
# the errors imposed by floating point number representation.
#
# Our general workflow is to read the data into a Pandas dataframe (essentially a table), and then follow the
# algorithms described in the paper using Theano. Some identifier naming conventions in this source code:
#
# - The prefix v*_ indicates that it is a Variable in the Theano sense of the word. Our transformations are performed
#   on a column-by-column basis, so v0_band1 is actually a list containing the band1 values for all of the pixels.
# - The prefix e*_ indicates that it is an expression for use in Theano.
# - The prefix *[\d+]_ indicates that it corresponds to a numbered formula in the whitepaper. Unless the number is 0,
#   in which case it is from our original data.
#
# For example, v2_mean_vis is a Theano Variable; data created as a byproduct of
# implementing formula 2 from the whitepaper so that it can be used in e2_whiteness_test,
# an expression also derived from formula 2 of the whitepaper.
#
# These expressions and variables are used to dynamically compile optimized functions
# during runtime. They are given columns from the table as input, and the vectors thus
# produced are stored as new columns in the table. This table can then be written to a new
# file, preserving both intermediary and final results for each pixel.
#
# #########################################################################

"""
Constants rom [Zhu 2012]. A variety of constants sprinkled throughout the white-paper.
"""
GEOTIFF_VALUE_SIZE = 0xFFFF
C1_MIN_BAND7_TOA_REFLECTANCE_OF_CLOUDS = 0.03
C1_MAX_BT_OF_CLOUDS = 27.0  # Brightness Temperature, degrees Celcius
C1_MAX_NDSI_OF_CLOUDS = 0.8  # Normalized Difference Snow Index, Scalar
C1_MAX_NDVI_OF_CLOUDS = 0.8  # Normalized Difference Vegetation Index, Scalar
C2_WHITENESS_THRESHOLD = 0.7  # See description for formula 2
C3_HOT_OFFSET = 0.16  # For formula 3. Has been doubled for computation reasons
NDVI_WATER_LAND_THRESHOLD = 0.10
C7_MIN_SWIR_OF_CLOUD = 0.03
C8_PERCENTILE_FOR_CLEARSKY_WATER_TEMPERATURE = 82.5
C10_MAX_WATER_REFLECTANCE = 0.11
C13_LOWER_PERCENTILE_FOR_CLEARSKY_LAND = 17.5
C13_UPPER_PERCENTILE_FOR_CLEARSKY_LAND = 82.5
C20151_CIRRUS_REFLECTANCE_THRESHOLD = 0.4  # Value of (scaled) band9 that converges with cirrus cloud reflectance

# ########################################### Theano Variables ##############################################
# These correspond to columns in the dataframe. Not all such columns are represented, only those needed in formulae.
# For example, we begin with v0 values.
# Values calculated by formula2 are labelled v2, and so on.
v1 = tt.dvector()
v2 = tt.dvector()
v3 = tt.dvector()
v4 = tt.dvector()
v0_blue = tt.dvector('v0_blue')
v0_green = tt.dvector('v0_green')
v0_red = tt.dvector('v0_red')
v0_nir = tt.dvector('v0_nir')
v0_swir1 = tt.dvector('v0_swir1')
v0_swir2 = tt.dvector('v0_swir2')
v0_cirrus = tt.dvector('v0_cirrus')
v0_tirs1_bt = tt.dvector('v0_tirs1_bt')
v1_ndsi = tt.dvector('v1_ndsi')
v1_ndvi = tt.dvector('v1_ndvi')
v2_whiteness = tt.dvector('v2_whiteness')
v7_clearsky_water = tt.dvector('v7_clearsky_water')
v10_brightness_prob = tt.dvector('v10_brightness_prob')
v20151_cirrus_cloud_probability = tt.dvector('v0_cirrus')
v14_l_temperature_prob = tt.dvector('v14_l_temperature_prob')
v15_variability_prob = tt.dvector('v15_variability_prob')


def calculate_ndvi(nir_reflectance: pandas.Series, red_reflectance: pandas.Series) -> pandas.Series:
    e1_ndvi = (v0_nir - v0_red) / (v0_nir + v0_red)
    return theano.function([v0_nir, v0_red], e1_ndvi)(nir_reflectance, red_reflectance)


def calculate_ndsi(green_reflectance: pandas.Series, swir1_reflectance: pandas.Series) -> pandas.Series:
    e1_ndsi = (v0_green - v0_swir1) / (v0_green + v0_swir1)
    return theano.function([v0_green, v0_swir1], e1_ndsi)(green_reflectance, swir1_reflectance)


def calculate_whiteness(blue_reflectance: pandas.Series, green_reflectance: pandas.Series,
                        red_reflectance: pandas.Series) -> pandas.Series:
    v_mv = tt.dvector('mean_vis')
    e_mv = (v0_blue + v0_green + v0_red) / 3
    ds_mv = theano.function([v0_blue, v0_green, v0_red], e_mv)(blue_reflectance, green_reflectance, red_reflectance)
    e2_whiteness = 1 - (tt.abs_((v0_blue - v_mv) / v_mv)
                        + tt.abs_((v0_green - v_mv) / v_mv)
                        + tt.abs_((v0_red - v_mv) / v_mv))
    return theano.function([v0_blue, v0_green, v0_red, v_mv], e2_whiteness)(blue_reflectance, green_reflectance,
                                                                            red_reflectance, ds_mv)


def calculate_perceptual_whiteness(blue_reflectance: pandas.Series, green_reflectance: pandas.Series,
                                   red_reflectance: pandas.Series) -> pandas.Series:
    v_mv = tt.dvector('mean_vis')
    e_mv = 0.2126 * v0_red + 0.7152 * v0_green + 0.0722 * v0_blue
    ds_mv = theano.function([v0_blue, v0_green, v0_red], e_mv)(blue_reflectance, green_reflectance, red_reflectance)
    e2_whiteness = 1 - (tt.abs_((v0_blue - v_mv) / v_mv)
                        + tt.abs_((v0_green - v_mv) / v_mv)
                        + tt.abs_((v0_red - v_mv) / v_mv))
    return theano.function([v0_blue, v0_green, v0_red, v_mv], e2_whiteness)(blue_reflectance, green_reflectance,
                                                                            red_reflectance, ds_mv)


def calculate_hot(blue_reflectance: pandas.Series, red_reflectance: pandas.Series) -> pandas.Series:
    e3_hot_test = v0_blue - 0.5 * v0_red
    return theano.function([v0_blue, v0_red], e3_hot_test)(blue_reflectance, red_reflectance)


def calculate_b4b5(nir_reflectance: pandas.Series, swir1_reflectance: pandas.Series) -> pandas.Series:
    e4_b4b5_test = v0_nir / v0_swir1
    return theano.function([v0_nir, v0_swir1], e4_b4b5_test)(nir_reflectance, swir1_reflectance)


def test_basic(swir2_reflectance: pandas.Series, tirs1_bt: pandas.Series, ndsi: pandas.Series, ndvi: pandas.Series):
    e1_basic_test = (1 / (1 + tt.exp(200 * (C1_MIN_BAND7_TOA_REFLECTANCE_OF_CLOUDS - v0_swir2)))
                     * 1 / (1 + tt.exp(200 * (v0_tirs1_bt - C1_MAX_BT_OF_CLOUDS)))
                     * 1 / (1 + tt.exp(200 * (v1_ndsi - C1_MAX_NDSI_OF_CLOUDS)))
                     * 1 / (1 + tt.exp(200 * (v1_ndvi - C1_MAX_NDVI_OF_CLOUDS))))
    return theano.function([v0_swir2, tirs1_bt, v1_ndsi, v1_ndvi],
                           e1_basic_test)(swir2_reflectance, tirs1_bt, ndsi, ndvi)


def calculate_water(nir_reflectance: pandas.Series, ndvi: pandas.Series):
    e_water = logistic_function(v1_ndvi, .2, 9) * logistic_function(v0_nir, 0.11, 9)
    return theano.function([v1_ndvi, v0_nir], e_water)(ndvi, nir_reflectance)


def calculate_pcp(basic: pandas.Series, whiteness: pandas.Series, hot: pandas.Series, b4b5: pandas.Series):
    e_pcp = v1 * v2 * v3 * v4
    return theano.function([v1, v2, v3, v4], e_pcp)(basic, whiteness, hot, b4b5)


def calculate_clearsky_water(water: pandas.Series, swir2_reflectance: pandas.Series) -> pandas.Series:
    e7_clearsky_water = v1 * logistic_function(v0_swir2, 0.03, 20)
    return theano.function([v1, v0_swir2], e7_clearsky_water)(water, swir2_reflectance)


def calculate_water_temperature(clearsky_water: pandas.Series, tirs1_bt: pandas.Series,
                                percentile_theshold=82.5) -> float:
    """
    This formulation is significantly different from that which is presented in the whitepaper: a necessity driven by
    the use of logistic functions instead of boolean masks. The idea behind the original formulation was that it masked
    out pixels that do not have clear skies over them. The pixels that remain have an unobstructed view of water. Even
    these pixels have atmospheric influences that make the water appear colder, therefore the upper level
    (82.5th percentile) is taken to exclude such influences.

    First, we seek to reduce the atmospheric influences on our final data by creating a set of weights. We use the
    minimum and maximum TIRS1 values to rescale the tirs1_bt data series from 0 to 1. Then, we find the 82.5th
    percentile of the series within that scaling. We want to create a logistic function that is near-zero at zero, and
    near-one at the percentile.

    We then take the weighted average of the tirs1_bt series, using two sets of weights: the set we just created, and
    the set from clearsky_water. The result is our estimated water temperature.
    """
    clarity_of_atmosphere = logistic_percentile(tirs1_bt, percentile_theshold)
    return doubly_weighted_average(tirs1_bt, clearsky_water, clarity_of_atmosphere)


def calculate_w_temperature_prob(water_temperature: float, tirs1_bt: pandas.Series,
                                 normalization_factor=4.0) -> pandas.Series:
    expr = (water_temperature - v0_tirs1_bt) / normalization_factor
    return theano.function([v0_tirs1_bt], expr)(tirs1_bt)


def calculate_brightness_prob(swir1_reflectance: pandas.Series, max_water_swir1_reflectance=0.11) -> pandas.Series:
    expr = tt.min(v0_swir1, max_water_swir1_reflectance) / max_water_swir1_reflectance
    return theano.function([v0_swir1], expr)(swir1_reflectance)


def calculate_w_cloud_prob(w_temperature_prob: pandas.Series, brightness_prob: pandas.Series) -> pandas.Series:
    return theano.function([v1, v2], v1 * v2)(w_temperature_prob, brightness_prob)


def calculate_clearsky_land(pcp: pandas.Series, water: pandas.Series) -> pandas.Series:
    expr = (1 - v1) * (1 - v2)
    return theano.function([v1, v2], expr)(pcp, water)


def calculate_land_temperature_prob(clearsky_land: pandas.Series, tirs1_bt: pandas.Series,
                     lower_percentile=17.5, upper_percentile=82.5) -> (float, float):
    t_lo = doubly_weighted_average(tirs1_bt, clearsky_land, logistic_percentile(tirs1_bt, lower_percentile))
    t_hi = doubly_weighted_average(tirs1_bt, clearsky_land, logistic_percentile(tirs1_bt, upper_percentile))
    expr = (t_hi + 4 - v0_tirs1_bt) / (t_hi + 4 - (t_lo - 4))
    return theano.function([v0_tirs1_bt], expr)(tirs1_bt)


# ###################### Auxillary functions #######################################################################


def logistic_function(lt: float, gt: float, width_factor=1.0):
    """
    An important note here is that one or both of lt and gt need to be a Theano variable representing a float.
    For example, suppose I want to make a comparison similar to 0.1 < SWIR1 < 0.5. I break that into two parts:
        p1 = 0.1 < SWIR1
        p2 = SWIR1 < 0.5
        p3 = p1 and p2
    which then becomes, in terms of logistic functions:
        v = tt.dvector()
        p1 = logistic_function(0.1, v)
        p2 = logistic_function(v, 0.5)
        p3 = p1 * p2
        p4 = theano.function([v], p3)(SWIR1)
    which can of course be shortened to one or two lines if desired. The result is a vector that is greater than 0.5
    approaching 1 when 0.1 < SWIR1 < 0.5, and is less than 0.5 approaching 0 when it is not.
    """
    return 1 / (1 + tt.exp(width_factor * (lt - gt)))


def logistic_percentile(s: pandas.Series, percentile: float, inverted=False, scaling_series=None) -> pandas.Series:
    """
    This creates a logistic function that is near-zero at the series' minimum, and near-one as values reach and exceed
    the given percentile in the series. It then returns the series evaluated according to this function.

    In other words, it creates a vector of weights that may be applied to the original series. These weights are in the
    interval [0,1]. At the lowest values in the series, the weight is at or near 0. As values increase, so too do the
    weights until, at or above the given percentile, the weight is at or near 1.

    Internal to the function, we can approximate this by setting the logistic function's threshold value to be
    half the percentile, and then determine its scaling factor with the hyperbolic function. We scale the series from
    zero to one so that this hyperbolic function has consistent results. We then evaluate the scaled series according
    to the logistic function.

    If inverted is set to True, then the graph is flipped. E.g. instead of f(x), 1-f(x).

    scaling_series is an optional parameter provided for convenience. If a pandas.Series is given, then that series
    is treated as a vector of scalars/weights/factors/whatever that is simply multiplied in after all else is done.
    """
    scaled = scale_from_zero_to_one(s)
    # noinspection PyTypeChecker
    percentile = numpy.percentile(scaled, percentile)
    expr = logistic_function(v1, percentile / 2, width_factor=(9.5 / percentile))
    expr = 1 - expr if inverted else expr
    if scaling_series is None:
        return theano.function([v1], expr)(scaled)
    else:
        return theano.function([v1, v2], expr * v2)(scaled, scaling_series)


def weighted_sum(s: pandas.Series, weights: pandas.Series) -> float:
    return theano.function([v1, v2], v1 * v2)(s, weights).sum()


def doubly_weighted_average(s: pandas.Series, weight_set1: pandas.Series, weight_set2: pandas.Series) -> float:
    total_weight = weighted_sum(weight_set1, weight_set2)
    return theano.function([v1, v2, v3], v1 * v2 * v3)(s, weight_set1, weight_set2).sum() / total_weight


def scale_from_zero_to_one(s: pandas.Series) -> pandas.Series:
    """
    This function returns a series parallel to the one given, scaled linearly such that its minimum value is now 0
    and its maximum value is now 1.
    """
    expr = (v1 - s.min()) / (s.max() - s.min())
    return theano.function([v1], expr)(s)


def probabilistic_or(p1, p2):
    return p1 + p2 - (p1 * p2)

