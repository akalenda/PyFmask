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
##########################################################################

import numpy
import pandas
import theano
import theano.tensor as tt

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
v0_bt1 = tt.dvector('v0_bt1')
v1_ndsi = tt.dvector('v1_ndsi')
v1_ndvi = tt.dvector('v1_ndvi')
v2_whiteness = tt.dvector('v2_whiteness')
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
                     * 1 / (1 + tt.exp(200 * (v0_bt1 - C1_MAX_BT_OF_CLOUDS)))
                     * 1 / (1 + tt.exp(200 * (v1_ndsi - C1_MAX_NDSI_OF_CLOUDS)))
                     * 1 / (1 + tt.exp(200 * (v1_ndvi - C1_MAX_NDVI_OF_CLOUDS))))
    return theano.function([v0_swir2, v0_bt1, v1_ndsi, v1_ndvi], e1_basic_test)(swir2_reflectance, tirs1_bt, ndsi, ndvi)


def logistic_function(lt: float, gt: float, width_factor=1.0):
    return 1 / (1 + tt.exp(width_factor * (lt - gt)))


def probabilistic_or(probability_a, probability_b):
    return probability_a + probability_b - probability_a * probability_b


def calculate_water(nir_reflectance: pandas.Series, ndvi: pandas.Series):
    land_vs_water = logistic_function(v1_ndvi, .2, 9) * logistic_function(v0_nir, 0.11, 9)
    cloud_vs_water = logistic_function(v1_ndvi, .3, 9) * logistic_function(v0_nir, 0.05, 9)
    e_water = probabilistic_or(land_vs_water, cloud_vs_water)
    return theano.function([v1_ndvi, v0_nir], e_water)(ndvi, nir_reflectance)


def test_clearsky_water(water: pandas.Series, swir2_reflectance: pandas.Series) -> pandas.Series:
    e7_clearsky_water = v1 * logistic_function(v0_swir2, 0.03, 20)
    return theano.function([v1, v0_swir2], e7_clearsky_water)(water, swir2_reflectance)


def calculate_w_temperature_prob(swir1_reflectance_of_clearskies_over_water: pandas.Series,
                                 tirs1_bt: pandas.Series) -> pandas.Series:
    # noinspection PyTypeChecker
    c8_t_water = numpy.percentile(swir1_reflectance_of_clearskies_over_water,
                                  C8_PERCENTILE_FOR_CLEARSKY_WATER_TEMPERATURE)
    e9_w_temperature_prob = (c8_t_water - v0_bt1) / 4
    return theano.function([v0_bt1], e9_w_temperature_prob)(tirs1_bt)


def calculate_pcp(basic: pandas.Series, whiteness: pandas.Series, hot: pandas.Series, b4b5: pandas.Series):
    e_pcp = v1 * v2 * v3 * v4
    return theano.function([v1, v2, v3, v4], e_pcp)(basic, whiteness, hot, b4b5)


def fmask(df: pandas.DataFrame) -> pandas.DataFrame:
    # ############################### DataFrame column aliases ##########################################
    df_swir1 = df['band6_reflectance_corrected']
    df_cirrus = df['band9_reflectance_corrected']
    df_bt1 = df['band10_bt']

    # ################ Formulae from [Zhu 2012] #######################################################
    """
    This test cuts out pixels that are clearly just snow or vegetation, or that are too warm to be clouds.
    """
    print('Formula1')

    """
    Formula 5 from [Zhu 2012] is split into three parts, as each may be useful in its own right.
    This one is true if the pixel suggests thin clouds over water.
    """
    print('Formula5a')

    """
    This one suggests clear skies over water if true.
    """
    print('Formula5b')

    """
    This one evaluates to True if it is definitely water, either with clear skies or thin cloud. False if it is land, 
    thick clouds over land, or thick clouds over water.
    """
    print('Formula5c')

    """
    This test produces true values for pixels that have a high probability of being cloud.
    It labels it as a Potential Cloud Pixel (PCP).
    """
    print('Formula6')
    df['pcp'] = df['basic_test'] & df['whiteness_test'] & df['hot_test'] & df['b4b5_test']

    """
    This further refines the Water Test of formula 5 to take advantage of the newer, second short-wave infrared band.
    """
    # TODO: Shouldn't this just be folded into the original test then?
    print('Formula7')

    """
    For pixels which are water under clear skies, estimate the temperature
    """
    print('Formula8')
    # TODO: What if all the water is under clouds? What if there's no water at all?
    # noinspection PyTypeChecker

    """
    """
    print('Formula10')
    e10_brightness_prob = tt.clip(v0_swir1 / C10_MAX_WATER_REFLECTANCE, -999999, 1.0)
    df['brightness_prob'] = theano.function([v0_swir1], e10_brightness_prob)(df_swir1)

    """
    From [Zhu, 2015]
    This uses the cirrus cloud band 9 to account for high-altitude clouds.
    See: https://landsat.usgs.gov/how-is-landsat-8s-cirrus-band-9-used
    """
    print('2015, Formula1')
    e20151_cirrus_cloud_probability = v0_cirrus / C20151_CIRRUS_REFLECTANCE_THRESHOLD
    df['cirrus_cloud_probability'] = theano.function([v0_cirrus], e20151_cirrus_cloud_probability)(df_cirrus)

    """
    """
    print('Formula11 replaced by 2015 Formula 2')
    e11_w_cloud_prob = v10_brightness_prob + v20151_cirrus_cloud_probability
    df['w_cloud_prob'] = theano.function([v10_brightness_prob, v20151_cirrus_cloud_probability],
                                         e11_w_cloud_prob)(df['brightness_prob'], df['cirrus_cloud_probability'])

    """
    """
    print('Formula12')
    df['clearsky_land'] = ~df['pcp'] & ~df['water_test']

    """
    """
    print('Formula13')
    df13_clearskyland = df[df['clearsky_land']]
    df13_clearskyland_bt = df13_clearskyland['band6_reflectance_corrected']
    # noinspection PyTypeChecker
    c13_t_lo = numpy.percentile(df13_clearskyland_bt, C13_LOWER_PERCENTILE_FOR_CLEARSKY_LAND)
    # noinspection PyTypeChecker
    c13_t_hi = numpy.percentile(df13_clearskyland_bt, C13_UPPER_PERCENTILE_FOR_CLEARSKY_LAND)

    """
    """
    print('Formula14')
    c14_temperature_magnitude = c13_t_hi - c13_t_lo
    e14_l_temperature_prob = (c13_t_hi + 4 - v0_bt1) / c14_temperature_magnitude
    df['l_temperature_prob'] = theano.function([v0_bt1], e14_l_temperature_prob)(df_bt1)

    """
    """
    print("Formula15")
    # TODO: The whitepaper explanation is weird about this one. It's talking about saturation of one band, and another
    # band being larger than the other... but I think it's basically just saying that negative values for ndvi and
    # ndsi are cropped to zero. At which point the absolute values don't do anything. And we don't even need to modify
    # the ndsi/ndvi values, we can just make zero a minimum for our max function. Is that right???
    e15_variability_prob = (tt.max([0, v1_ndvi, v1_ndsi, v2_whiteness]))
    df['variability_prob'] = theano.function([v1_ndvi, v1_ndsi, v2_whiteness],
                                             e15_variability_prob)(df['ndvi'], df['ndsi'], df['whiteness'])

    """
    """
    print("Formula16")
    e16_l_cloud_prob = v14_l_temperature_prob * v15_variability_prob
    df['l_cloud_prob'] = theano.function([v14_l_temperature_prob, v15_variability_prob],
                                         e16_l_cloud_prob)(df['l_temperature_prob'], df['variability_prob'])

    return df