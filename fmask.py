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

from Landsat8Scene import LandsatScene

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
v2_mean_vis = tt.dvector('v2_mean_vis')
v2_whiteness = tt.dvector('v2_whiteness')
v10_brightness_prob = tt.dvector('v10_brightness_prob')
v20151_cirrus_cloud_probability = tt.dvector('v0_cirrus')
v14_l_temperature_prob = tt.dvector('v14_l_temperature_prob')
v15_variability_prob = tt.dvector('v15_variability_prob')

# ############################### Theano Expressions ######################################
e1_ndsi = (v0_green - v0_swir1) / (v0_green + v0_swir1)
e1_basic_test = tt.and_(
    tt.and_(
        tt.lt(C1_MIN_BAND7_TOA_REFLECTANCE_OF_CLOUDS, v0_swir2),
        tt.lt(v0_bt1, C1_MAX_BT_OF_CLOUDS)
    ),
    tt.and_(
        tt.lt(v1_ndsi, C1_MAX_NDSI_OF_CLOUDS),
        tt.lt(v1_ndvi, C1_MAX_NDVI_OF_CLOUDS)
    )
)


def fmask(df: pandas.DataFrame) -> pandas.DataFrame:
    # ############################### DataFrame column aliases ##########################################
    df_blue = df['band2_reflectance_corrected']
    df_green = df['band3_reflectance_corrected']
    df_red = df['band4_reflectance_corrected']
    df_nir = df['band5_reflectance_corrected']
    df_swir1 = df['band6_reflectance_corrected']
    df_swir2 = df['band7_reflectance_corrected']
    df_cirrus = df['band9_reflectance_corrected']
    df_bt1 = df['band10_bt']
    df_bt2 = df['band11_bt']
    # TODO: Currently we are not using bt2. The original fmask used a single TIRS band for brightness temperature,
    # whereas Landsat8 provides two bands that collectively cover a wider spectrum. We could probably average the two
    # and use that with the original formulae. But since the two bands provide greater resolution, we should look into
    # how those two bands respond differently to various surfaces (clouds, water, land, vegetation, rocks, etc).

    # ################ Formulae from [Zhu 2012] #######################################################
    """
    This test cuts out pixels that are clearly just snow or vegetation, or that are too warm to be clouds.
    """
    print('Formula1')
    df['ndsi'] = theano.function([v0_green, v0_swir1], e1_ndsi)(df_green, df_swir1)
    df['basic_test'] = theano.function([v0_swir2, v0_bt1, v1_ndsi, v1_ndvi], e1_basic_test)(df_swir2, df_bt1,
                                                                                            df['ndsi'], df['ndvi'])



    """
    This test cuts out pixels that have too much color saturation (e.g. they are not white). The idea is
    that their blue/yellow/red values should be fairly close to one another.
    """
    print('Formula2')
    e2_mean_vis = (v0_blue + v0_green + v0_red) / 3
    e2_whiteness = (tt.abs_(v0_blue - v2_mean_vis) / v2_mean_vis + tt.abs_(v0_green - v2_mean_vis) / v2_mean_vis
                    + tt.abs_(v0_red - v2_mean_vis) / v2_mean_vis)
    df['mean_vis'] = theano.function([v0_blue, v0_green, v0_red], e2_mean_vis)(df_blue, df_green, df_red)
    df['whiteness'] = theano.function([v0_blue, v0_green, v0_red, v2_mean_vis],
                                      e2_whiteness)(df_blue, df_green, df_red, df['mean_vis'])

    """
    This test identifies thin cloud or haze.
    Blue light scatters more efficiently these substances. Therefore, if there is significantly more red
    than blue in the pixel, then we can eliminate the possibility of it being cloud. This may include false
    positives for rocks, turbid water, snow or ice.
    """
    print('Formula3')
    e3_hot_test = 2.0 * v0_blue - v0_red > C3_HOT_OFFSET
    df['hot_test'] = theano.function([v0_blue, v0_red], e3_hot_test)(df_blue, df_red)

    """
    This test cuts out objects that may look like thin cloud or haze in prior tests, but are in fact simply bright
    rocks. It may still include some clear-sky pixels.
    """
    print('Formula4')
    e4_b4b5_test = v0_nir > 0.75 * v0_swir1
    df['b4b5_test'] = theano.function([v0_nir, v0_swir1], e4_b4b5_test)(df_nir, df_swir1)

    """
    Formula 5 from [Zhu 2012] is split into three parts, as each may be useful in its own right.
    This one is true if the pixel suggests thin clouds over water.
    """
    print('Formula5a')
    e5_is_thincloud_or_turbidwater = tt.and_(v1_ndvi < NDVI_WATER_LAND_THRESHOLD, v0_nir < 0.05)
    df['is_thincloud_or_turbidwater'] = \
        theano.function([v1_ndvi, v0_nir], e5_is_thincloud_or_turbidwater)(df['ndvi'], df_nir)

    """
    This one suggests clear skies over water if true.
    """
    print('Formula5b')
    e5_is_clearsky_water = tt.and_(v1_ndvi < 0.01, v0_nir < 0.11)
    df['is_clearsky_water'] = theano.function([v1_ndvi, v0_nir], e5_is_clearsky_water)(df['ndvi'], df_nir)

    """
    This one evaluates to True if it is definitely water, either with clear skies or thin cloud. False if it is land, 
    thick clouds over land, or thick clouds over water.
    """
    print('Formula5c')
    df['water_test'] = df['is_thincloud_or_turbidwater'] | df['is_clearsky_water']

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
    e7_clearsky_water = v0_swir2 < C7_MIN_SWIR_OF_CLOUD
    df['clearsky_water'] = df['water_test'] & theano.function([v0_swir2], e7_clearsky_water)(df_swir2)

    """
    For pixels which are water under clear skies, estimate the temperature
    """
    print('Formula8')
    # TODO: What if all the water is under clouds? What if there's no water at all?
    # noinspection PyTypeChecker
    c8_t_water = numpy.percentile(df[df['clearsky_water']]['band6_reflectance_corrected'],
                                  C8_PERCENTILE_FOR_CLEARSKY_WATER_TEMPERATURE)

    """
    """
    print('Formula9')
    e9_w_temperature_prob = (c8_t_water - v0_bt1) / 4
    df['w_temperature_prob'] = theano.function([v0_bt1], e9_w_temperature_prob)(df_bt1)

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

    """
    """
    print("Formula17")
    df
    c17_land_threshold = numpy.percentile()

    return df


# ######################### EXAMPLE USAGE ######################################
if __name__ == '__main__':
    print(fmask(LandsatScene(LandsatScene.EXAMPLE_SCENE_ID)
                .download_scene_from_aws(will_overwrite=False)
                .dataframe_generate()
                .dataframe_drop_dead_pixels()
                .dataframe_adjust_for_metadata()
                .dataframe))


def calculate_ndvi(nir_reflectance: pandas.Series, red_reflectance: pandas.Series) -> pandas.Series:
    e1_ndvi = (v0_nir - v0_red) / (v0_nir + v0_red)
    return theano.function([v0_nir, v0_red], e1_ndvi)(nir_reflectance, red_reflectance)


def convert_to_uint(data: pandas.Series, min_datum: float, max_datum: float) -> pandas.Series:
    v_data = tt.dvector('v_data')
    e_convert_to_uint = v_data / max_datum * (max_datum - min_datum) + min_datum
    return theano.function([v_data], e_convert_to_uint)(data).astype('uint16')
