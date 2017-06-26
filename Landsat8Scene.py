import re
from os import makedirs, path
from time import strptime

import numpy
import pandas
import rasterio
import theano
import urllib3
from theano import tensor as tt

import fmask
from progressbar import print_progress_bar

# ####### Theano variables ################
_v_pixel_value = tt.ivector('pixelVal')
_v_band_scaling_multiplier = tt.dscalar('bandMult')
_v_band_scaling_additive = tt.dscalar('bandAdd')
_v_toa_uncorrected_reflectance = tt.dvector('toaCalc')
_v_solar_zenith_angle = tt.dscalar('zenith')
_v_solar_elevation_angle = tt.dscalar('solarElev')
_v_k1 = tt.dscalar('k1')
_v_k2 = tt.dscalar('k2')
_v_toa_radiance = tt.dvector('radiance')

# ####### Theano expressions ################
_e_conversion_to_toa = _v_band_scaling_multiplier * _v_pixel_value + _v_band_scaling_additive
_e_correction_for_sun_angle = _v_toa_uncorrected_reflectance / tt.cos(_v_solar_zenith_angle)
_e_conversion_to_bt = _v_k2 / tt.log(_v_k1 / _v_toa_radiance + 1)  # tt.log is the natural log

# ####### Theano functions ################
_f_toa_uncorrected = theano.function([_v_band_scaling_multiplier, _v_pixel_value, _v_band_scaling_additive],
                                     _e_conversion_to_toa)
_f_toa_corrected = theano.function([_v_toa_uncorrected_reflectance, _v_solar_zenith_angle], _e_correction_for_sun_angle)
_f_brightness_temp = theano.function([_v_k1, _v_k2, _v_toa_radiance], _e_conversion_to_bt)


class LandsatScene:
    """
    This class is largely concerned with fetching Landsat satellite imagery, extracting their raw data, and presenting
    them as Pandas DataFrames that can be easily worked with in a computation pipeline.

    See: https://landsat.usgs.gov/
    See: https://aws.amazon.com/public-datasets/landsat/
    """

    EXAMPLE_SCENE_IDS = [
        'LC80440342014077LGN00',
        'LC81400472015145LGN00'
    ]
    DATA_DIRECTORY = "./data/"

    """
    See: https://landsat.usgs.gov/what-are-band-designations-landsat-satellites
    """
    LANDSAT8_BANDS = [
        1,  # VIS: Visible spectrum, ultra blue (coastal/aerosol)
        2,  # VIS: Visible spectrum, blue
        3,  # VIS: Visible spectrum, green
        4,  # VIS: Visible spectrum, red
        5,  # NIR: Near infrared
        6,  # SWIR 1: Shortwave infrared
        7,  # SWIR 2: Shortwave infrared
        # 8,  # Panchromatic -- ignored for now, as it is of different length
        9,  # Cirrus
        10,  # TIRS 1: Thermal infrared
        11,  # TIRS 2: Thermal infrared
    ]

    def __init__(self, scene_id: str):
        """
        :param scene_id: Corresponds to UTM coordinates with date-time information. Example: 'LC81400472015273LGN00'
            @see https://landsat.usgs.gov/what-are-naming-conventions-landsat-scene-identifiers
            It is assumed that the GeoTIFFs for this scene are in a directory under the same name.
            E.g.: './<scene_id>/<scene_id>_B1.TIF'
        """
        self.scene_id = scene_id
        self.scene_id_breakdown = {
            'sensor': self.scene_id[1],
            'satellite': self.scene_id[2],
            'wrs_path': self.scene_id[3:6],
            'wrs_row': self.scene_id[6:9],
            'year': self.scene_id[9:13],
            'julian_day_of_year': self.scene_id[13:16],
            'ground_station_id': self.scene_id[16:19],
            'archive_version_number': self.scene_id[19:21]
        }
        self.derived_datetime = strptime(self.scene_id[9:16], '%Y%j')
        self.dataframe = None
        self._metadata = None
        self._profile = None

    def download_scene_from(self, url: str, will_overwrite=None, http_pool_manager=urllib3.PoolManager()):
        # TODO: Batch download version
        if self.scene_appears_downloaded():
            if will_overwrite is False:
                return self
            if will_overwrite is None:
                print("Scene {} already has a directory, which suggests it has been downloaded. Replace? ")
                if input("Replace (Y/n)? ").strip()[0] != 'Y':
                    print("Aborting download")
                    return self
        response_text = http_pool_manager.request('GET', url + 'index.html', preload_content=False).data.decode('utf-8')
        metadata_filename = re.search('"L.*_MTL.txt"', response_text).group()[1:-1]
        geotiff_filenames = [match.group()[1:-1] for match in re.finditer('"L.*.TIF"', response_text)]
        n = len(geotiff_filenames)
        print_progress_bar(0, n, 'Downloading' + self.scene_id)
        for i in range(n):
            self._download(url, geotiff_filenames[i], http_pool_manager)
            print_progress_bar(i, n, 'Downloading ' + self.scene_id)
        self._download(url, metadata_filename, http_pool_manager)
        print_progress_bar(n, n, 'Downloading ' + self.scene_id)
        return self

    def _download(self, remote_directory: str, filename: str, http_pool_manager: urllib3.PoolManager):
        """
        Helper method for download_scene_from
        Retrieves files from the remote directory and transfers them to a local directory
        :return: self
        """
        local_directory = '{0}{1}/'.format(self.DATA_DIRECTORY, self.scene_id)
        local_url = local_directory + filename
        remote_url = remote_directory + filename
        data = http_pool_manager.request('GET', remote_url, preload_content=False).data
        if not path.exists(local_directory):
            makedirs(local_directory)
        with open(local_url, 'wb') as download_target:
            download_target.write(data)
        return self

    def download_scene_from_aws(self, will_overwrite=None):
        """
        Attempts to download the data we need for this scene from Amazon Web Services
        :return: The return value of _download_scene_from method
        """
        return self.download_scene_from(self.get_aws_directory_url(), will_overwrite=will_overwrite)

    def dataframe_generate(self, bands=LANDSAT8_BANDS):
        """
        Creates a DataFrame, in which each row corresponds to a pixel in the scene, and each column corresponds to a
        band. The index can be used to compute a pixel's x-y coordinates within the image, which can then be used
        to compute its global spatial coordinates if needed.
            This dataframe can later be retrieved from self.dataframe
        :param bands: A list of the bands to load. The default bands are 1 through 7.
        :return: self
        """
        # TODO: Add support for band 8. And any other bands?
        if self.dataframe is not None:
            return self.dataframe
        if self._metadata is None:
            self._get_metadata()
        df = None
        print_progress_bar(0, len(bands), "Loading {} to dataframe".format(self.scene_id), length=50)
        for band_number in self.LANDSAT8_BANDS:
            file_path = '{0}{1}/{1}_B{2}.TIF'.format(self.DATA_DIRECTORY, self.scene_id, band_number)
            with rasterio.open(file_path) as image:
                self._profile = image.profile
                if df is None:
                    df = pandas.DataFrame(index=numpy.arange(image.width * image.height))
                df['band%i' % band_number] = image.read().flatten()
            print_progress_bar(band_number, len(bands), "Loading {} to dataframe:".format(self.scene_id), length=50)
        self.dataframe = df
        return self

    def dataframe_drop_dead_pixels(self):
        """
        Invokes to_dataframe method, with the addition that it removes from the dataframe any pixels which are purely
         zero across all bands. This should be because the pixels are outside of the satellite camera's field of view:
         A hypothesis we should likely test at some point.
        """
        self.dataframe = self.dataframe.loc[(self.dataframe != 0).any(1)]
        return self

    def write_fmask_outputs(self):
        ndvi_series = fmask.calculate_ndvi(self._read_series_from_csv('band5reflectance'),
                                           self._read_series_from_csv('band4reflectance'))
        self._write_series_to_csv(ndvi_series, 'ndvi')

    def write_fmask_inputs(self):
        """
        The GeoTIFFs we read have uint16 values, not in the units we want. But each scene is accompanied by a
        metadata file that has values -- some global, some for specific bands -- that we can use to convert the pixel
        values into floating point values of the appropriate units. Therefore, typically when using the data, you would
        not want to use for example the "band2" data series, but instead the "band2_reflectance_corrected" series.

        See: https://landsat.usgs.gov/using-usgs-landsat-8-product
        :return: self
        """
        solar_zenith_angle = float(self._metadata['SUN_AZIMUTH'])
        for df in self._generate_dataframe_chunks():
            for band_number in self.LANDSAT8_BANDS:
                bn = 'band' + str(band_number)
                try:
                    radiance_mult = float(self._metadata['RADIANCE_MULT_BAND_%i' % band_number])
                    radiance_add = float(self._metadata['RADIANCE_ADD_BAND_%i' % band_number])
                    series_radiance = _f_toa_uncorrected(radiance_mult, df[bn], radiance_add)
                    self._append_series_to_csv(series_radiance, bn + 'radiance')
                    try:
                        k1 = float(self._metadata['K1_CONSTANT_BAND_%i' % band_number])
                        k2 = float(self._metadata['K2_CONSTANT_BAND_%i' % band_number])
                        self._append_series_to_csv(_f_brightness_temp(k1, k2, series_radiance), bn + 'bt')
                    except KeyError as _:
                        pass
                    del series_radiance
                except KeyError as _:
                    pass
                try:
                    reflectance_mult = float(self._metadata['REFLECTANCE_MULT_BAND_%i' % band_number])
                    reflectance_add = float(self._metadata['REFLECTANCE_ADD_BAND_%i' % band_number])
                    series_reflect_uncorrected = _f_toa_uncorrected(reflectance_mult, df[bn], reflectance_add)
                    series_reflect_corrected = _f_toa_corrected(series_reflect_uncorrected, solar_zenith_angle)
                    self._append_series_to_csv(series_reflect_corrected, bn + 'reflectance')
                    del series_reflect_uncorrected, series_reflect_corrected
                except KeyError as _:
                    pass
        return self

    # ################################### HELPERS ##############################################

    def _generate_dataframe_chunks(self, chunk_size=100 * 1000) -> pandas.DataFrame:
        l = 0
        i = 0
        num_chunks = int(numpy.math.ceil(len(self.dataframe) / chunk_size))
        print()
        while l < len(self.dataframe):
            print_progress_bar(i, num_chunks, prefix="Processing image into CSVs: ")
            r = l + chunk_size
            yield self.dataframe[l:r]
            l = r
        print_progress_bar(i, num_chunks, prefix="Processing image into CSVs: ")

    def _get_metadata(self):
        metadata = dict()
        with open('{0}{1}/{1}_MTL.txt'.format(self.DATA_DIRECTORY, self.scene_id)) as metadata_file:
            for line in metadata_file.readlines():
                tokens = line.split("=")
                if len(tokens) > 1:
                    key, value = tokens[0].strip(), tokens[1].strip()
                    if key != "GROUP" and key != "END":
                        metadata[key] = value
        self._metadata = metadata
        return metadata

    def get_local_directory_url(self) -> str:
        """
        :return: The directory on the local machine where we expect to find the data for this LandsatScene
        """
        return '{0}{1}/'.format(self.DATA_DIRECTORY, self.scene_id)

    def get_aws_directory_url(self) -> str:
        """
        Calculated from the scene_id given to constructor
        :return: The directory on the remote Amazon Web Services S3 Bucket whence we may expect to download
        """
        return 'http://landsat-pds.s3.amazonaws.com/L{0}/{1}/{2}/{3}/'.format(
            self.scene_id_breakdown['satellite'],
            self.scene_id_breakdown['wrs_path'],
            self.scene_id_breakdown['wrs_row'],
            self.scene_id
        )

    def scene_appears_downloaded(self) -> bool:
        if not path.exists(self.DATA_DIRECTORY):
            makedirs(self.DATA_DIRECTORY)
            return False
        return path.exists('{0}{1}/'.format(self.DATA_DIRECTORY, self.scene_id))

    def _write_series_to_csv(self, ds: pandas.Series, filename: str):
        ds.to_csv(path='{0}{1}.csv'.format(self.get_local_directory_url(), filename), mode='w', index=True)

    def _append_series_to_csv(self, ds: pandas.Series, filename: str):
        pandas.Series(ds).to_csv(path='{0}{1}.csv'.format(self.get_local_directory_url(), filename),
                                 mode='a', index=True)

    def _read_series_from_csv(self, filename: str):
        return pandas.Series.from_csv(path='{0}{1}.csv'.format(self.get_local_directory_url(), filename))


# ######################### EXAMPLE USAGE ######################################
if __name__ == '__main__':
    print(LandsatScene(LandsatScene.EXAMPLE_SCENE_IDS[1])
          .download_scene_from_aws(will_overwrite=False)
          .dataframe_generate()
          # .dataframe_drop_dead_pixels()
          .write_fmask_inputs()
          .write_fmask_outputs())
