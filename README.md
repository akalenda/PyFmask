# PyFmask

The code in this repository is very much work-in-progress, very 
experimental. That doesn't mean you shouldn't use it,
though! We encourage you to give it a try. 
If you have a use for FMask, we want to hear from you.
Any problems, stumbling blocks, or questions can be brought up as a 
[Github Issue](https://github.com/akalenda/PyFmask/issues) that 
may well focus our efforts and lead to improvement of the library.

## Quick Start

You will need the following installed:

- [Python3](https://www.python.org/downloads/)
- [Theano](http://deeplearning.net/software/theano/install.html)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)
- [Rasterio](https://github.com/mapbox/rasterio)

We do not yet have a proper Python package. You will need to download the repository
and localize it to your current working directory. I.e:

    git clone https://github.com/akalenda/PyFmask.git
    cd PyFmask
    python3

Its use is flexible, but to get started quickly use `Landsat8Scene.py`'s convenience methods:

    from Landsat8Scene import Landsat8Scene
    from Landsat8Scene import EXAMPLE_SCENE_IDS 
    for scene in EXAMPLE_SCENE_IDS:
        (LandsatScene(scene)
         .download_scene_from_aws(will_overwrite=False)
         .dataframe_generate()
         .dataframe_drop_dead_pixels()
         .calculate_fmask_inputs()
         .calculate_fmask_outputs()
         .dataframe_write_series_to_geotiff('ndvi')
         .dataframe_write_series_to_geotiff('water'))

## Background

[Landsat8](https://landsat.gsfc.nasa.gov/landsat-8/) is a [remote sensing](https://en.wikipedia.org/wiki/Remote_sensing) 
platform that orbits the Earth. It scans images of the surface in 
[twelve spectral bands](https://landsat.gsfc.nasa.gov/landsat-8/landsat-8-bands/). These images are made available
through the [U.S. Geological Survey](https://landsat.usgs.gov/frequently-asked), as well as through a public
[S3 bucket](https://aws.amazon.com/s3/) on Amazon Web Services: 
[Landsat on AWS](https://aws.amazon.com/public-datasets/landsat/).

Remote sensing data is useful for many applications. [Google Earth](https://www.google.com/earth/), 
[MapBox](https://www.mapbox.com/), and [Libra](https://libra.developmentseed.org/) are a few interesting examples.
Other applications may be to use artificial intelligence, machine learning, and other tools to search for interesting
developments such as construction, deforestation, erosion, landfills, and so on.

## What is FMask? How does it fit in?

Clouds often obscure the surface phenomenae scientists are interested in. They can also introduce errors into machine
learning algorithms if not accounted for. Therefore, it is desirable to remove clouds from the picture -- or at the 
very least ignore them.

FMask can be thought of as an [expert system](https://en.wikipedia.org/wiki/Expert_system) used to identify where
clouds are in a Landsat picture. It was developed using empirical observations of clouds in remote sensing. Its
creation is described in two papers:

- Zhu, Zhe, Shixiong Wang, and Curtis E. Woodcock. 
"Improvement and expansion of the Fmask algorithm: cloud, cloud shadow, and snow detection for Landsats 4–7, 8, 
and Sentinel 2 images." Remote Sensing of Environment 159 (2015): 269-277.

- Zhu, Zhe, and Curtis E. Woodcock. "Object-based cloud and cloud shadow detection in Landsat imagery." 
Remote Sensing of Environment 118 (2012): 83-94.

Implementations of FMask already exist. Its authors provide a 
[MatLab implementation and standalone binaries](https://github.com/prs021/fmask). There is also a 
[python-fmask](http://pythonfmask.org/en/latest/), and a 
[C implementation](https://github.com/USGS-EROS/espa-cloud-masking). 

However, for various reasons these
implementations do not fulfill our needs, as will be explained in a future update to this readme.

