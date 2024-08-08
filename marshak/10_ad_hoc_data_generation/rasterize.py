import numpy as np
from rasterio import features

def rasterize_shapes_to_array(shapes: list,
                              attributes: list,
                              profile: dict,
                              all_touched: bool = False,
                              dtype: str = np.float32) -> np.ndarray:
    """
    Takes a list of geometries and attributes to create an array. Roughly the
    inverse, in spirit, to `get_geopandas_features_from_array`.  For example,
    `shapes = df.geometry` and `attributes = df.label`, where df is a geopandas
    GeoDataFrame. We note the array is initialized as array of zeros.

    Parameters
    ----------
    shapes : list
        List of Shapely geometries.
    attributes : list
        List of attributes corresponding to shapes.
    profile : dict
        Rasterio profile in which shapes will be projected into, importantly
        the transform and dimensions specified.
    all_touched : bool
        Whether factionally covered pixels are written with specific value or
        ignored. See `rasterio.features.rasterize`.
    dtype : str
        The initial array is np.zeros and dtype can be specified as a numpy
        dtype or appropriate string.

    Returns
    -------
    np.ndarray:
        The output array determined with profile.
    """
    out_arr = np.zeros((profile['height'], profile['width']), dtype=dtype)

    # this is where we create a generator of geom, value pairs to use in
    # rasterizing
    shapes = [(geom, value) for geom, value in zip(shapes, attributes)]
    burned = features.rasterize(shapes=shapes,
                                out=out_arr,
                                transform=profile['transform'],
                                all_touched=all_touched)

    return burned