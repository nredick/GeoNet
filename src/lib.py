import itertools
import time

import alphashape
import cv2
import fiona as fio
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject
from sklearn import preprocessing
from tqdm import tqdm


def filter_images_with_mask(X, Y):
    filtered_X = []
    filtered_Y = []

    for i in range(len(X)):
        if np.any(Y[i] != 0):  # Check if any pixel in the mask is non-zero
            filtered_X.append(X[i])
            filtered_Y.append(Y[i])

    return filtered_X, filtered_Y


def normalize_images(images):
    normalized_images = np.zeros_like(images, dtype=np.float32)
    num_images, _, _, channels = images.shape

    # Reshape the images to a 2D array for vectorized normalization
    reshaped_images = images.reshape(num_images, -1)

    # Perform per-channel normalization to [0, 1] range
    for c in range(channels):
        channel = images[:, :, :, c]
        normalized_channel = cv2.normalize(
            channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        normalized_images[:, :, :, c] = normalized_channel

    return normalized_images


# define a function to unpack a generator
def unpack_gen(gen):
    stack = []  # create an empty list to store the data
    while True:  # loop until the generator is exhausted
        try:
            stack.append(next(gen))  # append the next batch (1) to the list
        except StopIteration:
            break
    return np.stack(stack, axis=1)  # stack the list of batches into a single array


def img_reshape(arr, tile_size):
    # Check if the array is square
    if arr.shape[1] == arr.shape[2]:
        # Reshape the array
        new_shape = (tile_size, tile_size, -1)
        return arr.reshape(new_shape)
    else:
        # Pad the array to make it square
        max_dim = max(arr.shape[1], arr.shape[2])
        pad_width = ((0, 0), (0, max_dim - arr.shape[1]), (0, max_dim - arr.shape[2]))
        padded_arr = np.pad(arr, pad_width, mode="constant")
        # Reshape the padded array
        new_shape = (tile_size, tile_size, -1)
        return padded_arr.reshape(new_shape)


def reproject_raster(in_path, out_path, to_crs):
    # reproject raster to project crs
    with rio.open(in_path) as src:
        if src.crs == to_crs:
            print(f"{time.ctime()}: {in_path} is already in target CRS.")
            return in_path

        src_crs = src.crs
        transform, width, height = calculate_default_transform(
            src_crs, to_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()

        kwargs.update(
            {"crs": to_crs, "transform": transform, "width": width, "height": height}
        )

        with rio.open(out_path, "w", **kwargs) as dst:
            for i in tqdm(range(1, src.count + 1)):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=to_crs,
                    resampling=Resampling.nearest,
                )

    return out_path


def encode(data_path, feature):
    # create a numeric unique value for each attribute/feature in the data feature
    # vector = data.copy()

    data = gpd.read_file(data_path)
    data.columns = map(str.lower, data.columns)

    data = data.dropna(subset=["geometry"])

    feature = feature.lower()

    le = preprocessing.LabelEncoder()
    le.fit(data[feature])

    data[f"{feature}_encoded"] = le.transform(data[feature])
    data[f"{feature}_encoded"] += 1

    data.to_file(data_path)


def get_windows(window_shape, image_shape):
    win_rows, win_cols = window_shape
    img_rows, img_cols = image_shape
    offsets = itertools.product(
        range(0, img_cols, win_cols), range(0, img_rows, win_rows)
    )
    image_window = rio.windows.Window(
        col_off=0, row_off=0, width=img_cols, height=img_rows
    )

    for col_off, row_off in offsets:
        window = rio.windows.Window(
            col_off=col_off, row_off=row_off, width=win_cols, height=win_rows
        )

        yield window.intersection(image_window)


def to_raster(
    data_path,
    output_path,
    feature_id,
    pixel_size,
    dtype="float64",
    windows_shape=(1024, 1024),
):
    if feature_id:
        encode(data_path, feature_id)

    with fio.open(data_path) as features:
        crs = features.crs
        xmin, ymin, xmax, ymax = features.bounds
        transform = rio.Affine.from_gdal(xmin, pixel_size, 0, ymax, 0, -pixel_size)
        out_shape = (int((ymax - ymin) / pixel_size), int((xmax - xmin) / pixel_size))

        with rio.open(
            output_path,
            "w",
            height=out_shape[0],
            width=out_shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            tiled=True,
            options=["COMPRESS=LZW"],
        ) as raster:
            for window in get_windows(windows_shape, out_shape):
                window_transform = rio.windows.transform(window, transform)
                # can be smaller than windows_shape at the edges
                window_shape = (window.height, window.width)
                window_data = np.zeros(window_shape)

                for feature in features:
                    value = feature["properties"][f"{feature_id}_encoded"]
                    geom = feature["geometry"]
                    d = rasterize(
                        [(geom, value)],
                        all_touched=False,
                        out_shape=window_shape,
                        transform=window_transform,
                    )
                    window_data += d  # sum values up

                raster.write(window_data, window=window, indexes=1)


def closestDivisibleNumber(n, m):
    if n % m == 0:
        return n

    return n - (n % m)


def crop_center(img, cropx, cropy):
    threeD = False
    try:
        _, x, y = img.shape
        threeD = True
    except:
        x, y = img.shape

    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2

    if threeD:
        return img[:, startx : startx + cropx, starty : starty + cropy]

    return img[startx : startx + cropx, starty : starty + cropy]


def concave_hull(dataframe, degree=0.001):
    """Create a single concave hull of an input GeoPandas DataFrame"""
    flat_list = []

    # Iterate over each geometry in the DataFrame
    for geom in dataframe["geometry"]:
        # Check if the geometry is a MultiPolygon
        if geom.geom_type == "MultiPolygon":
            # Iterate over each polygon within the MultiPolygon
            for polygon in geom.geoms:
                # Extract the exterior coordinates of the polygon
                flat_list.extend(list(polygon.exterior.coords))
        else:
            # Extract the exterior coordinates of the geometry
            flat_list.extend(list(geom.exterior.coords))

    # Create the concave hull
    vertices = [(x, y) for x, y in flat_list]
    # alpha = alphashape.optimizealpha(vertices) / 2
    hull = alphashape.alphashape(vertices, degree)

    # Create a GeoDataFrame with the concave hull
    result = gpd.GeoDataFrame(geometry=[hull], crs=dataframe.crs)

    return result


# def to_raster(
#     data_path,
#     output_path,
#     feature_id,
#     pixel_size=PIXEL_SIZE,
#     dtype="float64",
#     windows_shape=(1024, 1024),
# ):
#     encode(data_path, feature_id)

#     with fio.open(data_path) as features:
#         crs = features.crs
#         xmin, ymin, xmax, ymax = features.bounds
#         transform = rio.Affine.from_gdal(xmin, pixel_size, 0, ymax, 0, -pixel_size)
#         out_shape = (int((ymax - ymin) / pixel_size), int((xmax - xmin) / pixel_size))

#         with rio.Env(CHECK_DISK_FREE_SPACE="NO"):

#             with rio.open(
#                 output_path,
#                 "w",
#                 height=out_shape[0],
#                 width=out_shape[1],
#                 count=1,
#                 dtype=dtype,
#                 crs=crs,
#                 transform=transform,
#                 tiled=True,
#                 options=["COMPRESS=LZW"],
#             ) as raster:
#                 for window in get_windows(windows_shape, out_shape):
#                     window_transform = rio.windows.transform(window, transform)
#                     # can be smaller than windows_shape at the edges
#                     window_shape = (window.height, window.width)
#                     window_data = np.zeros(window_shape)

#                     for feature in features:
#                         value = feature["properties"][f"{feature_id}_encoded"]
#                         geom = feature["geometry"]
#                         d = rasterize(
#                             [(geom, value)],
#                             all_touched=False,
#                             out_shape=window_shape,
#                             transform=window_transform,
#                         )
#                         window_data += d  # sum values up


#                     raster.write(window_data, window=window, indexes=1)
