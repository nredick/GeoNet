import alphashape
import geopandas as gpd


def concave_hull(dataframe, degree=0.001):
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
    hull = alphashape.alphashape(vertices, degree)

    # Create a GeoDataFrame with the concave hull
    result = gpd.GeoDataFrame(geometry=[hull], crs=dataframe.crs)

    # todo figure out how to add a buffer to the polygon

    return result
