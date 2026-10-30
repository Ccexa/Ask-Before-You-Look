from typing import Dict, Tuple
from shapely.geometry import Polygon, MultiPolygon, box, shape
from shapely.affinity import translate
from PIL import Image, ImageDraw


as_discrete = {
    'Others': 0,
    'T-G1': 1,
    'T-G2': 1,
    'T-G3': 1
}

palette = [
    0, 0, 0,
    255, 255, 255
] + [0, 0, 0] * (256-2)

def read_mask(
        geodata:Dict,
        location:Tuple[int, int] | Tuple[float, float],
        ds:float|int,
        roi_size:Tuple[int, int] | int
)->Image:
    '''
        geodata: geojson annotation from qupath
        location: top, left coordiantes at level 0
        ds: downscale factor at a specific level
        roi_size: the actual size region of interset 

        return: PIL.Image mask
    '''

    top, left = location
    roi_size = roi_size if isinstance(roi_size, tuple) else (roi_size, roi_size)
    h, w = roi_size
    mask = Image.new(mode='P', size=(w, h), color=0)
    drawer = ImageDraw.Draw(mask)



    H, W  = h * ds, w * ds
    bbox = box(minx=0, miny=0, maxx=W, maxy=H)

    for feature in geodata['features']:
        polygon = shape(feature['geometry'])
        try:
            classification = feature["properties"]["classification"]["name"]
        except:
            classification = None
        if classification is None:
            continue
        if classification in as_discrete.keys():
            fill_color = as_discrete[classification]
        else:
            fill_color = as_discrete['Others']

        translate_polygon = translate(polygon, xoff=-left, yoff=-top)
        clipped_polygon = translate_polygon.intersection(bbox)

        if not clipped_polygon.is_valid:
            continue

        if isinstance(clipped_polygon, Polygon):
            coords = [(x/ds, y/ds) for (x, y) in clipped_polygon.exterior.coords]
            if len(coords) >= 3:
                drawer.polygon(coords, fill=fill_color)
        elif isinstance(clipped_polygon, MultiPolygon):
            for polygon_  in clipped_polygon.geoms:
                coords = [(x/ds, y/ds) for (x, y) in polygon_.exterior.coords]
                if len(coords) >= 3:
                    drawer.polygon(coords ,fill=fill_color)

    mask.putpalette(palette)
    return mask
