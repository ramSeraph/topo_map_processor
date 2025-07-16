# Topo Map Processor [![PyPI - Latest Version](https://img.shields.io/pypi/v/topo_map_processor)](https://pypi.org/project/topo_map_processor/) [![GitHub Tag](https://img.shields.io/github/v/tag/ramSeraph/topo_map_processor?filter=v*)](https://github.com/ramSeraph/topo_map_processor/releases/latest)


A Python utility for processing topographic maps.

## Description

This project provides a collection of command-line tools and a Python library to process, analyze, and manipulate topographic map data, specifically for creating web-mappable tiles from GeoTIFF files. The tools allow you to create tiles, partition large tile sets, and update existing tile sets efficiently.

## Installation

It is recommended to use a virtual environment. [uv](https://github.com/astral-sh/uv) is a fast, modern Python package installer and is the recommended way to manage this project.

### For Library or Development

If you want to use `topo_map_processor` as a library or contribute to its development, install it from PyPI into a virtual environment.

**Using `uv` (Recommended):**
```bash
# Install uv if you don't have it
pip install uv

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the package from PyPI
uv pip install topo_map_processor
```

**Using `pip`:**
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package from PyPI
pip install topo_map_processor
```

To contribute, clone the repository and install in editable mode:
```bash
git clone https://github.com/ramSeraph/topo_map_processor.git
cd topo_map_processor
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

This project can be used as a library but also provides a set of command-line tools to help with managing the datasets, tiling and creating `pmtiles` files.

### Library Workflow

The core of this project is the `TopoMapProcessor` class, which provides a structured workflow for converting a raw scanned topographic map into a Cloud-Optimized GeoTIFF (COG) ready for tiling. The process is designed to be adaptable to different map series by subclassing and implementing key methods.

The main steps, executed by the `processor.process()` method, are as follows:

1.  **Image Preparation (`rotate`):** To minimize computation, the processor first creates a smaller, shrunken version of the full-resolution map. On this small image, it locates the main contour of the map's frame (the border area). By calculating the angle of this contour's minimum bounding box, it determines the skew of the map. The full-resolution image is then rotated using this angle to make the frame perfectly horizontal, correcting for any tilt introduced during scanning.
2.  **Corner Detection (`get_corners`):** Once the image is straightened, the processor uses the same map frame contour to identify the four corner regions. By narrowing the search to these smaller areas, it can apply more precise logic to find the exact pixel coordinates of the map's neatline corners. This is a critical step that often requires custom implementation in a subclass (`get_intersection_point`) to handle the specific visual features of the map's corners (e.g., crosses, specific colors, patterns).
3.  **Grid Line Removal (`remove_grid_lines`):** If configured, the processor can detect and remove grid lines (graticules) from the map image. This is achieved by implementing the `locate_grid_lines` method. The removal helps create a cleaner final map by inpainting the areas where the lines were.
4.  **Georeferencing (`georeference`):** Using the detected corner coordinates and the corresponding real-world geographic coordinates (provided via the `index_map`), the tool creates Ground Control Points (GCPs). It then uses `gdal_translate` to create an initial georeferenced GeoTIFF.
5.  **Warping and Cutting (`warp`):** The georeferenced image is then warped into the standard `EPSG:3857` projection used by most web maps. During this step, it's also precisely cropped to the map's boundary (neatline) using a cutline derived from the corner points. The output is a clean, map-only image, correctly projected.
6.  **Exporting (`export`):** The final warped image is converted into a Cloud-Optimized GeoTIFF (COG), which is highly efficient for web-based tiling. A corresponding bounds file (`.geojsonl`) is also created, defining the geographic extent of the map sheet.

The output of this library workflow—a directory of COG `.tif` files and a directory of `.geojsonl` bounds files—serves as the direct input for the command-line tools like `tile`, `collect-bounds`, and `partition`, which handle the final steps of creating a complete, partitioned web map tile set.

### Library Usage

The `TopoMapProcessor` class provides a framework for processing individual map sheets. To use it, you need to create a subclass and implement the required methods.

Here's an example of how to use the `TopoMapProcessor` class, based on the provided `parse_jica.py` script:

```python
from topo_map_processor import TopoMapProcessor, LineRemovalParams

class SampleProcessor(TopoMapProcessor):

    def __init__(self, filepath, extra, index_map):
        super().__init__(filepath, extra, index_map)
        # ... (additional initialization)

    def get_inter_dir(self):
        return Path('data/inter')

    def get_gtiff_dir(self):
        return Path('export/gtiffs')

    def get_bounds_dir(self):
        return Path('export/bounds')

    def get_crs_proj(self):
        return '+proj=tmerc +lat_0=0 +lon_0=84 +k=0.9999 +x_0=500000 +y_0=0 +units=m +ellps=evrst30 +towgs84=293.17,726.18,245.36,0,0,0,0 +no_defs'

    def get_scale(self):
        return 25000

    def get_intersection_point(self, img, direction, anchor_angle):
        # ... (implementation for finding intersection points which are the corners of the mapframe)
        # you are expected to pick from the various locate intersection points implementations in the library or write your own

    def locate_grid_lines(self):
        # ... (implementation for locating grid lines so as to remove them)
        # you are expected to pick from the various locate grid lines implementations in the library or write your own

# Example of how to run the processor
def process_files():
    # ... (load image files and special cases)
    for filepath in image_files:
        extra = special_cases.get(filepath.name, {})
        index_map = get_index_map_jica()
        processor = SampleProcessor(filepath, extra, index_map)
        processor.process()
```

The library is configurable and most parameters can be overridden at a per sheet level using the `extra` dictionary.

The `index_map` is used to map specific file names to their corresponding metadata, which can be useful for processing specific datasets. In particular the index_map is expected to contain the geojson geometry of the map sheet in the `geometry` key, which is used to create the bounds file and georeference the sheet. The geometry is expected to be in counter clockwise order starting at the top left corner of the map sheet.

The end result is a Cloud optimized GeoTIFF file, a bounds file in GeoJSON format from which the tiles can be generated, using the `tile` command-line tool, and a set of tiles in the specified directory.

I would recommend looking at the [nepal_survey_maps](https://github.com/ramSeraph/nepal_survey_maps) repository for a sample implementation

### Corner Detection Strategies

The `get_intersection_point` method is the most critical part to customize for a new map series. The library provides several helper methods that can be used to implement this logic, each suited for different types of corner features:

-   **`get_nearest_intersection_point`**: This method is useful when the map corners are defined by the intersection of two perpendicular lines (like a simple crosshair). It works by:
    1.  Detecting all horizontal and vertical lines in the corner region.
    2.  Finding all intersection points between these lines.
    3.  Identifying the point closest to the expected corner position, while also matching an expected angle and distance from an anchor point.

-   **`get_4way_intersection_point`**: This is a more robust version for crosshair-style corners. It specifically looks for intersections where four lines meet, making it less prone to errors from stray lines in the map collar.

-   **`get_biggest_contour_corner`**: This strategy is designed for corners that are marked by a solid shape or a filled area (e.g., a colored square or circle). It works by:
    1.  Isolating the specific color of the corner feature.
    2.  Finding the largest contour (shape) of that color in the corner region.
    3.  Returning the outermost point of that contour as the corner.

-   **`get_nearest_intersection_point_from_biggest_corner_contour`**: This is a hybrid approach. It first uses the "biggest contour" method to locate a general corner area (e.g., a colored box) and then searches for a precise line intersection *within* that area. This is effective for maps that have both a colored corner block and a fine crosshair inside it.

By combining these methods in your subclass's implementation of `get_intersection_point`, you can build a reliable corner detection system for your specific map type.

### Grid Line Detection Strategies

Similar to corner detection, the `locate_grid_lines` method can be implemented using a few key helper methods provided by the library. The primary goal is to identify the precise pixel paths of the map's grid lines (graticules) so they can be removed.

-   **`locate_grid_lines_using_transformer`**: This is the main strategy for identifying grid lines. It leverages the georeferencing information (the `GCPTransformer`) established from the map corners. The process is as follows:
    1.  It defines a regular grid in the map's real-world coordinate system (e.g., a line every 1000 meters).
    2.  It uses the transformer to project this theoretical grid back onto the pixel space of the image.
    3.  This results in a highly accurate set of lines that represent where the grid lines should be, even accounting for distortions in the map.

-   **`get_grid_line_corrections`**: This method serves as a refinement step to improve accuracy. While the transformer provides a very close estimate, this function can correct for minor printing or scanning inaccuracies. For each projected grid intersection, it searches a small area in the actual image for the *true* intersection of printed lines, adjusting the point to match the visual data. This ensures that the lines targeted for removal are precisely the ones printed on the map.

### Command-Line Tools

This project provides several command-line tools to work with topographic maps. These tools are dynamically loaded, and you can get the most up-to-date usage information by running the tool with the `--help` flag.

#### `collect-bounds`

Collects individual GeoJSONL bound files into a single GeoJSON FeatureCollection.

```bash
collect-bounds --bounds-dir <directory> --output-file <file>
```

-   `--bounds-dir`: Directory containing GeoJSONL files. (Required)
-   `--output-file`: Output GeoJSON file path. (Required)

#### `partition`

Partitions a large set of tiles into smaller PMTiles files. This is useful for managing large datasets, for example, to stay within file size limits for services like GitHub Releases.

```bash
partition --to-pmtiles-prefix <prefix> --from-tiles-dir <dir> --name <name> --description <desc> --max-zoom <zoom> [--min-zoom <zoom>] [--from-pmtiles-prefix <prefix>] [--only-disk] (--attribution <text> | --attribution-file <file>)
```

-   `--to-pmtiles-prefix`: Prefix for the output PMTiles files. (Required)
-   `--from-tiles-dir`: Directory containing the input tiles. (Required)
-   `--name`: Name of the mosaic. (Required)
-   `--description`: Description of the mosaic. (Required)
-   `--max-zoom`: Maximum zoom level for the mosaic. (Required)
-   `--min-zoom`: Minimum zoom level for the mosaic (default: 0).
-   `--from-pmtiles-prefix`: Prefix for input PMTiles files (required if not using `--only-disk`).
-   `--only-disk`: Only read tiles from the local disk.
-   `--attribution`: Attribution text for the mosaic. (Required, unless `--attribution-file` is used)
-   `--attribution-file`: File containing attribution text. (Required, unless `--attribution` is used)

#### `retile`

Retiles a specific list of sheets (GeoTIFFs) and updates the corresponding tiles in an existing tile set. It can determine which other sheets are affected and need to be pulled in to correctly retile the area.

```bash
retile --retile-list-file <file> --bounds-file <file> --max-zoom <zoom> [--min-zoom <zoom>] [--tiffs-dir <dir>] [--tiles-dir <dir>] [--from-pmtiles-prefix <prefix>] [--sheets-to-pull-list-outfile <file>]
```

-   `--retile-list-file`: File containing the list of sheets to retile. (Required)
-   `--bounds-file`: GeoJSON file containing the list of available sheets and their geographic bounds. (Required)
-   `--max-zoom`: Maximum zoom level to create tiles for. (Required)
-   `--min-zoom`: Minimum zoom level to create tiles for (default: 0).
-   `--tiffs-dir`: Directory where the GeoTIFFs are located.
-   `--tiles-dir`: Directory where the tiles will be created.
-   `--from-pmtiles-prefix`: Prefix for the PMTiles source from which to pull existing tiles.
-   `--sheets-to-pull-list-outfile`: If provided, the script will only calculate the full list of sheets that need to be processed (including adjacent ones) and write it to this file, then exit.

#### `tile`

Creates web map tiles from a directory of GeoTIFF files. It combines them into a virtual raster (`.vrt`) for efficient processing.

```bash
tile <tiles-dir> <tiffs-dir> <max-zoom> [<min-zoom>]
```

-   `<tiles-dir>`: Directory to store the output tiles. (Required)
-   `<tiffs-dir>`: Directory containing the input GeoTIFF files. (Required)
-   `<max-zoom>`: Maximum zoom level for tiling. (Required)
-   `<min-zoom>`: Minimum zoom level for tiling (default: 0).

#### `update-bounds`

Updates a main GeoJSON bounds file with new or modified bounds from individual `.geojsonl` files.

```bash
update-bounds --bounds-file <file> --bounds-dir <dir> --retile-list-file <file>
```

-   `--bounds-file`: The main GeoJSON bounds file to update. (Required)
-   `--bounds-dir`: Directory containing the new or updated individual `.geojsonl` files. (Required)
-   `--retile-list-file`: A file containing the list of sheet names that have been updated (corresponding to the files in `--bounds-dir`). (Required)

### Running Tools Directly with `uvx`

You can run the command-line tools directly from PyPI without a persistent installation using `uvx`. This is convenient for one-off tasks.

```bash

# Example: Run the 'collect-bounds' tool
uvx --from topo_map_processor collect-bounds --bounds-dir /path/to/bounds --output-file bounds.geojson

# tile and retile have a dependency gdal python library,
# and some special handling is required so as to install a version of gdal python bindings which matches the version of gdal installed in your system
# Also numpy and pillow dependencies are required so that the gdal python bindings have some of the resampling features enabled
# Example: Get help for the 'tile' command
GDAL_VERSION=$(gdalinfo --version | cut -d"," -f1 | cut -d" " -f2)
uvx --with numpy --with pillow --with gdal==$GDAL_VERSION --from topo_map_processor tile --help
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is under the UNLICENSE
