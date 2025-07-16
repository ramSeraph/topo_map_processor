# Topo Map Processor

A Python utility for processing topographic maps.

## Description

This project provides a collection of command-line tools and a Python library to process, analyze, and manipulate topographic map data, specifically for creating web-mappable tiles from GeoTIFF files. The tools allow you to create tiles, partition large tile sets, and update existing tile sets efficiently.

## Installation

To install the necessary dependencies, it is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

This project can be used as a library but also provides as a set of command-line tools to help with managing the datasets, tiling and creating `pmtiles` files.

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

The library is configurable and most parameters can be overriden at a per sheet level using the `extra` dictionary.

The `index_map` is used to map specific file names to their corresponding metadata, which can be useful for processing specific datasets. In particular the index_map is expected to contain the geojson geometry of the map sheet in the `geometry` key, which is used to create the bounds file and georeference the sheet. The geometry is expected to be in counter clockwise order startiung at the top left corner of the map sheet.

The end result is a Cloud optimized GeoTIFF file, a bounds file in GeoJSON format from which the tiles can be generated, using the `tile` command-line tool, and a set of tiles in the specified directory.

I would recommend looking at the [nepal_survey_maps](https://github.com/ramSeraph/nepal_survey_maps) repository for a sample implementation

### Command-Line Tools

This project provides several command-line tools to work with topographic maps.

#### `tile`

Creates tiles from a directory of GeoTIFF files.

```bash
tile --tiles-dir <tiles-dir> --tiffs-dir <tiffs-dir> --max-zoom <max-zoom> [--min-zoom <min-zoom>]
```

-   `<tiles-dir>`: Directory to store the output tiles.
-   `<tiffs-dir>`: Directory containing the input GeoTIFF files.
-   `<max-zoom>`: Maximum zoom level for tiling.
-   `<min-zoom>`: Minimum zoom level for tiling (default: 0).

#### `partition`

Partitions a large set of tiles into smaller PMTiles files, which is useful for managing large datasets.

```bash
partition --to-pmtiles-prefix <prefix> --from-tiles-dir <dir> --name <name> --description <desc> --max-zoom <max-zoom> [--from-pmtiles-prefix <prefix>] [--only-disk] [--min-zoom <min-zoom>] (--attribution <attr> | --attribution-file <file>)
```

#### `retile`

Retiles a specific list of sheets (GeoTIFFs) and updates the corresponding tiles.

```bash
retile --retile-list-file <file> --bounds-file <file> --max-zoom <max-zoom> --tiles-dir <dir> --tiffs-dir <dir> [--min-zoom <min-zoom>] [--from-pmtiles-prefix <prefix>]
```

#### `collect-bounds`

Collects individual GeoJSONL bound files into a single GeoJSON FeatureCollection.

```bash
collect-bounds <bounds-dir> <output-file>
```

-   `<bounds-dir>`: Directory containing `.geojsonl` files.
-   `<output-file>`: The output GeoJSON file.

#### `update-bounds`

Updates a GeoJSON bounds file with new information from a directory of individual bounds files.

```bash
update-bounds <bounds-file> <bounds-dir> <retile-list-file>
```

-   `<bounds-file>`: The GeoJSON file to update.
-   `<bounds-dir>`: Directory with the new `.geojsonl` files.
-   `<retile-list-file>`: A file containing the list of sheets that have been updated.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is under UNLICENSE
