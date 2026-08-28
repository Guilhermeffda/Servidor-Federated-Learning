# TACO Data

## Scope

This directory documents the reproducible acquisition and initial structural
audit of **TACO (Trash Annotations in Context)**. The current audit observes the
official aggregate COCO annotations and local image availability. It does not
modify annotations or images and does not perform geometric validation of
bounding boxes or segmentations.

## Official Source

- Authors' repository: <https://github.com/pedropro/TACO>
- Primary artifact: Zenodo Record `3587843`
- DOI: `10.5281/zenodo.3587843`

## Dataset Version / Artifact Audited

The audited artifact is the official `TACO.zip` obtained from Zenodo:

- Size: `2719810615` bytes
- Expected and observed MD5: `e9149407d883e21a8d224feef8210920`
- Observed SHA-256:
  `51cd6b009cac1e94decf058e470b1fef64148f9c89923d96dcd54d4db61ceae9`

The canonical aggregate annotation file audited was:

`data/raw/taco/extracted/TACO/data/annotations.json`

Its dataset root, used to resolve every `images[].file_name`, was:

`data/raw/taco/extracted/TACO/data/`

## Acquisition Method

`scripts/download_dataset.py` queries the public Zenodo record, validates the
published checksum, downloads the archive as a resumable stream, verifies MD5
and SHA-256, and extracts it with path-traversal protection. Raw data and local
acquisition metadata under `data/raw/taco/` are not versioned.

## Local Directory Structure

The relevant observed structure is:

```text
data/raw/taco/
├── archives/
│   └── TACO.zip
├── extracted/
│   ├── TACO/
│   │   ├── data/
│   │   │   ├── annotations.json
│   │   │   └── batch_1/ ... batch_15/
│   │   └── detector/
│   └── __MACOSX/
├── metadata/
└── dataset/                 # legacy Flickr smoke-test artifacts
```

`__MACOSX/` is ZIP packaging metadata and is not part of the logical dataset or
its statistics. The old `data/raw/taco/dataset/` Flickr artifacts are retained
locally but are not a source of truth for this audit.

Fifteen auxiliary `annotations.json` files exist under `batch_1/` through
`batch_15/`. They were reported as auxiliary artifacts and were not combined
with or added to the aggregate statistics. Four CSV files were observed under
`TACO/detector/taco_config/`.

## Annotation Format

The aggregate annotations use COCO JSON. The top-level keys observed are:

`annotations`, `categories`, `images`, `info`, `licenses`,
`scene_annotations`, and `scene_categories`.

## COCO Structure

Every one of the 1,500 image records contains:

`id`, `file_name`, `width`, `height`, `coco_url`, `date_captured`,
`flickr_640_url`, `flickr_url`, and `license`.

Every one of the 4,784 annotation records contains:

`id`, `image_id`, `category_id`, `bbox`, `segmentation`, `area`, and
`iscrowd`.

Every one of the 60 category records contains:

`id`, `name`, and `supercategory`.

All 4,784 annotations contain both `bbox` and `segmentation`. Their geometry was
not validated in this audit.

## Dataset Statistics

| Observation | Count |
|---|---:|
| Referenced images | 1,500 |
| Locally available, non-empty images | 1,500 |
| Missing images | 0 |
| Invalid/unsafe image path references | 0 |
| Images without annotations | 0 |
| Annotation records | 4,784 |
| Category records | 60 |
| Unique supercategories | 28 |
| Duplicate `file_name` values | 0 |
| Duplicate image ID values | 0 |
| Duplicate annotation ID values | 2 |
| Duplicate category ID values | 0 |
| Orphan annotation image references | 0 |
| Orphan annotation category references | 0 |

## Categories

Counts below are observed annotation instances in the aggregate JSON.

| ID | Category | Supercategory | Instances |
|---:|---|---|---:|
| 0 | Aluminium foil | Aluminium foil | 62 |
| 1 | Battery | Battery | 2 |
| 2 | Aluminium blister pack | Blister pack | 6 |
| 3 | Carded blister pack | Blister pack | 1 |
| 4 | Other plastic bottle | Bottle | 50 |
| 5 | Clear plastic bottle | Bottle | 285 |
| 6 | Glass bottle | Bottle | 104 |
| 7 | Plastic bottle cap | Bottle cap | 209 |
| 8 | Metal bottle cap | Bottle cap | 80 |
| 9 | Broken glass | Broken glass | 138 |
| 10 | Food Can | Can | 34 |
| 11 | Aerosol | Can | 10 |
| 12 | Drink can | Can | 229 |
| 13 | Toilet tube | Carton | 5 |
| 14 | Other carton | Carton | 93 |
| 15 | Egg carton | Carton | 11 |
| 16 | Drink carton | Carton | 45 |
| 17 | Corrugated carton | Carton | 64 |
| 18 | Meal carton | Carton | 30 |
| 19 | Pizza box | Carton | 3 |
| 20 | Paper cup | Cup | 67 |
| 21 | Disposable plastic cup | Cup | 104 |
| 22 | Foam cup | Cup | 13 |
| 23 | Glass cup | Cup | 6 |
| 24 | Other plastic cup | Cup | 2 |
| 25 | Food waste | Food waste | 8 |
| 26 | Glass jar | Glass jar | 6 |
| 27 | Plastic lid | Lid | 77 |
| 28 | Metal lid | Lid | 10 |
| 29 | Other plastic | Other plastic | 273 |
| 30 | Magazine paper | Paper | 12 |
| 31 | Tissues | Paper | 42 |
| 32 | Wrapping paper | Paper | 12 |
| 33 | Normal paper | Paper | 82 |
| 34 | Paper bag | Paper bag | 27 |
| 35 | Plastified paper bag | Paper bag | 0 |
| 36 | Plastic film | Plastic bag & wrapper | 451 |
| 37 | Six pack rings | Plastic bag & wrapper | 5 |
| 38 | Garbage bag | Plastic bag & wrapper | 31 |
| 39 | Other plastic wrapper | Plastic bag & wrapper | 260 |
| 40 | Single-use carrier bag | Plastic bag & wrapper | 61 |
| 41 | Polypropylene bag | Plastic bag & wrapper | 3 |
| 42 | Crisp packet | Plastic bag & wrapper | 39 |
| 43 | Spread tub | Plastic container | 9 |
| 44 | Tupperware | Plastic container | 4 |
| 45 | Disposable food container | Plastic container | 38 |
| 46 | Foam food container | Plastic container | 15 |
| 47 | Other plastic container | Plastic container | 6 |
| 48 | Plastic glooves | Plastic glooves | 4 |
| 49 | Plastic utensils | Plastic utensils | 37 |
| 50 | Pop tab | Pop tab | 99 |
| 51 | Rope & strings | Rope & strings | 29 |
| 52 | Scrap metal | Scrap metal | 20 |
| 53 | Shoe | Shoe | 7 |
| 54 | Squeezable tube | Squeezable tube | 7 |
| 55 | Plastic straw | Straw | 157 |
| 56 | Paper straw | Straw | 4 |
| 57 | Styrofoam piece | Styrofoam piece | 112 |
| 58 | Unlabeled litter | Unlabeled litter | 517 |
| 59 | Cigarette | Cigarette | 667 |

## Supercategories

| Supercategory | Categories | Instances |
|---|---:|---:|
| Aluminium foil | 1 | 62 |
| Battery | 1 | 2 |
| Blister pack | 2 | 7 |
| Bottle | 3 | 439 |
| Bottle cap | 2 | 289 |
| Broken glass | 1 | 138 |
| Can | 3 | 273 |
| Carton | 7 | 251 |
| Cigarette | 1 | 667 |
| Cup | 5 | 192 |
| Food waste | 1 | 8 |
| Glass jar | 1 | 6 |
| Lid | 2 | 87 |
| Other plastic | 1 | 273 |
| Paper | 4 | 148 |
| Paper bag | 2 | 27 |
| Plastic bag & wrapper | 7 | 850 |
| Plastic container | 5 | 72 |
| Plastic glooves | 1 | 4 |
| Plastic utensils | 1 | 37 |
| Pop tab | 1 | 99 |
| Rope & strings | 1 | 29 |
| Scrap metal | 1 | 20 |
| Shoe | 1 | 7 |
| Squeezable tube | 1 | 7 |
| Straw | 2 | 161 |
| Styrofoam piece | 1 | 112 |
| Unlabeled litter | 1 | 517 |

## Scene Metadata

No explicit scene/background tag fields are present directly in the aggregate
COCO `images` records. The aggregate JSON does, however, contain separate
top-level `scene_annotations` and `scene_categories` structures directly
associated with images through `image_id` and `background_ids`.

Observed scene context:

- Scene annotation records: 4,296
- Scene category records: 7
- Images with scene values: 1,496
- Images without scene values: 4
- Images with multiple distinct scene/background IDs: 900
- Scene records containing multiple background IDs: 1,486
- Orphan scene image IDs: 0

| Background ID | Name | Occurrences |
|---:|---|---:|
| 0 | Clean | 37 |
| 1 | Indoor, Man-made | 223 |
| 2 | Pavement | 1,260 |
| 3 | Sand, Dirt, Pebbles | 1,809 |
| 4 | Trash | 29 |
| 5 | Vegetation | 2,123 |
| 6 | Water | 390 |
| 7 | Unmatched in `scene_categories` | 114 |

These values are reported as provided. No scene label was inferred from image
content or batch names.

## Missing / Unavailable Images

No referenced image was missing: all 1,500 `file_name` paths resolved safely to
existing, non-empty local files.

## Metadata / Provenance

All 1,500 image records contain `flickr_url`, `flickr_640_url`, `coco_url`,
`date_captured`, and `license` fields. Observed values in the image-level
`license` field were:

| Value | Image records |
|---|---:|
| `"CC"` | 319 |
| `"ODBL (c) OpenLitterMap & Contributors"` | 466 |
| `null` | 715 |

These are metadata observations only and are not a legal interpretation. The
software repository license and individual image provenance/licensing are
distinct matters.

## Known Issues

- Annotation IDs `309` and `4040` each occur twice. The records were reported
  but not modified.
- `background_id` `7` occurs 114 times in `scene_annotations`, but the observed
  `scene_categories` define only IDs 0 through 6.
- Category ID `35` (`Plastified paper bag`) has zero annotation instances.
- `__MACOSX/` packaging metadata is present and intentionally excluded.
- Auxiliary per-batch annotation files exist but are not included in aggregate
  statistics.

No duplicate image IDs, duplicate category IDs, orphan annotation references,
unsafe image paths, or missing image files were observed.

## Reproduction

From the `taco_data/` directory:

```powershell
py scripts\download_dataset.py
py scripts\inspect_dataset.py
```

An explicit in-workspace annotation override may be used for diagnostics:

```powershell
py scripts\inspect_dataset.py --annotations path\to\annotations.json
```

## Notes

GPS and geolocation are outside the current methodological scope of this project
and are not dataset requirements. 