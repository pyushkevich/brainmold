
# BrainMold: Generator for 3D Printed Brain Molds

`BrainMold` is a tool to generate 3D printable brain molds from MRI and brain mask images. It uses the Convert3D tool (`c3d`) to process input data and create mold designs.

## Features

- Generate hemisphere and slab molds for 3D printing
- Optional support for additional segmentation images (dots, avoidance regions)
- Customizable mold parameters like slit spacing, mold resolution, wall and floor thickness

## Installation

```bash
pip install picsl_brainmold
```

Ensure that `Convert3D` is installed and available in your system's PATH.

## Usage

Run the tool with the following arguments:

```bash
python -m picsl_brainmold --subject <ID> --hemisphere <L|R> --input <mri> <mask> --output <output_dir>
```

### Required Arguments

- `--subject, -s`: Subject ID (used in naming files)
- `--hemisphere, -h`: Hemisphere (left or right)
- `--input, -i`: Input MRI and brain mask
- `--output, -o`: Output directory for generated files

### Optional Arguments

- `--dots, -d`: Dots segmentation image
- `--avoidance, -a`: Avoidance segmentation image
- `--no-hemi-mold`: Skip hemisphere mold printing
- `--slab`: Generate mold for a specific slab (default: all slabs)

## Example

```bash
python -m picsl_brainmold --subject 001 --hemisphere L --input mri.nii.gz mask.nii.gz --output ./output
```

This will generate the hemisphere and slab molds for subject 001's left hemisphere using the provided MRI and mask.

## Output

The tool generates the following outputs:
- Hemisphere mold image
- Slab mold images (if applicable)
- Text files with mold cut plane information
- PNG and TEX files for slab cut visualization

## License

This project is licensed under the MIT License.

