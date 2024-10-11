# PICSL BrainMold

PICSL BrainMold is a tool designed to generate 3D printed brain molds, slab masks, and paper templates for precise brain cutting. This tool can be used for processing MRI brain scans to create molds for hemispheres and masks for slabs, with support for additional processing such as dot placement guidance.

## Installation

You can install `picsl_brainmold` via pip:

```bash
pip install picsl_brainmold
```

## Usage

The `picsl_brainmold` tool provides three main commands to guide the user through the process of generating brain molds, slabs, and paper templates:

1. **Mold Mode**: Generate a cutting mold for a hemisphere.
2. **Slabs Mode**: Generate masks for individual slabs.
3. **Paper Mode**: Generate a paper template for block cutting.

### Command Line Interface

#### General Syntax

```bash
python -m picsl_brainmold <command> [options]
```

Where `<command>` is one of:

- `mold`: For generating the cutting mold for a hemisphere.
- `slabs`: For generating slab masks.
- `paper`: For generating a paper template for cutting.

### Mold Mode

Generates a cutting mold for one hemisphere of the brain.

```bash
python -m picsl_brainmold mold -s <subject_id> -H <hemisphere> -i <mri_file> <mask_file> -w <work_dir> [options]
```

Required options:
- `-s`, `--subject`: The subject ID, used in naming files.
- `-H`, `--hemisphere`: Hemisphere to process ("L" for left or "R" for right).
- `-i`, `--input`: The input MRI file and brain mask (generate using ITK-SNAP).
- `-w`, `--work`: The work directory for output (will be created if needed).

Optional options:
- `-g`, `--guide`: Generate guide images for dot placement at a given depth (in mm) from the anterior.

### Slabs Mode

Generates masks for individual slabs based on a dots segmentation image. It is recommended that all dots be placed on the same plane within each slab, using the dot guidance image.

```bash
python -m picsl_brainmold slabs -w <work_dir> -d <dots_image>
```

Required options:
- `-w`, `--work`: The work directory (from previous run of the tool).
- `-d`, `--dots`: The dots segmentation image file.

### Paper Mode

Generates a paper template for block cutting based on the generated slabs.

```bash
python -m picsl_brainmold paper -w <work_dir>
```

Required options:
- `-w`, `--work`: The work directory.

## Example Workflow

Here’s an example workflow for processing a brain scan and generating the necessary files:

1. **Generate the brain mold:**

```bash
python -m picsl_brainmold mold -s subj001 -H L -i subj001_reslice.nii.gz subj001_hemisphere_mask.nii.gz -g 0.2 -w ./work
```

2. **Generate slab masks:**

```bash
python -m picsl_brainmold slabs -w ./work -d subj001_cortex_dots_final.nii.gz
```

3. **Generate a paper template:**

```bash
python -m picsl_brainmold paper -w ./work
```

## License

This project is licensed under the MIT License.
