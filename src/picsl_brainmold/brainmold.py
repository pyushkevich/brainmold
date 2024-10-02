from picsl_c3d import Convert3D
import math
import argparse
import os

class Parameters:
    def __init__(self):
        # Things related to input and output files
        self.fn_input_mri = ""
        self.fn_input_seg = ""
        self.fn_input_dots = ""
        self.fn_input_avoidance = ""
        self.fn_output_root = ""
        self.id_string = ""

        # Side of the hemisphere
        self.side = 'N'

        # Parameters for mold creation
        self.slit_spacing_mm = 10.0
        self.slit_width_mm = 1.6
        self.mold_resolution_mm = 0.4
        self.mold_wall_thickness_mm = 3
        self.mold_floor_thickness_mm = 3

        # The amount of dilation and erosion applied to the mask as a
        # preprocessing step
        self.preproc_dilation = 5
        self.preproc_erosion = 3

        # Flags
        self.flag_print_hemisphere_mold = True

        # Specific slab to analyze
        self.selected_slab = -1

    def _generate_filename(self, subdir, suffix, slab=None):
        """Helper function to generate filenames and create directories."""
        # Generate the base filename
        if slab is not None:
            filename = f"{self.id_string}_slab{slab:02d}_{suffix}"
        else:
            filename = f"{self.id_string}_{suffix}"

        # Full path for the output directory
        if subdir:
            filepath = os.path.join(self.fn_output_root, subdir, filename)
        else:
            filepath = os.path.join(self.fn_output_root, filename)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        return filepath

    def fn_hemi_mold_image(self):
        return self._generate_filename(None, "hemi_mold.nii.gz")

    def fn_hemi_volume_image(self):
        return self._generate_filename(None, "hemi_volume.nii.gz")

    def fn_hemi_c3d_log(self):
        return self._generate_filename("logs", "hemi_c3d_log.txt")

    def fn_slab_mold_image(self, slab):
        return self._generate_filename(None, "mold.nii.gz", slab)

    def fn_slab_volume_image(self, slab):
        return self._generate_filename(None, "volume.nii.gz", slab)

    def fn_slab_c3d_log(self, slab):
        return self._generate_filename("logs", "c3d_log.txt", slab)

    def fn_slab_optimization_log(self, slab):
        return self._generate_filename("logs", "opt_log.txt", slab)

    def fn_slab_cutplane_json(self, slab):
        return self._generate_filename(None, "cutplanes.json", slab)

    def fn_slab_cut_png(self, slab):
        return self._generate_filename("tex", "cuts.png", slab)

    def fn_slab_cut_texlet(self, slab):
        return self._generate_filename("tex", "cuts.tex", slab)

    def fn_slab_with_dots_volume_image(self, slab):
        return self._generate_filename(None, "mask_with_dots.nii.gz", slab)

    def fn_global_texfile(self):
        return self._generate_filename("tex", "print_template.tex")


def make_mold(param:Parameters):

    # Initialize API and redirect output
    api = Convert3D()

    # Read the mask image and dots, avoidance images
    api.execute(f"-verbose {param.fn_input_seg} -popas mask")

    if param.fn_input_dots:
        api.execute(f"{param.fn_input_dots} -swapdim LPI -popas dots")
    else:
        api.execute("-push mask -scale 0 -swapdim LPI -popas dots")

    if param.fn_input_avoidance:
        api.execute(f"{param.fn_input_avoidance} -swapdim LPI -popas avoid")
    else:
        api.execute("-push mask -scale 0 -swapdim LPI -popas avoid")

    # Trim and resample hemisphere
    trim_radius = max(param.mold_wall_thickness_mm, max(param.mold_floor_thickness_mm, 5.))
    print("Extracting hemisphere main connected component and resampling")

    r_dil, r_ero = param.preproc_dilation, param.preproc_erosion
    api.execute(f"-push mask -swapdim LPI -thresh 1 1 1 0 -as comp_raw "
                f"-dilate 1 {r_dil}x{r_dil}x{r_dil} -dilate 0 {r_ero}x{r_ero}x{r_ero} -as comp "
                f"-trim {trim_radius}mm -resample-mm {param.mold_resolution_mm}mm -as H")

    # Extract image details
    i_comp_resampled = api.peek(-1)
    sz = i_comp_resampled.GetSize()
    sp = i_comp_resampled.GetSpacing()
    off_y = (sz[1] * sp[1] / 2 - 5)

    # Compute slab positions
    idx_center = [ sz[i] * 0.5 for i in range(3)]
    p_center = i_comp_resampled.TransformContinuousIndexToPhysicalPoint(idx_center)
    y_center = -p_center[1]

    # Number of slabs
    n_slabs = int((off_y / param.slit_spacing_mm) * 2)
    print(f"Slabs {n_slabs}, center slab coordinates: {y_center}")

    slabs = [{'y0': y_center + (n_slabs / 2. - i) * param.slit_spacing_mm - param.slit_spacing_mm,
              'y1': y_center + (n_slabs / 2. - i) * param.slit_spacing_mm}
             for i in range(n_slabs)]

    # Print the main mold if requested
    if param.flag_print_hemisphere_mold:
        print("Generating hemisphere mold")
        api.execute("-clear -push H -cmp -popas z -popas y -popas x "
                    "-push H -scale 0 -shift 4")

        for i in range(n_slabs + 1):
            y_cut = slabs[i]['y0'] if i < n_slabs else slabs[i - 1]['y1']
            api.execute(f"-push y -shift {-y_cut} -abs -stretch 0 {param.slit_width_mm} -4 4 -clip -4 4 -min")

        api.execute(f"-as res1 -o {param.fn_hemi_volume_image()}")

        ext_dir_1 = "RPS" if param.side == 'R' else "LPS"
        ext_dir_2 = "LPS" if param.side == 'R' else "RPS"
        floor_offset_vox = (trim_radius - param.mold_floor_thickness_mm) / param.mold_resolution_mm
        api.execute(f"-clear -push H -swapdim {ext_dir_1} -cmv -pick 0 "
                    f"-stretch {floor_offset_vox - 1.0} {floor_offset_vox + 1.0} 4 -4 -clip -4 4 -swapdim LPI -as base")

        api.execute("-clear -push H -push base -push res1 -max "
                    "-copy-transform -pad 5x5x5 5x5x5 -4 -as mold")

        p3 = math.ceil(param.mold_wall_thickness_mm / param.mold_resolution_mm)
        api.execute(f"-clear -push comp -thresh -inf 0.5 4 -4 "
                    f"-swapdim {ext_dir_1} -extrude-seg -swapdim LPI -dup "
                    f"-swapdim {ext_dir_2} -extrude-seg "
                    f"-thresh -inf 0 0 1 -dilate 0 0x{p3}x{p3} -stretch 0 1 4 -4 -swapdim LPI -min "
                    f"-insert mold 1 -background 4 -reslice-identity "
                    f"-push mold -min -as carved -thresh 0 inf 1 0")

        # Save final output
        api.execute(f"-clear -push comp -o {param.fn_hemi_volume_image()}")
        api.execute(f"-push carved -o {param.fn_hemi_mold_image()}")


class BrainMoldLauncher:
    
    def __init__(self, parser):
        
        # Required arguments
        parser.add_argument("--subject", "-s", required=True, help="Subject ID, used in naming files")
        parser.add_argument("--hemisphere", "-H", choices=["L", "R"], required=True, help="Which hemisphere (left or right)")
        parser.add_argument("--input", "-i", nargs=2, required=True, metavar=('mri', 'mask'), help="Input MRI and brain mask")
        parser.add_argument("--output", "-o", required=True, help="Output directory")

        # Optional arguments
        parser.add_argument("--dots", "-d", help="Dots segmentation image")
        parser.add_argument("--avoidance", "-a", help="Avoidance segmentation image")
        parser.add_argument("--no-hemi-mold", action="store_true", help="Skip hemisphere mold printing")
        parser.add_argument("--slab", type=int, default=-1, help="Only generate for one slab (negative for none)")
        
        parser.set_defaults(func = lambda args : self.run(args))
        
    def run(self, args):
        
        # Set the parameters from arguments
        param = Parameters()
        param.id_string = args.subject
        param.side = args.hemisphere
        param.fn_input_mri = args.input[0]
        param.fn_input_seg = args.input[1]
        param.fn_output_root = args.output
        param.fn_input_dots = args.dots
        param.fn_input_avoidance = args.avoidance
        param.flag_print_hemisphere_mold = not args.no_hemi_mold
        param.selected_slab = args.slab
        
        make_mold(param)

