from picsl_c3d import Convert3D
import math
import argparse
import os
import json
import io
import numpy as np
import SimpleITK as sitk

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
        
        # Depth at which the guide is generated
        self.dot_guide_depth = None

        # Flags
        self.flag_print_hemisphere_mold = True

        # Specific slab to analyze
        self.selected_slab = -1
        
    def to_dict(self):
        
        def rebase(k, v):
            if k.startswith('fn_') and k != 'fn_output_root' and v is not None and os.path.exists(v):
                print(k, v)
                return os.path.relpath(v, self.fn_output_root)
            else:
                return v
        
        return { k : rebase(k,v) for k, v in self.__dict__.items()}
    
    def from_dict(self, d):        
        for k, v in self.__dict__.items():
            if k in d:
                v = d[k]
                if k.startswith('fn_') and k != 'fn_output_root':
                    v = os.path.abspath(os.path.join(self.fn_output_root, v))
                self.__dict__[k] = v
        
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

    def fn_dot_guide_image(self):
        return self._generate_filename(None, "dots_guide.nii.gz")

    def fn_hemi_c3d_log(self):
        return self._generate_filename("logs", "hemi_c3d_log.txt")

    def fn_slab_mold_image(self, slab):
        return self._generate_filename(None, "mold.nii.gz", slab)

    def fn_slab_volume_image(self, slab):
        return self._generate_filename(None, "volume.nii.gz", slab)

    def fn_config_json(self):
        return os.path.join(self.fn_output_root, 'config.json')

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

    def fn_slab_tex_png(self, slab, flip):
        return self._generate_filename("tex", f"template_{flip}.png", slab)

    def fn_global_texfile(self):
        return self._generate_filename("tex", "print_template_no_cuts.tex")


def make_mold(id:str, side:str, 
              fn_input_mri:str, fn_input_seg:str, fn_output_root:str, 
              dot_guide_depth=None):
    
    # Set up the parameters data structure
    param = Parameters()
    param.id_string, param.side = id, side
    param.fn_input_mri = fn_input_mri
    param.fn_input_seg = fn_input_seg
    param.fn_output_root = fn_output_root
    param.dot_guide_depth = dot_guide_depth

    with open(param.fn_hemi_c3d_log(), 'wt') as fn_log:
        
        # Initialize API and redirect output
        api = Convert3D(out=fn_log, err=fn_log)

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
        i_comp_resampled = api.get_image('H')
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
            
        # Generate the dots guide
        if param.dot_guide_depth:
            
            # Take the original image, the guides will be defined in this space. But we need to 
            # swapdim the image so we know what direction AP corresponds to
            # api.execute('-push mask -swapdim LPS -scale 0')
            api.execute(f'{param.fn_input_seg} -as src -swapdim LPS -o goo.nii.gz -scale 0')
            i_guide = api.peek(-1)
            
            # For each slab, tag a plane
            for i, s in enumerate(slabs):
                y_phy = s['y0'] - param.dot_guide_depth
                y_img = int(0.5 + i_guide.TransformPhysicalPointToContinuousIndex([0, -y_phy, 0])[1])
                print(f"Slab {i} : [{s['y0']:5.2f} - {s['y1']:5.2f}] marking slice y={y_img}")
                i_guide[:,y_img,:] = 128
                
            # Save the guide image
            api.push(i_guide)
            api.execute(f'-insert mask 1 -int 0 -reslice-identity -push mask -dilate 1 3x3x3 -thresh 0 0 1 0 -times '
                        f'-type uchar -o {param.fn_dot_guide_image()}')
            
            # Generate a file with the parameters and slab information
            state_dict = {
                'slabs': slabs,
                'param': param.to_dict()
            }
            with open(param.fn_config_json(), 'w') as fd:
                json.dump(state_dict, fd, indent=4, sort_keys=True)
            

def process_slab(param:Parameters, slab_id:int, slab: dict, 
                 i_hemi_raw, i_hemi_mask, i_dots, dot_coord):
    
    with open(param.fn_slab_c3d_log(slab_id), 'wt') as fn_log:
    
        # Create our own API
        api = Convert3D(out = fn_log, err = fn_log)
    
        # Extract the slab image region
        api.add_image("H", i_hemi_mask)
        api.add_image("dots", i_dots)
        api.execute(
            f"-verbose -clear -push H -cmp -pick 1 -thresh {slab['y0']} {slab['y1']} 255 0 "
            f"-push H -times -thresh 0 127 0 255 -info -trim 0vox -as slab ")
        
        # Check if the image is empty
        i_slab = api.get_image('slab')
        if i_slab.GetNumberOfPixels() == 0:
            print("  Empty slab encountered")
            return
        
        # Process each dot stat
        for dot_value, dot_center in dot_coord.items():

            # Check if the dot exists on this slide
            if dot_center[1] >= slab['y0'] and dot_center[1] <= slab['y1']:

                print(f"  Slab {slab_id} contains dot {dot_value} with physical coordinates {dot_center}")

                # Mark the dot with a 3x3x3 cube in the target volume
                api.execute(f'-clear -push slab -dup -push dots -thresh {dot_value} {dot_value} 1 0 '
                            f'-dilate 1 1x1x1 -dup -lstat -pop -centroid -int 0 -reslice-identity '
                            f'-centroid-mark {dot_value} -dilate {dot_value} 2x2x2 -composite -as slab')
            
        # Save the image
        output_filename = param.fn_slab_with_dots_volume_image(slab_id)
        api.execute(f"-push slab -type uchar -o {output_filename}")        
        
        
def read_config(fn_output_root:str):
    
    # Read the configuration and slab indices
    param = Parameters()
    param.fn_output_root = fn_output_root

    slabs = []
    with open(param.fn_config_json(), 'r') as fd:
        d = json.load(fd)
        try:
            param.from_dict(d['param'])
            slabs = d['slabs']
        except:
            print(f"Failed to read configuration from {param.fn_config_json()}")
    
    return param, slabs    


def make_slabs(fn_output_root:str, fn_dots:str, slab_index=None):
    
    # Read the configuration and extra parameter
    param, slabs = read_config(fn_output_root)
    param.fn_input_dots = fn_dots
        
    # Set up Convert3D
    sout = io.StringIO()
    api = Convert3D(out = sout)
    
    # Generate the necessary inputs to the slab extractor
    api.execute(f'{param.fn_input_seg} -swapdim LPI -thresh 1 1 1 0')
    i_comp_raw = api.peek(-1)
    trim_radius = max(param.mold_wall_thickness_mm, max(param.mold_floor_thickness_mm, 5.))
    api.execute(f'{param.fn_hemi_volume_image()} '
                f'-trim {trim_radius}mm -resample-mm {param.mold_resolution_mm}mm ')
    i_comp_resampled = api.peek(-1)
    api.execute(f'{param.fn_input_dots} -as D')
    i_dots = api.peek(-1)
    
    # We need to find the centroid of each dot in physical units
    a_dots = sitk.GetArrayFromImage(i_dots)
    dot_val = np.setdiff1d(np.unique(a_dots), [0., 128.])
    dot_coord = {}
    for i in dot_val:
        dot_index = np.stack(np.nonzero(a_dots == i), 1)
        dot_center_index = np.mean(dot_index, 0)
        dot_coord_lps = i_dots.TransformContinuousIndexToPhysicalPoint(np.flip(dot_center_index))
        dot_coord[i] = [ -x if i < 2 else x for (i, x) in enumerate(dot_coord_lps) ]
        print(f'Dot {i} has center {dot_coord[i]} in Nifti RAS physical coordinates')

    for i, slab in enumerate(slabs):
        if slab_index is None or i == slab_index:
            print(f'Generating mold for Slab {i:02d} of {len(slabs)} with range [{slab["y0"]:5.2f}, {slab["y1"]:5.2f}]')
            process_slab(param, i, slabs[i], i_comp_raw, i_comp_resampled, i_dots, dot_coord)
        


class BrainMoldLauncher:
    
    def __init__(self, parser):
        
        # Required arguments
        parser.add_argument("--subject", "-s", required=True, help="Subject ID, used in naming files")
        parser.add_argument("--hemisphere", "-H", choices=["L", "R"], required=True, help="Which hemisphere (left or right)")
        parser.add_argument("--input", "-i", nargs=2, required=True, metavar=('mri', 'mask'), help="Input MRI and brain mask")
        parser.add_argument("--work", "-w", required=True, help="Work directory for brainmold (will be created)")
        
        # Optional arguments
        parser.add_argument("--guide", "-g", default=None, type=float, metavar='depth',
                            help="Generate guide images for dot placement at given depth from the anterior, in mm")

        # Optional arguments
        # parser.add_argument("--dots", "-d", help="Dots segmentation image")
        # parser.add_argument("--avoidance", "-a", help="Avoidance segmentation image")
        # parser.add_argument("--no-hemi-mold", action="store_true", help="Skip hemisphere mold printing")
        # parser.add_argument("--slab", type=int, default=-1, help="Only generate for one slab (negative for none)")
        
        parser.set_defaults(func = lambda args : self.run(args))
        
    def run(self, args):
        
        # Set the parameters from arguments
        make_mold(id = args.subject, side=args.hemisphere,
                  fn_input_mri = args.input[0], 
                  fn_input_seg = args.input[1],
                  fn_output_root = args.work,
                  dot_guide_depth = args.guide)
        
        
class SlabGeneratorLauncher:
    
    def __init__(self, parser):
        
        # Required arguments
        parser.add_argument("--work", "-w", required=True, help="Work directory for brainmold (will be created)")
        parser.add_argument("--dots", "-d", required=True, help="Dots segmentation image")        
        parser.set_defaults(func = lambda args : self.run(args))
        
    def run(self, args):
        
        # Set the parameters from arguments
        make_slabs(fn_output_root = args.work, fn_dots = args.dots)

