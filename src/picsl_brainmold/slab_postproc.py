import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker
# import nibabel as nib
import numpy as np
import argparse
import glob
import os
import pathlib
import subprocess
import textwrap
from picsl_c3d import Convert3D

from picsl_brainmold.brainmold import Parameters, read_config

def make_paper_template(fn_output_root:str):
    
    # Read the configuration and extra parameter
    param, slabs = read_config(fn_output_root)

    # Find all the matching files in the directory
    slab_indices = []
    for i,_ in enumerate(slabs):
        if os.path.exists(param.fn_slab_with_dots_volume_image(i)):
            slab_indices.append(i)

    print(f"Found {len(slab_indices)} input files in {param.fn_output_root}")

    # Latex strings
    s_preamble = """
        \\documentclass{article}
        \\usepackage[table]{xcolor}
        \\usepackage{graphicx}
        \\usepackage[letterpaper, margin=0.25in]{geometry}
        \\renewcommand{\\arraystretch}{1.5}
        \\setlength\\tabcolsep{0.25in}
        \\begin{document}
        """

    s_slab = """
        \\begin{figure}
        \\centering
        {\\centering \\Large \\textbf{Specimen %s slab %02d}}
        \\newline \\newline
        \\includegraphics[width=%fin]{%s}
        \\hfill \\includegraphics[width=%fin]{%s}
        \\newline \\newline
        %s
        \\end{figure}
        """

    s_closing = """
        \\end{document}
        """

    # Label names
    dot_names = {
        1 : "VIS",
        2 : "MOT",
        3 : "PCIN",
        4 : "MF",
        5 : "ACIN",
        6 : "ORF",
        7 : "STEMP",
        8 : "IF",
        9 : "ANTIN",
        10 : "ATEMPP",
        11 : "VLT",
        12 : "SP",
        13 : "ANG",
        14 : "ERC",
        15 : "BA35",
        16 : "CA1",
        17 : "SUB",
        18 : "PHC",
        19 : "AACIN"
    }
    
    # Read the original MRI
    api = Convert3D()
    api.execute(f'{param.fn_input_mri} -popas MRI {param.fn_input_seg} -popas SEG')

    # Start writing LaTeX file
    with open(param.fn_global_texfile(), 'wt') as latex:

        # Write the preample
        latex.write(textwrap.dedent(s_preamble))

        # Repeat for all slab images
        for i_slab in slab_indices:
            
            # Load the slab and reslice MRI to it
            api.execute(f'{param.fn_slab_with_dots_volume_image(i_slab)} -as slab '
                        f'-push MRI -reslice-identity -stretch 1% 99% 0 255 -clip 0 255 -as MRIslab '
                        f'-push SEG -reslice-identity -popas SEGslab')

            # Read the image into a NUMPY array
            slab = api.get_image('slab')
            slab_arr = sitk.GetArrayFromImage(slab)

            # Read the MRI slab
            mri_slab = api.get_image('MRIslab')
            mri_slab_arr = sitk.GetArrayFromImage(mri_slab)

            # Read the segmentation slab
            seg_slab = api.get_image('SEGslab')
            seg_slab_arr = sitk.GetArrayFromImage(seg_slab)

            # Get physical (RAS) coordinates of the voxels
            LPS_mat = np.array(slab.GetDirection()).reshape(3,3) @ np.diag(slab.GetSpacing())
            LPS_mat_inv = np.linalg.inv(LPS_mat)
            LPS_off = np.array(slab.GetOrigin())
            LPS_off_inv = - LPS_mat_inv @ LPS_off
            LPS_to_RAS_mat = np.diag([-1.,-1.,1.])
            idx_ijk = np.flip(np.vstack(np.where(np.isfinite(slab_arr))), axis=0)
            idx_ras = LPS_to_RAS_mat @ ((LPS_mat @ idx_ijk).T + LPS_off).T
            x_ras = idx_ras[0,:].reshape(slab_arr.shape)
            y_ras = idx_ras[1,:].reshape(slab_arr.shape)
            z_ras = idx_ras[2,:].reshape(slab_arr.shape)

            # Get the unique dots
            dot_vals = np.array([int(x) for x in np.unique(slab_arr[:]) if x not in [0, 255]])

            # Report
            print(f"Making figures for slab {i_slab}; Dots: {dot_vals}")

            # Get the voxel coordinates of dot centers
            dot_ctrs = np.zeros((dot_vals.size,3))
            for i,d in enumerate(dot_vals):
                dot_ctrs[i,:] = np.mean(idx_ras[:,slab_arr.flat==d], axis=1)
                
            # Get the RAS y coordinate of the most anterior dot
            y_anterior_most = slabs[i_slab]['y1'] - (param.dot_guide_depth if param.dot_guide_depth is not None else 0.2)
            if dot_ctrs.shape[0] > 0:
                y_anterior_most = np.max(dot_ctrs[:,1])
                
            # Map this coordinate to the slide index
            j_anterior_most = int(np.round((LPS_mat_inv @ LPS_to_RAS_mat @ np.array([0,y_anterior_most,0]) + LPS_off_inv)[1]))
            
            # Collapse the image along the y axis for 2D printing
            # slab_proj = np.mean(slab_arr > 0, axis=1)
            
            # Show slab at the level of the most anterior dot
            slab_proj = seg_slab_arr[:,j_anterior_most,:]
            
            # Get the MRI at the depth of the most anterior dot - this is where the cutting will
            # be taking place most likely

            # Get the first slice of the MRI slab
            mri_first_slab = mri_slab_arr[:,j_anterior_most,:]
            mri_first_slab[slab_proj == 0] = np.nan
            mri_first_slab = np.flip(mri_first_slab, 0)
            
            # mri_first_slab = np.rot90(mri_first_slab)

            # Get the physical coordinates for this projection
            x_proj,z_proj = x_ras[:,0,:], z_ras[:,0,:]

            # Get the anterior-posterior extents of the slab
            ext_anterior = np.amax(y_ras[:])
            ext_posterior = np.amin(y_ras[:])

            # Code for contour plots
            # Plot extents in mm
            figsz_mm = [80., 140.]
            x_ctr, z_ctr = (x_proj[0,0] + x_proj[-1,-1]) / 2, (z_proj[0,0] + z_proj[-1,-1]) / 2
            figext_mm = np.array([
                [ x_ctr - figsz_mm[0]*0.5, x_ctr + figsz_mm[0]*0.5 ],
                [ z_ctr - figsz_mm[1]*0.5, z_ctr + figsz_mm[1]*0.5 ] ])

            fn_out = ["",""]
            fig_width = [0,0]
            for flip in (0,1):
                fig = plt.figure(figsize=(figsz_mm[0]/(0.8*25.4),figsz_mm[1]/(0.9*25.4)))
                fig_width[flip] = fig.get_size_inches()[0]
                ax = fig.add_axes((0.15,0.075,0.8,0.9))
                plt.xlim(figext_mm[0,flip], figext_mm[0,1-flip])
                plt.ylim(figext_mm[1,0], figext_mm[1,1])

                plt.contour(x_proj, z_proj, slab_proj, (0.5,))
                # Show an overlay of the MRI to help visualize the sucli and gyri wrt dots
                plt.imshow(mri_first_slab, cmap='gray', alpha=0.65, extent=[x_proj.min(), x_proj.max(), z_proj.min(), z_proj.max()])
                #plt.scatter(dot_ctrs[:,0], dot_ctrs[:,2], c=dot_vals)
                #plt.gca().set_aspect('equal')

                mymarks = ['o','v','x','D','+','*']
                for i,d in enumerate(dot_vals):
                    dxy = dot_ctrs[i,[0,2]]
                    plt.scatter(dxy[0], dxy[1], color='r', marker=mymarks[i], s=40, label=f'Dot {int(d):d}', alpha=0.75, zorder=3)
                    plt.annotate("Dot " + str(d), dxy, dxy+10, weight='bold',
                                 path_effects=[pe.withStroke(linewidth=1, foreground="white")],
                                 arrowprops=dict(facecolor='black', arrowstyle="->", alpha=0.6))

                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10.))
                ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5.))
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10.))
                ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5.))
                ax.grid(color="gray", which="major", linestyle=':', linewidth=0.5)
                ax.grid(color="gray", which="minor", linestyle=':', linewidth=0.25, alpha=0.5)

                if len(dot_vals) > 0:
                    ax.legend()

                # Output filename
                fn_out[flip] = param.fn_slab_tex_png(i_slab, flip)
                fig.savefig(fn_out[flip], dpi=300)
                fig.clear()
                plt.close(fig)

            # Generate code for the dots depth table
            s_table = ""
            if len(dot_vals) > 0:
                s_table += "\\begin{tabular}{|c|c|p{.8in}|p{.8in}|} \\hline\n"
                s_table += "    \\textbf{Dot} & \\textbf{Region} &\\textbf{Depth from Anterior} & \\textbf{Depth from Posterior} \\\\ \\hline\n"
                for i,d in enumerate(dot_vals):
                    s_table += "    Dot %d & %s & %6.2f mm & %6.2f mm \\\\ \\hline\n" % (
                        d, dot_names[d],
                        ext_anterior - dot_ctrs[i,1],
                        dot_ctrs[i,1] - ext_posterior)
                s_table += "    \\end{tabular}"
            s_table += "\\\\ \\vspace{5mm} \\small{MRI and contour shown {%5.2f}mm from anterior}" % (slabs[i_slab]['y1'] - y_anterior_most)

            latex.write(textwrap.dedent(s_slab % (param.id_string, i_slab,
                                                fig_width[0], os.path.basename(fn_out[0]),
                                                fig_width[1], os.path.basename(fn_out[1]),
                                                s_table)))

        latex.write(textwrap.dedent(s_closing))


class PaperTemplateLauncher:
    
    def __init__(self, parser):
        parser.add_argument("--work", "-w", required=True, help="Work directory for brainmold (will be created)")
        parser.set_defaults(func = lambda args : self.run(args))
        
    def run(self, args):
        make_paper_template(args.work)
    
