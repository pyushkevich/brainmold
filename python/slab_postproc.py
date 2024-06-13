import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.ticker
import nibabel as nib
import numpy as np
import argparse
import glob
import os
import pathlib
import subprocess
import textwrap

parse = argparse.ArgumentParser()

parse.add_argument("-i", dest="id", type=str, required=True, help="Specimen ID")
parse.add_argument("-d", dest="dir", type=str, required=True, help="Brainmold output directory")

args = parse.parse_args()

# Get the whole hemisphere MRI file
parent_dir = os.path.dirname(args.dir) # the reslice file is located in the parent directory
mri_reslice = os.path.join(parent_dir, "%s_reslice.nii.gz" % (args.id))

# Find all the matching files in the directory
slab_indices = {}
for i in range(100):
    fimg = os.path.join(args.dir, "%s_slab%02d_mask_with_dots.nii.gz" % (args.id,i))
    if os.path.exists(fimg):
        slab_indices[i] = fimg

print("Found %d input files in %s" % (len(slab_indices), args.dir))

# Create output directory
tex_path = os.path.join(args.dir, "tex")
pathlib.Path(tex_path).mkdir(exist_ok=True)

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

# Start writing LaTeX file
latex = open(os.path.join(tex_path, "%s_print_template_nocuts.tex" % (args.id,)), "wt")
latex.write(textwrap.dedent(s_preamble))

# Repeat for all slab images
for i_slab, fimg in slab_indices.items():

    # Extract slab corresponding to the mask from whole hemisphere MRI
    tmp_slab = os.path.join(args.dir, ("%s_slab%02d.nii.gz" % (args.id, i_slab)))
    c3d_extract_mri_slab = "c3d %s %s -reslice-identity -o %s" % (fimg, mri_reslice, tmp_slab)
    result = subprocess.run(c3d_extract_mri_slab, shell=True, capture_output=True, text=True)

    # Read the image into a NUMPY array
    slab = sitk.ReadImage(fimg)
    slab_arr = sitk.GetArrayFromImage(slab)

    # Read the MRI slab
    mri_slab = nib.load(tmp_slab)
    mri_slab_arr = mri_slab.get_fdata()

    # Get physical (RAS) coordinates of the voxels
    LPS_mat = np.array(slab.GetDirection()).reshape(3,3) @ np.diag(slab.GetSpacing())
    LPS_off = np.array(slab.GetOrigin())
    LPS_to_RAS_mat = np.diag([-1.,-1.,1.])
    idx_ijk = np.flip(np.vstack(np.where(np.isfinite(slab_arr))), axis=0)
    idx_ras = LPS_to_RAS_mat @ ((LPS_mat @ idx_ijk).T + LPS_off).T
    x_ras = idx_ras[0,:].reshape(slab_arr.shape)
    y_ras = idx_ras[1,:].reshape(slab_arr.shape)
    z_ras = idx_ras[2,:].reshape(slab_arr.shape)

    # Get the unique dots
    dot_vals = np.array([x for x in np.unique(slab_arr[:]) if x not in [0, 255]])

    # Report
    print("Making figures for %s; Dots:" % (os.path.basename(fimg),), dot_vals)

    # Get the voxel coordinates of dot centers
    dot_ctrs = np.zeros((dot_vals.size,3))
    for i,d in enumerate(dot_vals):
        dot_ctrs[i,:] = np.mean(idx_ras[:,slab_arr.flat==d], axis=1)

    # Collapse the image along the y axis for 2D printing
    slab_proj = np.mean(slab_arr > 0, axis=1)

    # Get the first slice of the MRI slab
    mri_first_slab = mri_slab_arr[:,1,:]
    mri_first_slab = np.rot90(mri_first_slab)

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
        plt.imshow(mri_first_slab, cmap='gray', alpha=0.5, extent=[x_proj.min(), x_proj.max(), z_proj.min(), z_proj.max()])
        #plt.scatter(dot_ctrs[:,0], dot_ctrs[:,2], c=dot_vals)
        #plt.gca().set_aspect('equal')

        mymarks = ['o','v','x','D','+','*']
        for i,d in enumerate(dot_vals):
            dxy = dot_ctrs[i,[0,2]]
            plt.scatter(dxy[0], dxy[1], color='r', marker=mymarks[i], s=40, label='Dot ' + str(d))
            plt.annotate("Dot " + str(d), dxy, dxy+10,
                        arrowprops=dict(facecolor='black', arrowstyle="->"))

        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10.))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5.))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10.))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5.))
        ax.grid(color="gray", which="major", linestyle=':', linewidth=0.5)
        ax.grid(color="gray", which="minor", linestyle=':', linewidth=0.25, alpha=0.5)

        if len(dot_vals) > 0:
            ax.legend()

        # Output filename
        fn_out[flip] = os.path.join(args.dir, "tex", "%s_slab%02d_template_%d.png" % (args.id, i_slab, flip))
        fig.savefig(fn_out[flip], dpi=300)
        fig.clear()
        plt.close(fig)

    # Remove the temporary MRI slab file
    os.remove(tmp_slab)

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

    latex.write(textwrap.dedent(s_slab % (args.id, i_slab,
                                        fig_width[0], fn_out[0],
                                        fig_width[1], fn_out[1],
                                        s_table)))

latex.write(textwrap.dedent(s_closing))
