import argparse
from picsl_brainmold.brainmold import BrainMoldLauncher, SlabGeneratorLauncher
from picsl_brainmold.slab_postproc import PaperTemplateLauncher

# Create a parser
parse = argparse.ArgumentParser(
    prog="picsl_brainmold", description="PICSL BrainMold: Generator for 3D Printed Brain Molds")

# Add subparsers for the main commands
sub = parse.add_subparsers(dest='command', help='sub-command help', required=True)

# Add the CRASHS subparser commands
c_bm = BrainMoldLauncher(
    sub.add_parser('mold', help='Generate a cutting mold for a hemisphere'))

c_sg = SlabGeneratorLauncher(
    sub.add_parser('slabs', help='Generate masks for individual slabs'))

c_paper = PaperTemplateLauncher(
    sub.add_parser('paper', help='Generate a paper template for block cutting'))

# Parse the arguments
args = parse.parse_args()
args.func(args)
