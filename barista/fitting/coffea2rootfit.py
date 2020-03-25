
import uproot
import glob
from coffea import util
from collections import defaultdict
from functools import partial

def coffea2roofit(input_files, input_obj, output_file, output_obj=None, combine_datasets=True):
    '''
    Convert {dataset:Bcand_accumulator} output to TTree.
    Bcand_accumulator has dict-like structure {branch: array}
    input_obj = key or keys needed to find Bcand object
    datasets are added together.
    '''
    print("Welcome to coffea2roofit")
    print("Input files:")
    print(input_files)
    print("Output file:")
    print(output_file)

    # Make list of datasets and branches
    branches = []
    branch_types = {}
    input_stuff = util.load(input_files[0])
    for k1 in input_stuff[input_obj].keys():
        for branch_name in input_stuff[input_obj][k1].keys():
            branches.append(branch_name)
            branch_types[branch_name] = input_stuff[input_obj][k1][branch_name].value.dtype
        break

    if not output_obj:
        output_obj = input_obj # Use same name

    for input_file in input_files:
        print("Processing {}".format(input_file))
        input_stuff = util.load(input_file)
        for k1 in input_stuff[input_obj].keys():
            # Determine which tree to fill, and create it if it doesn't exist
            if combine_datasets:
                tree_name = output_obj
            else:
                tree_name = f"{output_obj}_{k1}"
            if not tree_name in output_file:
                print("Creating tree {}".format(tree_name))
                output_file[tree_name] = uproot.newtree(branch_types)

            # Make {branch : array} dict for filling
            bcand_accumulator = input_stuff[input_obj][k1]
            bcand_array = {}
            for branch in branches:
                bcand_array[branch] = bcand_accumulator[branch].value

            # Fill
            output_file[tree_name].extend(bcand_array)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a Bcand_accumulator to a TTree")
    parser.add_argument("--input_objs", "-j", type=str, required=True, help="Bcand_accumulator objects to convert (comma-separated)")
    parser.add_argument("--output_file", "-o", type=str, required=True, help="Output file")
    parser.add_argument("--input_files", "-i", type=str, required=True, help="glob string for list of input files")
    parser.add_argument("--combine_datasets", "-c", action="store_true", help="Combine datasets into one tree")
    args = parser.parse_args()

    input_files = glob.glob(args.input_files)
    input_objs = args.input_objs.split(",")
    with uproot.recreate(args.output_file) as output_file:
        for input_obj in input_objs:
            coffea2roofit(input_files, input_obj, output_file, combine_datasets=args.combine_datasets)

