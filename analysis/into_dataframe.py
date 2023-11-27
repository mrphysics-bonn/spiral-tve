# Imports

import argparse
import os
import numpy as np
from dipy.io.image import load_nifti
import pandas as pd

def main(args):

    if args.approach == 'DIVIDE' and args.dtd == 'OP':
        map_flat = np.asarray([0])
    else:
        # load dtd-metrics maps
        filepath_map = args.map_file
        map = load_nifti(filepath_map)[0]

        # transform 3D vectors into 1D vectors:
        if args.approach == 'QTI' and args.dtd == 'MD':
            fac = 1e3
        else:
            fac = 1
        # if dtd == 'MKi':
        #     map = map/map.max()
        map_cut = map[:,:,:]*fac
        map_flat = map_cut.flatten()
        map_flat = map_flat[~np.isnan(map_flat)]
        map_flat = map_flat[map_flat!=0]

    data = {
        'ROI': [args.region]*len(map_flat),
        'Sequence': [args.sequence]*len(map_flat),
        'Approach': [args.approach]*len(map_flat),
        args.dtd: map_flat
    }
    df = pd.DataFrame(data)

    file_path = "dataframe_"+args.dtd+".csv"
    if os.path.exists(file_path):
        df_read = pd.read_csv(file_path)
        df_combined = pd.concat([df_read, df], ignore_index=True)
        df_combined.to_csv(file_path,index=False)
    else:
        df.to_csv(file_path,index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Save parameter map into Pandas Dataframe',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-map_file', '--map_file', type=str, help='Input map file')
    parser.add_argument('-region', '--region', type=str, help='ROI')
    parser.add_argument('-dtd', '--dtd', type=str, help='Diffusion distribution metric. Could be of type MD, FA, uFA, MKi, OP.')
    parser.add_argument('-sequence', '--sequence', type=str, help='Name of the used sequence')
    parser.add_argument('-approach', '--approach', type=str, help='Tensor valued encoding approach. Could be of type DIVIDE or QTI.')

    args = parser.parse_args()

    main(args)
