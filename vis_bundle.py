#!/usr/bin/env python

import numpy as np
import nibabel as nib

from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects
from dipy.tracking.streamline import transform_streamlines

from imageio import mimsave
from tqdm import tqdm
from argparse import ArgumentParser, RawTextHelpFormatter


from mrathon.streamline_visualisation import StreamlineVisualiser


DESC = "Plot streamlines using dipy / fury."
EPILOG = "---\n Author: danjorg@kth.se"


if __name__ == "__main__":

    ap = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)
    ap.add_argument("--window-size", required=True, type=int, help="Number of streamlines to be shown at a time.")
    ap.add_argument("--plot-size", type=int, nargs=2, default=[600, 600], help="Resolution of render window.")
    ap.add_argument("--outname", type=str, help="Path to output file.")
    ap.add_argument("--smoothness", type=float, default=0.001, help="Weighting of new camera position wrt previous.")
    ap.add_argument("--initial-smoothness", type=float, default=0.5, help="Smoothness during fillup phase.")
    ap.add_argument("--reset-camera", action='store_true', help="Activate resetting of camera in rendering.")
    ap.add_argument("--sl-per-frame", type=int, default=20, help="Number of streamlines per frame.")
    ap.add_argument("--fps", type=int, default=24, help="Frames per second.")
    ap.add_argument("--tractogram", type=str, help="Input file for tractogram to be displayed.")
    ap.add_argument("--regions", type=str, nargs='*', help="Path to a segmentation file being displayed in order " +
                                                           "to have a static reference during updates.")
    ap.add_argument("--reference-tractogram", type=str, help="Use for deciding on static camera position.")

    args = vars(ap.parse_args())

    sl_vis = StreamlineVisualiser(plot_size=args['plot_size'], show=args['outname'] is None,
                                  update_camera=not args['reference_tractogram'], reset_camera=args['reset_camera'],
                                  window_size=args['window_size'] // args['sl_per_frame'])

    if not args['tractogram']:

        # get DATA
        fetch_bundles_2_subjects()
        dix = read_bundles_2_subjects(subj_id='subj_1', metrics=['fa'],
                                      bundles=['cg.left', 'cst.right'])

        fa = dix['fa']
        affine = dix['affine']
        bundle = dix['cg.left'] + dix['cst.right']
        bundle_native = transform_streamlines(bundle, np.linalg.inv(affine))

        # fa_actor = actor.slicer(fa)
        # scene.add(fa_actor)

    else:

        tracto = nib.streamlines.load(args['tractogram'])

        # create static vis of a segmentation
        if args['regions']:
            affine = sl_vis.load_and_add_regions(args['regions'])
            bundle_native = transform_streamlines(tracto.streamlines, np.linalg.inv(tracto.affine))
        else:
            bundle_native = transform_streamlines(tracto.streamlines, np.linalg.inv(tracto.affine))

    max_iter = len(bundle_native) // args['sl_per_frame']

    # set camera based on reference streamlines
    if args['reference_tractogram']:
        sl_vis.load_and_add_reference_tractogram(args['reference_tractogram'])
        # scene.rm(sl_actor)

    # vis and render
    for i in tqdm(range(0, max_iter)):
        sl_vis.add_and_render(bundle_native[i*args['sl_per_frame']:(i+1)*args['sl_per_frame']],
                              w=args['smoothness'] if i > sl_vis.window_size else args['initial_smoothness'])

    if sl_vis.show:
        sl_vis.sm.start()

    if args['outname']:
        mimsave(args['outname'], sl_vis.frames, fps=args['fps'])
