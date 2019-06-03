#!/usr/bin/env python

from time import sleep
import numpy as np
import nibabel as nib

from fury import window, actor
from fury.window import ShowManager

from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects
from dipy.tracking.streamline import transform_streamlines

from queue import Queue
from imageio import mimsave

from tqdm import tqdm

from argparse import ArgumentParser, RawTextHelpFormatter


DESC = "Plot streamlines using dipy / fury."
EPILOG = "---\n Author: danjorg@kth.se"


TIME_INC = 0.1


def add_and_render(_sm,  _scene, _streamlines, _q, prev_pos, prev_view, w_pos, reset, update_camera, _show):
    if _show:
        sleep(TIME_INC)
    _sl_actor = actor.line(_streamlines)
    _scene.add(_sl_actor)
    _q.put(_sl_actor)
    if update_camera:
        _view_axis, _pos_vec = get_pos_vector(_streamlines, prev_pos, prev_view, w_pos)
        _scene.set_camera(position=_pos_vec + 80*_view_axis,
                          focal_point=_pos_vec,
                          view_up=(0, 0, 1))
    if reset:
        _scene.reset_camera()
    if _show:
        _sm.render()

    if update_camera:
        return _view_axis, _pos_vec
    else:
        return None, None


def get_pos_vector(_streamlines, prev_pos, prev_view, w_pos):
    """

    Parameters
    ----------
    w_pos : object
    prev_view
    prev_pos
    _streamlines

    Returns
    -------
    view_axis : 3-vector
        best axis to look at the data
    center : 3-vector
        center of mass of the given streamlines
    """

    # center of data
    center = np.mean(np.concatenate(_streamlines, axis=0), axis=0)

    # automatic camera position
    cov = np.cov(np.concatenate(_streamlines, axis=0).T)
    evals, evecs = np.linalg.eig(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    view = -1 * evecs[:, -1] if evecs[0, -1] < 0 else evecs[:, -1]

    def weighting(vec, prev, w):
        return (1 - w) * prev + w * vec

    return weighting(view, prev_view, w_pos), weighting(center, prev_pos, w_pos)


def load_and_add_regions(region_list, _scene):

    _affine = None

    for region in region_list:
        seg = nib.load(region)
        if _affine is not None:
            if np.all(seg.affine != _affine):
                raise ValueError('Non-matching affines!')
        else:
            _affine = seg.affine
        seg_actor = actor.contour_from_roi(seg.get_data(), affine=seg.affine)
        _scene.add(seg_actor)

    return _affine


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

    scene = window.Scene()

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
            affine = load_and_add_regions(args['regions'], scene)
            bundle_native = transform_streamlines(tracto.streamlines, np.linalg.inv(tracto.affine))
        else:
            bundle_native = transform_streamlines(tracto.streamlines, np.linalg.inv(tracto.affine))

    sm = ShowManager(scene=scene, size=args['plot_size'], reset_camera=False)

    maxsize = args['window_size'] // args['sl_per_frame']
    q = Queue(maxsize=maxsize)

    i = 0
    max_iter = len(bundle_native) // args['sl_per_frame']

    frames = []
    show = args['outname'] is None

    pos_vec = np.array([0, 0, 0])
    view_axis = np.array([0, 0, 0])

    # set camera based on reference streamlines
    if args['reference_tractogram']:
        tracto = nib.streamlines.load(args['reference_tractogram'])
        streamlines = transform_streamlines(tracto.streamlines, np.linalg.inv(tracto.affine))
        sl_actor = actor.line(streamlines, opacity=0.3)
        scene.add(sl_actor)
        view_axis, pos_vec = get_pos_vector(streamlines, prev_view=view_axis, prev_pos=pos_vec, w_pos=1)
        scene.set_camera(position=pos_vec + 80*view_axis,
                         focal_point=pos_vec,
                         view_up=(0, 0, 1))
        scene.reset_camera()
        # scene.rm(sl_actor)

    # fill the queue
    while i < max_iter and not q.full():
        view_axis, pos_vec = add_and_render(sm, scene,
                                            bundle_native[i * args['sl_per_frame']:(i + 1) * args['sl_per_frame']], q,
                                            prev_pos=pos_vec, prev_view=view_axis,
                                            w_pos=args['initial_smoothness'], reset=args['reset_camera'], _show=show,
                                            update_camera=not args['reference_tractogram'])
        if not show:
            frames.append(window.snapshot(scene, size=args['plot_size'])[::-1, ::-1])
        i += 1

    # update the queue
    for i in tqdm(range(i, max_iter)):
        scene.rm(q.get())  # remove oldest bunch of streamlines
        view_axis, pos_vec = add_and_render(sm, scene,
                                            bundle_native[i*args['sl_per_frame']:(i+1)*args['sl_per_frame']], q,
                                            prev_pos=pos_vec, prev_view=view_axis,
                                            w_pos=args['smoothness'], reset=args['reset_camera'], _show=show,
                                            update_camera=not args['reference_tractogram'])
        if not show:
            frames.append(window.snapshot(scene, size=args['plot_size'])[::-1, ::-1])

    if show:
        sm.start()

    if args['outname']:
        mimsave(args['outname'], frames, fps=args['fps'])
