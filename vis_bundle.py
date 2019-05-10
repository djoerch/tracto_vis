#!/usr/bin/env python

from time import sleep
import numpy as np

from fury import window, actor
from fury.window import ShowManager

from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects
from dipy.tracking.streamline import transform_streamlines, length

from queue import Queue


from argparse import ArgumentParser, RawTextHelpFormatter


DESC = "Plot streamlines using dipy / fury."
EPILOG = "---\n Author: danjorg@kth.se"


SL_INC = 20
TIME_INC = 0.1


def add_and_render(_sm,  _scene, _streamlines, _q):
    sleep(TIME_INC)
    _sl_actor = actor.line(_streamlines)
    _scene.add(_sl_actor)
    _q.put(_sl_actor)
    pos_vec = get_pos_vector(_streamlines)
    _scene.set_camera(position=100*pos_vec,
                      focal_point=50*pos_vec,
                      view_up=(0.18, 0.00, 0.98))
    _scene.reset_camera()
    _sm.render()
    return


def get_pos_vector(_streamlines):
    # automatic camera position
    cov = np.cov(np.concatenate(_streamlines, axis=0).T)
    evals, evecs = np.linalg.eig(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    if evecs[0, -1] < 0:
        return -1 * evecs[:, -1]
    else:
        return evecs[:, -1]


if __name__ == "__main__":

    ap = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)
    ap.add_argument("--window-size", required=True, type=int, help="Number of streamlines to be shown at a time.")
    ap.add_argument("--plot-size", type=int, nargs=2, default=[600, 600], help="Resolution of render window.")

    args = vars(ap.parse_args())

    # get DATA
    fetch_bundles_2_subjects()
    dix = read_bundles_2_subjects(subj_id='subj_1', metrics=['fa'],
                                  bundles=['cg.left', 'cst.right'])

    fa = dix['fa']
    affine = dix['affine']
    bundle = dix['cg.left'] + dix['cst.right']
    bundle_native = transform_streamlines(bundle, np.linalg.inv(affine))

    fa_actor = actor.slicer(fa)

    scene = window.Scene()
    # scene.set_camera(position=(-176.42, 118.52, 128.20),
    #                  focal_point=(113.30, 128.31, 76.56),
    #                  view_up=(0.18, 0.00, 0.98))
    # scene.add(fa_actor)
    # scene.set_camera(view_up=(0.18, 0.00, 0.98))

    sm = ShowManager(scene=scene, size=args['plot_size'], reset_camera=False)

    maxsize = args['window_size'] // SL_INC
    q = Queue(maxsize=maxsize)

    i = 0
    max_iter = len(bundle_native) // SL_INC

    # fill the queue
    while i < max_iter and not q.full():
        add_and_render(sm, scene, bundle_native[i * SL_INC:(i + 1) * SL_INC], q)
        i += 1

    # update the queue
    while i < max_iter:
        scene.rm(q.get())  # remove oldest bunch of streamlines
        add_and_render(sm, scene, bundle_native[i*SL_INC:(i+1)*SL_INC], q)
        i += 1
