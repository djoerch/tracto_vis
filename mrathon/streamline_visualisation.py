
import numpy as np
import nibabel as nib

from queue import Queue
from dipy.tracking.streamline import transform_streamlines
from time import sleep
from fury import window, actor


class StreamlineVisualiser:

    TIME_INC = 0.1

    def __init__(self, plot_size, show, reset_camera, update_camera, window_size):

        self.scene = window.Scene()
        self.plot_size = plot_size
        self.sm = window.ShowManager(scene=self.scene, size=plot_size, reset_camera=False)
        self.show = show
        self.reset_camera = reset_camera
        self.update_camera = update_camera

        self.pos_vec = np.array([0, 0, 0])
        self.view_axis = np.array([0, 0, 0])

        self.q = Queue(maxsize=window_size)

        self.frames = []

    @property
    def window_size(self):
        return self.q.maxsize

    @property
    def view_up(self):
        return [0, 0, 1]

    def add_and_render(self, streamlines, w):
        """
        Add given streamlines to the scene and render the update.

        Parameters
        ----------
        streamlines : nib.streamlines.array_sequence.ArraySequence
            streamlines to update the scene with
        w : float
            weighting factor for the new camera settings

        Returns
        -------
        None
        """

        # remove oldest bunch of streamlines
        if self.q.full():
            self.scene.rm(self.q.get())

        if self.show:
            sleep(StreamlineVisualiser.TIME_INC)

        # add new streamlines to the scene and the queue
        sl_actor = actor.line(streamlines)
        self.scene.add(sl_actor)
        self.q.put(sl_actor)

        # update camera settings
        if self.update_camera:
            self.apply_camera_update(streamlines, w)
        if self.reset_camera:
            self.scene.reset_camera()

        # render updated scene or collect 2D snapshot of the scene
        if self.show:
            self.sm.render()
        else:
            self.frames.append(window.snapshot(self.scene, size=self.plot_size)[::-1, ::-1])

    def apply_camera_update(self, streamlines, w):
        """
        Compute a camera update and perform it.

        Parameters
        ----------
        streamlines : nib.streamlines.array_sequence.ArraySequence
            streamlines based on which to perform the camera update
        w : float
            weight for the new camera settings

        Returns
        -------
        None
        """

        # center of mass of the streamlines
        center = np.mean(np.concatenate(streamlines, axis=0), axis=0)

        # PCA-like description of the streamlines
        cov = np.cov(np.concatenate(streamlines, axis=0).T)
        evals, evecs = np.linalg.eig(cov)

        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]

        # define the view axis as the axis with 'lowest variation' in the data (i.e. in the sl-points)
        view = -1 * evecs[:, -1] if evecs[0, -1] < 0 else evecs[:, -1]

        def weighting(vec, prev, _w):
            return (1 - _w) * prev + _w * vec

        # smoothly update the view axis on the data and the focal point, i.e. the center of the data
        self.view_axis = weighting(view, self.view_axis, w)
        self.pos_vec = weighting(center, self.pos_vec, w)

        # update the camera settings
        self.scene.set_camera(position=self.pos_vec + 80 * self.view_axis,
                              focal_point=self.pos_vec,
                              view_up=self.view_up)

    def load_and_add_regions(self, region_list, transparency=1):
        """
        Load segmentations and add them to the scene as surfaces.

        Parameters
        ----------
        region_list : list[str]
            list of strings, each of which represents the path to a binary segmentation file (.nii)
        transparency : float
            scalar determining transparency of the mask. (full transparency means 0.0)

        Returns
        -------
        affine : 4x4 ndarray
            Affine of all processed ROIs which are guaranteed to have the same affine.
        """
        affine = None

        for region in region_list:

            # load segmentation
            seg = nib.load(region)

            # check if affine matches the previous ones
            if affine is not None:
                if np.all(seg.affine != affine):
                    raise ValueError('Non-matching affines!')
            else:
                affine = seg.affine

            # add segmentation to the scene
            seg_actor = actor.contour_from_roi(seg.get_data(), affine=seg.affine, opacity=transparency)
            self.scene.add(seg_actor)

        return affine

    def load_and_add_reference_tractogram(self, path_to_tractogram):
        """
        Load tractogram from given path and add all streamlines to the scene.
        The camera setup is determined based on this tractogram. The streamlines are visualised
        with transparency.

        Parameters
        ----------
        path_to_tractogram : str
            path to tractogram file (.trk)

        Returns
        -------
        None
        """
        # get streamlines in voxel space
        tracto = nib.streamlines.load(path_to_tractogram)
        streamlines = transform_streamlines(tracto.streamlines, np.linalg.inv(tracto.affine))

        # add streamlines to the scene
        sl_actor = actor.line(streamlines, opacity=0.3)
        self.scene.add(sl_actor)

        # update camera settings
        self.apply_camera_update(streamlines, w=1)
        self.scene.reset_camera()
