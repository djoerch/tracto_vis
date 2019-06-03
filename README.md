# Example calls

### live visualisation

`python vis_bundle.py --window-size 500 --plot-size 1000 1000 --fps 12 --sl-per-frame 10 --reset-camera`

### with gif creation

`python vis_bundle.py --window-size 500 --plot-size 1000 1000 --outname sl_video_example.gif --fps 12 --sl-per-frame 100
--reset-camera`

### live vis with reference tractogram and ROIs

`python vis_bundle.py --window-size 100 --plot-size 1000 1000 --fps 12 --sl-per-frame 10
--tractogram tractogram.trk --regions seed_1.nii target_1.nii --reference-tractogram tractogram3.trk`