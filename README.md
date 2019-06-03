# Install

`pip install git+ssh://git@bitbucket.org/djoerch/mrathon`

# Example calls

### live visualisation

`vis_bundle --window-size 500 --plot-size 1000 1000 --fps 12 --sl-per-frame 10 --reset-camera`

### with gif creation

`vis_bundle --window-size 500 --plot-size 1000 1000 --outname sl_video_example.gif --fps 12 --sl-per-frame 100
--reset-camera`

### live vis with reference tractogram and ROIs

`vis_bundle --window-size 100 --plot-size 1000 1000 --fps 12 --sl-per-frame 10
--tractogram tractogram.trk --regions seed_1.nii target_1.nii --reference-tractogram tractogram3.trk`