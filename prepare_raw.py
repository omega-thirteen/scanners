from mne import concatenate_raws
from mne.channels import read_montage
from mne.datasets.eegbci import load_data
from mne.io import read_raw_edf


with open('data/channel-names.txt') as f:
    ch_names = [line.strip() for line in f]


# A montage contains the spatial location of each electrode on an
# idealized spherical model of the human head.

montage = read_montage('standard_1005', ch_names=ch_names)

# First parameter is the subject's ID number; second parameter lists the
# recording(s) to be loaded. Below, the locations of all three runs of
# the first experiment are added.

edf_files = load_data(1, [3, 7, 11])

raw_objects = [read_raw_edf(file, preload=True) for file in edf_files]

raw = concatenate_raws(raw_objects, preload=True)

# Channel names do not follow standard practice in original EDF files.

renamer = {old: new for old, new in zip(raw.ch_names[:-1], ch_names)}
raw.rename_channels(renamer)
raw.set_montage(montage)
