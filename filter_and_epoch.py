#!/usr/bin/env python

from prepare_raw import add_montage, fix_channels, load_subject
from mne import find_events, Epochs
from mne.preprocessing import ICA
from os.path import splitext
from numpy.random import RandomState


# 'runs' selects which versions of the experiment to epoch. Available
# choices are: [3, 7, 11], [4, 8, 12], [5, 9, 13], and [6, 10, 14]

baseline = None
begin = int(input('Enter first subject: '))
end = int(input('Enter last subject: '))
event_id = dict(left_fist=2, right_fist=3)
random_state = RandomState(42)
runs = [3, 7, 11]
tmax = 0.5
tmin = -0.2

# EDF files for subjects 88, 89, 92, 100 have overlapping events, which
# read_raw_edf() cannot handle.

for num in range(begin, end):
    if num in {88, 89, 92, 100}:
        continue
    for run in runs:
        raw = load_subject(num, run)
        fix_channels(raw)
        add_montage(raw)
        # Band-pass filter to capture the relevant signal (alpha and
        # beta ranges). Butterworth filter is implied by method='iir'
        # with iir_params=None or left out.
        raw.filter(7.0, 30.0, method='iir', n_jobs=2)
        ica = ICA(n_components=0.95, random_state=random_state)
        ica.fit(raw, decim=3)
        ica.apply(raw)
        events = find_events(raw, consecutive=False)
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=False,
                        picks=None, baseline=baseline, preload=True)
        filename = splitext(raw.info['filename'])[0]
        epochs.save(filename + '-epo.fif')
