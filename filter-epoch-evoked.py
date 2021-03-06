#!/usr/bin/env python

from mne import find_events, Epochs, write_evokeds
from mne.preprocessing import ICA
from multiprocessing import cpu_count
from numpy.random import RandomState
from os.path import splitext
from prepare_raw import add_montage, fix_channels, load_subject


# 'runs' selects which versions of the experiment to epoch. Available
# choices are: [3, 7, 11], [4, 8, 12], [5, 9, 13], and [6, 10, 14]

baseline = None
begin = int(input('Enter first subject: '))
end = int(input('Enter last subject: '))
event_id = dict(left_fist=2, right_fist=3)
n_cores = cpu_count()
random_state = RandomState(42)
runs = [3, 7, 11]
tmax = 0.5
tmin = -0.2

with open('data/misc/channel-names.txt') as f:
    ch_names = f.read().splitlines()[:14] + ['STI 014']

# EDF files for subjects 88, 89, 92, 100 have overlapping events, which
# read_raw_edf() cannot handle.

for num in range(begin, end):
    if num in {88, 89, 92, 100}:
        continue
    for run in runs:
        raw = load_subject(num, run)
        fix_channels(raw)
        add_montage(raw)
        raw.pick_channels(ch_names)

        # Band-pass filter to capture the relevant signal (alpha, beta,
        # and mu ranges). Butterworth filter is implied by method='iir'
        # with iir_params=None or left out.

        raw.filter(7.0, 30.0, method='iir', n_jobs=n_cores)
        ica = ICA(n_components=0.95, random_state=random_state)
        ica.fit(raw, decim=3)
        ica.apply(raw)
        events = find_events(raw, consecutive=False)
        epochs = Epochs(raw,
                        events,
                        event_id,
                        tmin,
                        tmax,
                        baseline=baseline,
                        preload=True,
                        proj=False)
        evoked_avg = [epochs[cond].average() for cond in ['left_fist',
                                                          'right_fist']]
        filename = splitext(raw.info['filename'])[0]
        epochs.save(filename + '-epo.fif')
        write_evokeds(filename + '-ave.fif', evoked_avg)
