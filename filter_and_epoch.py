#!/usr/bin/env python

import prepare_raw as pr
from mne import find_events, Epochs


# 'runs' selects which versions of the experiment to epoch. Available
# choices are: [3, 7, 11], [4, 8, 12], [5, 9, 13], and [6, 10, 14]
baseline = (None, 0)
begin = 1
end = 110
event_id = dict(left_fist=2, right_fist=3)
runs = [3, 7, 11]
tmax = 0.5
tmin = -0.2

for num in range(begin, end):
    for run in runs:
        raw = pr.load_subject(num, run)
        pr.fix_channels(raw)
        pr.add_montage(raw)
        # Band-pass filter to capture the relevant signal (alpha and
        # beta ranges). Butterworth filter is implied by method='iir'
        # with iir_params=None or left out.
        raw.filter(7.0, 30.0, method='iir', n_jobs=2)
        events = find_events(raw, consecutive=False)
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=None, baseline=baseline, preload=True)
        epochs.save('{}-epo.fif'.format(raw.info['filename'][:-4]))
