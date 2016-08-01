from glob import glob
from mne import read_epochs, read_evokeds


# The following block collects all epochs into an HDF5 file:

epoch_files = sorted(glob('**/*-epo.fif', recursive=True))

for f in epoch_files:
    df = read_epochs(f,
                     proj=False,
                     preload=True).to_data_frame(index=None)
    df.rename(columns={'STI 014': 'event'}, inplace=True)
    df.to_hdf('data/misc/epochs.h5',
              'epochs',
              format='t',
              append=True,
              complib='blosc',
              complevel=9)

# The following block collects all evokeds into an HDF5 file:

evoked_files = sorted(glob('**/*-ave.fif', recursive=True))
for f in evoked_files:
    for cond in ['left_fist', 'right_fist']:
        df = read_evokeds(f,
                          condition=cond,
                          baseline=None,
                          kind='average',
                          proj=False,
                          verbose=None).to_data_frame(index=None)
        df.rename(columns={'STI 014': 'event'}, inplace=True)
        df.to_hdf('data/misc/evokeds.h5',
                  'evokeds',
                  format='t',
                  append=True,
                  complib='blosc',
                  complevel=9)
