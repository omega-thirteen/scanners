from mne import concatenate_raws
from mne.channels import read_montage
from mne.datasets.eegbci import load_data
from mne.io import read_raw_edf


# EDF files for subjects 88, 89, 92, 100 have overlapping events, which
# read_raw_edf() cannot handle.

def load_subject(id_num, runs):
    '''
    Loads raw EEG recordings for one subject and at least one run of
    experiments.

    Arguments:
        id_num: int, the subject's ID number
        runs: int or list of ints -- which experiment(s) to read data from

    Returns:
        MNE Raw object
    '''
    edf_files = load_data(id_num, runs)
    if len(edf_files) > 1:
        raw_objects = [read_raw_edf(file, preload=True) for file in edf_files]
        mne_raw = concatenate_raws(raw_objects, preload=True)
    else:
        mne_raw = read_raw_edf(edf_files[0], preload=True)
    return mne_raw


def fix_channels(mne_raw, channels='data/misc/channel-names.txt'):
    '''
    1) Fixes channel names to comply with 'standard_1005' format.
    2) Fixes stimulus channel type.
    Arguments:
        mne_raw: MNE Raw object

    Returns:
        MNE Raw object (modified in place)
    '''
    ch_names = open(channels).read().splitlines()
    renamer = {old: new for old, new in zip(mne_raw.ch_names[:-1], ch_names)}
    mne_raw.rename_channels(renamer)
    mne_raw.set_channel_types({'STI 014': 'stim'})


def add_montage(mne_raw, channels='data/misc/channel-names.txt'):
    '''
    Creates 'standard_1005' montage with corrected channel names and
    adds it to MNE raw object.

    Arguments:
        mne_raw: MNE Raw object

    Returns:
        MNE Raw object (modified in place)
    '''
    ch_names = open(channels).read().splitlines()
    montage = read_montage('standard_1005', ch_names=ch_names)
    mne_raw.set_montage(montage)
