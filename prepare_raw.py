from mne import concatenate_raws
from mne.channels import read_montage
from mne.datasets.eegbci import load_data
from mne.io import read_raw_edf


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
    raw_objects = [read_raw_edf(file, preload=True) for file in edf_files]
    mne_raw = concatenate_raws(raw_objects, preload=True)
    return mne_raw


def fix_channels(mne_raw):
    '''
    Fixes channel names to comply with 'standard_1005' format.

    Arguments:
        mne_raw: MNE Raw object

    Returns:
        MNE Raw object
    '''
    with open('data/channel-names.txt') as f:
        ch_names = [line.strip() for line in f]

    renamer = {old: new for old, new in zip(mne_raw.ch_names[:-1], ch_names)}
    mne_raw.rename_channels(renamer)
    return mne_raw


def add_montage(mne_raw):
    '''
    Creates 'standard_1005' montage with corrected channel names and
    adds it to MNE raw object.

    Arguments:
        mne_raw: MNE Raw object

    Returns:
        MNE Raw object
    '''
    with open('data/channel-names.txt') as f:
        ch_names = [line.strip() for line in f]

    montage = read_montage('standard_1005', ch_names=ch_names)
    mne_raw.set_montage(montage)
