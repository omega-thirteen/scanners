Scanners
======
Predicting motion based on electroencephalographic recordings (EEG).

## Introduction
The [electroencephalograph](http://en.wikipedia.org/wiki/Electroencephalography), 
or EEG, is a common medical device first developed to diagnose and
monitor neurological disorders such as epilepsy. To record brain
activity, up to 128 electrodes are affixed to the patient's scalp at
[standardized points](http://en.wikipedia.org/wiki/10-20_system_\(EEG\)). 
When large numbers of neurons below an electrode fire in unison, the
difference in electric potential (measured in microvolts) is registered
between that electrode and a reference electrode, which is usually
attached to one or both earlobes. The EEG can be administered while the
patient performs simple tasks -- moving arms or legs, observing images
on a screen, listening to music -- or even while he or she is asleep.
The EEG's notable advantages, those being its low cost and minimally
invasive nature, lend it to applications beyond the hospital setting.
The nascent field of [brain-computer
interfaces](http://en.wikipedia.org/wiki/Brain-computer_interface#Non-invasive_BCIs) 
(BCI) is one such application.

## Dataset
[EEG Motor Movement/Imagery Dataset](https://physionet.org/pn4/eegmmidb/), or EEGMMIDB, from [PhysioNet](https://physionet.org):

>Subjects performed different motor/imagery tasks while 64-channel EEG
>were recorded using the [BCI2000 system](http://www.bci2000.org). Each
>subject performed 14 experimental runs: two one-minute baseline runs
>(one with eyes open, one with eyes closed), and three two-minute runs
>of each of the four following tasks: A target appears on either the
>left or the right side of the screen. The subject opens and closes
>the corresponding fist until the target disappears. Then the subject
>relaxes. A target appears on either the left or the right side of the
>screen. The subject imagines opening and closing the corresponding fist
>until the target disappears. Then the subject relaxes. A target appears
>on either the top or the bottom of the screen. The subject opens and
>closes either both fists (if the target is on top) or both feet (if the
>target is on the bottom) until the target disappears. Then the subject
>relaxes. A target appears on either the top or the bottom of the screen.
>The subject imagines opening and closing either both fists (if the
>target is on top) or both feet (if the target is on the bottom) until
>the target disappears. Then the subject relaxes.

## Separating wheat from chaff
Easy and cheap to obtain though they may be, EEG recordings do have
[inherent drawbacks](http://en.wikipedia.org/wiki/Electroencephalography#/Limitations). 
For one, the noise-to-signal ratio is uncomfortably high, and 
[artifacts](http://en.wikipedia.org/wiki/Electroencephalography#/Artifacts) 
are difficult to avoid: electrical interference, both endogenous
(heartbeat, eye movement, changes in skin conductivity due to sweating)
and exogenous (AC power mains), can make interpretation of recordings
difficult or impossible.

Fortunately, these problems can be substantially mitigated with
the aid of certain mathematical techniques. Low-pass and high-pass
[filters](http://en.wikipedia.org/wiki/Butterworth_filter) were
applied to the raw signal at 30 Hz and 7 Hz, respectively, to
remove high-frequency noise (including that produced by [mains
electricity](http://en.wikipedia.org/wiki/Mains_electricity) at 60
Hz) as well as low-frequency artifacts caused by heartbeat and eye
movement. Conveniently, the band from 7-30 Hz is likely to carry
all the information we seek: normal, waking neural oscillations
are dominated by [alpha](http://en.wikipedia.org/wiki/Alpha_wave)
and [beta](http://en.wikipedia.org/wiki/Beta_wave) waves. More
to the point, motor activity is typically marked by activity
within the [mu band](http://en.wikipedia.org/wiki/Mu_wave),
from 7.5-12.5 Hz. While one is at rest, neurons in the [motor
cortex](http://en.wikipedia.org/wiki/Motor_cortex) tend to fire in
unison. At the onset of motion, however, these neurons' activity becomes
desynchronized, and therein lies the key to successful prediction.

[Independent component
analysis](http://en.wikipedia.org/wiki/Independent_component_analysis)
(the [FastICA implementation](http://en.wikipedia.org/wiki/FastICA),
to be specific) was then performed to achieve [signal
separation](http://en.wikipedia.org/wiki/Blind_signal_separation).
Why is this necessary? Imagine you have placed three microphones in
a room where a lively cocktail party is being held. With multiple
conversations occurring simultaneously, each microphone is bound
to pick up at least some of the sound captured by the other two.
If one can safely assume that the component signals (the voices
of individual people at the cocktail party) are [statistically
independent](http://en.wikipedia.org/wiki/Statistical_independence) and
that the component signals' values are not normally distributed, then
ICA can reliably reproduce those signals.

## Crossing the streams
In their initial continuous form, EEG recordings are
not suitable ground on which to build a predictive
algorithm. The stream must be sectioned into
[epochs](http://en.wikipedia.org/wiki/Quantitative_electroencephalography#Fourier_analysis_of_EEG), 
time-locked series which contain the event of interest.
Like-kind epochs can then be averaged across multiple recordings
and compared between and among experimental subjects. These
averages are known in the literature as [event-related
potentials](http://en.wikipedia.org/wiki/Event-related_potential).

## Classification procedure
At this early stage, I have limited the scope of my project to the first
version of the experiments found in EEGMMIDB, in which each subject was
instructed to clench the corresponding fist when an image appeared on
either the left or right side of a screen.

From the recordings of 109 subjects who participated in the
experiment, four sets were removed from the dataset because of
formatting discrepancies. Those remaining were divided into epochs
(200 milliseconds before the event and 500 milliseconds after).
Only the fourteen electrodes located above the [primary motor
cortex](http://en.wikipedia.org/wiki/Primary_motor_cortex) were used in
this context.

After packaging the data in a compatible form using
[MNE](http://mne-tools.github.io/stable/index.html) and standardizing
each feature, I divided them into training and testing sets
(80% and 20%, respectively) and then employed scikit-learn's
[SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html), 
specifying one of two applicable loss functions, and
[RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html) 
to tune parameters for the classifier.

Here are the results produced by SGDClassifier with ['hinge' loss
function](http://en.wikipedia.org/wiki/Hinge_loss), which gives a linear
support vector machine:

```
RandomizedSearchCV took 12.69 seconds for 20 candidate parameter settings.
Model with rank: 1
Mean validation score: 0.499 (std: 0.002)
Parameters: {'n_iter': 9, 'alpha': 17.884975133166026, 'shuffle': True}

Model with rank: 2
Mean validation score: 0.499 (std: 0.002)
Parameters: {'n_iter': 10, 'alpha': 0.12765806765435153, 'shuffle': False}

Model with rank: 3
Mean validation score: 0.499 (std: 0.002)
Parameters: {'n_iter': 18, 'alpha': 0.1495637472206022, 'shuffle': False}
```

It's plain to see that the classifier is performing no better
than chance (a score of approximately 50%). I suspect this
stems from the format of the input data. MNE provides an
[EpochsVectorizer](http://mne-tools.github.io/stable/generated/mne.decoding.EpochsVectorizer.html) 
class that, as the name suggests, reduces the dimensions of each epoch.
Thus far I have not been able to make it work as expected.


## Download
```
$ git clone https://github.com/omega-thirteen/scanners.git
```
or, using your SSH key:
```
$ git clone git@github.com:omega-thirteen/scanners.git
```
