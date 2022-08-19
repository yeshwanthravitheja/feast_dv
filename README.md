# FEAST Module for DV-GUI 

Install DV-GUI and learn how to make a DV module from source from https://inivation.gitlab.io/dv/dv-docs/


## What does it do?
This module implements Feature Extraction using Adaptive Selection Thresholds(FEAST) an unsupervised online learning algorithm to extract most common spatio-temporal features from the event data of a neuromorphic vision sensor(DAVIS). 


## Getting started

To compile the module as is, run

```
cmake .
make
sudo make install
```

After this, you should be able to add the module to your DV configuration in the DV software

Checkout more information regarding FEAST from the paper:
```
@article{afshar2020event,
  title={Event-based feature extraction using adaptive selection thresholds},
  author={Afshar, Saeed and Ralph, Nicholas and Xu, Ying and Tapson, Jonathan and Schaik, Andr{\'e} van and Cohen, Gregory},
  journal={Sensors},
  volume={20},
  number={6},
  pages={1600},
  year={2020},
  publisher={MDPI}
}
```
