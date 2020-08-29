# DV Example Refractory Period Module

This is the solution to a tutorial on https://inivation.gitlab.io/dv/dv-docs/


## What does it do?
This module implements a simple Refractory Period Filter. This means, that any pixel that fires an event gets inhibited from firing more events for a configurable time period.

## Getting started

To compile the module as is, run

```
cmake .
make
sudo make install
```

After this, you shoul dbe able to add the module to your DV configuration in the DV software
