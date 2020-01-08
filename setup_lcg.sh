#!/usr/bin/env bash

source env_lcg.sh

pip install --user coffea

# progressbar, sliders, etc.
jupyter nbextension enable --py widgetsnbextension
