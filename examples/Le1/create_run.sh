#!/bin/bash

SLDF="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cp $SLDF/flamelet_setup.ulf .
cp $SLDF/flamelet_template.ulf .
cp $SLDF/flamelet_mixtureEntries.ulf .
cp $SLDF/ch4_smooke.xml .
cp $SLDF/input.yml .
