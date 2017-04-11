#!/bin/bash

set -eu

window=`xdotool search --name $1 | head -n 1`

xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Up
sleep 0.2
xdotool key --window $window Up
sleep 0.2
xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Return
# Uncomment for using vision as input
#sleep 0.2
#xdotool key --window $window F2b
