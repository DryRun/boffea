#!/bin/bash
if [ -z "$1" ]; then
	PORT=7778;
else
	PORT="$1";
fi
ssh -N -L localhost:${PORT}:localhost:${PORT} dryu@brux7.hep.brown.edu
