#!/bin/bash
if [ -z "$1" ]; then
	PORT=7778;
else
	PORT="$1";
fi
ssh -L localhost:${PORT}:localhost:${PORT} dyu7@brux20campus.hep.brown.edu -p 2222 "cd BFrag/boffea; echo 'Setting up env...'; source env.sh; echo 'Launching notebook...'; source notebookserver.sh ${PORT};"

