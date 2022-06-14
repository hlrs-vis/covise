#!/bin/bash
echo "building and installing cef"
cd $1/cef
cmake .
make
mkdir -p $1/ALL
stow -t $1/ALL -d $1/cef Release Resources
