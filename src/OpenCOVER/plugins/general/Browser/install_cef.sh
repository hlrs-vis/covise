#!/bin/bash

mkdir -p $1/ALL
stow -t $1/ALL -d $2 Release Resources
