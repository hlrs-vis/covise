#! /bin/sh

files=""
for i in `cat "$2"`; do
   files="$files $i"
done

${COVISEDIR}/archive/makearchive.sh "$1" $files
