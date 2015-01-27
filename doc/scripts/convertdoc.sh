#! /bin/bash

if [ -z "$2" ]; then
        moddir=`unset PWD; /bin/pwd`
else
        moddir="$2"
fi

name=`basename "$moddir"`

if [ -z "$1" ]; then
        category=`cat "$moddir"/"$name".pro | awk 'BEGIN { FS="= " }; /^CATEGORY.*=/ { print $2 }'`
        docdir=$COVISEDIR/doc/refguide/modules/"$category"/"$name"
else
        docdir="$1"
fi

mkdir -p "$moddir"/doc

for i in "$docdir"/*.gif; do
        test -r "$i" && giftopnm "$i" | pnmtopng > "$moddir/doc"/`basename "$i" .gif`.png
done


#csplit -k -z "$docdir"/"${name}".tex '%^\\label%+1' '/^\\subsubsection{Parameters}/' '{*}' '/^\\subsubsection{Input Ports}/' '{*}' '/^\\subsubsection{Output Ports}/' '{*}' '/^\\subsubsection{/'

#test -f xx00 && mv xx00 "$2"/doc/beforetable.tex.in
#test -f xx01 && mv xx01 "$2"/doc/parameters.tex.in
#test -f xx02 && mv xx02 "$2"/doc/inports.tex.in
#test -f xx03 && mv xx03 "$2"/doc/outports.tex.in
#test -f xx04 && mv xx04 "$2"/doc/aftertable.tex.in

cd "$moddir"/doc
awk -f $COVISEDIR/doc/scripts/moddocsplit.awk < "$docdir"/"${name}".tex
