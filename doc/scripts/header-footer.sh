#! /bin/bash

while [ ! -z "$1" ]; do
   if [ "$1" = "-n" ]; then
      shift
      next="$1"
      shift
   elif [ "$1" = "-p" ]; then
      shift
      prev="$1"
      shift
   else
      level="$1"
      shift
   fi
done

if [ -z "$level" ]; then
        level=3
fi

$COVISEDIR/doc/scripts/htmlheader.sh -n "$next" -p "$prev" "$level"
sed -e '1,/^<!-- *Table of Child-Links-->$/d' -e'/^<BR><HR>$/,$d' -e '/^<ADDRESS>$/,$d'
#sed -e '0,/^<!-- *Table of Child-Links-->$/d; /<ADDRESS>/,$d' | sed -n -e :a -e '1,10!{P;N;D;};N;ba'
$COVISEDIR/doc/scripts/htmlfooter.sh -n "$next" -p "$prev" "$level"

