#! /bin/bash

cd "$1/.."
[ -r CMakeLists.txt ] || exit 0

pwd

DIR=`unset PWD; /bin/pwd`
export NAME=`basename $DIR`
export TARGET="$NAME"
unset GREP_OPTIONS
export CATEGORY=$(grep -m 1 -i ADD_COVISE_MODULE CMakeLists.txt | sed -e 's/^.*( *//' -e 's/ .*$//')

cd doc || exit 0

export DOCDIR=$COVISEDIR/doc
export OUTDIR=$DOCDIR/html/modules/$CATEGORY/$TARGET

if [ "`uname`" = "Darwin" ]; then
   EXTRGX=-E
else
   EXTRGX=-r
fi

mkdir -p $OUTDIR/pict

cp -f pict/*.png $OUTDIR/pict > /dev/null 2>&1

$COVISEDIR/doc/scripts/maketex.sh > $NAME.tex

echo $CATEGORY:$DIR/doc/$NAME >> ${COVISEDIR}/doc/refguide/modulelist.txt

YEAR=`date +%Y`

latex2html \
              -transparent -local_icons \
              -html_version 4.0,table \
              -t "COVISE Online Documentation" \
              -split 4 -link 4 \
              -no_footnode -no_navigation -no_info \
              -address "&copy 1993-$YEAR HLRS, RRZK, Visenso" \
              -mkdir -dir $OUTDIR $TARGET.tex > /dev/null 2>&1

rm -f $OUTDIR/index.html
$COVISEDIR/doc/scripts/htmlheader.sh 3 > $OUTDIR/$TARGET.html.new
echo "<p>Module category: <a href=\"../index.html\">$CATEGORY</a></p><hr>" >> $OUTDIR/$TARGET.html.new
cat $OUTDIR/$TARGET.html \
   | sed $EXTRGX -e '1,/^(ableofchildlinks|<!-- *Table of Child-Links *-->)$/d' \
   | sed -e '/<ADDRESS>/,$d' \
   | sed -n -e :a -e '1,1!{P;N;D;};N;ba' >> $OUTDIR/$TARGET.html.new
$COVISEDIR/doc/scripts/htmlfooter.sh 3 >> $OUTDIR/$TARGET.html.new
mv $OUTDIR/$TARGET.html.new $OUTDIR/$TARGET.html
