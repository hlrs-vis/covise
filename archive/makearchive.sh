#! /bin/bash

#set -v

# make sure that globs are case sensitive
shopt -u nocaseglob
LANG=C


acceptfile() {
   case "$1" in
      */.svn/*)
         return 1
         ;;
      */core|*/core.*)
         return 1
         ;;
      */WARNINGS|*.pl|*/Makefile|*/Makefile.*|*.in|*.tex|*.log|*.aux|*/html.make)
         if [ ! -z "$DOCDIST" ]; then
             return 1
         else
             return 0
         fi
         ;;
      *)
         return 0
         ;;
   esac
}

processpath() {
        local files
        local links
        local f


	    files=`find "$1" -type f -print`
            links=`find "$1" -type l -print`
            for f in $files $links; do   # ftc = files to check
                if acceptfile $f; then
                   echo "$f" >> "$FILELIST"
                fi
            done

}

processglob() {
   local f

   case "$1" in
      *'*'*)
      for f in `eval echo "$1"`; do

         #
         # $f can be a directory, too ...  :-((
         #
         #echo ASTERISKS
         processpath "$f"
      done
      ;;
      *)
      #echo ANYTHING ELSE
      processpath $1
      ;;
   esac
}



ARCHIVE=$1
shift

if [ "$SRCDIR" = "" ]; then
   DIR=`unset PWD; /bin/pwd`
else
   DIR="$SRCDIR"
fi
DIR=`echo $DIR| sed -e s,//,/,g`

BASEDIR=$COVISEDIR

cd ${BASEDIR}
BASEDIR=`unset PWD; /bin/pwd`
BASEDIR=`echo $BASEDIR| sed -e s,//,/,g`

if [ "${DIR}" = "${BASEDIR}" ]; then
        RELPWD=""
else
        RELPWD=${DIR#${BASEDIR}/}/
fi

cd ${BASEDIR}

FILELIST="${ARCHIVE}.filelist-base"
touch $FILELIST

while [ "$1" ]; do
    F=`echo $1 | sed -e 's,/[^/][^/]*/\.\.,,' -e 's,[^/][^/]*/\.\./,,'`
    if [ "$F" = "${F#/}" ]; then
       FILE="${RELPWD}$F"
    else
       FILE="${F#${BASEDIR}/}"
    fi
    # canonify paths containing /../
    FILE=`echo $FILE | sed -e 's,/[^/][^/]*/\.\.,,' -e 's,[^/][^/]*/\.\./,,'`

    #
    # $files usually contains wildcards - hence we have to assure
    # that no core or contrib files will be archived ...  :-(
    #
    processglob "$FILE"
    shift
done

cd ${EXTERNLIBS}/../.. || exit 1

FILELIST="${ARCHIVE}.filelist-extlib"
touch $FILELIST
for i in $EXTLIBDEP; do
   BASEARCHSUFFIX=`echo ${ARCHSUFFIX} | sed -e s/opt$// -e s/mpi$// -e s/opt$//`
   for j in `grep ^$i ${BASEDIR}/archive/extlibs-${BASEARCHSUFFIX}-opt.txt`; do
      if [ -d "extern_libs/${ARCHSUFFIX}/" ]; then
         EXTLIBARCH=${ARCHSUFFIX}
      else
         EXTLIBARCH=${BASEARCHSUFFIX}
      fi
      if [ -d "extern_libs/${EXTLIBARCH}/$j" ]; then
         processglob "extern_libs/${EXTLIBARCH}/$j/*"
      else
         processglob "extern_libs/${EXTLIBARCH}/$j"
      fi
   done
done
set verbose
