#! /usr/local/bin/bash

DESTDIR="${COVISEDIR}/${ARCHSUFFIX}/system-lib"
rm -rf "$DESTDIR" || true
mkdir -p "$DESTDIR" || true

export DYLD_LIBRARY_PATH="$DESTDIR:$DYLD_LIBRARY_PATH"
export DYLD_FRAMEWORK_PATH="$DESTDIR:$DYLD_FRAMEWORK_PATH"

function error() {
   echo error: "$1"
   return 1
}

# exclude libraries from these directories
WHITELIST=(/opt/X11 /usr/lib /lib /System ${EXTERNLIBS} ${DESTDIR} ${COVISEDIR})

declare -A TOCOPY

function handleobj() {
   local lib
   local d
   local copy

   #echo "$1: ${TOCOPY[$1]}"

   if [ "${TOCOPY[$1]}" != "" ]; then
      return 0
   fi
   TOCOPY["$1"]="-"

   for lib in $(otool -L "$1" \
      | grep '^\t.*\(compatibility version.*current version.*\)$' \
      | sed -e 's/^\t//' -e 's/ (compatibility version.*current version.*)$//'); do
      copy="yes"
      for d in "${WHITELIST[@]}"; do
         if [ "${lib#$d}" != "$lib" ]; then
            copy="no"
            TOCOPY["$lib"]="0"
            continue
         fi
      done
      if [ "$copy" = "yes" ]; then
         handleobj "$lib"
         TOCOPY["$lib"]="1"
      fi
   done

   return 0
}

for i in "$@"; do
   if [ ! -x "$i" -o ! -f "$i" ]; then
      if [ ! -h "$i" ]; then
         continue
      fi
   fi

   handleobj "$i"
done

for lib in ${!TOCOPY[*]}; do
   #echo "${TOCOPY[$lib]}:" "$lib"
   if [ "${TOCOPY["$lib"]}" != "1" ]; then
      continue
   fi
   if echo "$lib" | grep '\.framework/' > /dev/null; then
      framework=$(echo $lib| sed -e 's,\.framework/.*,.framework,')
      #echo Framework: $framework
      base=$(basename "$framework")
      ditto "$framework" "$DESTDIR/$base"
   else
      if [ -f "$lib" ]; then
         cp "$lib" "$DESTDIR"
         base=$(basename "$lib")
         chmod u+rwx "$DESTDIR/$base"
      else
         echo Not found: $lib
      fi
   fi
done

exit 0
