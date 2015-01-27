#! /bin/bash

source ${COVISEDIR}/.env.sh

bundle="$1"
appname="$(basename $1 .app)"
libdir="$bundle/Contents/lib"
frameworkdir="$bundle/Contents/Frameworks"
debug=0

function msg() {
    echo "$@" 1>&2
}

function debug() {
    test "$debug" != "0" && msg "$1"
}

function error() {
    msg "$@"
    exit 1
}

function dylibs() {
id=$(otool -D "$1" | grep -v ':$')
otool -L "$1" \
   | grep '^	' \
   | cut -f2 -d'	' \
   | cut -f1 -d' ' \
   | grep -v "^$id$"
}

function unresolved() {
    dylibs "$1" \
   | grep -v '^[[:space:]]*/usr/lib/.*$' \
   | grep -v '^[[:space:]]*/System/Library/.*$'
}

function resolve() {
   local lib="$1"

   local tail="${lib##*.framework}"
   local pathname="${lib%$tail}"
   local framework="${pathname##*/}"
   if [ "${lib##*.framework/}" != "$lib" ]; then
      oldifs="$IFS"
      IFS=":"
      for d in $frameworkdir:$DYLD_FRAMEWORK_PATH:/Library/Frameworks; do
         IFS="$oldifs"
         if [ -d "$d/$framework" ]; then
            echo f "$d/$framework"
            return 0
         fi
      done
      IFS="$oldifs"

      return 1
   fi

   if [ "${lib#*/}" != "$lib" ]; then
      if [ -r "$lib" ]; then
         echo l "$lib"
         return 0
      fi

      lib="${lib##*/}"
   fi

   oldifs="$IFS"
   IFS=":"
   for d in $libdir:$DYLD_LIBRARY_PATH; do
      IFS="$oldifs"
      if [ -r "$d/$lib" ]; then
         echo l "$d/$lib"
         return 0
      fi
   done
   IFS="$oldifs"

   echo "e $lib"

   return 1
}

function fixup() {
   local shobj="$1"

   level=$(($level+1))

   for i in $(unresolved "$shobj"); do
      ret=$(resolve "$i") || error "Could not find $i"
      local libtype="$(echo $ret | cut -f1 -d' ')"
      local src="$(echo $ret | cut -f2- -d' ')"
      local base
      local dst
      local tail
      case "$libtype" in
         l)
            mkdir -p "$libdir"
            base="${i##*/}"
            dst="$libdir/$base"
            msg "$shobj: $i -> @executable_path/../lib/$base"
            install_name_tool -change "$i" "@executable_path/../lib/$base" "$shobj"
            if [ ! -r "$dst" ]; then
               cp "$src" "$dst"
               install_name_tool -id "@executable_path/../lib/$base" "$dst"
               fixup "$dst"
            else
               debug "l: already have $dst"
            fi
            ;;
         f)
            mkdir -p "$frameworkdir"
            base="${src##*/}"
            dst="$frameworkdir/$base"
            tail="${i##*.framework}"
            tail="${tail#/}"
            tail="${tail#/}"
            install_name_tool -change "$i" "@executable_path/../Frameworks/$base/$tail" "$shobj"
            msg "$shobj: $i -> @executable_path/../lib/$base/$tail"
            if [ ! -r "$dst" ]; then
               cp -HR "$src" "$dst"
               install_name_tool -id "@executable_path/../Frameworks/$base/$tail" "$dst/$tail"
               fixup "$dst/$tail"
            else
               debug "f: already have $dst"
            fi
            ;;
         *)
            error "Script error"
            ;;
      esac
   done

   level=$(($level-1))
}

level=0

fixup "$bundle/Contents/MacOS/$appname"

unset noglob
set nullglob
for i in "$bundle/Contents/PlugIns/"*.so "$bundle/Contents/Plugins/"*.dylib; do
   fixup "$i"
done
