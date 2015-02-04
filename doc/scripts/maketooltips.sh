#! /bin/bash

. ${COVISEDIR}/scripts/covise-env.sh

for m in ${COVISEDIR}/${ARCHSUFFIX}/bin/*/*; do
   test -d "$m" && continue
   catmod=`echo "$m" | sed -e s,^"${COVISEDIR}/${ARCHSUFFIX}/bin",, -e s,^/,,`
   cat=`echo "$catmod" | sed -e 's,/.*,,'`
   mod=`echo "$catmod" | sed -e 's,.*/,,'`
   desc=`"$m" -d | grep ^Desc: | sed -e 's/^Desc:[[:space:]]*"//' -e 's/"$//'`
   echo "$cat/$mod:  $desc"
done
