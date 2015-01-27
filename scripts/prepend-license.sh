#! /bin/bash

LICENSE="${COVISEDIR}/scripts/license-header.txt"
ENDPATTERN="License:.LGPL"
if [ ! -r "$LICENSE" ]; then
   exit 1
fi

for f in "$@"; do
   if head -n 15 "$f" | grep -q "$ENDPATTERN"; then
      :
      #sed -e '1,/'$ENDPATTERN'/d' "$f" \
      #   | cat "$LICENSE" - > "$f.license" && mv "$f.license" "$f"
   else
      cat "$LICENSE" "$f" > "$f.license" && mv "$f.license" "$f"
   fi
done
