#! /bin/bash
for f in "$@"; do
   dos2unix "$f"
   clang-format -style="{BasedOnStyle: WebKit, PointerBindsToType: false, BreakBeforeBraces: Allman}" -i "$f"
   #bcpp -fnc ${COVISEDIR}/scripts/bcpp.cfg -fi "$f" > "$f.indent" && mv "$f.indent" "$f"
   #astyle --mode=c -s3 -b -o -O --convert-tabs --break-blocks --suffix='.indent' "$f" && rm -f "$f.indent"
done
