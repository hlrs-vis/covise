#! /bin/sh

PLATFORM=${COVISEDIR}/DIST/DIST.$ARCHSUFFIX/Platform
BASEARCH=`basename $ARCHSUFFIX opt`

case $BASEARCH in
    sgin32)
        printf "DESC SGI IRIX 6.5.6 or later, 32 bit" > $PLATFORM
        ;;
    sgi64)
        printf "DESC SGI IRIX 6.5.6 or later, 64 bit" > $PLATFORM
        ;;
    linux)
        printf "DESC SuSE Linux 7.3 on IA32, gcc 2.95" > $PLATFORM
        ;;
    gcc3)
       printf "DESC Red Hat Linux 8 on IA32, gcc 3" > $PLATFORM
        ;;
    amd64)
       printf "DESC Fedora Core 3 on x86_64, 64 bit" > $PLATFORM
       ;;
    *)
        # Use descriptions from README-ARCHSUFFIX by default
        printf "DESC %s" "$(fgrep $BASEARCH ${COVISEDIR}/README-ARCHSUFFIX.txt | grep -v \# | tr -s '[:blank:]' | cut -f 2- -d\ )" > $PLATFORM
        ;;
esac

if [ "$BASEARCH" != "$ARCHSUFFIX" ]; then
        echo " (optimized)" >> $PLATFORM
else
        echo " (debug)" >> $PLATFORM
fi
