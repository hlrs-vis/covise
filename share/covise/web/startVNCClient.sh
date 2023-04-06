#!/usr/bin/env bash

port=5911
ip="$(/sbin/ifconfig \
   | grep '\<inet\>' \
   | sed -n '1p' \
   | tr -s ' ' \
   | cut -d ' ' -f3 \
   | cut -d ':' -f2)"

echo $ip

if [ -f $COVISEDIR/share/covise/web/noVNC/utils/novnc_proxy ]; then
    if  ps -e | grep [O]penCOVER>/dev/null ; then
	echo "cover is running, starting JS-vnc client" $ip
	($COVISEDIR/share/covise/web/noVNC/utils/novnc_proxy --vnc $ip:$port > /dev/null )&
    else
	echo OpenCOVER not running
    fi
fi

echo use the following link: http://$HOSTNAME.hlrs.de:6080/vnc.h:tml?host=$HOSTNAME".hlrs.de&port="$port"&password=password&logging=debug"
