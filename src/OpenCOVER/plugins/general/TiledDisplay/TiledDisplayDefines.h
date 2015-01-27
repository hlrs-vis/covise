/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_DEFINES_H
#define TILED_DISPLAY_DEFINES_H

#ifdef __APPLE__ // workaround for crummy OpenTransportProviders.h
#undef TCP_NODELAY
#undef TCP_MAXSEG
#undef TCP_NOTIFY_THRESHOLD
#undef TCP_ABORT_THRESHOLD
#undef TCP_CONN_NOTIFY_THRESHOLD
#undef TCP_CONN_ABORT_THRESHOLD
#undef TCP_OOBINLINE
#undef TCP_URGENT_PTR_TYPE
#undef TCP_KEEPALIVE
#endif

#if 0
#for QMAKE, set to one if you enable jpeg encoding(also change the define below)
encode_jpeg=0
#endif

#undef TILE_ENCODE_JPEG

#endif
