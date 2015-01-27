/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NETWORK_BUFFER_H
#define _NETWORK_BUFFER_H

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

struct _network_buffer
{

    int max_size;
    int size;
    int pos;

    void *data;
};
typedef struct _network_buffer network_buffer;

EXTERN network_buffer *network_buffer_new(int init_size);
EXTERN void network_buffer_delete(network_buffer **nb);

#endif
