/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_DATA_DESCR_H
#define _LIBAPPL_APPL_DATA_DESCR_H

#include <network/network.h>

#include <appl/appl_data_spec.h>
#include <appl/appl_datatype.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

#define APPL_DATA_DESCR_SPEC_DELETABLE 0x0001

/******************************************************************************
 *  Data description                                                          *
 ******************************************************************************/

struct _appl_data_descr_t
{

    char *name;
    appl_datatype_t type;

    void *spec; /* Further specification of data type (see above) */

    int flags;
};
typedef struct _appl_data_descr_t appl_data_descr_t;

EXTERN appl_data_descr_t *appl_data_descr_new();
EXTERN void appl_data_descr_delete(appl_data_descr_t **add);
EXTERN void appl_data_descr_init(appl_data_descr_t *add);

EXTERN int appl_data_descr_write(appl_data_descr_t *add, sock_t sock);
EXTERN int appl_data_descr_read_ip(appl_data_descr_t *add, sock_t sock);
EXTERN appl_data_descr_t *appl_data_descr_read(sock_t sock);

EXTERN int appl_data_descr_printf(appl_data_descr_t *add, int off);

#endif
