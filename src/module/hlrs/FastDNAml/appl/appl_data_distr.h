/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_DATA_DISTR_H
#define _LIBAPPL_APPL_DATA_DISTR_H

#include <network/network.h>

#include <appl/appl_data.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

EXTERN void *appl_data_distr_create_repr(appl_data_t *data);
EXTERN void appl_data_distr_add_repr(appl_data_t *data, void *repr, int proc);

EXTERN appl_data_t *appl_data_distr_sub_data(appl_data_t *data, int proc);

EXTERN int appl_data_distr_write_proc(appl_data_t *data, int proc, sock_t sock);

#endif
