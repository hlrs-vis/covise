/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_H
#define _LIBAPPL_H

#include <appl/appl_funcs.h>
#include <appl/appl_param.h>
#include <appl/appl_data.h>
#include <appl/appl_data_repr.h>
#include <appl/appl_data_distr.h>
#include <appl/appl_footprint.h>
#include <appl/appl_command.h>
#include <appl/appl_state.h>
#include <appl/appl_helper.h>

#include <network/network.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

struct _appl_t
{
    appl_footprint_t *footprint; /* Application footprint */
    appl_funcs_t *funcs;
    appl_param_list_t *params; /* List of registered application params */
    appl_data_list_t *data; /* List of registered application data */
    int state; /* Internal state of application */
    int mode;
    int threshold; /* Threshold of output level */
    sock_t sock;
};
typedef struct _appl_t appl_t;

EXTERN appl_t *appl_new();
EXTERN void appl_delete(appl_t **a);

EXTERN int appl_try_server_connect(appl_t *appl, const char *server_name, short port);

EXTERN void appl_init_values(appl_t *appl);
EXTERN appl_t *appl_init(int *argc, char ***argv, appl_funcs_t *funcs, appl_param_list_t *apl, appl_data_list_t *adl);

EXTERN int appl_start(appl_t *appl);
EXTERN void appl_finish(appl_t **appl);

EXTERN void appl_register_param(appl_t *appl, char *name, appl_datatype_t type, char changeable, void *corr);
EXTERN void appl_register_data(appl_t *app, char *name, appl_datatype_t type, char changeable, void *corr);

EXTERN int appl_get_state(appl_t *appl);
EXTERN void appl_set_state(appl_t *appl, int state);

EXTERN int appl_write_info(appl_t *appl);
EXTERN appl_t *appl_read_info(sock_t sock);

EXTERN int appl_print_info(appl_t *appl);

#endif
