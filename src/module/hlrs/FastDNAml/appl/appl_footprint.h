/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_FOOTPRINT_H
#define _LIBAPPL_APPL_FOOTPRINT_H

#include <appl/appl_param.h>
#include <appl/appl_funcs.h>

#include <network/network.h>

#ifndef MAX_HOSTNAME_LEN
#define MAX_HOSTNAME_LEN 1024
#endif

struct _appl_footprint_t
{

    char *name;
    char *hostname;

    int num_processes;

    char **hnames;
};
typedef struct _appl_footprint_t appl_footprint_t;

extern appl_footprint_t *appl_footprint_new();
extern void appl_footprint_delete(appl_footprint_t **afp);

extern int appl_footprint_gather(appl_footprint_t *afp, int *argc, char ***argv, appl_funcs_t *funcs);

extern int appl_footprint_write(appl_footprint_t *afp, sock_t sock);
extern int appl_footprint_read_ip(appl_footprint_t *afp, sock_t sock);
extern appl_footprint_t *appl_footprint_read(sock_t sock);

extern void appl_footprint_print(appl_footprint_t *afp);

#endif
