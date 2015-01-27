/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_FUNCS_H
#define _LIBAPPL_APPL_FUNCS_H

/* #include <appl/appl.h> */

struct _appl_t;

typedef int (*appl_get_num_procs_func)();
typedef char **(*appl_get_hostnames_func)();

typedef int (*appl_start_func)(struct _appl_t *appl, int *argc, char ***argv);
typedef void (*appl_finish_func)(struct _appl_t *appl);

struct _appl_funcs_t
{

    appl_get_num_procs_func get_num_procs;
    appl_get_hostnames_func get_hostnames;

    appl_start_func start;
    appl_finish_func finish;
};
typedef struct _appl_funcs_t appl_funcs_t;

extern appl_funcs_t *appl_funcs_new();
extern void appl_funcs_delete(appl_funcs_t **af);

extern int appl_funcs_get_num_procs_generic();
extern char **appl_funcs_get_hostnames_generic();

#endif
