/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_DATAGROUP_H
#define _LIBAPPL_APPL_DATAGROUP_H

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

/******************************************************************************
 *  Data group elem                                                           *
 ******************************************************************************/

typedef appl_data_t appl_data_group_elem_t;

EXTERN appl_data_group_elem_t *appl_data_group_elem_new();
EXTERN void appl_data_group_elem_delete(appl_data_group_elem_t **adge);

/******************************************************************************
 *  Data group                                                                *
 ******************************************************************************/

struct _appl_data_group_t
{
    int num_elems;
    int max_num_elems;
    appl_data_group_elem_t **elems;
};
typedef struct _appl_data_group_t appl_data_group_t;

EXTERN appl_data_group_t *appl_data_group_new();
EXTERN void appl_data_group_delete(appl_data_group_t **adg);

EXTERN void appl_data_group_add_elem(appl_data_group_t *adg, appl_data_group_elem_t *adge);
EXTERN void appl_data_group_add(appl_data_group_t *adg, appl_data_t *data);

#endif
