/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_DATA_SPEC_H
#define _LIBAPPL_APPL_DATA_SPEC_H

#include <appl/appl_data_descr.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

struct _appl_data_descr_t;

/******************************************************************************
 * Specification of non-scalar datatype                                       *
 ******************************************************************************/

/* Array specification: Number of elements and type description */

struct _appl_data_spec_array
{
    int num;
    struct _appl_data_descr_t *descr;
};
typedef struct _appl_data_spec_array appl_data_spec_array;

EXTERN appl_data_spec_array *appl_data_spec_array_new(int num);
EXTERN void appl_data_spec_array_delete(appl_data_spec_array **adsa);

EXTERN int appl_data_spec_array_write(appl_data_spec_array *adsa, sock_t sock);
EXTERN int appl_data_spec_array_read_ip(appl_data_spec_array *adsa, sock_t sock);

/* Specification for group of data elements */

struct _appl_data_spec_group
{
    int max_num_elems;
    int num_elems;
    struct _appl_data_descr_t **descr;
};
typedef struct _appl_data_spec_group appl_data_spec_group;

EXTERN appl_data_spec_group *appl_data_spec_group_new();
EXTERN void appl_data_spec_group_delete(appl_data_spec_group **adsg);

EXTERN void appl_data_spec_group_add(appl_data_spec_group *adsg, struct _appl_data_descr_t *descr);

EXTERN int appl_data_spec_group_write(appl_data_spec_group *adsg, sock_t sock);
EXTERN int appl_data_spec_group_read_ip(appl_data_spec_group *adsg, sock_t sock);

/* Specification of a distributed data type */

struct _appl_data_spec_distr
{

    int max_num_elems;
    int num_elems;

    int *procs; /* Array with processor numbers */
    struct _appl_data_descr_t **descr; /* Array with data descriptions */
};
typedef struct _appl_data_spec_distr appl_data_spec_distr;

EXTERN appl_data_spec_distr *appl_data_spec_distr_new();
EXTERN void appl_data_spec_distr_delete(appl_data_spec_distr **adsd);

EXTERN void appl_data_spec_distr_init(appl_data_spec_distr *adsd);

EXTERN void appl_data_spec_distr_add(appl_data_spec_distr *adsd, struct _appl_data_descr_t *descr, int proc);

EXTERN struct _appl_data_descr_t *appl_data_spec_distr_find(appl_data_spec_distr *adsd, int p);

EXTERN int appl_data_spec_distr_write(appl_data_spec_distr *adsd, sock_t sock);
EXTERN int appl_data_spec_distr_read_ip(appl_data_spec_distr *adsf, sock_t sock);

#endif
