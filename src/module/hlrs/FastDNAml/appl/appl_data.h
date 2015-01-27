/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_DATA_H
#define _LIBAPPL_APPL_DATA_H

#include <stdio.h>

#include <network/network.h>

#include <appl/appl_data_descr.h>
#include <appl/appl_datatype.h>
#include <appl/appl_fastDNAml_types.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

#define APPL_DATA_DESCR_DELETABLE 0x0001
#define APPL_DATA_REPR_DELETABLE 0x0002

/******************************************************************************
 *  Data                                                                      *
 ******************************************************************************/

struct _appl_data_t
{
    struct _appl_data_descr_t *descr;
    void *repr; /* Host representation of datatype or
								  a list with further datatype entries
								  (eg a group of datatypes.)           */

    int flags;
};
typedef struct _appl_data_t appl_data_t;

EXTERN appl_data_t *appl_data_new();
EXTERN appl_data_t *appl_data_new_from_descr(appl_data_descr_t *descr);
EXTERN void appl_data_delete(appl_data_t **ad);
EXTERN void appl_data_delete_non_destr(appl_data_t **ad);

#if 0
EXTERN int appl_data_read_header_ip(appl_data_t *data, sock_t sock);
EXTERN int appl_data_write_header(appl_data_t *data, sock_t sock);
#endif

EXTERN int appl_data_write(appl_data_t *data, sock_t sock);
EXTERN int appl_data_read_ip(appl_data_t *data, sock_t sock);
EXTERN appl_data_t *appl_data_read(sock_t sock);

EXTERN int appl_data_printf(appl_data_t *data);

/******************************************************************************
 *  Data list element                                                         *
 ******************************************************************************/

struct _appl_data_list_elem
{
    struct _appl_data_list_elem *prev, *next;
    appl_data_t *data;
};
typedef struct _appl_data_list_elem appl_data_list_elem;

EXTERN appl_data_list_elem *appl_data_list_elem_new();
EXTERN void appl_data_list_elem_delete(appl_data_list_elem **adle);

EXTERN int appl_data_list_elem_printf(appl_data_list_elem *adle, int off);

/******************************************************************************
 *  Data list                                                                 *
 ******************************************************************************/

struct _appl_data_list_t
{
    int num_elems;
    appl_data_list_elem *start, *end;
};
typedef struct _appl_data_list_t appl_data_list_t;

EXTERN appl_data_list_t *appl_data_list_new();
EXTERN void appl_data_list_delete(appl_data_list_t **adl);

EXTERN void appl_data_list_insert(appl_data_list_t *adl, appl_data_t *ad);

EXTERN appl_data_t *appl_data_list_find(appl_data_list_t *adl, char *name);

EXTERN int appl_data_list_write(appl_data_list_t *adl, sock_t sock);
EXTERN int appl_data_list_read_ip(appl_data_list_t *adl, sock_t sock);
EXTERN appl_data_list_t *appl_data_list_read(sock_t sock);

EXTERN int appl_data_list_printf(appl_data_list_t *adl);

/* Misc. functions */

EXTERN char *appl_treeString(char *treestr, appl_tree *tr, appl_nodeptr p, int form);

#endif
