/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_PARAM_H
#define _LIBAPPL_APPL_PARAM_H

#include <appl/appl_datatype.h>

#include <network/network.h>

/******************************************************************************
 * Parameter description                                                      *
 ******************************************************************************/

struct _appl_param_descr_t
{
    char *name;
    int num;
    appl_datatype_t type;
    char changeable;
};
typedef struct _appl_param_descr_t appl_param_descr_t;

extern appl_param_descr_t *appl_param_descr_new();
extern void appl_param_descr_delete(appl_param_descr_t **apd);
extern void appl_param_descr_delete_no_destroy(appl_param_descr_t *apd);

extern void appl_param_descr_init(appl_param_descr_t *apd);

extern int appl_param_descr_write(appl_param_descr_t *apd, sock_t sock);
extern int appl_param_descr_read_ip(appl_param_descr_t *apd, sock_t sock);
extern appl_param_descr_t *appl_param_descr_read(sock_t sock);

struct _appl_param_t
{
    appl_param_descr_t descr;
    void *correspondence;
};
typedef struct _appl_param_t appl_param_t;

extern appl_param_t *appl_param_new();
extern void appl_param_delete(appl_param_t **ap);

/******************************************************************************
 * Parameter description list element                                         *
 ******************************************************************************/

struct _appl_param_list_elem
{
    struct _appl_param_list_elem *prev, *next;
    appl_param_t *param;
};
typedef struct _appl_param_list_elem appl_param_list_elem;

extern appl_param_list_elem *appl_param_list_elem_new();
extern void appl_param_list_elem_delete(appl_param_list_elem **aple);

/******************************************************************************
 * Parameter list                                                             *
 ******************************************************************************/

struct _appl_param_list_t
{
    int num_elems;
    appl_param_list_elem *start;
    appl_param_list_elem *end;
};
typedef struct _appl_param_list_t appl_param_list_t;

extern appl_param_list_t *appl_param_list_new();
extern void appl_param_list_delete(appl_param_list_t **apl);

extern void appl_param_list_insert(appl_param_list_t *apl, appl_param_t *ap);
extern void appl_param_list_remove(appl_param_list_t *apl, appl_param_t *ap);

extern void
appl_param_list_register(appl_param_list_t *apl,
                         char *name,
                         appl_datatype_t type,
                         char changeable,
                         void *corr);

int appl_param_list_write(appl_param_list_t *apl, sock_t sock);
int appl_param_list_read_ip(appl_param_list_t *apl, sock_t sock);
appl_param_list_t *appl_param_list_read(sock_t sock);

int appl_param_list_print(appl_param_list_t *apl);

#endif
