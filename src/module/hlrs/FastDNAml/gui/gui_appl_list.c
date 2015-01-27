/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>

#include "gui_appl.h"
#include "gui_appl_list.h"

gui_appl_list_t *
gui_appl_list_new()
{
    gui_appl_list_t *rv = NULL;

    rv = (gui_appl_list_t *)malloc(sizeof(gui_appl_list_t));

    if (rv != NULL)
        gui_appl_list_init(rv);

    return rv;
}

void
gui_appl_list_delete(gui_appl_list_t **gal)
{
    int i;

    if (*gal)
    {

        if ((*gal)->appls != NULL)
        {
            for (i = 0; i < (*gal)->num; i++)
                gui_appl_delete(&(*gal)->appls[i]);
            free((*gal)->appls);
            (*gal)->appls = NULL;
        }

        free(*gal);
        *gal = NULL;
    }
}

void
gui_appl_list_init(gui_appl_list_t *gal)
{
    if (gal != NULL)
    {
        gal->max_num = 10;
        gal->num = 0;
        gal->appls = (gui_appl_t **)malloc(sizeof(gui_appl_t *) * gal->max_num);
    }
}

gui_appl_t *
gui_appl_list_find(gui_appl_list_t *gal, int applID)
{
    gui_appl_t *rv = NULL;
    int i;

    for (i = 0; i < gal->num; i++)
    {
        if (gal->appls[i]->applID == applID)
            break;
    }

    if (gal->appls[i]->applID == applID)
    {
        rv = gal->appls[i];
    }

    return rv;
}

void
gui_appl_list_add(gui_appl_list_t *gal, appl_t *appl, int applID)
{
    gui_appl_t *ga;

    if (gal != NULL && appl != NULL)
    {
        ga = gui_appl_new();
        if (ga != NULL)
        {
            ga->appl = appl;
            ga->applID = applID;
            gal->appls[gal->num] = ga;
            if (++(gal->num) > gal->max_num)
            {
                gal->max_num += 10;
                gal->appls = (gui_appl_t **)realloc((void *)gal->appls, sizeof(gui_appl_t *) * gal->max_num);
            }
        }
    }
}

int
gui_appl_list_write(gui_appl_list_t *gal, sock_t sock)
{
    int rv = 0;

    return rv;
}

int
gui_appl_list_read_ip(gui_appl_list_t *gal, sock_t sock)
{
    int rv = 0;
    int applID;
    appl_t *newAppl;
    int num_appls;
    int err;

    int i;

    if (gal != NULL)
    {
        err = sock_read_int(sock, &num_appls);

        printf("There will be %d application infos waiting.\n", num_appls);

        /* Pre-alloc neccessary space */

        if (gal->max_num < num_appls)
        {
            gal->max_num = num_appls + 1;
            gal->appls = (gui_appl_t **)realloc((void *)(gal->appls), sizeof(gui_appl_t *) * gal->max_num);
        }

        for (i = 0; i < num_appls; i++)
        {

            printf("\nReading application %d.\n", i);

            sock_read_int(sock, &applID);
            newAppl = appl_read_info(sock);

            if (newAppl == NULL)
            {
                err = 1;
                break;
            }

            gui_appl_list_add(gal, newAppl, applID);
        }
    }

    rv = err;

    return rv;
}

gui_appl_list_t *
gui_appl_list_read(sock_t sock)
{
    gui_appl_list_t *rv = NULL;

    rv = gui_appl_list_new();

    if (gui_appl_list_read_ip(rv, sock) != 0)
    {
        gui_appl_list_delete(&rv);
    }

    return rv;
}

int
gui_appl_list_printf(gui_appl_list_t *gal)
{
    int rv = 0;
    int i;

    printf("Number of applications:    %d\n", gal->num);

    for (i = 0; i < gal->num; i++)
    {
        appl_print_info(gal->appls[i]->appl);
    }

    return rv;
}
