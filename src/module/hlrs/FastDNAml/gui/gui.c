/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>

#include <network/network.h>
#include <appl/appl.h>

#include "gui.h"

#include "gui_command.h"

#ifndef ENTER
#define ENTER printf("Entering %s.\n", __FUNCTION__)
#endif

#ifndef LEAVE
#define LEAVE printf("Leaving %s.\n", __FUNCTION__)
#endif

gui_t *
gui_new()
{
    gui_t *rv = NULL;

    rv = (gui_t *)malloc(sizeof(gui_t));
    if (rv != NULL)
        gui_init(rv);

    return rv;
}

void
gui_delete(gui_t **g)
{
    if (*g)
    {
        if ((*g)->state == GUI_STATE_CONNECTED)
            sock_close((*g)->sock);
        if ((*g)->appl_list != NULL)
            gui_appl_list_delete(&(*g)->appl_list);
        free(*g);
        *g = NULL;
    }
}

void
gui_init(gui_t *g)
{
    if (g != NULL)
    {
        g->state = GUI_STATE_INITED;
        g->sock = -1;
        g->appl_list = gui_appl_list_new();
    }
}

int
gui_handle_command(gui_t *gui, gui_command_t *comm)
{
    int rv = 0;
    appl_data_descr_t *newDescr;
    appl_data_t *newData;
    gui_appl_t *appl;

    ENTER;

    switch (comm->command)
    {

    case APPL_COMMAND_DATA_SET:

        printf("Application is setting data.\n");

        appl = gui_appl_list_find(gui->appl_list, comm->ID);

        if (appl != NULL)
        {

            newDescr = appl_data_descr_read(gui->sock);
            printf("Data will be named %s.\n", newDescr->name);
            newData = appl_data_list_find(appl->appl->data, newDescr->name);

            if (newData != NULL)
            {
                printf("Found data in application list.\n");
                appl_data_read_ip(newData, gui->sock);

                if (newData->descr->type == APPL_DATATYPE_TREE || newData->descr->type == APPL_DATATYPE_TREE_PTR)
                {
                    printf("Tree is %s\n", (char *)newData->repr);
                }

                /* Action on read data goes here. */
            }
            else
            {
                printf("Data to be send is NOT registered.\n");
            }
        }
        else
        {
            printf("No matching application.\n");
        }

        break;

    case APPL_COMMAND_APPLICATION_DISCONNECT:
        printf("An application just disconnected.\n");
        break;

    default:
        break;
    }

    LEAVE;

    return rv;
}

int
gui_get_tree_from_appl(gui_t *gui, gui_appl_t *appl)
{
    int rv = 0;

    appl_data_t *data;
    gui_command_t command;

    if (appl != NULL)
    {

        data = appl_data_list_find(appl->appl->data, "tree");

        if (data != NULL)
        {

            printf("Found desired data.\n");

            gui_command_init(&command);
            command.command = APPL_COMMAND_DATA_GET;
            command.ID = appl->applID;
            command.data = (void *)data->descr;

            /* Send the header... */

            gui_command_header_write(&command, gui->sock);

            /* ... and the actual data description. */

            appl_data_descr_write(data->descr, gui->sock);
        }
        else
        {
            printf("Data not in inventory.\n");
        }
    }

    return rv;
}

int
gui_get_tree_from_applID(gui_t *gui, int applID)
{
    int rv = 0;

    gui_appl_t *appl;
    appl_data_t *data;
    gui_command_t command;

    if (gui->state == GUI_STATE_CONNECTED)
    {

        if (gui->appl_list != NULL && gui->appl_list->num > 0)
        {

            appl = gui_appl_list_find(gui->appl_list, applID);

            if (appl != NULL)
            {
                rv |= gui_get_tree_from_appl(gui, appl);
            }
            else
            {
                printf("Application with ID %d not found.\n", applID);
            }
        }
        else
        {
            printf("No suitable application.\n");
        }
    }

    return rv;
}

int
gui_get_tree(gui_t *gui)
{
    int rv = 0;
    appl_data_t *data;
    gui_command_t command;
    int i;

    if (gui->state == GUI_STATE_CONNECTED)
    {

        if (gui->appl_list != NULL && gui->appl_list->num > 0)
        {
            for (i = 0; i < gui->appl_list->num; i++)
            {
                gui_get_tree_from_appl(gui, gui->appl_list->appls[i]);
            }
        }
    }

    return rv;
}

int
gui_set_data(gui_t *gui, appl_data_t *data)
{
    gui_command_t command;

    command.command = APPL_COMMAND_DATA_SET;
    command.data = (void *)data;

    /* Send the header... */
    gui_command_header_write(&command, gui->sock);

    /* ... and the actual data description. */
    appl_data_descr_write(data->descr, gui->sock);

    /* ... and last but not least the data itself... */

    appl_data_write(data, gui->sock);
}

int
gui_main_loop(gui_t *gui)
{
    int rv = 0;
    sock_t sock;
    gui_command_t *new_command;
    char looping;
    char stop_send;
    int i;
    appl_data_t *dat, *sub_dat;

    sock = gui->sock;

    looping = 1;
    stop_send = 0;
    i = 0;

    while (looping)
    {

        new_command = NULL;

        if (gui->state == GUI_STATE_CONNECTED)
            new_command = gui_command_wait(gui, 10);

        if (new_command != NULL)
        {
            printf("There was a command for GUI.\n");
            gui_handle_command(gui, new_command);
            gui_command_delete(&new_command);
        }

        if (gui->state == GUI_STATE_DISCONNECTED)
        {
            printf("Server disconnected GUI.\n");
            looping = 0;
        }

        if (stop_send == 0 && gui->state == GUI_STATE_CONNECTED && i < 1)
        {
            printf("Sending tree.\n");
            gui_get_tree(gui);
            printf("\n\n");
            stop_send = 0;
            i++;
        }

        dat = appl_data_list_find(gui->appl_list->appls[0]->appl->data, "datum");
        if (dat != NULL)
        {
            printf("Setting datum...\n");
            if (dat->descr->type == APPL_DATATYPE_DISTRIBUTED)
            {

                sub_dat = appl_data_distr_sub_data(dat, 0);

                if (sub_dat != NULL && sub_dat->descr->type == APPL_DATATYPE_INT)
                {
                    *((int *)(sub_dat->repr)) = 42;
                    gui_set_data(gui, sub_dat);
                }
            }
        }
    }

    return rv;
}
