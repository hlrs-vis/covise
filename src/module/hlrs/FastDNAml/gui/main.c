/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>

#include <network/network.h>

#include <appl/appl.h>

#include "gui.h"
#include "gui_appl_list.h"

int
main(int argc, char **argv)
{
    gui_t *gui;
    gui_appl_list_t *gal;

    sock_t server_sock;
    int num_apps;
    appl_t *appl = NULL;

    server_sock = sock_connect("localhost", 31011);

    if (server_sock >= 0)
    {

        gui = gui_new();
        gui->sock = server_sock;
        gui->state = GUI_STATE_CONNECTED;

        gui_appl_list_read_ip(gui->appl_list, gui->sock);
        if (gui->appl_list != NULL)
        {
            gui_appl_list_printf(gui->appl_list);
        }

        printf("Application list successfully read.\n");

        /* appl = appl_read_info(gui->sock); */

        if (appl != NULL)
        {
            appl_print_info(appl);
        }

        gui_main_loop(gui);

        gui_delete(&gui);
    }
    else
    {
        fprintf(stderr, "Cannot connect to server.");
        fflush(stderr);
    }

    return 0;
}
