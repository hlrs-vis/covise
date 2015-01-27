/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GUI_SERVER_H
#define _GUI_SERVER_H

#include <network/network.h>

struct _gui_server_info
{
    char *hostname;
};
typedef struct _gui_server_info gui_server_info;

extern gui_server_info *gui_server_info_new();
extern void gui_server_info_delete(gui_server_info **gsi);

extern int gui_server_info_write(gui_server_info *gsi, sock_t sock);
extern int gui_server_info_read_ip(gui_server_info *gsi, sock_t sock);

#endif
