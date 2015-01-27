/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_COMMAND_H
#define _LIBAPPL_APPL_COMMAND_H

#include <network/network.h>

#include <appl/appl.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

#define APPL_COMMAND_PROTOCOL_VER 0

#define APPL_COMMAND_START 0x1000
#define APPL_COMMAND_HALT 0x1001
#define APPL_COMMAND_STOP 0x1002
#define APPL_COMMAND_END APPL_COMMAND_STOP

#define APPL_COMMAND_PARAM_SET 0x2000
#define APPL_COMMAND_PARAM_GET 0x2001

#define APPL_COMMAND_DATA_SET 0x3000
#define APPL_COMMAND_DATA_GET 0x3001

#define APPL_COMMAND_REGISTER_DATA 0x3100
#define APPL_COMMAND_UNREGISTER_DATA 0x3101

#define APPL_COMMAND_SERVER_DISCONNECT 0x3200
#define APPL_COMMAND_APPLICATION_DISCONNECT 0x3300

#define APPL_COMMAND_INVALID 0xffff

struct _appl_t;

struct _appl_command_t
{
    int command; /* Actual command */
    int ID; /* Every command gets an ID */
    int protocol_version; /* Version of application protocol */

    int TAN; /* Command belongs to this transaction number */
    int seq; /* Transactions consist of sequences of commands */

    void *data; /* Special data that go with the command (return values etc.) */

    char destructable; /* Indicate whether the data can be deleted after use or not. */
};
typedef struct _appl_command_t appl_command_t;

EXTERN appl_command_t *appl_command_new();
EXTERN void appl_command_delete(appl_command_t **c);

EXTERN int appl_command_read_header_ip(appl_command_t *c, sock_t sock);
EXTERN appl_command_t *appl_command_read_header(sock_t sock);

EXTERN int appl_command_write_header(appl_command_t *c, sock_t sock);

EXTERN int appl_command_register_data(appl_data_t *data, sock_t sock);
EXTERN int appl_command_unregister_data(appl_data_t *data, sock_t sock);

EXTERN appl_command_t *appl_command_get_header(struct _appl_t *appl, int *state);
EXTERN int appl_command_handle(struct _appl_t *appl, appl_command_t **c, int *state);
EXTERN int appl_command_check(struct _appl_t *appl);

#endif
