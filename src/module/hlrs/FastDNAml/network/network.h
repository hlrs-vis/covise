/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NETWORK_H
#define _NETWORK_H

#ifdef HAVE_STDINT_H
#include <stdint.h>
#else
typedef unsigned short uint16_t;
#endif

#include <sys/types.h>
#include <sys/socket.h>

#include <netinet/in.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

#define LIBNETWORK_BYTE_ORDER_UNDEFINED 0x0000
#define LIBNETWORK_BYTE_ORDER_1234 0x1234
#define LIBNETWORK_BYTE_ORDER_4321 0x4321

#define LIBNETWORK_SIZEOF_SHORT 2
#define LIBNETWORK_SIZEOF_INT 4
#define LIBNETWORK_SIZEOF_LONG 8

#define LIBNETWORK_SIZEOF_FLOAT 4
#define LIBNETWORK_SIZEOF_DOUBLE 8

typedef int sock_t;

EXTERN sock_t setup_network(uint16_t port);

EXTERN sock_t sock_open(uint16_t port);
EXTERN void sock_close(sock_t sock);
EXTERN void sock_shutdown(sock_t sock);

EXTERN int sock_set_opt(sock_t sock, int optname, const void *optval);

EXTERN int sock_read_bytes_no_check(sock_t sock, char *buffer, size_t len);
EXTERN int sock_read_bytes(sock_t sock, char *buffer, size_t len);
EXTERN int sock_write_bytes_no_check(sock_t sock, char *buffer, size_t len);
EXTERN int sock_write_bytes(sock_t sock, char *buffer, size_t len);

EXTERN int sock_read_byte(sock_t sock, char *val);
EXTERN int sock_write_byte(sock_t sock, char *val);

EXTERN int sock_read_int(sock_t sock, int *val);
EXTERN int sock_read_ints(sock_t sock, int *buf, int hm);
EXTERN int sock_write_int(sock_t sock, int *val);
EXTERN int sock_write_ints(sock_t sock, int *buf, int hm);

EXTERN int sock_read_double(sock_t sock, double *val);
EXTERN int sock_read_doubles(sock_t sock, double *buf, int hm);
EXTERN int sock_write_double(sock_t sock, double *val);
EXTERN int sock_write_doubles(sock_t sock, double *buf, int hm);

EXTERN char *sock_read_string(sock_t sock);
EXTERN int sock_write_string(sock_t sock, const char *str);

EXTERN sock_t sock_connect(char *hostname, uint16_t port);
EXTERN sock_t sock_connect_with_port(char *hostname, uint16_t rem_port, uint16_t loc_port);

EXTERN sock_t connection_wait(sock_t sock, struct sockaddr *addr, size_t *len);
EXTERN sock_t connection_wait_timeout(sock_t sock, struct sockaddr *addr, size_t *len, time_t milli_secs);

/*** Misc. functions ***/

EXTERN void swap_bytes(void *buf, int num);

/*** Conversion functions ***/

/* Integer values */

EXTERN int netshort_to_short(void *buf, short *val);
EXTERN int short_to_netshort(short *val, void *buf);
EXTERN int netshorts_to_shorts(void *buf, short *vals, int hm);
EXTERN int shorts_to_netshorts(short *vals, void *buf, int hm);

EXTERN int netint_to_int(void *buf, int *val);
EXTERN int int_to_netint(int *val, void *buf);
EXTERN int netints_to_ints(void *buf, int *vals, int hm);
EXTERN int ints_to_netints(int *vals, void *buf, int hm);

EXTERN int netlong_to_long(void *buf, long *val);
EXTERN int long_to_netlong(long *val, void *buf);
EXTERN int netlongs_to_longs(void *buf, long *vals, int hm);
EXTERN int longs_to_netlongs(long *vals, void *buf, int hm);

/* Floating point values */

EXTERN int netdouble_to_double(void *buf, double *val);
EXTERN int double_to_netdouble(double *val, void *buf);
EXTERN int netdoubles_to_doubles(void *buf, double *vals, int hm);
EXTERN int doubles_to_netdoubles(double *vals, void *buf, int hm);

#endif
