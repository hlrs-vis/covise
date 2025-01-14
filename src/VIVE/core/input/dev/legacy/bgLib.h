/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __LV_H
#define __LV_H

/*
 * IMPORTANT.
 * 1.
 *  The bglv data structure has been modified to allow for future expansion
 *  capabilities.  If you have a FlyBox/BeeBox/CerealBox with an EPROM
 *  revision below 3.05, you can simply
 *     #define REV300
 *  instead of
 *     #define REV305
 *  and the data structure will be the old size.  For the future you should
 *  use this file as is, but make sure to compile all files that include
 *  this.  (i.e., if you don't do this you will be passing the wrong
 *  size of data structure....)
 *
 * 2.
 *  For the new series of SGI hardware (O2/Octane/Origin) baud rates of
 *  115200 are supported, and the termios data structure has been modified
 *  in IRIX 6.xx
 *  You can
 *    #define _OLD_TERMIOS
 *  and this will link to an old termios data structure, and will disable
 *  the use of 115200 baud.  We recommend that you don't do this on new
 *  machine.  (Of course if you are still running IRIX 5.xx then you will
 *  have to do this.
 */

#include <time.h>
#ifdef WIN32
#include <windows.h>
#endif

#define REV305

#define FLYBOX 1
#define BEEBOX 2
#define CEREALBOX 3
#define CAB 4
#define DRIVEBOX 5

#define FB_NOBLOCK 1
#define FB_BLOCK 2

#define AIC1 0x01
#define AIC2 0x02
#define AIC3 0x04
#define AIC4 0x08
#define AIC5 0x10
#define AIC6 0x20
#define AIC7 0x40
#define AIC8 0x80

#define AOC1 0x01
#define AOC2 0x02
#define AOC3 0x04
/*
 *  Extra analog outputs to be available with next release
 *  of LV board.
 */
#define AOC4 0x08
#define AOC5 0x10
#define AOC6 0x20
#define AOC7 0x40
#define AOC8 0x80

#define DIC1 0x10
#define DIC2 0x20
#define DIC3 0x40

/*
 *  For use with JunctionBox
 */
#define MD16 0x10
#define MD32 0x20
#define MD48 0x30
#define MD56 0x40
#define MD64 0x50
#define MD80 0x60
#define MD96 0x70
#define MD112 0x80

#define MPDIG 0x08

#define MP112i 0x00
#define MP112o 0x10
#define MP56io 0x20

#define DOC1 0x10
#define DOC2 0x20
#define DOC3 0x40

/*
 *  Baud 115200 available for use with Irix 6.x new termios
 *  structure.  Supported with rev 3.07 EPROM's.  Useful on
 *  O2/Octane/Origin systems.
 */
#define BAUD1152 0x10
#define BAUD576 0x70
#define BAUD384 0x60
#define BAUD192 0x50
#define BAUD96 0x40
#define BAUD48 0x30
#define BAUD24 0x20

#ifdef _OLD_TERMIOS
#define BAUD12 0x10
#endif

#define BG_OFFSET 0x21

/*
 *  Define some commands
 */

#define BG_BURST 'B' /* Burst mode                  */
#define BG_BURST_SET 'b' /* Burst mode rate set         */
#define BG_CONFIG 'c' /* Configure 3.07 + EPROMs     */
#define BG_CONT 'C' /* Continuous buffered         */
#define BG_DEFAULT 'd' /* Reset to Default            */
#define BG_PACKET 'p' /* One input and one output    */
#define BG_ONCE 'o' /* One input                   */
#define BG_ONCE_CS 'O' /* One input with check sum    */
#define BG_RESET_FB 'r' /* Reset 3 chars with offset   */
#define BG_RESET_FB_O 'R' /* Reset (rev 2.2 no offset)   */
#define BG_STOP 'S' /* Stop burst mode             */
#define BG_SETUP 's' /* Setup rev 3.0 eprom         */
#define BG_TEST1 'T' /* Test (and copyright)        */
#define BG_TEST2 't' /* Test (and copy, and rev #)  */

typedef struct rs_struct
{
    int wrt; /* write error */
    int rd; /* read error  */
    int len; /* string length error  */
    int nl; /* last char error  */
    int cycles; /* numer of cycles */
    int thou; /* thousands of cycles */
} RS_ERR;

typedef struct REVISION
{
    int major; /*  Software major revision             */
    int minor; /*  Software minor revision             */
    int bug; /*  Software bug revision               */
    char alpha; /*  EPROM alpha revision                */
    int year;
} revision;

/*
 *  For v3.0 software, define a new structure
 */
typedef struct BGLV_STRUCT
{
    int n_analog_in; /*  Number of analog inputs (8 max)     */
    int analog_in; /*  Analog input selector               */
    int n_dig_in; /*  Number of digital inputs (24 max)   */
    int dig_in; /*  Digital input selector              */
    int n_analog_out; /*  Number of analog outputs (3 max)    */
    int analog_out; /*  Analog out channel selector         */
    int n_dig_out; /*  Number of digital outputs (24 max)  */
    int dig_out; /*  Digital output selector             */
    float ain[8]; /*  Analog input data                   */
#ifdef REV300
    int aout[3]; /*  Analog output data                  */
#endif
    int din[3]; /*  Digital input data                  */
    int dout[3]; /*  Digital output data                 */
    long count;
    int str_len; /*  Length of string to expect          */
    int baud; /*  Baud rate selected                  */
    char mode[2]; /*  Mode to send - rev 2.2              */
    time_t tag;
    int port;
    int box_type; /*  Device type                         */
#ifdef WIN32
    HANDLE sp_fd; /*  Serial port file descriptor         */
#else
    int sp_fd; /*  Serial port file descriptor         */
#endif
    revision Rev; /*  Software major revision             */
#ifdef REV305
    int aout[8]; /*  For next generation board           */
    int mp_dig_in; /*  Multiplex dig input, overrides dig  */
    int mp_dig_out; /*  Multiplex dig output, overrides dig */
    int mp_din[14]; /*  Multiplex inputs  values            */
    int mp_dout[14]; /*  Multiplex outputs values            */
    int n_enc; /*  Number of encoders                  */
    int enc_sel; /*  Encoder selection                   */
    long enc_abs_val[4]; /*  Absolute value of encoder           */
    int enc_inc_val[4]; /*  Incremental value of encoder        */
    float sparef[16]; /*  Reserved for BG expansion           */
    int sparei[16]; /*  Reserved for BG expansion           */
#endif
} bglv;

extern int pack_data(bglv *bgp, char *out_buf);
extern int send_outputs(bglv *bgp);
extern int open_lv(bglv *bgp, char *p, int flag);
#ifdef WIN32
int set_baud(HANDLE sp_fd, int b);
#else
int set_baud(int sp_fd, int b);
#endif
extern int convert_serial(bglv *bgp, char *str);
extern void no_answer();
extern int check_setup(bglv *bgp);
extern int parse_year(char *s);
extern int check_rev(bglv *bgp);
extern int init_lv(bglv *bgp);
extern void close_lv(bglv *bgp);
extern int r_cs(bglv *bgp, char *str);
#ifdef WIN32
extern int get_ack(HANDLE sp_fd);
extern int w_lv(HANDLE sp_fd, char *mode);
#else
extern int get_ack(int sp_fd);
extern int w_lv(int sp_fd, char *mode);
#endif
extern int r_lv(bglv *bgp);
extern int check_inputs(bglv *bgp);
#endif /* __LV_H */
