/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include "polhemusdrvr.h"

#ifdef __APPLE__
#define NO_TERMIO
#else
#include <termio.h>
#endif

void send_fastrak_cmd(int desc, const char *cmd_buf);

int
fastrackOpen(char *portname)
{
#ifndef NO_TERMIO
    static struct termio termconf;
    int desc;

    desc = open(portname, O_RDWR);

    if (desc >= 0)
    {
        printf("Port /dev/ttyd3 open.\n");
    }
    else
        perror("Open port:");

    termconf.c_iflag = 0;
    termconf.c_oflag = 0;
    termconf.c_cflag = B9600 | CS8 | CREAD | CLOCAL;
    termconf.c_lflag = ICANON;
    termconf.c_line = 0;
    termconf.c_cc[VTIME] = 0;
    termconf.c_cc[VMIN] = 47; /* max packet size */

    if (ioctl(desc, TCSETAW, &termconf) == -1)
    {
        perror("Tracker-TermSetup");
        exit(0);
    }
    return desc;
#else
    return 0;
#endif
}

/******************************************************************************/
void
fastrackReset(int desc)
{
    send_fastrak_cmd(desc, "R1");
}

/******************************************************************************/
void
fastrackSetHemisphere(int desc, float H1, float H2, float H3)
{

    char str[500];
    sprintf(str, "H1,0.0,0.0,-1.0");
    //  printf("%s\n",str);
    sprintf(str, "H1,%f,%f,%f", H1, H2, H3);
    //  printf("%s\n",str);
    send_fastrak_cmd(desc, "H1,0.0,0.0,-1.0");
}

/******************************************************************************/
void
fastrackSetPositionFilter(int desc, float f, float flow, float fhigh, float factor)
{
    char str[500];
    sprintf(str, "x,%10f,%10f,%10f,%10f", f, flow, fhigh, factor);
    //  printf("%s\n",str);
    send_fastrak_cmd(desc, str);
}

/******************************************************************************/
void
fastrackSetAttitudeFilter(int desc, float f, float flow, float fhigh, float factor)
{
    char s[500];
    sprintf(s, "v,%10f,%10f,%10f,%10f", f, flow, fhigh, factor);
    //  printf("%s\n",s);
    send_fastrak_cmd(desc, s);
}

/******************************************************************************/
void
fastrackSetAsciiFormat(int desc)
{
    send_fastrak_cmd(desc, "F");
}

/******************************************************************************/
void
fastrackDisableContinuousOutput(int desc)
{
    send_fastrak_cmd(desc, "c");
}

/******************************************************************************/
void
fastrackSetUnitToInches(int desc)
{
    send_fastrak_cmd(desc, "U");
}

/******************************************************************************/
void
fastrackSetUnitToCentimeters(int desc)
{
    send_fastrak_cmd(desc, "u");
}

/******************************************************************************/
void
fastrackSetReferenceFrame(int desc, float Ox, float Oy, float Oz,
                          float Xx, float Xy, float Xz, float Yx, float Yy, float Yz)
{
    char s[1000];

    sprintf(s, "A1,%1f,%1f,%1f,%1f,%1f,%1f,%1f,%1f,%1f",
            Ox, Oy, Oz, Xx, Xy, Xz, Yx, Yy, Yz);

    //  printf("%s\n",s);
    send_fastrak_cmd(desc, s);
}

/******************************************************************************/
void
fastrackSetOutputToQuaternions(int desc)
{
    send_fastrak_cmd(desc, "O1,2,11,1");
}

/******************************************************************************/
void
fastrackGetSingleRecord(int desc, PolhemusRecord *record)
{
    char buf[100];
    char oc, er;
    int st;
    float x, y, z, w, q1, q2, q3;
    int ok = FALSE;

    send_fastrak_cmd(desc, "P");

    while (ok == FALSE)
    {
        send_fastrak_cmd(desc, "P");
        if (read(desc, buf, 54) != 54)
        {
            cerr << "fasttrackGetSingleRecord: short read" << endl;
        }

        /* kartesiche Koordinaten und Quaternionen */
        if (sscanf(buf, "%c%i%c%f%f%f%f%f%f%f", &oc, &st, &er, &x, &y, &z, &w, &q1, &q2, &q3) != 10)
        {
            cerr << "fasttrackGetSingleRecord: sscanf failed" << endl;
        }

        if (buf[0] == '0' && buf[1] == '1' && buf[53] == '\n')
        {
            ok = TRUE;
            //  printf("Datensatz ist gut\n\n");
            record->x = x;
            record->y = y;
            record->z = z;
            record->w = w;
            record->q1 = q1;
            record->q2 = q2;
            record->q3 = q3;
        }
        else
        {
            printf("Datensatz ist schlecht\n\n");
            ok = FALSE;
        }
    }
}

/******************************************************************************/
/******************************************************************************/
void
send_fastrak_cmd(int desc, const char *cmd_buf)
{
    char crlf[3];

    /* code to add a CR-LF pair to the end of a command */
    /* if it's needed */

    sprintf(crlf, "\r\n");

    switch ((int)cmd_buf[0])
    {
    case 'P':
    case 'C':
    case 'c':
        if (write(desc, cmd_buf, 1) != 1)
        {
            cerr << "send_fasttrak_cmd: short write1" << endl;
        }
        break;
    default:
        strcat((char *)cmd_buf, crlf);
        if ((unsigned int)write(desc, cmd_buf, strlen(cmd_buf)) != strlen(cmd_buf))
        {
            cerr << "send_fasttrak_cmd: short write2" << endl;
        }
    }
} /* end send_fastrak */
