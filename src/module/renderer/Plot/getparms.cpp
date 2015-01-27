/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: getparms.c,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 * Read a parameter file
 */

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "extern.h"
#include "globals.h"
#include "noxprotos.h"

static char readbuf[512];

int getparms(int, char *plfile)
{
    int linecount = 0, errpos = 0, errcnt = 0;
    // char s[256];
    FILE *pp;
    struct stat statb;
    double a, b, c, d, x, y;

    /* check to make sure this is a file and not a dir */
    if (stat(plfile, &statb))
    {
        sprintf(buf, "Can't stat file %s", plfile);
        errwin(buf);
        return 0;
    }
    if (!S_ISREG(statb.st_mode))
    {
        sprintf(buf, "File %s is not a regular file", plfile);
        errwin(buf);
        return 0;
    }
    if ((pp = fopen(plfile, "r")) == NULL)
    {
        sprintf(readbuf, "Can't open parameter file %s", plfile);
        errwin(readbuf);
        return 0;
    }
    else
    {
        errcnt = 0;
        while (fgets(readbuf, 511, pp) != NULL)
        {
            linecount++;
            if (readbuf[0] == '#')
            {
                continue;
            }
            if (strlen(readbuf) <= 1)
            {
                continue;
            }
            lowtoupper(readbuf);
            if (debuglevel == 1)
            {
                printf("%s", readbuf);
            }
            errpos = 0;
            scanner(readbuf, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
            if (errpos)
            {
                printf("Error at line %d: %s\n", linecount, readbuf);
                errcnt++;
                if (errcnt > 5)
                {
                    if (yesno("Lots of errors, cancel?", NULL, NULL, NULL))
                    {
                        fclose(pp);
                        return 0;
                    }
                    else
                    {
                        errcnt = 0;
                    }
                }
            }
        }
        fclose(pp);
    }
    return 1;
}

void read_param(char *pbuf)
{
    int errpos = 0;
    double a, b, c, d, x, y;
    extern int gotparams, gotread, readsrc, readtype;
    extern char paramfile[], readfile[];

    if (pbuf[0] == '#')
    {
        return;
    }
    lowtoupper(pbuf);
    scanner(pbuf, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
    if (gotparams && paramfile[0])
    {
        if (!getparms(cg, paramfile))
        {
        }
        gotparams = 0;
    }
    else if (gotread && readfile[0])
    {
        if (getdata(cg, readfile, readsrc, readtype))
        {
            /* drawgraph(); */
        }
        gotread = 0;
    }
}
