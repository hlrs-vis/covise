/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: io.c,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 * input error checking, fexists()
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pwd.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

extern void errwin(const char *s);
extern int yesno(const char *msg1, const char *s1, const char *s2, const char *helptext);

static char readbuf[80];

int ibounds(int x, int lower, int upper, char *name)
{
    int test;

    test = ((x >= lower) && (x <= upper));
    if (!test)
    {
        sprintf(readbuf, " in %s : parameter must be in (%d , %d)", name, lower, upper);
        errwin(readbuf);
    }
    return (test);
}

int fbounds(double x, double lower, double upper, char *name)
{
    int test;

    test = ((x >= lower) && (x <= upper));
    if (!test)
    {
        sprintf(readbuf, "In %s : parameter must be in [%lf, %lf]", name, lower, upper);
        errwin(readbuf);
    }
    return (test);
}

int fexists(char *to)
{
    struct stat stto;
    char tbuf[256];

    if (stat(to, &stto) == 0)
    {
        sprintf(tbuf, "Overwrite %s?", to);
        if (!yesno(tbuf, NULL, NULL, NULL))
        {
            return (1);
        }
        return (0);
    }
    return (0);
}

int isdir(char *f)
{
    struct stat st;

    stat(f, &st);
    return (S_ISDIR(st.st_mode));
}

int sortstrcmp(char **str1, char **str2)
{
    return (strcmp(*str1, *str2));
}

/* TODO this needs some work */
void expand_tilde(char *buf)
{
    char buf2[MAXPATHLEN];
    char *home;
    if (buf[0] == '~')
    {
        if (strlen(buf) == 1)
        {
            home = getenv("HOME");
            if (home == NULL)
            {
                errwin("Couldn't find $HOME!");
                return;
            }
            else
            {
                strcpy(buf, home);
                strcat(buf, "/");
            }
        }
        else if (buf[1] == '/')
        {
            home = getenv("HOME");
            if (home == NULL)
            {
                errwin("Couldn't find $HOME!");
                return;
            }
            strcpy(buf2, home);
            strcat(buf2, "/");
            strcat(buf2, buf + 1);
            strcpy(buf, buf2);
        }
        else
        {
            char tmp[128], *pp = tmp, *q = buf + 1;
            struct passwd *pent;

            while (*q && (*q != '/'))
            {
                *pp++ = *q++;
            }
            *pp = 0;
            if ((pent = getpwnam(tmp)) != NULL)
            {
                strcpy(buf2, pent->pw_dir);
                strcat(buf2, "/");
                strcat(buf2, q);
                strcpy(buf, buf2);
            }
            else
            {
                errwin("No user by that name");
            }
        }
    }
}
