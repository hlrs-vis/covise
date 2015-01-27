/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: files.c,v 1.11 1994/10/09 04:42:11 pturner Exp pturner $
 *
 * read data files
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "globals.h"
#include "noxprotos.h"
#include <Xm/Xm.h>

#if defined(HAVE_NETCDF) || defined(HAVE_MFHDF)

#include <netcdf.h>
#endif

#define MAXERR 50
#define MAX_LINE_LEN 512
/*
 * number of doubles to allocate for each call to realloc
 */
#define BUFSIZE 512

int realtime = 0;
int change_gno; /* if the graph number changes on read in */
static int cur_gno; /* if the graph number changes on read in */
int change_type; /* current set type */
static int cur_type; /* current set type */

static int readerror = 0; /* number of errors */
static int readline = 0; /* line number in file */
// static int readfile = 0;	/* number of file read, not used */

extern void errwin(const char *s);
extern int readrawspice(int gno, char *fn, FILE *fp);
extern void update_status_popup(Widget w, XtPointer client_data, XtPointer call_data);
extern int yesno(const char *msg1, const char *s1, const char *s2, const char *helptext);
extern void log_resultsCB(Widget w, XtPointer client_data, XtPointer call_data);
extern void drawgraph(void);
extern void set_plotstr_string(plotstr *pstr, char *buf);
extern void set_left_footer(const char *s);
extern "C" {
extern void cfree(void *);
}

extern void log_results(const char *buf);

int getdata(int gno, char *fn, int src, int type)
{
    FILE *fp = NULL;
    int retval;
    struct stat statb;

    switch (src)
    {
    case DISK:
        /* check to make sure this is a file and not a dir */
        if (stat(fn, &statb))
        {
            sprintf(buf, "Can't open file %s", fn);
            errwin(buf);
            return 0;
        }
        if (!S_ISREG(statb.st_mode))
        {
            sprintf(buf, "File %s is not a regular file", fn);
            errwin(buf);
            return 0;
        }
        fp = fopen(fn, "r");
        readline = 0;
        break;
    case PIPE:
        fp = (FILE *)popen(fn, "r");
        readline = 0;
        break;
    case 2:
        fp = stdin;
        readline = 0;
        break;
    }
    if (fp == NULL)
    {
        sprintf(buf, "Can't open file %s", fn);
        errwin(buf);
        return 0;
    }
    cur_gno = gno;
    change_type = cur_type = type;
    retval = -1;
    while (retval == -1)
    {
        retval = 0;
        switch (cur_type)
        {
        case XY:
            retval = readxy(cur_gno, fn, fp, 0);
            break;
        case NXY:
            retval = readnxy(cur_gno, fn, fp);
            break;
        case IHL:
            retval = readihl(cur_gno, fn, fp);
            break;
        case BIN:
            retval = readbinary(cur_gno, fn, fp);
            break;
        case XYDX:
        case XYDY:
        case XYDXDX:
        case XYDYDY:
        case XYDXDY:
        case XYZ:
        case XYRT:
        case XYHILO:
        case XYBOXPLOT:
        case XYUV:
        case XYBOX:
            retval = readxxyy(cur_gno, fn, fp, cur_type);
            break;
        case XYSTRING:
            retval = readxystring(cur_gno, fn, fp);
            break;
        case BLOCK:
            retval = readblockdata(cur_gno, fn, fp);
            break;
        case RAWSPICE:
            retval = readrawspice(cur_gno, fn, fp);
            break;
        }
    }
    if (src == PIPE)
    {
        pclose(fp);
    }
    else
    {
        if (fp != stdin) /* leave stdin open */
        {
            fclose(fp);
        }
    }
    update_status_popup(NULL, NULL, NULL);
    return retval;
}

int getdata_step(int gno, char *fn, int src, int type)
{
    static FILE *fp;
    int retval;

    if (fp == NULL)
    {
        switch (src)
        {
        case DISK:
            fp = fopen(fn, "r");
            break;
        case PIPE:
            fp = (FILE *)popen(fn, "r");
            break;
        case 2:
            fp = stdin;
            break;
        case 3:
            if (fp)
            {
                if (src == PIPE)
                {
                    pclose(fp);
                }
                else
                {
                    if (fp != stdin) /* leave stdin open */
                    {
                        fclose(fp);
                    }
                }
            }
            fp = NULL;
            return (0);
        }
    }
    if (fp == NULL)
    {
        sprintf(buf, "Can't open file %s", fn);
        errwin(buf);
        fp = NULL;
        return 0;
    }
    cur_gno = gno;
    change_type = cur_type = type;
    retval = -1;
    while (retval == -1)
    {
        retval = 0;
        switch (cur_type)
        {
        case XY:
            retval = readxy(cur_gno, fn, fp, 1);
            break;
        case NXY:
            retval = readnxy(cur_gno, fn, fp);
            break;
        case IHL:
            retval = readihl(cur_gno, fn, fp);
            break;
        case BIN:
            retval = readbinary(cur_gno, fn, fp);
            break;
        case XYDX:
        case XYDY:
        case XYDXDX:
        case XYDYDY:
        case XYDXDY:
        case XYZ:
        case XYRT:
        case XYHILO:
        case XYBOXPLOT:
        case XYBOX:
            retval = readxxyy(cur_gno, fn, fp, cur_type);
            break;
        case XYSTRING:
            retval = readxystring(cur_gno, fn, fp);
            break;
        case BLOCK:
            retval = readblockdata(cur_gno, fn, fp);
            break;
        case RAWSPICE:
            retval = readrawspice(cur_gno, fn, fp);
            break;
        }
    }
    if (retval != -2)
    {
        /* means it returned because a single set was
       * read */
        if (src == PIPE)
        {
            pclose(fp);
        }
        else
        {
            if (fp != stdin) /* leave stdin open */
            {
                fclose(fp);
            }
        }
    }
    return retval;
}

int scanline(char *buf, int xformat, double *x, double *y)
{
    int pstat, mo, da, yr, hr, mi;
    double sec;
    switch (xformat)
    {
    case GENERAL:
    case DECIMAL:
        return sscanf(buf, "%lf %lf", x, y);
    case MMDDYY:
        if ((pstat = sscanf(buf, "%d-%d-%d %lf", &mo, &da, &yr, y)) != 4)
        {
            return 0;
        }
        else
        {
            *x = julday(mo, da, yr, 12, 0, 0.0);
            printf("%d %d %d %lf %lf\n", mo, da, yr, *x, *y);
            return 4;
        }
    case YYMMDD:
        if ((pstat = sscanf(buf, "%d-%d-%d %lf", &yr, &mo, &da, y)) != 4)
        {
            return 0;
        }
        else
        {
            *x = julday(mo, da, yr, 12, 0, 0.0);
            printf("%d %d %d %lf %lf\n", mo, da, yr, *x, *y);
            return 4;
        }
    case MMDDYYHMS:
        if ((pstat = sscanf(buf, "%d-%d-%d %d:%d:%lf %lf",
                            &mo, &da, &yr, &hr, &mi, &sec, y)) != 7)
        {
            return 0;
        }
        else
        {
            *x = julday(mo, da, yr, hr, mi, sec);
            return 7;
        }
    case YYMMDDHMS:
        if ((pstat = sscanf(buf, "%d-%d-%d %d:%d:%lf %lf",
                            &yr, &mo, &da, &hr, &mi, &sec, y)) != 7)
        {
            return 0;
        }
        else
        {
            *x = julday(mo, da, yr, hr, mi, sec);
            return 7;
        }
    default:
        return sscanf(buf, "%lf %lf", x, y);
    }
}

/*
 * read file type 0
 */
int readxy(int gno, char *fn, FILE *fp, int readone)
{
    extern int readxformat; /* TODO to globals.h */
    int i = 0, ll, j, pstat, readset = 0, retval = 0;
    double *x, *y;

    x = (double *)calloc(BUFSIZE, sizeof(double));
    y = (double *)calloc(BUFSIZE, sizeof(double));
    if (x == NULL || y == NULL)
    {
        errwin("Insufficient memory for set");
        cxfree(x);
        cxfree(y);
        return (0);
    }
    while (fgets(buf, MAX_LINE_LEN, fp) != NULL)
    {
        readline++;
        ll = strlen(buf);
        if ((ll > 0) && (buf[ll - 1] != '\n'))
        {
            /* must have a newline
          * char at end of line */
            readerror++;
            fprintf(stderr, "No newline at line #%1d: %s\n", readline, buf);
            if (readerror > MAXERR)
            {
                if (yesno("Lots of errors, abort?", NULL, NULL, NULL))
                {
                    cxfree(x);
                    cxfree(y);
                    return (0);
                }
                else
                {
                    readerror = 0;
                }
            }
            continue;
        }
        if (buf[0] == '#')
        {
            continue;
        }
        if (strlen(buf) < 2) /* blank line */
        {
            continue;
        }
        if (buf[0] == '@')
        {
            change_gno = -1;
            change_type = cur_type;
            read_param(buf + 1);
            if (change_gno >= 0)
            {
                cur_gno = gno = change_gno;
            }
            if (change_type != cur_type)
            {
                cur_type = change_type;
                retval = -1;
                break; /* exit this module and store any set */
            }
            continue;
        }
        convertchar(buf);
        /* count the number of items scanned */
        if ((pstat = scanline(buf, readxformat, &x[i], &y[i])) >= 1)
        {
            /* supply x if missing (y winds up in x) */
            if (pstat == 1)
            {
                y[i] = x[i];
                x[i] = i;
            }
            if (realtime == 1 && inwin)
            {
                drawpolysym(&x[i], &y[i], 1, 3, 0, 0, 1.0);
            }
            /* got x and y so increment */
            i++;
            if (i % BUFSIZE == 0)
            {
                x = (double *)realloc(x, (i + BUFSIZE) * sizeof(double));
                y = (double *)realloc(y, (i + BUFSIZE) * sizeof(double));
            }
        }
        else
        {
            if (i != 0)
            {
                if ((j = nextset(gno)) == -1)
                {
                    cxfree(x);
                    cxfree(y);
                    return (readset);
                }
                activateset(gno, j);
                settype(gno, j, XY);
                setcol(gno, x, j, i, 0);
                setcol(gno, y, j, i, 1);
                setcomment(gno, j, fn);
                log_results(fn);
                updatesetminmax(gno, j);
                if (realtime == 2 && inwin)
                {
                    drawsetxy(gno, g[gno].p[j], j);
                }
                readset++;
            }
            else
            {
                readerror++;
                fprintf(stderr, "Error at line #%1d: %s", readline, buf);
                if (readerror > MAXERR)
                {
                    if (yesno("Lots of errors, abort?", NULL, NULL, NULL))
                    {
                        cxfree(x);
                        cxfree(y);
                        return (0);
                    }
                    else
                    {
                        readerror = 0;
                    }
                }
            }
            i = 0;
            x = (double *)calloc(BUFSIZE, sizeof(double));
            y = (double *)calloc(BUFSIZE, sizeof(double));
            if (x == NULL || y == NULL)
            {
                errwin("Insufficient memory for set");
                cxfree(x);
                cxfree(y);
                return (readset);
            }
            if (readone)
            {
                return (-2);
            }
        }
    }
    if (i != 0)
    {
        if ((j = nextset(gno)) == -1)
        {
            cxfree(x);
            cxfree(y);
            return (readset);
        }
        activateset(gno, j);
        settype(gno, j, XY);
        setcol(gno, x, j, i, 0);
        setcol(gno, y, j, i, 1);
        setcomment(gno, j, fn);
        log_results(fn);
        updatesetminmax(gno, j);
        if (realtime == 2 && inwin)
        {
            /*
          * TODO ??? drawsetxy(g[gno].p[j]);
          */
        }
        readset++;
    }
    else
    {
        cxfree(x);
        cxfree(y);
    }
    if (retval == -1)
    {
        return retval;
    }
    else
    {
        return readset;
    }
}

/*
 * read the first set found in a file to set setno
 */
int read_set_fromfile(int gno, int setno, char *fn, int src)
{
    FILE *fp = NULL;
    struct stat statb;
    int readline = 0;
    int i = 0, pstat, retval = 0;
    double *x, *y;

    switch (src)
    {
    case DISK:
        /* check to make sure this is a file and not a dir */
        if (stat(fn, &statb))
        {
            sprintf(buf, "Can't stat file %s", fn);
            errwin(buf);
            return 0;
        }
        if (!S_ISREG(statb.st_mode))
        {
            sprintf(buf, "File %s is not a regular file", fn);
            errwin(buf);
            return 0;
        }
        fp = fopen(fn, "r");
        readline = 0;
        break;
    case PIPE:
        fp = (FILE *)popen(fn, "r");
        readline = 0;
        break;
    case 2:
        fp = stdin;
        readline = 0;
        break;
    }
    if (fp == NULL)
    {
        sprintf(buf, "Can't open file %s", fn);
        errwin(buf);
        return 0;
    }
    softkillset(gno, setno);
    x = (double *)calloc(BUFSIZE, sizeof(double));
    y = (double *)calloc(BUFSIZE, sizeof(double));
    if (x == NULL || y == NULL)
    {
        errwin("Insufficient memory for set");
        cxfree(x);
        cxfree(y);
        goto breakout;
    }
    while (fgets(buf, MAX_LINE_LEN, fp) != NULL)
    {
        readline++;
        if (buf[strlen(buf) - 1] != '\n')
        {
            /* must have a newline char
          * at end of line */
            readerror++;
            fprintf(stderr, "No newline at line #%1d: %s", readline, buf);
            continue;
        }
        if (buf[0] == '#')
        {
            continue;
        }
        if (buf[0] == '@')
        {
            continue;
        }
        convertchar(buf);
        /* count the number of items scanned */
        if ((pstat = sscanf(buf, "%lf %lf", &x[i], &y[i])) >= 1)
        {
            /* supply x if missing (y winds up in x) */
            if (pstat == 1)
            {
                y[i] = x[i];
                x[i] = i;
            }
            i++;
            if (i % BUFSIZE == 0)
            {
                x = (double *)realloc(x, (i + BUFSIZE) * sizeof(double));
                y = (double *)realloc(y, (i + BUFSIZE) * sizeof(double));
            }
        }
    }
    activateset(gno, setno);
    settype(gno, setno, XY);
    setcol(gno, x, setno, i, 0);
    setcol(gno, y, setno, i, 1);
    setcomment(gno, setno, fn);
    log_results(fn);
    updatesetminmax(gno, setno);
    retval = 1;

breakout:
    ;

    if (src == PIPE)
    {
        pclose(fp);
    }
    else
    {
        if (fp != stdin) /* leave stdin open */
        {
            fclose(fp);
        }
    }
    return retval;
}

/*
 * read IHL format
 */
int readihl(int gno, char *fn, FILE *fp)
{
    int i, j, pstat, npts;
    double *x, *y, tmp;

    i = 0;
    pstat = 0;
    if ((j = nextset(gno)) == -1)
    {
        return 0;
    }
    if (fgets(buf, MAX_LINE_LEN, fp) == NULL)
    {
        errwin("Can't read from file");
        killset(gno, j);
        return 0;
    }
    readline++;
    pstat = sscanf(buf, "%d", &npts);
    if (npts == 0)
    {
        errwin("Number of points = 0");
        killset(gno, j);
        return 0;
    }
    activateset(gno, j);
    settype(gno, j, XY);
    setlength(gno, j, npts);
    setcomment(gno, j, fn);
    log_results(fn);
    x = getx(gno, j);
    y = gety(gno, j);
    for (i = 0; i < npts; i++)
    {
        if (fgets(buf, MAX_LINE_LEN, fp) == NULL)
        {
            errwin("Premature EOF");
            updatesetminmax(gno, j);
            return 1;
        }
        readline++;
        convertchar(buf);
        pstat = sscanf(buf, "%lf %lf %lf", &tmp, &x[i], &y[i]);
    }
    updatesetminmax(gno, j);
    return 1;
}

/*
 * read x1 y1 y2 ... y30 formatted files
 * note that the maximum number of sets is 30
 */
#define MAXSETN 30

int readnxy(int gno, char *fn, FILE *fp)
{
    int i, j, pstat, cnt, scnt[MAXSETN], setn[MAXSETN],
        retval = 0;
    double *x[MAXSETN], *y[MAXSETN], xval, yr[MAXSETN];
    char *s, buf[1024], tmpbuf[1024];
    int do_restart = 0;

/* if more than one set of nxy data is in the file,
    * leap to here after each is read - the goto is at the
    * bottom of this module.
    */
restart:
    ;

    i = 0;
    pstat = 0;
    cnt = 0;
    while ((fgets(buf, MAX_LINE_LEN, fp) != NULL) && ((buf[0] == '#') || (buf[0] == '@')))
    {
        readline++;
        if (buf[0] == '@')
        {
            change_gno = -1;
            read_param(buf + 1);
            if (change_gno >= 0)
            {
                cur_gno = gno = change_gno;
            }
        }
    }
    convertchar(buf);

    /*
    * count the columns
    */
    strcpy(tmpbuf, buf);
    s = tmpbuf;
    while ((s = strtok(s, " \t\n")) != NULL)
    {
        cnt++;
        s = NULL;
    }
    if (cnt > MAXPLOT)
    {
        errwin("Maximum number of columns exceeded, reading first 31");
        cnt = 31;
    }
    s = buf;
    s = strtok(s, " \t\n");
    if (s == NULL)
    {
        errwin("Read ended by a blank line at or near the beginning of file");
        return 0;
    }
    pstat = sscanf(s, "%lf", &xval);
    if (pstat == 0)
    {
        errwin("Read ended, non-numeric found on line at or near beginning of file");
        return 0;
    }
    s = NULL;
    for (j = 0; j < cnt - 1; j++)
    {
        s = strtok(s, " \t\n");
        if (s == NULL)
        {
            yr[j] = 0.0;
            errwin("Number of items in column incorrect");
        }
        else
        {
            yr[j] = atof(s);
        }
        s = NULL;
    }
    if (cnt > 1)
    {
        for (i = 0; i < cnt - 1; i++)
        {
            if ((setn[i] = nextset(gno)) == -1)
            {
                for (j = 0; j < i; j++)
                {
                    killset(gno, setn[j]);
                }
                return 0;
            }
            activateset(gno, setn[i]);
            settype(gno, setn[i], XY);
            x[i] = (double *)calloc(BUFSIZE, sizeof(double));
            y[i] = (double *)calloc(BUFSIZE, sizeof(double));
            if (x[i] == NULL || y[i] == NULL)
            {
                errwin("Insufficient memory for set");
                cxfree(x[i]);
                cxfree(y[i]);
                for (j = 0; j < i + 1; j++)
                {
                    killset(gno, setn[j]);
                }
                return (0);
            }
            *(x[i]) = xval;
            *(y[i]) = yr[i];
            scnt[i] = 1;
        }
        while (!do_restart && (fgets(buf, MAX_LINE_LEN, fp) != NULL))
        {
            readline++;
            if (buf[0] == '#')
            {
                continue;
            }
            if (strlen(buf) < 2)
            {
                continue;
            }
            if (buf[0] == '@')
            {
                change_gno = -1;
                change_type = cur_type;
                read_param(buf + 1);
                if (change_gno >= 0)
                {
                    cur_gno = gno = change_gno;
                }
                if (change_type != cur_type)
                {
                    cur_type = change_type;
                    retval = -1;
                    break; /* exit this module and store any set */
                }
                continue;
            }
            convertchar(buf);
            s = buf;
            s = strtok(s, " \t\n");
            if (s == NULL)
            {
                continue;
            }
            /* check for set separator */
            pstat = sscanf(s, "%lf", &xval);
            if (pstat == 0)
            {
                do_restart = 1;
                continue;
            }
            else
            {
                s = NULL;
                for (j = 0; j < cnt - 1; j++)
                {
                    s = strtok(s, " \t\n");
                    if (s == NULL)
                    {
                        yr[j] = 0.0;
                        errwin("Number of items in column incorrect");
                    }
                    else
                    {
                        yr[j] = atof(s);
                    }
                    s = NULL;
                }
                for (i = 0; i < cnt - 1; i++)
                {
                    *(x[i] + scnt[i]) = xval;
                    *(y[i] + scnt[i]) = yr[i];
                    scnt[i]++;
                    if (scnt[i] % BUFSIZE == 0)
                    {
                        x[i] = (double *)realloc(x[i], (scnt[i] + BUFSIZE) * sizeof(double));
                        y[i] = (double *)realloc(y[i], (scnt[i] + BUFSIZE) * sizeof(double));
                    }
                }
            }
        }
        for (i = 0; i < cnt - 1; i++)
        {
            setcol(gno, x[i], setn[i], scnt[i], 0);
            setcol(gno, y[i], setn[i], scnt[i], 1);
            setcomment(gno, setn[i], fn);
            log_results(fn);
            updatesetminmax(gno, setn[i]);
        }
        if (!do_restart)
        {
            if (retval == -1)
            {
                return retval;
            }
            else
            {
                return 1;
            }
        }
        else
        {
            do_restart = 0;
            goto restart;
        }
    }
    return 0;
}

int readbinary(int gno, char *fn, FILE *fp)
{
    int i, j, setn, nsets = 0, npts;
    double *x, *y;
    float *xf, *yf;

    /*
       fread(&type, sizeof(int), 1, fp);
   */
    fread(&nsets, sizeof(int), 1, fp);
    if (nsets > g[gno].maxplot)
    {
        sprintf(buf, "Not enough sets: have %d, need %d", g[gno].maxplot, nsets);
        errwin(buf);
        return 0;
    }
    for (i = 0; i < nsets; i++)
    {
        fread(&npts, sizeof(int), 1, fp);
        if (npts > 0)
        {
            x = (double *)calloc(npts, sizeof(double));
            if (x == NULL)
            {
                errwin("Can't calloc in readbinary");
                return 0;
            }
            y = (double *)calloc(npts, sizeof(double));
            if (y == NULL)
            {
                errwin("Can't calloc in readbinary");
                cxfree(x);
                return 0;
            }
            xf = (float *)calloc(npts, sizeof(float));
            if (xf == NULL)
            {
                errwin("Can't calloc in readbinary");
                return 0;
            }
            yf = (float *)calloc(npts, sizeof(float));
            if (yf == NULL)
            {
                errwin("Can't calloc in readbinary");
                cxfree(xf);
                return 0;
            }
            fread(xf, sizeof(float), npts, fp);
            fread(yf, sizeof(float), npts, fp);
            for (j = 0; j < npts; j++)
            {
                x[j] = xf[j];
                y[j] = yf[j];
            }
            cfree(xf);
            cfree(yf);
            if ((setn = nextset(gno)) == -1)
            {
                cxfree(x);
                cxfree(y);
                return 0;
            }
            activateset(gno, setn);
            settype(gno, setn, XY);
            setcol(gno, x, setn, npts, 0);
            setcol(gno, y, setn, npts, 1);
            setcomment(gno, setn, fn);
            log_results(fn);
            updatesetminmax(gno, setn);
        }
    }
    return 1;
}

int readxystring(int, char *, FILE *)
{
    return 0;
}

/*
 * read file types using dx and/or dy
 */
int readxxyy(int gno, char *fn, FILE *fp, int type)
{
    int i = 0, j = 0, pstat, readset = 0, retval = 0;
    double *x, *y, *dx, *dy, *dz, *dw;
    double xtmp, ytmp, dxtmp, dytmp, dztmp, dwtmp;

    x = y = dx = dy = dz = dw = NULL;
    x = (double *)calloc(BUFSIZE, sizeof(double));
    y = (double *)calloc(BUFSIZE, sizeof(double));
    switch (type)
    {
    case XYZ:
    case XYRT:
    case XYDX:
    case XYDY:
        dx = (double *)calloc(BUFSIZE, sizeof(double));
        break;
    case XYDXDX:
    case XYDYDY:
    case XYDXDY:
    case XYUV:
        dx = (double *)calloc(BUFSIZE, sizeof(double));
        dy = (double *)calloc(BUFSIZE, sizeof(double));
        break;
    case XYHILO:
    case XYBOX:
        dx = (double *)calloc(BUFSIZE, sizeof(double));
        dy = (double *)calloc(BUFSIZE, sizeof(double));
        dz = (double *)calloc(BUFSIZE, sizeof(double));
        break;
    case XYBOXPLOT:
        dx = (double *)calloc(BUFSIZE, sizeof(double));
        dy = (double *)calloc(BUFSIZE, sizeof(double));
        dz = (double *)calloc(BUFSIZE, sizeof(double));
        dw = (double *)calloc(BUFSIZE, sizeof(double));
        break;
    default:
        dx = (double *)calloc(BUFSIZE, sizeof(double));
        dy = (double *)calloc(BUFSIZE, sizeof(double));
        break;
    }
    if (x == NULL || y == NULL)
    {
        errwin("Insufficient memory for set");
        cxfree(x);
        cxfree(y);
        cxfree(dx);
        cxfree(dy);
        cxfree(dz);
        cxfree(dw);
        return (0);
    }
    while (fgets(buf, MAX_LINE_LEN, fp) != NULL)
    {
        readline++;
        if (buf[0] == '#')
        {
            continue;
        }
        if (strlen(buf) < 2)
        {
            continue;
        }
        if (buf[0] == '@')
        {
            change_gno = -1;
            change_type = cur_type;
            read_param(buf + 1);
            if (change_gno >= 0)
            {
                cur_gno = gno = change_gno;
            }
            if (change_type != cur_type)
            {
                if (change_type != cur_type)
                {
                    cur_type = change_type;
                    retval = -1;
                    break; /* exit this module and store any set */
                }
            }
            continue;
        }
        convertchar(buf);
        /* count the number of items scanned */
        if ((pstat = sscanf(buf, "%lf %lf %lf %lf %lf %lf", &xtmp, &ytmp, &dxtmp, &dytmp, &dztmp, &dwtmp)) >= 1)
        {
            /* got x and y so increment */
            x[i] = xtmp;
            y[i] = ytmp;
            if (type == XYDX || type == XYDY || type == XYZ || type == XYRT)
            {
                dx[i] = dxtmp;
            }
            else if (type == XYHILO || type == XYBOX)
            {
                dx[i] = dxtmp;
                dy[i] = dytmp;
                dz[i] = dztmp;
            }
            else if (type == XYBOXPLOT)
            {
                dx[i] = dxtmp;
                dy[i] = dytmp;
                dz[i] = dztmp;
                dw[i] = dwtmp;
            }
            else
            {
                dx[i] = dxtmp;
                dy[i] = dytmp;
            }
            i++;
            if (i % BUFSIZE == 0)
            {
                x = (double *)realloc(x, (i + BUFSIZE) * sizeof(double));
                y = (double *)realloc(y, (i + BUFSIZE) * sizeof(double));
                switch (type)
                {
                case XYDX:
                case XYDY:
                case XYZ:
                case XYRT:
                    dx = (double *)realloc(dx, (i + BUFSIZE) * sizeof(double));
                    break;
                case XYDXDX:
                case XYDYDY:
                case XYDXDY:
                case XYUV:
                    dx = (double *)realloc(dx, (i + BUFSIZE) * sizeof(double));
                    dy = (double *)realloc(dy, (i + BUFSIZE) * sizeof(double));
                    break;
                case XYHILO:
                case XYBOX:
                    dx = (double *)realloc(dx, (i + BUFSIZE) * sizeof(double));
                    dy = (double *)realloc(dy, (i + BUFSIZE) * sizeof(double));
                    dz = (double *)realloc(dz, (i + BUFSIZE) * sizeof(double));
                    break;
                case XYBOXPLOT:
                    dx = (double *)realloc(dx, (i + BUFSIZE) * sizeof(double));
                    dy = (double *)realloc(dy, (i + BUFSIZE) * sizeof(double));
                    dz = (double *)realloc(dz, (i + BUFSIZE) * sizeof(double));
                    dw = (double *)realloc(dz, (i + BUFSIZE) * sizeof(double));
                    break;
                default:
                    dx = (double *)realloc(dx, (i + BUFSIZE) * sizeof(double));
                    dy = (double *)realloc(dy, (i + BUFSIZE) * sizeof(double));
                    break;
                }
            }
        }
        else
        {
            if (i != 0)
            {
                if ((j = nextset(gno)) == -1)
                {
                    cxfree(x);
                    cxfree(y);
                    cxfree(dx);
                    cxfree(dy);
                    cxfree(dz);
                    cxfree(dw);
                    return readset;
                }
                activateset(gno, j);
                settype(gno, j, type);
                setcol(gno, x, j, i, 0);
                setcol(gno, y, j, i, 1);
                setcol(gno, dx, j, i, 2);
                setcol(gno, dy, j, i, 3);
                setcol(gno, dz, j, i, 4);
                setcol(gno, dw, j, i, 5);
                setcomment(gno, j, fn);
                log_results(fn);
                updatesetminmax(gno, j);
                readset++;
            }
            else
            {
                readerror++;
                fprintf(stderr, "Error at line #%1d: %s", readline, buf);
                if (readerror > MAXERR)
                {
                    if (yesno("Lots of errors, abort?", NULL, NULL, NULL))
                    {
                        cxfree(x);
                        cxfree(y);
                        cxfree(dx);
                        cxfree(dy);
                        cxfree(dz);
                        cxfree(dw);
                        return (0);
                    }
                    else
                    {
                        readerror = 0;
                    }
                }
            }
            i = 0;
            x = (double *)calloc(BUFSIZE, sizeof(double));
            y = (double *)calloc(BUFSIZE, sizeof(double));
            switch (type)
            {
            case XYDX:
            case XYZ:
            case XYRT:
            case XYDY:
                dx = (double *)calloc(BUFSIZE, sizeof(double));
                break;
            case XYDXDX:
            case XYDYDY:
            case XYDXDY:
            case XYUV:
                dx = (double *)calloc(BUFSIZE, sizeof(double));
                dy = (double *)calloc(BUFSIZE, sizeof(double));
                break;
            case XYHILO:
            case XYBOX:
                dx = (double *)calloc(BUFSIZE, sizeof(double));
                dy = (double *)calloc(BUFSIZE, sizeof(double));
                dz = (double *)calloc(BUFSIZE, sizeof(double));
                break;
            case XYBOXPLOT:
                dx = (double *)calloc(BUFSIZE, sizeof(double));
                dy = (double *)calloc(BUFSIZE, sizeof(double));
                dz = (double *)calloc(BUFSIZE, sizeof(double));
                dw = (double *)calloc(BUFSIZE, sizeof(double));
                break;
            default:
                dx = (double *)calloc(BUFSIZE, sizeof(double));
                dy = (double *)calloc(BUFSIZE, sizeof(double));
                break;
            }
            if (x == NULL || y == NULL)
            {
                errwin("Insufficient memory for set");
                cxfree(x);
                cxfree(y);
                cxfree(dx);
                cxfree(dy);
                cxfree(dz);
                cxfree(dw);
                killset(gno, j);
                return (readset);
            }
        }
    }
    if (i != 0)
    {
        if ((j = nextset(gno)) == -1)
        {
            cxfree(x);
            cxfree(y);
            cxfree(dx);
            cxfree(dy);
            cxfree(dz);
            cxfree(dw);
            return readset;
        }
        activateset(gno, j);
        settype(gno, j, type);
        setcol(gno, x, j, i, 0);
        setcol(gno, y, j, i, 1);
        setcol(gno, dx, j, i, 2);
        setcol(gno, dy, j, i, 3);
        setcol(gno, dz, j, i, 4);
        setcol(gno, dw, j, i, 5);
        setcomment(gno, j, fn);
        log_results(fn);
        updatesetminmax(gno, j);
        readset++;
    }
    else
    {
        cxfree(x);
        cxfree(y);
        cxfree(dx);
        cxfree(dy);
        cxfree(dz);
        cxfree(dw);
    }
    if (retval == -1)
    {
        return retval;
    }
    else
    {
        return readset;
    }
}

void kill_blockdata(void)
{
    int j;
    if (blockdata != NULL)
    {
        for (j = 0; j < maxblock; j++)
        {
            cxfree(blockdata[j]);
        }
    }
}

int alloc_blockdata(int ncols)
{
    int j;
    if (blockdata != NULL)
    {
        kill_blockdata();
    }
    if (ncols < MAXPLOT)
    {
        ncols = MAXPLOT;
    }
    blockdata = (double **)malloc(ncols * sizeof(double *));
    if (blockdata != NULL)
    {
        maxblock = ncols;
        for (j = 0; j < maxblock; j++)
        {
            blockdata[j] = NULL;
        }
    }
    else
    {
        errwin("alloc_blockdata(): Error, unable to allocate memory for block data");
    }
    return (0);
}

/*
 * read block data
 */
int readblockdata(int, char *, FILE *fp)
{
    int i = 0, j, k, ncols = 0, pstat;
    int first = 1, readerror = 0;
    double **data = NULL;
    char tmpbuf[2048], *s, tbuf[256];
    int linecount = 0;

    i = 0;
    pstat = 0;
    while ((s = fgets(buf, MAX_LINE_LEN, fp)) != NULL)
    {
        readline++;
        linecount++;
        if (buf[0] == '#')
        {
            continue;
        }
        if (buf[0] == '@')
        {
            read_param(buf + 1);
            continue;
        }
        if ((int)strlen(buf) > 1)
        {
            convertchar(buf);
            if (first) /* count the number of columns */
            {
                ncols = 0;
                strcpy(tmpbuf, buf);
                s = tmpbuf;
                while (*s == ' ' || *s == '\t' || *s == '\n')
                {
                    s++;
                }
                while ((s = strtok(s, " \t\n")) != NULL)
                {
                    ncols++;
                    s = NULL;
                }
                if (ncols < 1 || ncols > maxblock)
                {
                    errwin("Column count incorrect");
                    return 0;
                }
                data = (double **)malloc(sizeof(double *) * maxblock);
                if (data == NULL)
                {
                    errwin("Can't allocate memory for block data");
                    return (0);
                }
                for (j = 0; j < ncols; j++)
                {
                    data[j] = (double *)calloc(BUFSIZE, sizeof(double));
                    if (data[j] == NULL)
                    {
                        errwin("Insufficient memory for block data");
                        for (k = 0; k < j; k++)
                        {
                            cxfree(data[k]);
                        }
                        cxfree(data);
                        return 0;
                    }
                }
                first = 0;
            }
            s = buf;
            while (*s == ' ' || *s == '\t' || *s == '\n')
            {
                s++;
            }
            for (j = 0; j < ncols; j++)
            {
                s = strtok(s, " \t\n");
                if (s == NULL)
                {
                    data[j][i] = 0.0;
                    sprintf(tbuf, "Number of items in column incorrect at line %d, line skipped", linecount);
                    errwin(tbuf);
                    readerror++;
                    if (readerror > MAXERR)
                    {
                        if (yesno("Lots of errors, abort?", NULL, NULL, NULL))
                        {
                            for (k = 0; k < ncols; k++)
                            {
                                cxfree(data[k]);
                            }
                            cxfree(data);
                            return (0);
                        }
                        else
                        {
                            readerror = 0;
                        }
                    }
                    /* skip the rest */
                    goto bustout;
                }
                else
                {
                    data[j][i] = atof(s);
                }
                s = NULL;
            }
            i++;
            if (i % BUFSIZE == 0)
            {
                for (j = 0; j < ncols; j++)
                {
                    data[j] = (double *)realloc(data[j], (i + BUFSIZE) * sizeof(double));
                    if (data[j] == NULL)
                    {
                        errwin("Insufficient memory for block data");
                        for (k = 0; k < j; k++)
                        {
                            cxfree(data[k]);
                        }
                        cxfree(data);
                        return 0;
                    }
                }
            }
        }
    bustout:
        ;
    }
    for (j = 0; j < ncols; j++)
    {
        blockdata[j] = data[j];
    }
    cxfree(data);
    blocklen = i;
    blockncols = ncols;
    return 1;
}

void create_set_fromblock(int, int type, char *cols)
{
    int i;
    int setno, graphno;
    int cx, cy, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    double *tx, *ty, *t2, *t3, *t4, *t5;
    int nc, *coli;
    char *s, buf[256];
    strcpy(buf, cols);
    s = buf;
    nc = 0;
    coli = (int *)malloc(maxblock * sizeof(int *));
    while ((s = strtok(s, ":")) != NULL)
    {
        coli[nc] = atoi(s);
        coli[nc]--;
        nc++;
        s = NULL;
    }
    if (nc == 0)
    {
        errwin("No columns scanned in column string");
        free(coli);
        return;
    }
    for (i = 0; i < nc; i++)
    {
        if (coli[i] < 0 || coli[i] >= blockncols)
        {
            errwin("Incorrect column specification");
            free(coli);
            return;
        }
    }

    cx = coli[0];
    cy = coli[1];
    if (cx >= blockncols)
    {
        errwin("Column for X exceeds the number of columns in block data");
        free(coli);
        return;
    }
    if (cy >= blockncols)
    {
        errwin("Column for Y exceeds the number of columns in block data");
        free(coli);
        return;
    }
    switch (type)
    {
    case XY:
        break;
    case XYRT:
    case XYDX:
    case XYDY:
    case XYZ:
        c1 = coli[2];
        if (c1 >= blockncols)
        {
            errwin("Column for E1 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        break;
    case XYDXDX:
    case XYDYDY:
    case XYDXDY:
        c1 = coli[2];
        c2 = coli[3];
        if (c1 >= blockncols)
        {
            errwin("Column for E1 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        if (c2 >= blockncols)
        {
            errwin("Column for E2 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        break;
    case XYHILO:
    case XYBOX:
        c1 = coli[2];
        c2 = coli[3];
        c3 = coli[4];
        if (c1 >= blockncols)
        {
            errwin("Column for E1 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        if (c2 >= blockncols)
        {
            errwin("Column for E2 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        if (c3 >= blockncols)
        {
            errwin("Column for E3 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        break;
    case XYBOXPLOT:
        c1 = coli[2];
        c2 = coli[3];
        c3 = coli[4];
        c4 = coli[5];
        if (c1 >= blockncols)
        {
            errwin("Column for E1 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        if (c2 >= blockncols)
        {
            errwin("Column for E2 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        if (c3 >= blockncols)
        {
            errwin("Column for E3 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        if (c4 >= blockncols)
        {
            errwin("Column for E4 exceeds the number of columns in block data");
            free(coli);
            return;
        }
        break;
    }
    setno = -1;
    graphno = -1;

    if (graphno == -1)
    {
        graphno = cg;
    }
    if (setno == -1)
    {
        setno = nextset(graphno);
    }
    if (setno == -1)
    {
        return;
    }
    if (g[graphno].active == OFF)
    {
        set_graph_active(graphno);
    }
    activateset(graphno, setno);
    settype(graphno, setno, type);

    tx = (double *)calloc(blocklen, sizeof(double));
    ty = (double *)calloc(blocklen, sizeof(double));
    for (i = 0; i < blocklen; i++)
    {
        tx[i] = blockdata[cx][i];
        ty[i] = blockdata[cy][i];
    }
    setcol(graphno, tx, setno, blocklen, 0);
    setcol(graphno, ty, setno, blocklen, 1);

    switch (type)
    {
    case XY:
        sprintf(buf, "Cols %d %d", cx + 1, cy + 1);
        break;
    case XYRT:
    case XYDX:
    case XYDY:
    case XYZ:
        sprintf(buf, "Cols %d %d %d", cx + 1, cy + 1, c1 + 1);
        t2 = (double *)calloc(blocklen, sizeof(double));
        for (i = 0; i < blocklen; i++)
        {
            t2[i] = blockdata[c1][i];
        }
        setcol(graphno, t2, setno, blocklen, 2);
        break;
    case XYDXDX:
    case XYDYDY:
    case XYDXDY:
        sprintf(buf, "Cols %d %d %d %d", cx + 1, cy + 1, c1 + 1, c2 + 1);
        t2 = (double *)calloc(blocklen, sizeof(double));
        t3 = (double *)calloc(blocklen, sizeof(double));
        for (i = 0; i < blocklen; i++)
        {
            t2[i] = blockdata[c1][i];
            t3[i] = blockdata[c2][i];
        }
        setcol(graphno, t2, setno, blocklen, 2);
        setcol(graphno, t3, setno, blocklen, 3);
        break;
    case XYHILO:
    case XYBOX:
        sprintf(buf, "Cols %d %d %d %d %d", cx + 1, cy + 1, c1 + 1, c2 + 1, c3 + 1);
        t2 = (double *)calloc(blocklen, sizeof(double));
        t3 = (double *)calloc(blocklen, sizeof(double));
        t4 = (double *)calloc(blocklen, sizeof(double));
        for (i = 0; i < blocklen; i++)
        {
            t2[i] = blockdata[c1][i];
            t3[i] = blockdata[c2][i];
            t4[i] = blockdata[c3][i];
        }
        setcol(graphno, t2, setno, blocklen, 2);
        setcol(graphno, t3, setno, blocklen, 3);
        setcol(graphno, t4, setno, blocklen, 4);
        break;
    case XYBOXPLOT:
        sprintf(buf, "Cols %d %d %d %d %d %d", cx + 1, cy + 1, c1 + 1, c2 + 1, c3 + 1, c4 + 1);
        t2 = (double *)calloc(blocklen, sizeof(double));
        t3 = (double *)calloc(blocklen, sizeof(double));
        t4 = (double *)calloc(blocklen, sizeof(double));
        t5 = (double *)calloc(blocklen, sizeof(double));
        for (i = 0; i < blocklen; i++)
        {
            t2[i] = blockdata[c1][i];
            t3[i] = blockdata[c2][i];
            t4[i] = blockdata[c3][i];
            t5[i] = blockdata[c4][i];
        }
        setcol(graphno, t2, setno, blocklen, 2);
        setcol(graphno, t3, setno, blocklen, 3);
        setcol(graphno, t4, setno, blocklen, 4);
        setcol(graphno, t5, setno, blocklen, 5);
        break;
    }

    free(coli);
    setcomment(graphno, setno, buf);
    log_results(buf);
    updatesetminmax(graphno, setno);
    update_status_popup(NULL, NULL, NULL);
    drawgraph();
}

/**/
/* Reads a rawspicefile */
/**/

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/* ASSUMED RAW FILE FORMAT:

   Both Spice and CAZM produce this same file format.  It looks as
   follows:

   Title
   Date
   Name
   Flags
   No. Variables: 3
   No. Points: 00061
Command
Variables:
0  time time
1  v(2) voltage
2  v(5) voltage
Values:
0	0e0
0e0
0e0
1	5e-10
0e0
0e0
2	1e-9
0e0
4.7e-1

For now, the only lines I pay attention to are "No. Variables",
"No. Points", "Variables", and "Values".  I scan for those exact
words and then assume that the information to follow is in the
format above.

Multiple runs may be present in a file.

If Variable 0 is time, then I assume this is a TRANSIENT run,
else if Variable 0 is Frequency, then I assume this is an AC
run, else I assume it is a TRANSFER run.
*/

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/*
 * note that the maximum number of sets is 30
 */
#define MAXSETN 30

int readrawspice(int gno, char *, FILE *fp)
{
    int i, j, setn[MAXSETN];
    double *x[MAXSETN], *y[MAXSETN], xval, yval;
    char inputlinetype[BUFSIZE];
    char inputline[BUFSIZE];
    char tmpstring1[BUFSIZE], tmpstring2[BUFSIZE];
    char *truncated;
    int numvariables, numpoints, tmpint;
    numvariables = 0;
    numpoints = 0;

    while (fgets(inputline, BUFSIZE, fp) != NULL)
    {
        readline++;
        strcpy(inputlinetype, "");
        sscanf(inputline, "%s%s", inputlinetype, tmpstring1);
        if (strcmp(inputlinetype, "Title:") == 0)
        {
            truncated = &inputline[6];
            for (i = 0; i < (int)strlen(truncated); i++)
            {
                if (truncated[i] == '\n')
                {
                    truncated[i] = '\0';
                    break;
                }
            }
            set_plotstr_string(&g[gno].labs.title, truncated);
        }
        if (strcmp(inputlinetype, "Date:") == 0)
        {
            truncated = &inputline[5];
            for (i = 0; i < (int)strlen(truncated); i++)
            {
                if (truncated[i] == '\n')
                {
                    truncated[i] = '\0';
                    break;
                }
            }
            set_plotstr_string(&g[gno].labs.stitle, truncated);
        }
        if (strcmp(inputlinetype, "No.") == 0)
        {
            if (strcmp(tmpstring1, "Variables:") == 0)
            {
                sscanf(inputline, "%s%s%d", inputlinetype, tmpstring1,
                       &numvariables);
                /*				printf("%d variables\n",numvariables);*/
            }
        }
        /**/
        /* Accounts for this variant .....*/
        /**/
        /*Variables:    0    TIME    seconds */
        /*			1        nearend vlts */
        /**/
        /**/
        if ((strcmp(inputlinetype, "Variables:") == 0) && (numvariables != 0)
            && (int)sscanf(inputline, "%s%d%s%s", inputlinetype, &tmpint,
                           tmpstring1, tmpstring2) == 4)
        {
            /* Read off x axis title and ingore for now */
            sscanf(inputline, "%s%d%s%s", inputlinetype, &tmpint, tmpstring1,
                   tmpstring2);
            for (i = 0; i < numvariables - 1; i++)
            {
                if ((setn[i] = nextset(gno)) == -1)
                {
                    for (j = 0; j < i; j++)
                    {
                        killset(gno, setn[j]);
                    }
                    return 0;
                }
                fgets(inputline, BUFSIZE, fp);
                readline++;
                sscanf(inputline, "%d%s%s", &tmpint, tmpstring1, tmpstring2);
                /*				printf("%d %s %s
            \n",tmpint,tmpstring1,tmpstring2); */
                strcat(tmpstring1, "   ");
                strcat(tmpstring1, tmpstring2);
                activateset(gno, setn[i]);
                settype(gno, setn[i], XY);
                setcomment(gno, setn[i], (char *)tmpstring1);
                log_results(tmpstring1);
            }
        }
        /**/
        /* Accounts for this variant .....*/
        /**/
        /*Variables:                      */
        /*             0    TIME    seconds */
        /*			1        nearend vlts */
        /**/
        /**/
        else if ((strcmp(inputlinetype, "Variables:") == 0) && (numvariables != 0) && (int)(sscanf(inputline, "%s%d%s%s",
                                                                                                   inputlinetype, &tmpint,
                                                                                                   tmpstring1, tmpstring2) == 1))
        {
            /* Read off x axis title and ingore for now */
            fgets(inputline, BUFSIZE, fp);
            readline++;
            sscanf(inputline, "%s%d%s%s", inputlinetype, &tmpint,
                   tmpstring1, tmpstring2);
            for (i = 0; i < numvariables - 1; i++)
            {
                if ((setn[i] = nextset(gno)) == -1)
                {
                    for (j = 0; j < i; j++)
                    {
                        killset(gno, setn[j]);
                    }
                    return 0;
                }
                fgets(inputline, BUFSIZE, fp);
                readline++;
                sscanf(inputline, "%d%s%s", &tmpint, tmpstring1, tmpstring2);
                /*				printf("%d %s %s
            \n",tmpint,tmpstring1,tmpstring2); */
                strcat(tmpstring1, "   ");
                strcat(tmpstring1, tmpstring2);
                activateset(gno, setn[i]);
                settype(gno, setn[i], XY);
                setcomment(gno, setn[i], (char *)tmpstring1);
                log_results(tmpstring1);
            }
        }
        if (strcmp(inputlinetype, "Values:") == 0)
        {
            /**/
            /* Read in actual values until end of file */
            /**/
            for (i = 0; i < numvariables - 1; i++)
            {
                /**/
                /* Allocate initial memory for each array */
                /**/
                x[i] = (double *)calloc(BUFSIZE, sizeof(double));
                y[i] = (double *)calloc(BUFSIZE, sizeof(double));
                if (x[i] == NULL || y[i] == NULL)
                {
                    errwin("Insufficient memory for set; Clearing data");
                    cxfree(x[i]);
                    cxfree(y[i]);
                    for (j = 0; j < i + 1; j++)
                    {
                        killset(gno, setn[j]);
                    }
                    return 0;
                }
            }
            while (fgets(inputline, BUFSIZE, fp) != NULL)
            {
                readline++;
                /**/
                /* If not an incremental line, grab another line */
                /**/
                while (sscanf(inputline, "%d%lf", &tmpint, &xval) != 2)
                {
                    if (fgets(inputline, BUFSIZE, fp) == NULL)
                    {
                        readline++;
                        /**/
                        /* EOF or error in obtaining another line */
                        /**/
                        break;
                    }
                }
                for (j = 0; j < numvariables - 1; j++)
                {
                    fgets(inputline, BUFSIZE, fp);
                    readline++;
                    sscanf(inputline, "%lf", &yval);
                    x[j][numpoints] = xval;
                    y[j][numpoints] = yval;
                }
                numpoints++;
                /**/
                /* If I run out of space, add more at end of arrays */
                /**/
                if (numpoints % BUFSIZE == 0)
                {
                    for (j = 0; j < numvariables - 1; j++)
                    {
                        x[j] = (double *)realloc(x[j],
                                                 (numpoints + BUFSIZE) * sizeof(double));
                        y[j] = (double *)realloc(y[j],
                                                 (numpoints + BUFSIZE) * sizeof(double));
                    }
                }
                strcpy(inputline, "");
            }
            for (i = 0; i < numvariables - 1; i++)
            {
                setcol(gno, x[i], setn[i], numpoints, 0);
                setcol(gno, y[i], setn[i], numpoints, 1);
                updatesetminmax(gno, setn[i]);
            }
        }
    }
    sprintf(tmpstring1, "%d sets of %d data points read", numvariables,
            numpoints);
    if (inwin)
    {
        set_left_footer(tmpstring1);
    }
    return 1;
}

#if defined(HAVE_NETCDF) || defined(HAVE_MFHDF)

/*
 * read a variable from netcdf file into a set in graph gno
 * xvar and yvar are the names for x, y in the netcdf file resp.
 * return 0 on fail, return 1 if success.
 *
 * if xvar == NULL, then load the index of the point to x
 *
 */
int readnetcdf(int gno,
               int setno,
               char *netcdfname,
               char *xvar,
               char *yvar,
               int nstart,
               int nstop,
               int nstride)
{
    int cdfid; /* netCDF id */
    int ndims, nvars, ngatts, recdim;
    int err;
    int i, n, retval = 0;
    double *x, *y;
    float *xf, *yf;
    short *xs, *ys;
    long *xl, *yl;

    /* dimension id for unlimited dimension */
    int udim;

    /* variable ids */
    int x_id, y_id;

    /* variable shapes */
    int dims[2];
    long start[2];
    long count[2];

    nc_type xdatatype = 0;
    nc_type ydatatype = 0;
    int xndims, xdim[10], xnatts;
    int yndims, ydim[10], ynatts;
    long nx, ny;

    long size;
    char name[256];

    extern int ncopts;
    ncopts = 0; /* no crash on error */

    /*
    * get a set if on entry setno == -1, if setno=-1, then fail
    */
    if (setno == -1)
    {
        if ((setno = nextset(gno)) == -1)
        {
            return 0;
        }
    }
    else
    {
        if (isactive(cg, setno))
        {
            killset(gno, setno);
        }
    }
    /*
    * open the netcdf file and locate the variable to read
    */
    if ((cdfid = ncopen(netcdfname, NC_NOWRITE)) == -1)
    {
        errwin("Can't open file.");
        return 0;
    }
    if (xvar != NULL)
    {
        if ((x_id = ncvarid(cdfid, xvar)) == -1)
        {
            char ebuf[256];
            sprintf(ebuf, "readnetcdf(): No such variable %s for X", xvar);
            errwin(ebuf);
            return 0;
        }
        ncvarinq(cdfid, x_id, NULL, &xdatatype, &xndims, xdim, &xnatts);
        ncdiminq(cdfid, xdim[0], NULL, &nx);
        if (xndims != 1)
        {
            errwin("Number of dimensions for X must be 1.");
            return 0;
        }
    }
    if ((y_id = ncvarid(cdfid, yvar)) == -1)
    {
        char ebuf[256];
        sprintf(ebuf, "readnetcdf(): No such variable %s for Y", yvar);
        errwin(ebuf);
        return 0;
    }
    ncvarinq(cdfid, y_id, NULL, &ydatatype, &yndims, ydim, &ynatts);
    ncdiminq(cdfid, ydim[0], NULL, &ny);
    if (yndims != 1)
    {
        errwin("Number of dimensions for Y must be 1.");
        return 0;
    }
    if (xvar != NULL)
    {
        n = nx < ny ? nx : ny;
    }
    else
    {
        n = ny;
    }
    if (n <= 0)
    {
        errwin("Length of dimension == 0.");
        return 0;
    }
    /*
    * allocate for this set
    */
    x = (double *)calloc(n, sizeof(double));
    y = (double *)calloc(n, sizeof(double));
    if (x == NULL || y == NULL)
    {
        errwin("Insufficient memory for set");
        cxfree(x);
        cxfree(y);
        ncclose(cdfid);
        return 0;
    }
    start[0] = 0;
    count[0] = n;
    /* This will retrieve whole file, modify
    * these values to get subset. This will only
    * work for single-dimension vars.  You need
    * to add dims to start & count for
    * multi-dimensional. */

    /*
    * read the variables from the netcdf file
    */
    if (xvar != NULL)
    {
        /* TODO should check for other data types here */
        /* TODO should check for NULL on the callocs() */
        /* TODO making assumptions about the sizes of shorts and longs */
        switch (xdatatype)
        {
        case NC_SHORT:
            xs = (short *)calloc(n, sizeof(short));
            ncvarget(cdfid, x_id, start, count, (void *)xs);
            for (i = 0; i < n; i++)
            {
                x[i] = xs[i];
            }
            cfree(xs);
            break;
        case NC_LONG:
            xl = (long *)calloc(n, sizeof(long));
            ncvarget(cdfid, x_id, start, count, (void *)xl);
            for (i = 0; i < n; i++)
            {
                x[i] = xl[i];
            }
            cfree(xl);
            break;
        case NC_FLOAT:
            xf = (float *)calloc(n, sizeof(float));
            ncvarget(cdfid, x_id, start, count, (void *)xf);
            for (i = 0; i < n; i++)
            {
                x[i] = xf[i];
            }
            cfree(xf);
            break;
        case NC_DOUBLE:
            ncvarget(cdfid, x_id, start, count, (void *)x);
            break;
        default:
            errwin("Data type not supported");
            cxfree(x);
            cxfree(y);
            ncclose(cdfid);
            return 0;
            break;
        }
    } /* just load index */
    else
    {
        for (i = 0; i < n; i++)
        {
            x[i] = i + 1;
        }
    }
    switch (ydatatype)
    {
    case NC_SHORT:
        ys = (short *)calloc(n, sizeof(short));
        ncvarget(cdfid, y_id, start, count, (void *)ys);
        for (i = 0; i < n; i++)
        {
            y[i] = ys[i];
        }
        break;
    case NC_LONG:
        yl = (long *)calloc(n, sizeof(long));
        ncvarget(cdfid, y_id, start, count, (void *)yl);
        for (i = 0; i < n; i++)
        {
            y[i] = yl[i];
        }
        break;
    case NC_FLOAT:
        /* TODO should check for NULL here */
        yf = (float *)calloc(n, sizeof(float));
        ncvarget(cdfid, y_id, start, count, (void *)yf);
        for (i = 0; i < n; i++)
        {
            y[i] = yf[i];
        }
        cfree(yf);
        break;
    case NC_DOUBLE:
        ncvarget(cdfid, y_id, start, count, (void *)y);
        break;
    default:
        errwin("Data type not supported");
        cxfree(x);
        cxfree(y);
        ncclose(cdfid);
        return 0;
        break;
    }
    ncclose(cdfid);

    /*
    * initialize stuff for the newly created set
    */
    activateset(gno, setno);
    settype(gno, setno, XY);
    setcol(gno, x, setno, n, 0);
    setcol(gno, y, setno, n, 1);

    sprintf(buf, "File %s x = %s y = %s", netcdfname, xvar == NULL ? "Index" : xvar, yvar);
    setcomment(gno, setno, buf);
    log_results(buf);
    updatesetminmax(gno, setno);
    return 1;
}
#endif /* HAVE_NETCDF */

#ifdef HAVE_HDF
/* the netCDF interface is OK for now, no need to implement this */
#endif /* HAVE_HDF */
