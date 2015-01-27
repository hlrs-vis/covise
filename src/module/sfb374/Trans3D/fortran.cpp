/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          initia.cpp  -  fortran like I/O
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include <string>
#ifndef __sgi
#include <cstdio>
#include <cstdarg>
#else
#include <stdio.h>
#include <stdarg.h>
#endif
#include <iostream>

#include "fortran.h"
#include "error.h"

#define MAXSTREAM 30

fstream *fp[MAXSTREAM];
char AppPath[200];
char FilePath[200];

// ***************************************************************************
// initialize streams
// ***************************************************************************

void init_stream()
{
    for (int i = 0; i < MAXSTREAM; i++)
        fp[i] = 0;
}

// ***************************************************************************
// open stream
// ***************************************************************************

int unitopen(int unit, const char *pname, int imode, bool bApp)
{
    char buffer[250], *pstr;

    if (fp[unit])
        return 1; // already open
    if (unit == 0 || unit == 6)
        return 0; // screen output

    pstr = buffer;
    if (bApp == true)
    {
        strcpy(buffer, AppPath);
        pstr += strlen(AppPath);
    }
    if (*pname == '\0')
        sprintf(pstr, "fort%i.txt", unit);
    else
        strcpy(pstr, pname);

    fp[unit] = new fstream(pstr, imode);
    if (!fp[unit]->is_open())
    {
        delete fp[unit];
        fp[unit] = 0;
    }
    if (!fp[unit])
    {
        // war : WarningFunction("konnte Datei %s nicht oeffnen\n",-1,pname);
        WarningFunction("konnte Datei %s nicht oeffnen\n", pname);
        return -1;
    }
    return 0;
}

// ***************************************************************************
// output in FORTRAN style
// ***************************************************************************

int unitfortranwrite(int unit, const char *pstr, ...)
{
    char buffer[200], *pc;
    va_list argptr;

    va_start(argptr, pstr);
    vsprintf(buffer, pstr, argptr);
    va_end(argptr);
    pc = strchr(buffer, 'e');
    while (pc != 0)
    {
        if (isdigit(*(pc - 1)) != 0 && strchr("+-", *(pc + 1)) != 0)
            *pc = 'D';
        pc = strchr(pc + 1, 'e');
    }

    if (unit == 0 || unit == 6)
    {
        cout << buffer;
        cout.flush();
    }
    else
    {
        if (!fp[unit])
            if (unitopen(unit, "", ios::out) < 0)
                return -1;
        (*fp[unit]) << buffer;
    }
    return 0;
}

// ***************************************************************************
// text output
// ***************************************************************************

int unitwrite(int unit, const char *pstr, ...)
{
    char buffer[500];
    va_list argptr;

    va_start(argptr, pstr);
    vsprintf(buffer, pstr, argptr);
    va_end(argptr);

    if (unit == 9 || unit == 29) // ray tracing
        return 0;
    if (unit == 0 || unit == 6)
    {
        cout << buffer;
        cout.flush();
        return 0;
    }
    if (!fp[unit])
        if (unitopen(unit, "", ios::out) < 0)
            return -1;
    (*fp[unit]) << buffer;

    return 0;
}

// ***************************************************************************
// text output with newline
// ***************************************************************************

int unitwriteln(int unit, const char *pstr, ...)
{
    char buffer[500];
    va_list argptr;

    va_start(argptr, pstr);
    vsprintf(buffer, pstr, argptr);
    va_end(argptr);
    strcat(buffer, "\n");
    return unitwrite(unit, buffer);
}

// ***************************************************************************
// output integer
// ***************************************************************************

int unitwrite(int unit, int i, const char *pstr)
{
    char buffer[200];

    if (pstr == 0)
        sprintf(buffer, "%i", i);
    else
        sprintf(buffer, pstr, i);
    if (unit == 0 || unit == 6)
    {
        cout << buffer;
        cout.flush();
        return 0;
    }
    if (!fp[unit])
        if (unitopen(unit, "", ios::out) < 0)
            return -1;
    (*fp[unit]) << buffer;

    return 0;
}

// ***************************************************************************
// output bool
// ***************************************************************************

int unitwrite(int unit, bool b, const char *pstr)
{
    return unitwrite(unit, (int)b, pstr);
}

// ***************************************************************************
// output floating value
// ***************************************************************************

int unitwrite(int unit, prec r, const char *pstr)
{
    char buffer[200];

    if (pstr == 0)
        sprintf(buffer, "%lf", (double)r);
    else
        sprintf(buffer, pstr, (double)r);
    if (unit == 0 || unit == 6)
    {
        cout << buffer;
        cout.flush();
        return 0;
    }
    if (!fp[unit])
        if (unitopen(unit, "", ios::out) < 0)
            return -1;
    (*fp[unit]) << buffer;

    return 0;
}

// ***************************************************************************
// read character
// ***************************************************************************

char readchar(int unit)
{
    char c;

    if (unit == 0 || unit == 6)
    {
        cin >> c;
        return c;
    }
    if (!fp[unit])
        if (unitopen(unit, "") < 0)
            return -1;
    c = fp[unit]->get();
    return c;
}

// ***************************************************************************
// read integer (if bFortran==true: starting at ',= \t\n')
// ***************************************************************************

int readint(int unit, bool bFortran)
{
    char buffer[20], c;
    int i;

    if (unit == 0 || unit == 6)
    {
        cin >> i;
        return i;
    }
    if (!fp[unit])
        if (unitopen(unit, "") < 0)
            return -1;
    buffer[0] = ' ';
    do
    {
        if (fp[unit]->eof())
            return -1;
        c = fp[unit]->get();
        if ((isdigit(c) != 0 || strchr("+-", c) != 0)
            && strchr(",= \t\n\r", buffer[0]) == 0 && bFortran == true)
            c = '?';
        buffer[0] = c;
    } while (isdigit(buffer[0]) == 0 && strchr("+-", buffer[0]) == 0);

    i = 0;
    do
    {
        i++;
        buffer[i] = fp[unit]->get();
        if (tolower(buffer[i]) == 'd')
            buffer[i] = 'e';
    } while (isdigit(buffer[i]) != 0 || buffer[i] == 'e');
#ifndef WIN32
    if ((buffer[i] == '\r') || (buffer[i] == '\n'))
    {
        c = fp[unit]->get();
        if ((c != '\r') && (c != '\n'))
            fp[unit]->putback(c);
    }
#endif
    buffer[i] = '\0';
    sscanf(buffer, "%i", &i);

    return i;
}

void unitread(int unit, int &i) { i = readint(unit); }

// ***************************************************************************
// read bool (if bFortran==true: starting at ',= \t\n')
// ***************************************************************************

bool readbool(int unit, bool bFortran)
{
    return (readint(unit, bFortran) > 0);
}

void unitread(int unit, bool &b) { b = readbool(unit); }

// ***************************************************************************
// read floating value (if bFortran==true: starting at ',= \t\n')
// ***************************************************************************

prec readreal(int unit, bool bFortran)
{
    char buffer[20], c;
    int i;
    double d;

    if (unit == 0 || unit == 6)
    {
        cin >> d;
        return (prec)d;
    }
    if (!fp[unit])
        if (unitopen(unit, "") < 0)
            return -1;
    buffer[0] = ' ';
    do
    {
        if (fp[unit]->fail())
            return -1;
        c = fp[unit]->get();
        if ((isdigit(c) != 0 || strchr("+-.", c) != 0)
            && strchr(",= \t\n\r", buffer[0]) == 0 && bFortran == true)
            c = '?';
        buffer[0] = c;
    } while (isdigit(buffer[0]) == 0 && strchr("+-.", buffer[0]) == 0);

    i = 0;
    do
    {
        i++;
        buffer[i] = fp[unit]->get();
        if (tolower(buffer[i]) == 'd')
            buffer[i] = 'e';
    } while (isdigit(buffer[i]) != 0 || strchr("e.+-", buffer[i]) != 0);
#ifndef WIN32
    if ((buffer[i] == '\r') || (buffer[i] == '\n'))
    {
        c = fp[unit]->get();
        if ((c != '\r') && (c != '\n'))
            fp[unit]->putback(c);
    }
#endif
    buffer[i] = '\0';
    sscanf(buffer, "%lf", &d);

    return (prec)d;
}

void unitread(int unit, prec &p) { p = readreal(unit); }

// ***************************************************************************
// read line
// ***************************************************************************

char *unitreadln(int unit, char *pstr)
{
    char buffer[100], *ptab = 0;

    if (unit == 0 || unit == 6)
        cin.getline(buffer, 99);
    else
    {
        if (!fp[unit])
            if (unitopen(unit, "") < 0)
                return pstr;
        fp[unit]->getline(buffer, 99);
    }
#ifndef WIN32 // get rid of tailing \n and \rs
    char c = buffer[strlen(buffer) - 1];
    while ((c == '\r') || (c == '\n'))
    {
        buffer[strlen(buffer) - 1] = '\0';
        c = buffer[strlen(buffer) - 1];
    }
#endif

    if (pstr != 0)
        strcpy(pstr, buffer);
    if ((ptab = strchr(pstr, '\t')) == 0)
        ptab = pstr;
    else
        ptab++;
    return ptab;
}

// ***************************************************************************
// close stream
// ***************************************************************************

void unitclose(int unit)
{
    if (fp[unit])
    {
        delete fp[unit];
        fp[unit] = 0;
    }
    return;
}

// ***************************************************************************
// close all streams
// ***************************************************************************

void unitcloseall(void)
{
    for (int i = 0; i < MAXSTREAM; i++)
        unitclose(i);
    return;
}
