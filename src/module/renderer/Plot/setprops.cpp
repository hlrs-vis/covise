/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: setprops.c,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 * setprop - set properties of graphs and sets
 *
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include "globals.h"

extern int check_err;
/* defined in checkon.c TODO blongs in
 * globals.h */

extern int yesno(const char *msg1, const char *s1, const char *s2, const char *helptext);
extern void errwin(const char *s);
extern int checkon(int prop, int old_val, int new_val);

void set_prop(int gno, ...)
{
    va_list var;
    int prop, allsets = 1;
    int i, j, startg, endg, starts = 0, ends = maxplot - 1;
    double dprop;
    double dprop1, dprop2;
    char *cprop;
    char buf[256];

    if (gno == -1)
    {
        startg = 0;
        endg = maxgraph - 1;
    }
    else
    {
        startg = endg = gno;
    }

    va_start(var, gno);
    while ((prop = va_arg(var, int)) != 0)
    {
        switch (prop)
        {
        case SETS:
            allsets = 1;
            starts = 0;
            ends = maxplot - 1;
            break;
        /* when range is activated
                       case RANGE:
                  starts = va_arg(var, int);
                  ends = va_arg(var, int);
                  break;
            */
        case SET:
            switch (prop = va_arg(var, int))
            {
            case SETNUM:
                prop = va_arg(var, int);
                if (prop == -1)
                {
                    allsets = 1;
                    starts = 0;
                    ends = maxplot - 1;
                }
                else
                {
                    allsets = 0;
                    starts = ends = prop;
                }
                break;
            }
            break;
        case ACTIVE:
            prop = va_arg(var, int);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    if (prop == ON) /* could have been ignored */
                    {
                        if (g[i].p[j].deact && (g[i].p[j].ex[0] != NULL))
                        {
                            g[i].p[j].deact = 0;
                            g[i].p[j].active = ON;
                        }
                    }
                    else if (prop == OFF)
                    {
                        g[i].p[j].active = OFF;
                    }
                    else if (prop == IGNORE)
                    {
                        g[i].p[j].active = OFF;
                        g[i].p[j].deact = 1;
                    }
                }
            }
            break;
        case TYPE:
            prop = va_arg(var, int);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].type = prop;
                }
            }
            break;
        case MISSINGP:
            dprop = va_arg(var, double);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].missing = prop;
                }
            }
            break;
        case FONTP:
            prop = va_arg(var, int);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].font = prop;
                }
            }
            break;
        case PREC:
            prop = va_arg(var, int);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].prec = prop;
                }
            }
            break;
        case FORMAT:
            prop = va_arg(var, int);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].format = prop;
                }
            }
            break;
        case LINEWIDTH:
            prop = va_arg(var, int);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].linew = checkon(LINEWIDTH, g[i].p[j].linew, prop);
                    if (check_err)
                    {
                        return;
                    }
                }
            }
            break;
        case LINESTYLE:
            prop = va_arg(var, int);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].lines = checkon(LINESTYLE, g[i].p[j].lines, prop);
                    if (check_err)
                    {
                        return;
                    }
                }
            }
            break;
        case COLOR:
            prop = va_arg(var, int);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].color = prop;
                }
            }
            break;
        case XYZ:
            dprop1 = va_arg(var, double);
            dprop2 = va_arg(var, double);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].zmin = dprop1;
                    g[i].p[j].zmax = dprop2;
                }
            }
            break;
        case COMMENT:
            cprop = va_arg(var, char *);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    strcpy(g[i].p[j].comments, cprop);
                }
            }
            break;
        case FILL:
            switch (prop = va_arg(var, int))
            {
            case TYPE:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].fill = prop;
                    }
                }
                break;
            case WITH:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].fillusing = prop;
                    }
                }
                break;
            case COLOR:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].fillcolor = prop;
                    }
                }
                break;
            case PATTERN:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].fillpattern = prop;
                    }
                }
                break;
            default:
                sprintf(buf, "Attribute not found in setprops()-FILL, # = %d", prop);
                errwin(buf);
                break;
            }
            break;
        case SKIP:
            prop = va_arg(var, int);
            for (i = startg; i <= endg; i++)
            {
                if (allsets)
                {
                    ends = g[i].maxplot - 1;
                }
                for (j = starts; j <= ends; j++)
                {
                    g[i].p[j].symskip = prop;
                }
            }
            break;
        case SYMBOL:
            switch (prop = va_arg(var, int))
            {
            case TYPE:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].sym = prop;
                    }
                }
                break;
            case FILL:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].symfill = prop;
                    }
                }
                break;
            case CENTER:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].symdot = prop;
                    }
                }
                break;
            case SIZE:
                dprop = va_arg(var, double);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].symsize = dprop;
                    }
                }
                break;
            case SKIP:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].symskip = prop;
                    }
                }
                break;
            case CHAR:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].symchar = prop;
                    }
                }
                break;
            case COLOR:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].symcolor = prop;
                    }
                }
                break;
            case LINEWIDTH:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].symlinew = prop;
                    }
                }
                break;
            case LINESTYLE:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].symlines = prop;
                    }
                }
                break;
            default:
                sprintf(buf, "Attribute not found in setprops()-SYMBOL, # = %d", prop);
                errwin(buf);
                break;
            }
            break;
        case ERRORBAR:
            switch (prop = va_arg(var, int))
            {
            case LENGTH:
                dprop = va_arg(var, double);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].errbarper = dprop;
                    }
                }
                break;
            case TYPE:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].errbarxy = prop;
                    }
                }
                break;
            case LINEWIDTH:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].errbar_linew = prop;
                    }
                }
                break;
            case LINESTYLE:
                prop = va_arg(var, int);
                for (i = startg; i <= endg; i++)
                {
                    if (allsets)
                    {
                        ends = g[i].maxplot - 1;
                    }
                    for (j = starts; j <= ends; j++)
                    {
                        g[i].p[j].errbar_lines = prop;
                    }
                }
                break;
            case RISER:
                prop = va_arg(var, int);
                switch (prop)
                {
                case ACTIVE:
                    prop = va_arg(var, int);
                    for (i = startg; i <= endg; i++)
                    {
                        if (allsets)
                        {
                            ends = g[i].maxplot - 1;
                        }
                        for (j = starts; j <= ends; j++)
                        {
                            g[i].p[j].errbar_riser = prop;
                        }
                    }
                    break;
                case LINEWIDTH:
                    prop = va_arg(var, int);
                    for (i = startg; i <= endg; i++)
                    {
                        if (allsets)
                        {
                            ends = g[i].maxplot - 1;
                        }
                        for (j = starts; j <= ends; j++)
                        {
                            g[i].p[j].errbar_riser_linew = prop;
                        }
                    }
                    break;
                case LINESTYLE:
                    prop = va_arg(var, int);
                    for (i = startg; i <= endg; i++)
                    {
                        if (allsets)
                        {
                            ends = g[i].maxplot - 1;
                        }
                        for (j = starts; j <= ends; j++)
                        {
                            g[i].p[j].errbar_riser_lines = prop;
                        }
                    }
                    break;
                default:
                    sprintf(buf, "Attribute not found in setprops()-RISER, # = %d", prop);
                    errwin(buf);
                    break;
                }
                break;
            default:
                sprintf(buf, "Attribute not found in setprops()-ERRORBAR, # = %d", prop);
                errwin(buf);
                break;
            }
            break;
        default:
            sprintf(buf, "Attribute not found in setprops()-top, # = %d", prop);
            errwin(buf);
            break;
        }
    }
    va_end(var);
}
