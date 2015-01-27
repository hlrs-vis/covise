/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: checkon.c,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 * routines for sanity checks
 *
 */

#include <stdio.h>
#include <math.h>
#include "globals.h"
#include "noxprotos.h"

extern void errwin(const char *s);
int check_err = 0;

int checkon_ticks(int /*gno*/)
{
    /* TODO this is done in drawticks.c and the following may be unnecessary
       int i, n1, n2;
       double dx = g[gno].w.xg2 - g[gno].w.xg1;
       double dy = g[gno].w.yg2 - g[gno].w.yg1;
       for (i = 0; i < MAXAXES; i++) {
      if (g[gno].t[i].active == ON) {
          n1 = n2 = 0;
          if (g[gno].t[i].t_flag == ON) {
         if (i % 2 == 0) {
             n1 = dx / g[gno].t[i].tmajor;
             n2 = dx / g[gno].t[i].tminor;
   if (n1 > 500) {
   errwin("Too many X-axis major ticks (> 500)");
   return 0;
   }
   if (n2 > 1000) {
   errwin("Too many X-axis minor ticks (> 1000)");
   return 0;
   }
   } else {
   n1 = dy / g[gno].t[i].tmajor;
   n2 = dy / g[gno].t[i].tminor;
   if (n1 > 500) {
   errwin("Too many Y-axis major ticks (> 500)");
   return 0;
   }
   if (n2 > 1000) {
   errwin("Too many Y-axis minor ticks (> 1000)");
   return 0;
   }
   }
   }
   }
   }
   */
    return 1;
}

int checkon_world(int gno)
{
    double dx = g[gno].w.xg2 - g[gno].w.xg1;
    double dy = g[gno].w.yg2 - g[gno].w.yg1;

    if (g[gno].type == LOGX || g[gno].type == LOGXY)
    {
        if (g[gno].w.xg1 <= 0)
        {
            errwin("World X-min <= 0.0");
            return 0;
        }
        if (g[gno].w.xg2 <= 0)
        {
            errwin("World X-max <= 0.0");
            return 0;
        }
    }
    if (dx <= 0.0)
    {
        errwin("World DX <= 0.0");
        return 0;
    }
    if (g[gno].type == LOGY || g[gno].type == LOGXY)
    {
        if (g[gno].w.yg1 <= 0.0)
        {
            errwin("World Y-min <= 0.0");
            return 0;
        }
        if (g[gno].w.yg2 <= 0.0)
        {
            errwin("World Y-max <= 0.0");
            return 0;
        }
    }
    if (dy <= 0.0)
    {
        errwin("World DY <= 0.0");
        return 0;
    }
    return 1;
}

/* possibly defunct as reversal of axes depends on xv1 > xv2 */
int checkon_viewport(int gno)
{
    double dx = g[gno].v.xv2 - g[gno].v.xv1;
    double dy = g[gno].v.yv2 - g[gno].v.yv1;

    if (dx <= 0.0)
    {
        errwin("Viewport DX <= 0.0");
        return 0;
    }
    if (dy <= 0.0)
    {
        errwin("Viewport DY <= 0.0");
        return 0;
    }
    return 1;
}

#define MAX_FONT 10
#define MAX_JUST 2
#define MAX_ARROW 3
#define MAX_PATTERN 30
#define MAX_PREC 10

int checkon(int prop, int old_val, int new_val)
{
    char buf[256];
    int retval = old_val;
    check_err = 0;
    switch (prop)
    {
    case LINEWIDTH:
        if (new_val >= 0 && new_val <= MAX_LINEWIDTH)
        {
            retval = new_val;
        }
        else
        {
            sprintf(buf, "LINEWIDTH out of bounds, should be from 0 to %d", MAX_LINEWIDTH);
            check_err = 1;
        }
        break;
    case LINESTYLE:
        if (new_val >= 0 && new_val <= MAX_LINESTYLE)
        {
            retval = new_val;
        }
        else
        {
            sprintf(buf, "LINESTYLE out of bounds, should be from 0 to %d", MAX_LINESTYLE);
            check_err = 1;
        }
        break;
    case COLOR: /* TODO use MAX_COLOR */
        if (new_val >= 0 && new_val < 16)
        {
            retval = new_val;
        }
        else
        {
            sprintf(buf, "COLOR out of bounds, should be from 0 to %d", 16 - 1);
            check_err = 1;
        }
        break;
    case JUST:
        if (new_val >= 0 && new_val <= MAX_JUST)
        {
            retval = new_val;
        }
        else
        {
            sprintf(buf, "JUST out of bounds, should be from 0 to %d", MAX_JUST);
            check_err = 1;
        }
        break;
    case FONTP:
        if (new_val >= 0 && new_val < MAX_FONT)
        {
            retval = new_val;
        }
        else
        {
            sprintf(buf, "FONT out of bounds, should be from 0 to %d", MAX_FONT - 1);
            check_err = 1;
        }
        break;
    case ARROW:
        if (new_val >= 0 && new_val <= MAX_ARROW)
        {
            retval = new_val;
        }
        else
        {
            sprintf(buf, "ARROW out of bounds, should be from 0 to %d", MAX_ARROW);
            check_err = 1;
        }
        break;
    case PATTERN:
        if (new_val >= 0 && new_val < MAX_PATTERN)
        {
            retval = new_val;
        }
        else
        {
            sprintf(buf, "PATTERN out of bounds, should be from 0 to %d", MAX_PATTERN - 1);
            check_err = 1;
        }
        break;
    case SYMBOL:
        if (new_val >= 0 && new_val < MAXSYM)
        {
            retval = new_val;
        }
        else
        {
            sprintf(buf, "SYMBOL out of bounds, should be from 0 to %d", MAXSYM - 1);
            check_err = 1;
        }
        break;
    case PREC:
        if (new_val >= 0 && new_val < MAX_PREC)
        {
            retval = new_val;
        }
        else
        {
            sprintf(buf, "PREC out of bounds, should be from 0 to %d", MAX_PREC - 1);
            check_err = 1;
        }
        break;
    }
    if (check_err)
    {
        errwin(buf);
    }
    return retval;
}
