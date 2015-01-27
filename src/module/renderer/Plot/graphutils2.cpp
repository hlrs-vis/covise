/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: graphutils2.c,v 1.2 1994/09/20 02:29:29 pturner Exp pturner $
 *
 * utilities for graphs - second batch, scrolling, world stack and arrange
 *
 */

#include <stdio.h>
#include <math.h>
#include "extern.h"
#include "globals.h"

extern void initialize_screen(int *argc, char **argv);
extern void set_program_defaults(void);
extern void initialize_cms_data(void);
extern int getparms(int gno, char *plfile);
// extern void set_printer_proc(Widget w, XtPointer client_data, XtPointer call_data);
extern void set_graph_active(int gno);
extern int argmatch(char *s1, char *s2, int atleast);
extern void realloc_plots(int maxplot);
extern int alloc_blockdata(int ncols);
extern int alloc_blockdata(int ncols);
extern void realloc_graph_plots(int gno, int maxplot);
extern void realloc_graphs(void);
extern void realloc_lines(int n);
extern void realloc_boxes(int n);
extern void realloc_strings(int n);
extern void read_param(char *pbuf);
extern int getdata(int gno, char *fn, int src, int type);
extern int getdata_step(int gno, char *fn, int src, int type);
extern void create_set_fromblock(int gno, int type, char *cols);
extern int fexists(char *to);
extern void do_writesets(int gno, int setno, int imbed, char *fn, char *format);
extern void do_writesets_binary(int gno, int setno, char *fn);
extern int isdir(char *f);
extern int activeset(int gno);
extern void defaulty(int gno, int setno);
extern void default_axis(int gno, int method, int axis);
extern void defaultx(int gno, int setno);
extern void set_printer(int device, char *prstr);
extern void defaultgraph(int gno);
extern void set_plotstr_string(plotstr *pstr, char *buf);
extern void arrange_graphs(int grows, int gcols);
extern void hselectfont(int f);
extern void defineworld(double x1, double y1, double x2, double y2, int mapx, int mapy);
extern int islogx(int gno);
extern int islogy(int gno);
extern void viewport(double x1, double y1, double x2, double y2);
extern void runbatch(char *bfile);
extern void do_hardcopy(void);
extern void do_main_loop(void);
extern int yesno(const char *msg1, const char *s1, const char *s2, const char *helptext);
extern void errwin(const char *s);
extern void killset(int gno, int setno);
extern void set_default_graph(int gno);
extern void do_copyset(int gfrom, int j1, int gto, int j2);
extern void drawgraph(void);
extern void fswap(double *x, double *y);
extern void iswap(int *x, int *y);
extern void updatesetminmax(int gno, int setno);
extern void getsetminmax(int gno, int setno, double *x1, double *x2, double *y1, double *y2);
extern void setfixedscale(double xv1, double yv1, double xv2, double yv2,
                          double *xg1, double *yg1, double *xg2, double *yg2);
extern void do_clear_boxes(void);
extern void do_clear_text(void);
extern void update_set_lists(int gno);
extern void update_world(int gno);
extern void update_view(int gno);
extern void update_status(int gno, int itemtype, int itemno);
extern void update_ticks(int gno);
extern void update_autos(int gno);
extern void updatelegends(int gno);
extern void updatesymbols(int gno, int value);
extern void update_label_proc(void);
extern void update_locator_items(int gno);
extern void update_draw(void);
extern void update_frame_items(int gno);
extern void update_graph_items(void);
extern void update_hotlinks(void);
extern void update_all(int gno);
extern void set_action(int act);
extern void set_stack_message(void);
extern int getFormat_index(int f);
extern void autoscale_set(int gno, int setno, int axis);
extern void autoscale_graph(int gno, int axis);

void set_graph_active(int gno);
void push_world(void);

void arrange_graphs(int grows, int gcols)
{
    int i, j;
    double dvx = 0.9 / gcols;
    double dvy = 0.9 / grows;
    int gtmp = 0;

    if (gcols != 1 || grows != 1)
    {
        for (i = 0; i < gcols; i++)
        {
            for (j = 0; j < grows; j++)
            {
                g[gtmp].v.xv1 = 0.1 + i * dvx;
                g[gtmp].v.xv2 = 0.1 + ((i + 1) * dvx);
                g[gtmp].v.xv2 -= 0.07;
                g[gtmp].v.yv1 = 0.1 + j * dvy;
                g[gtmp].v.yv2 = 0.1 + ((j + 1) * dvy);
                g[gtmp].v.yv2 -= 0.07;
                set_graph_active(gtmp);
                update_all(cg);
                gtmp++;
            }
        }
    }
}

/*
 * pan through world coordinates
 */

void gwindleft_proc(void)
{
    double dx;
    int i;
    if (ismaster)
        cm->sendCommandMessage(GWINDLEFT_PROC, 0, 0);

    if (scrolling_islinked)
    {
        for (i = 0; i < maxgraph; i++)
        {
            if (isactive_graph(i) && !islogx(i))
            {
                dx = scrollper * (g[i].w.xg2 - g[i].w.xg1);
                g[i].w.xg1 = g[i].w.xg1 - dx;
                g[i].w.xg2 = g[i].w.xg2 - dx;
            }
        }
    }
    else
    {
        if (!islogx(cg))
        {
            dx = scrollper * (g[cg].w.xg2 - g[cg].w.xg1);
            g[cg].w.xg1 = g[cg].w.xg1 - dx;
            g[cg].w.xg2 = g[cg].w.xg2 - dx;
        }
    }
    drawgraph();
}

void gwindright_proc(void)
{
    double dx;
    int i;
    if (ismaster)
        cm->sendCommandMessage(GWINDRIGHT_PROC, 0, 0);

    if (scrolling_islinked)
    {
        for (i = 0; i < maxgraph; i++)
        {
            if (isactive_graph(i) && !islogx(i))
            {
                dx = scrollper * (g[i].w.xg2 - g[i].w.xg1);
                g[i].w.xg1 = g[i].w.xg1 + dx;
                g[i].w.xg2 = g[i].w.xg2 + dx;
            }
        }
    }
    else
    {
        if (!islogx(cg))
        {
            dx = scrollper * (g[cg].w.xg2 - g[cg].w.xg1);
            g[cg].w.xg1 = g[cg].w.xg1 + dx;
            g[cg].w.xg2 = g[cg].w.xg2 + dx;
        }
    }
    drawgraph();
}

void gwinddown_proc(void)
{
    double dy;
    int i;
    if (ismaster)
        cm->sendCommandMessage(GWINDDOWN_PROC, 0, 0);

    if (scrolling_islinked)
    {
        for (i = 0; i < maxgraph; i++)
        {
            if (isactive_graph(i) && !islogy(i))
            {
                dy = scrollper * (g[i].w.yg2 - g[i].w.yg1);
                g[i].w.yg1 = g[i].w.yg1 - dy;
                g[i].w.yg2 = g[i].w.yg2 - dy;
            }
        }
    }
    else
    {
        if (!islogy(cg))
        {
            dy = scrollper * (g[cg].w.yg2 - g[cg].w.yg1);
            g[cg].w.yg1 = g[cg].w.yg1 - dy;
            g[cg].w.yg2 = g[cg].w.yg2 - dy;
        }
    }
    drawgraph();
}

void gwindup_proc(void)
{
    double dy;
    int i;
    if (ismaster)
        cm->sendCommandMessage(GWINDUP_PROC, 0, 0);

    if (scrolling_islinked)
    {
        for (i = 0; i < maxgraph; i++)
        {
            if (isactive_graph(i) && !islogy(i))
            {
                dy = scrollper * (g[i].w.yg2 - g[i].w.yg1);
                g[i].w.yg1 = g[i].w.yg1 + dy;
                g[i].w.yg2 = g[i].w.yg2 + dy;
            }
        }
    }
    else
    {
        if (!islogy(cg))
        {
            dy = scrollper * (g[cg].w.yg2 - g[cg].w.yg1);
            g[cg].w.yg1 = g[cg].w.yg1 + dy;
            g[cg].w.yg2 = g[cg].w.yg2 + dy;
        }
    }
    drawgraph();
}

void gwindshrink_proc(void)
{
    if (ismaster)
        cm->sendCommandMessage(GWINDSHRINK_PROC, 0, 0);
    double dx = shexper * (g[cg].w.xg2 - g[cg].w.xg1);
    double dy = shexper * (g[cg].w.yg2 - g[cg].w.yg1);

    if (!islogx(cg) && !islogy(cg))
    {
        g[cg].w.xg1 = g[cg].w.xg1 + dx;
        g[cg].w.xg2 = g[cg].w.xg2 - dx;
        g[cg].w.yg1 = g[cg].w.yg1 + dy;
        g[cg].w.yg2 = g[cg].w.yg2 - dy;
        drawgraph();
    }
    else
    {
        errwin("Shrink not implemented for log plots");
    }
}

void gwindexpand_proc(void)
{
    if (ismaster)
        cm->sendCommandMessage(GWINDEXPAND_PROC, 0, 0);
    double dx = shexper * (g[cg].w.xg2 - g[cg].w.xg1);
    double dy = shexper * (g[cg].w.yg2 - g[cg].w.yg1);

    if (!islogx(cg) && !islogy(cg))
    {
        g[cg].w.xg1 = g[cg].w.xg1 - dx;
        g[cg].w.xg2 = g[cg].w.xg2 + dx;
        g[cg].w.yg1 = g[cg].w.yg1 - dy;
        g[cg].w.yg2 = g[cg].w.yg2 + dy;
        drawgraph();
    }
    else
    {
        errwin("Expand not implemented for log plots");
    }
}

/*
 * set scroll amount
 */
void scroll_proc(int value)
{
    if (ismaster)
        cm->sendCommandMessage(SCROLL_PROC, value, 0);
    scrollper = value / 100.0;
    update_draw();
}

void scrollinout_proc(int value)
{
    if (ismaster)
        cm->sendCommandMessage(SCROLLINOUT_PROC, value, 0);
    shexper = value / 100.0;
    update_draw();
}

/*
 * world stack operations
 */
void push_and_zoom(void)
{
    if (ismaster)
        cm->sendCommandMessage(PUSH_AND_ZOOM, 0, 0);
    push_world();
    set_action(0);
    set_action(ZOOM_1ST);
}

void cycle_world_stack(void)
{
    int i;
    if (ismaster)
        cm->sendCommandMessage(CYCLE_WORLD_STACK, 0, 0);

    if (g[cg].ws_top == 0)
    {
        errwin("World stack empty");
    }
    else
    {
        if (g[cg].curw >= g[cg].ws_top)
        {
            g[cg].curw = 0;
        }
        g[cg].w.xg1 = g[cg].ws[g[cg].curw].w.xg1;
        g[cg].w.xg2 = g[cg].ws[g[cg].curw].w.xg2;
        g[cg].w.yg1 = g[cg].ws[g[cg].curw].w.yg1;
        g[cg].w.yg2 = g[cg].ws[g[cg].curw].w.yg2;
        for (i = 0; i < MAXAXES; i += 2)
        {
            g[cg].t[i].tmajor = g[cg].ws[g[cg].curw].t[i].xg1;
            g[cg].t[i].tminor = g[cg].ws[g[cg].curw].t[i].xg2;
            g[cg].t[i + 1].tmajor = g[cg].ws[g[cg].curw].t[i].yg1;
            g[cg].t[i + 1].tminor = g[cg].ws[g[cg].curw].t[i].yg2;
        }
        g[cg].curw++;
        drawgraph();
    }
}

void clear_world_stack(void)
{
    g[cg].ws_top = 0;
    g[cg].curw = 0;
    set_stack_message();
}

void show_world_stack(int n)
{
    int i;

    if (g[cg].ws_top == 0)
    {
        errwin("World stack empty");
    }
    else
    {
        if (n >= g[cg].ws_top)
        {
            errwin("Selected view greater than stack depth");
        }
        else if (n < 0)
        {
            errwin("Selected view less than zero");
        }
        else
        {
            g[cg].w.xg1 = g[cg].ws[n].w.xg1;
            g[cg].w.xg2 = g[cg].ws[n].w.xg2;
            g[cg].w.yg1 = g[cg].ws[n].w.yg1;
            g[cg].w.yg2 = g[cg].ws[n].w.yg2;
            for (i = 0; i < MAXAXES; i += 2)
            {
                g[cg].t[i].tmajor = g[cg].ws[n].t[i].xg1;
                g[cg].t[i].tminor = g[cg].ws[n].t[i].xg2;
                g[cg].t[i + 1].tmajor = g[cg].ws[n].t[i].yg1;
                g[cg].t[i + 1].tminor = g[cg].ws[n].t[i].yg2;
            }
            drawgraph();
        }
    }
}

/*
 * world stacks
 */

void add_world(int gno, double x1, double x2, double y1, double y2,
               double t1, double t2, double u1, double u2)
{
    int i;

    if (g[gno].ws_top < MAX_ZOOM_STACK)
    {
        g[gno].ws[g[gno].ws_top].w.xg1 = x1;
        g[gno].ws[g[gno].ws_top].w.xg2 = x2;
        g[gno].ws[g[gno].ws_top].w.yg1 = y1;
        g[gno].ws[g[gno].ws_top].w.yg2 = y2;
        for (i = 0; i < 3; i++)
        {
            g[gno].ws[g[gno].ws_top].t[i].xg1 = t1;
            g[gno].ws[g[gno].ws_top].t[i].xg2 = t2;
            g[gno].ws[g[gno].ws_top].t[i].yg1 = u1;
            g[gno].ws[g[gno].ws_top].t[i].yg2 = u2;
        }
        g[gno].ws_top++;
        set_stack_message();
    }
    else
    {
        errwin("World stack full");
    }
}

void push_world(void)
{
    int i;
    if (ismaster)
        cm->sendCommandMessage(PUSH_WORLD, 0, 0);

    if (g[cg].ws_top < MAX_ZOOM_STACK)
    {
        g[cg].ws[g[cg].ws_top].w.xg1 = g[cg].w.xg1;
        g[cg].ws[g[cg].ws_top].w.xg2 = g[cg].w.xg2;
        g[cg].ws[g[cg].ws_top].w.yg1 = g[cg].w.yg1;
        g[cg].ws[g[cg].ws_top].w.yg2 = g[cg].w.yg2;
        for (i = 0; i < MAXAXES; i += 2)
        {
            g[cg].ws[g[cg].ws_top].t[i / 2].xg1 = g[cg].t[i].tmajor;
            g[cg].ws[g[cg].ws_top].t[i / 2].xg2 = g[cg].t[i].tminor;
            g[cg].ws[g[cg].ws_top].t[i / 2].yg1 = g[cg].t[i + 1].tmajor;
            g[cg].ws[g[cg].ws_top].t[i / 2].yg2 = g[cg].t[i + 1].tminor;
        }
        g[cg].ws_top++;
        set_stack_message();
    }
    else
    {
        errwin("World stack full");
    }
}

void pop_world(void)
{
    int i;
    if (ismaster)
        cm->sendCommandMessage(POP_WORLD, 0, 0);

    if (g[cg].ws_top == 0)
    {
        errwin("World stack empty");
    }
    else
    {
        g[cg].ws_top--;
        g[cg].w.xg1 = g[cg].ws[g[cg].ws_top].w.xg1;
        g[cg].w.xg2 = g[cg].ws[g[cg].ws_top].w.xg2;
        g[cg].w.yg1 = g[cg].ws[g[cg].ws_top].w.yg1;
        g[cg].w.yg2 = g[cg].ws[g[cg].ws_top].w.yg2;
        for (i = 0; i < MAXAXES; i += 2)
        {
            g[cg].t[i].tmajor = g[cg].ws[g[cg].ws_top].t[i].xg1;
            g[cg].t[i].tminor = g[cg].ws[g[cg].ws_top].t[i].xg2;
            g[cg].t[i + 1].tmajor = g[cg].ws[g[cg].ws_top].t[i].yg1;
            g[cg].t[i + 1].tminor = g[cg].ws[g[cg].ws_top].t[i].yg2;
        }
        drawgraph();
    }
}

/*
 * set format string for locator
 */
static const char *fchar[3] = { "lf", "le", "g" };
static const char *typestr[6] = { "X, Y", "DX, DY", "DIST", "R, Theta", "VX, VY", "SX, SY" };
char locator_format[256] = { "G%1d: X, Y = [%.6g, %.6g]" };

void make_format(int gno)
{
    int type, locpx, locfx, locpy, locfy;

    type = g[gno].pt_type;
    locfx = getFormat_index(g[gno].fx);
    locfy = getFormat_index(g[gno].fy);
    locpx = g[gno].px;
    locpy = g[gno].py;
    switch (type)
    {
    case 0:
        if (locfx < 3 && locfy < 3)
        {
            sprintf(locator_format, "G%%1d: %s = [%%.%d%s, %%.%d%s]", typestr[type], locpx, fchar[locfx], locpy, fchar[locfy]);
        }
        else
        {
            locator_format[0] = 0;
        }
        break;
    case 2:
        locfx = locfx == DECIMAL ? 0 : locfx == EXPONENTIAL ? 1 : locfx == GENERAL ? 2 : 0;
        sprintf(locator_format, "G%%1d: %s = [%%.%d%s]", typestr[type], locpx, fchar[locfx]);
        break;
    case 1:
    case 3:
    case 4:
        locfx = locfx == DECIMAL ? 0 : locfx == EXPONENTIAL ? 1 : locfx == GENERAL ? 2 : 0;
        locfy = locfy == DECIMAL ? 0 : locfy == EXPONENTIAL ? 1 : locfy == GENERAL ? 2 : 0;
        sprintf(locator_format, "G%%1d: %s = [%%.%d%s, %%.%d%s]", typestr[type], locpx, fchar[locfx], locpy, fchar[locfy]);
        break;
    case 5:
        sprintf(locator_format, "G%%1d: %s = [%%d, %%d]", typestr[type]);
        break;
    }
}

void arrange_graphs2(int grows, int gcols, double vgap, double hgap,
                     double sx, double sy, double wx, double wy, int)
{
    int i, j;
    int gtmp = 0;

    if (gcols != 1 || grows != 1)
    {
        for (i = 0; i < gcols; i++)
        {
            for (j = 0; j < grows; j++)
            {
                g[gtmp].v.xv1 = sx + i * hgap + i * wx;
                g[gtmp].v.xv2 = sx + i * hgap + (i + 1) * wx;
                g[gtmp].v.yv1 = sy + j * vgap + j * wy;
                g[gtmp].v.yv2 = sy + j * vgap + (j + 1) * wy;
                if (!isactive_graph(gtmp))
                {
                    set_graph_active(gtmp);
                }
                gtmp++;
            }
        }
    }
}

void define_autos(int aon, int au, int ap, int ameth, int antx, int anty)
{
    int i, ming, maxg;
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(DEFINE_AUTOS, aon, au, ap, ameth, antx, anty, 0, 0, 0, 0);
    }

    if (au >= 0 && !isactive_set(cg, au))
    {
        errwin("Set not active");
        return;
    }
    if (ap)
    {
        ming = 0;
        maxg = maxgraph - 1;
    }
    else
    {
        ming = cg;
        maxg = cg;
    }
    if (ming == cg && maxg == cg)
    {
        if (!isactive_graph(cg))
        {
            errwin("Current graph is not active!");
            return;
        }
    }
    for (i = ming; i <= maxg; i++)
    {
        if (isactive_graph(i))
        {
            if (ameth == 0)
            {
                g[i].auto_type = AUTO;
            }
            else
            {
                g[i].auto_type = SPEC;
            }
            g[i].t[0].t_num = antx + 2;
            g[i].t[1].t_num = anty + 2;
            if (aon > -4)
            {
                /* -4 == just set parameters but don't
             * autoscale */
                if (au >= 0 && isactive_set(i, au))
                {
                    autoscale_set(i, au, aon);
                }
                else
                {
                    autoscale_graph(i, aon);
                }
            }
        }
    }
    if (aon != -4)
    {
        drawgraph();
    }
}

void define_arrange(int nrows, int ncols, int pack,
                    double vgap, double hgap, double sx, double sy, double wx, double wy)
{
    int i, j, k, gno, applyto = 0;
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(DEFINE_ARRANGE, nrows, ncols, pack, 0, 0, 0, 0, 0, 0, 0);
        cm->sendCommand_FloatMessage(DEFINE_ARRANGE, vgap, hgap, sx, sy, wx, wy, 0.0, 0.0, 0.0, 0.0);
    }

    if (nrows * ncols > maxgraph)
    {
        errwin("Too many rows and columns");
        return;
    }
    switch (pack)
    {
    case 0:
        for (j = 0; j < ncols; j++)
        {
            for (i = 0; i < nrows; i++)
            {
                gno = i + j * nrows;
                for (k = 0; k < 2; k++)
                {
                    g[gno].t[k].tl_flag = ON;
                }
            }
        }
        break;
    case 1:
        hgap = 0.0;
        for (j = 1; j < ncols; j++)
        {
            for (i = 0; i < nrows; i++)
            {
                gno = i + j * nrows;
                for (k = 1; k < 6; k += 2)
                {
                    g[gno].t[k].tl_flag = OFF;
                }
            }
        }
        break;
    case 2:
        vgap = 0.0;
        for (j = 0; j < ncols; j++)
        {
            for (i = 1; i < nrows; i++)
            {
                gno = i + j * nrows;
                for (k = 0; k < 6; k += 2)
                {
                    g[gno].t[k].tl_flag = OFF;
                }
            }
        }
        break;
    case 3:
        hgap = 0.0;
        vgap = 0.0;
        for (j = 1; j < ncols; j++)
        {
            for (i = 0; i < nrows; i++)
            {
                gno = i + j * nrows;
                for (k = 1; k < 6; k += 2)
                {
                    g[gno].t[k].tl_flag = OFF;
                }
            }
        }
        for (j = 0; j < ncols; j++)
        {
            for (i = 1; i < nrows; i++)
            {
                gno = i + j * nrows;
                for (k = 0; k < 6; k += 2)
                {
                    g[gno].t[k].tl_flag = OFF;
                }
            }
        }
        break;
    }
    arrange_graphs2(nrows, ncols, vgap, hgap, sx, sy, wx, wy, applyto);
    drawgraph();
}
