/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: graphutils.c,v 1.3 1994/11/04 02:52:22 pturner Exp pturner $
 *
 * utilities for graphs
 *
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
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
extern void do_clear_lines(void);
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

void flipxy(int gno);
void invertx(int gno);
void inverty(int gno);
void set_graph_active(int gno);
void update_all(int gno);

int iscontained(int gno, double wx, double wy)
{
    int i;
    double xconv(double x), yconv(double y);
    double x = xconv(wx), y = yconv(wy);

    for (i = 0; i < maxgraph; i++)
    {
        if (g[i].active == ON)
        {
            if ((g[i].v.xv1 <= x && g[i].v.xv2 >= x) && (g[i].v.yv1 <= y && g[i].v.yv2 >= y))
            {
                return i;
            }
        }
    }
    return gno;
}

/*
 * Count the number of active sets in graph gno
 */
int nactive(int gno)
{
    int i, cnt = 0;
    for (i = 0; i < g[gno].maxplot; i++)
    {
        if (isactive(gno, i))
        {
            cnt++;
        }
    }
    return cnt;
}

int isxreversed(int gno)
{
    return (g[gno].v.xv1 > g[gno].v.xv2);
}

int isyreversed(int gno)
{
    return (g[gno].v.yv1 > g[gno].v.yv2);
}

int islogx(int gno)
{
    return (g[gno].type == LOGX || g[gno].type == LOGXY);
}

int islogy(int gno)
{
    return (g[gno].type == LOGY || g[gno].type == LOGXY);
}

char *graph_types(int it, int which)
{
    static char s[128];

    strcpy(s, "UNKNOWN");
    switch (it)
    {
    case XY:
        if (which)
        {
            strcpy(s, "xy");
        }
        else
        {
            strcpy(s, "XY");
        }
        break;
    case LOGX:
        if (which)
        {
            strcpy(s, "logx");
        }
        else
        {
            strcpy(s, "LOG-LINEAR");
        }
        break;
    case LOGY:
        if (which)
        {
            strcpy(s, "logy");
        }
        else
        {
            strcpy(s, "LINEAR-LOG");
        }
        break;
    case LOGXY:
        if (which)
        {
            strcpy(s, "logxy");
        }
        else
        {
            strcpy(s, "LOG-LOG");
        }
        break;
    case XYFIXED:
        strcpy(s, "FIXED XY");
        break;
    case POLAR:
        strcpy(s, "polar");
        break;
    case BAR:
        if (which)
        {
            strcpy(s, "bar");
        }
        else
        {
            strcpy(s, "BAR");
        }
        break;
    case HBAR:
        if (which)
        {
            strcpy(s, "hbar");
        }
        else
        {
            strcpy(s, "HORIZONTAL BAR");
        }
        break;
    case PIE:
        strcpy(s, "pie");
        break;
    case STACKEDBAR:
        if (which)
        {
            strcpy(s, "stackedbar");
        }
        else
        {
            strcpy(s, "STACKED BAR");
        }
        break;
    case STACKEDHBAR:
        if (which)
        {
            strcpy(s, "stackedhbar");
        }
        else
        {
            strcpy(s, "STACKED HORIZONTAL BAR");
        }
        break;
    case STACKEDLINE:
        strcpy(s, "STACKED LINE");
        break;
    }
    return s;
}

int getFormat_index(int f)
{
    int i = 0;

    while (f != format_types[i] && format_types[i] != 0)
        i++;
    return i;
}

char *getFormat_types(int f)
{
    static char s[128];

    strcpy(s, "decimal");
    switch (f)
    {
    case DECIMAL:
        strcpy(s, "decimal");
        break;
    case EXPONENTIAL:
        strcpy(s, "exponential");
        break;
    case POWER:
        strcpy(s, "power");
        break;
    case GENERAL:
        strcpy(s, "general");
        break;
    case DDMMYY:
        strcpy(s, "ddmmyy");
        break;
    case MMDDYY:
        strcpy(s, "mmddyy");
        break;
    case MMYY:
        strcpy(s, "mmyy");
        break;
    case MMDD:
        strcpy(s, "mmdd");
        break;
    case MONTHDAY:
        strcpy(s, "monthday");
        break;
    case DAYMONTH:
        strcpy(s, "daymonth");
        break;
    case MONTHS:
        strcpy(s, "months");
        break;
    case MONTHL:
        strcpy(s, "monthl");
        break;
    case DAYOFWEEKS:
        strcpy(s, "dayofweeks");
        break;
    case DAYOFWEEKL:
        strcpy(s, "dayofweekl");
        break;
    case DAYOFYEAR:
        strcpy(s, "dayofyear");
        break;
    case HMS:
        strcpy(s, "hms");
        break;
    case MMDDHMS:
        strcpy(s, "mmddhms");
        break;
    case MMDDYYHMS:
        strcpy(s, "mmddyyhms");
        break;
    case DEGREESLON:
        strcpy(s, "degreeslon");
        break;
    case DEGREESMMLON:
        strcpy(s, "degreesmmlon");
        break;
    case DEGREESMMSSLON:
        strcpy(s, "degreesmmsslon");
        break;
    case MMSSLON:
        strcpy(s, "mmsslon");
        break;
    case DEGREESLAT:
        strcpy(s, "degreeslat");
        break;
    case DEGREESMMLAT:
        strcpy(s, "degreesmmlat");
        break;
    case DEGREESMMSSLAT:
        strcpy(s, "degreesmmsslat");
        break;
    case MMSSLAT:
        strcpy(s, "mmsslat");
        break;
    }
    return s;
}

void kill_graph(int gno)
{
    int i, j;

    if (gno == maxgraph)
    {
        for (i = 0; i < maxgraph; i++)
        {
            for (j = 0; j < g[i].maxplot; j++)
            {
                killset(i, j);
            }
            set_default_graph(i);
        }
    }
    else
    {
        for (i = 0; i < g[gno].maxplot; i++)
        {
            killset(gno, i);
        }
        set_default_graph(gno);
    }
}

void copy_graph(int from, int to)
{
    int i, j;
    plotarr *p;
    kill_graph(to);
    p = g[to].p;
    memcpy(&g[to], &g[from], sizeof(graph));
    g[to].p = p;
    set_graph_active(to); /* TODO compare maxplots */
    for (i = 0; i < g[from].maxplot; i++)
    {
        for (j = 0; j < MAX_SET_COLS; j++)
        {
            g[to].p[i].ex[j] = NULL;
        }
        g[to].p[i].active = OFF;
        if (isactive(from, i))
        {
            do_copyset(from, i, to, i);
        }
    }
}

void copy_graph_sets_only(int from, int to)
{
    int i, j;
    kill_graph(to);
    set_graph_active(to); /* TODO compare maxplots */
    for (i = 0; i < g[from].maxplot; i++)
    {
        for (j = 0; j < MAX_SET_COLS; j++)
        {
            g[to].p[i].ex[j] = NULL;
        }
        g[to].p[i].active = OFF;
        if (isactive(from, i))
        {
            do_copyset(from, i, to, i);
        }
    }
}

void swap_graph(int from, int to)
{
    graph gtmp;
    memcpy(&gtmp, &g[from], sizeof(graph));
    memcpy(&g[from], &g[to], sizeof(graph));
    memcpy(&g[to], &gtmp, sizeof(graph));
}

/*
 * for flipping
 */
void do_flipxy(void)
{
    flipxy(cg);
    drawgraph();
}

void flipxy(int gno)
{
    int i, j;
    tickmarks t;
    double *x, *y;

    for (i = 0; i < MAXAXES; i += 2)
    {
        memcpy(&t, &g[gno].t[i], sizeof(tickmarks));
        memcpy(&g[gno].t[i], &g[gno].t[i + 1], sizeof(tickmarks));
        memcpy(&g[gno].t[i + 1], &t, sizeof(tickmarks));
        if (g[gno].t[i].t_op == RIGHT)
        {
            g[gno].t[i].t_op = TOP;
        }
        else if (g[gno].t[i].t_op == LEFT)
        {
            g[gno].t[i].t_op = BOTTOM;
        }
        if (g[gno].t[i].tl_op == RIGHT)
        {
            g[gno].t[i].tl_op = TOP;
        }
        else if (g[gno].t[i].tl_op == LEFT)
        {
            g[gno].t[i].tl_op = BOTTOM;
        }
        if (g[gno].t[i + 1].t_op == TOP)
        {
            g[gno].t[i + 1].t_op = RIGHT;
        }
        else if (g[gno].t[i + 1].t_op == BOTTOM)
        {
            g[gno].t[i + 1].t_op = LEFT;
        }
        if (g[gno].t[i + 1].tl_op == TOP)
        {
            g[gno].t[i + 1].tl_op = RIGHT;
        }
        else if (g[gno].t[i + 1].tl_op == BOTTOM)
        {
            g[gno].t[i + 1].tl_op = LEFT;
        }
    }
    if (g[gno].type == LOGX)
    {
        g[gno].type = LOGY;
    }
    else if (g[gno].type == LOGY)
    {
        g[gno].type = LOGX;
    }
    fswap(&g[gno].w.xg1, &g[gno].w.yg1);
    fswap(&g[gno].w.xg2, &g[gno].w.yg2);
    fswap(&g[gno].dsx, &g[gno].dsy);
    iswap(&g[gno].fx, &g[gno].fy);
    iswap(&g[gno].px, &g[gno].py);
    for (i = 0; i < g[gno].maxplot; i++)
    {
        if (isactive(gno, i))
        {
            x = getx(gno, i); /* TODO really need to just swap pointers */
            y = gety(gno, i);
            for (j = 0; j < getsetlength(gno, i); j++)
            {
                fswap(&x[j], &y[j]);
            }
            updatesetminmax(gno, i);
        }
    }
    update_all(gno);
}

void do_invertx(void)
{
    invertx(cg);
    drawgraph();
}

void do_inverty(void)
{
    inverty(cg);
    drawgraph();
}

void invertx(int gno)
{
    fswap(&g[gno].v.xv1, &g[gno].v.xv2);
}

void inverty(int gno)
{
    fswap(&g[gno].v.yv1, &g[gno].v.yv2);
}

void get_graph_box(int i, boxtype *b)
{
    memcpy(b, &boxes[i], sizeof(boxtype));
}

void get_graph_line(int i, linetype *l)
{
    memcpy(l, &lines[i], sizeof(linetype));
}

void get_graph_string(int i, plotstr *s)
{
    memcpy(s, &pstr[i], sizeof(plotstr));
}

void get_graph_framep(int gno, framep *f)
{
    memcpy(f, &g[gno].f, sizeof(framep));
}

void get_graph_world(int gno, world *w)
{
    memcpy(w, &g[gno].w, sizeof(world));
}

void get_graph_view(int gno, view *v)
{
    memcpy(v, &g[gno].v, sizeof(view));
}

void get_graph_labels(int gno, labels *labs)
{
    memcpy(labs, &g[gno].labs, sizeof(labels));
}

void get_graph_plotarr(int gno, int i, plotarr *p)
{
    memcpy(p, &g[gno].p[i], sizeof(plotarr));
}

void get_graph_tickmarks(int gno, tickmarks *t, int a)
{
    memcpy(t, &g[gno].t[a], sizeof(tickmarks));
}

void get_graph_legend(int gno, legend *leg)
{
    memcpy(leg, &g[gno].l, sizeof(legend));
}

/*
 *
 * set graph props
 *
 */

void set_graph_box(int i, boxtype *b)
{
    memcpy(&boxes[i], b, sizeof(boxtype));
}

void set_graph_line(int i, linetype *l)
{
    memcpy(&lines[i], l, sizeof(linetype));
}

void set_graph_string(int i, plotstr *s)
{
    memcpy(&pstr[i], s, sizeof(plotstr));
}

void set_graph_active(int gno)
{
    g[gno].active = ON;
}

void set_graph_framep(int gno, framep *f)
{
    memcpy(&g[gno].f, f, sizeof(framep));
}

void set_graph_world(int gno, world *w)
{
    memcpy(&g[gno].w, w, sizeof(world));
}

void set_graph_view(int gno, view *v)
{
    memcpy(&g[gno].v, v, sizeof(view));
}

void set_graph_viewport(int gno, double vx1, double vy1, double vx2, double vy2)
{
    if (isxreversed(gno))
    {
        g[gno].v.xv2 = vx1;
        g[gno].v.xv1 = vx2;
    }
    else
    {
        g[gno].v.xv1 = vx1;
        g[gno].v.xv2 = vx2;
    }
    if (isyreversed(gno))
    {
        g[gno].v.yv2 = vy1;
        g[gno].v.yv1 = vy2;
    }
    else
    {
        g[gno].v.yv1 = vy1;
        g[gno].v.yv2 = vy2;
    }
}

void set_graph_labels(int gno, labels *labs)
{
    memcpy(&g[gno].labs, labs, sizeof(labels));
}

void set_graph_plotarr(int gno, int i, plotarr *p)
{
    memcpy(&g[gno].p[i], p, sizeof(plotarr));
}

void set_graph_tickmarks(int gno, tickmarks *t, int a)
{
    memcpy(&g[gno].t[a], t, sizeof(tickmarks));
}

void set_graph_legend(int gno, legend *leg)
{
    memcpy(&g[gno].l, leg, sizeof(legend));
}

void set_axis_prop(int whichgraph, int naxis, int prop, double val)
{
    int i, j, startg, stopg;

    if (whichgraph == -1)
    {
        startg = 0;
        stopg = maxgraph - 1;
    }
    else
    {
        startg = whichgraph;
        stopg = whichgraph;
    }
    for (j = startg; j <= stopg; j++)
    {
        switch (prop)
        {
        case ON:
        case OFF:
            switch (naxis)
            {
            case 6:
                for (i = 0; i < MAXAXES; i++)
                {
                    g[j].t[i].active = (int)val;
                }
                break;
            case 7:
                for (i = 0; i < MAXAXES; i += 2)
                {
                    g[j].t[i].active = (int)val;
                }
                break;
            case 8:
                for (i = 1; i < MAXAXES; i += 2)
                {
                    g[j].t[i].active = (int)val;
                }
                break;
            }
            break;
        case COLOR:
            switch (naxis)
            {
            case 6:
                for (i = 0; i < MAXAXES; i++)
                {
                    g[j].t[i].tl_color = (int)val;
                    g[j].t[i].t_drawbarcolor = (int)val;
                    g[j].t[i].t_color = (int)val;
                    g[j].t[i].t_mcolor = (int)val;
                    g[j].t[i].label.color = (int)val;
                }
                break;
            case 7:
                for (i = 0; i < MAXAXES; i += 2)
                {
                    g[j].t[i].tl_color = (int)val;
                    g[j].t[i].t_drawbarcolor = (int)val;
                    g[j].t[i].t_color = (int)val;
                    g[j].t[i].t_mcolor = (int)val;
                    g[j].t[i].label.color = (int)val;
                }
                break;
            case 8:
                for (i = 1; i < MAXAXES; i += 2)
                {
                    g[j].t[i].tl_color = (int)val;
                    g[j].t[i].t_drawbarcolor = (int)val;
                    g[j].t[i].t_color = (int)val;
                    g[j].t[i].t_mcolor = (int)val;
                    g[j].t[i].label.color = (int)val;
                }
                break;
            }
            break;
        case LINEWIDTH:
            switch (naxis)
            {
            case 6:
                for (i = 0; i < MAXAXES; i++)
                {
                    g[j].t[i].tl_linew = (int)val;
                    g[j].t[i].t_linew = (int)val;
                    g[j].t[i].t_mlinew = (int)val;
                    g[j].t[i].t_drawbarlinew = (int)val;
                }
                break;
            case 7:
                for (i = 0; i < MAXAXES; i += 2)
                {
                    g[j].t[i].tl_linew = (int)val;
                    g[j].t[i].t_linew = (int)val;
                    g[j].t[i].t_mlinew = (int)val;
                    g[j].t[i].t_drawbarlinew = (int)val;
                }
                break;
            case 8:
                for (i = 1; i < MAXAXES; i += 2)
                {
                    g[j].t[i].tl_linew = (int)val;
                    g[j].t[i].t_linew = (int)val;
                    g[j].t[i].t_mlinew = (int)val;
                    g[j].t[i].t_drawbarlinew = (int)val;
                }
                break;
            }
            break;
        case FONTP:
            switch (naxis)
            {
            case 6:
                for (i = 0; i < MAXAXES; i++)
                {
                    g[j].t[i].tl_font = (int)val;
                    g[j].t[i].label.font = (int)val;
                }
                break;
            case 7:
                for (i = 0; i < MAXAXES; i += 2)
                {
                    g[j].t[i].tl_font = (int)val;
                    g[j].t[i].label.font = (int)val;
                }
                break;
            case 8:
                for (i = 1; i < MAXAXES; i += 2)
                {
                    g[j].t[i].tl_font = (int)val;
                    g[j].t[i].label.font = (int)val;
                }
                break;
            }
            break;
        case CHAR:
            switch (naxis)
            {
            case 6:
                for (i = 0; i < MAXAXES; i++)
                {
                    g[j].t[i].tl_charsize = val;
                    g[j].t[i].label.charsize = val;
                }
                break;
            case 7:
                for (i = 0; i < MAXAXES; i += 2)
                {
                    g[j].t[i].tl_charsize = val;
                    g[j].t[i].label.charsize = val;
                }
                break;
            case 8:
                for (i = 1; i < MAXAXES; i += 2)
                {
                    g[j].t[i].tl_charsize = val;
                    g[j].t[i].label.charsize = val;
                }
                break;
            }
            break;
        }
    }
}

/*
    the following routines determine default scaling and tickmarks
*/

void defaultgraph(int gno)
{
    double x1, x2, y1, y2;
    double xgmax, xgmin, ygmax, ygmin;
    int i, first = 1;

    xgmax = xgmin = ygmax = ygmin = 0.0;
    for (i = 0; i < g[gno].maxplot; i++)
    {
        if (isactive_set(gno, i))
        {
            getsetminmax(gno, i, &x1, &x2, &y1, &y2);
            if (g[gno].type == STACKEDBAR)
            {
                if (first)
                {
                    xgmin = x1;
                    xgmax = x2;
                    ygmin = y1;
                    ygmax = y2;
                    first = 0;
                }
                else
                {
                    xgmin = (x1 < xgmin) ? x1 : xgmin;
                    xgmax = (x2 > xgmax) ? x2 : xgmax;
                    ygmin = (y1 < ygmin) ? y1 : ygmin;
                    ygmax += y2;
                }
            }
            else if (g[gno].type == STACKEDHBAR)
            {
                if (first)
                {
                    xgmin = x1;
                    xgmax = x2;
                    ygmin = y1;
                    ygmax = y2;
                    first = 0;
                }
                else
                {
                    ygmin = (y1 < ygmin) ? y1 : ygmin;
                    ygmax = (y2 > ygmax) ? y2 : ygmax;
                    xgmin = (x1 < xgmin) ? x1 : xgmin;
                    xgmax += x2;
                }
            }
            else
            {
                if (first)
                {
                    xgmin = x1;
                    xgmax = x2;
                    ygmin = y1;
                    ygmax = y2;
                    first = 0;
                }
                else
                {
                    xgmin = (x1 < xgmin) ? x1 : xgmin;
                    xgmax = (x2 > xgmax) ? x2 : xgmax;
                    ygmin = (y1 < ygmin) ? y1 : ygmin;
                    ygmax = (y2 > ygmax) ? y2 : ygmax;
                }
            }
        }
    }
    if (xgmin != xgmax)
    {
        g[gno].w.xg2 = xgmax;
        g[gno].w.xg1 = xgmin;
    }
    else
    {
        g[gno].w.xg1 = xgmin - 1.0;
        g[gno].w.xg2 = xgmin + 1.0;
    }
    if (ygmin != ygmax)
    {
        g[gno].w.yg2 = ygmax;
        g[gno].w.yg1 = ygmin;
    }
    else
    {
        g[gno].w.yg1 = ygmin - 1.0;
        g[gno].w.yg2 = ygmin + 1.0;
    }
    switch (g[gno].type)
    {
    case BAR:
        g[gno].w.xg1 -= (g[gno].w.xg2 - g[gno].w.xg1) * 0.05;
        g[gno].w.xg2 += (g[gno].w.xg2 - g[gno].w.xg1) * 0.05;
        if (g[gno].w.yg1 > 0.0)
        {
            g[gno].w.yg1 = 0.0;
        }
        break;
    case HBAR:
        g[gno].w.yg1 -= (g[gno].w.yg2 - g[gno].w.yg1) * 0.05;
        g[gno].w.yg2 += (g[gno].w.yg2 - g[gno].w.yg1) * 0.05;
        if (g[gno].w.xg1 > 0.0)
        {
            g[gno].w.xg1 = 0.0;
        }
        break;
    case STACKEDBAR:
        g[gno].w.xg1 -= (g[gno].w.xg2 - g[gno].w.xg1) * 0.05;
        g[gno].w.xg2 += (g[gno].w.xg2 - g[gno].w.xg1) * 0.05;
        if (g[gno].w.yg1 > 0.0)
        {
            g[gno].w.yg1 = 0.0;
        }
        break;
    case STACKEDHBAR:
        g[gno].w.yg1 -= (g[gno].w.yg2 - g[gno].w.yg1) * 0.05;
        g[gno].w.yg2 += (g[gno].w.yg2 - g[gno].w.yg1) * 0.05;
        if (g[gno].w.xg1 > 0.0)
        {
            g[gno].w.xg1 = 0.0;
        }
        break;
    case LOGX:
        if (g[gno].w.xg1 <= 0.0)
        {
            errwin("can't set graph type to log-linear, X minimum = 0.0");
            g[gno].type = XY;
        }
        break;
    case LOGY:
        if (g[gno].w.yg1 <= 0.0)
        {
            errwin("can't set graph type to linear-log, Y minimum = 0.0");
            g[gno].type = XY;
        }
        break;
    case LOGXY:
        if (g[gno].w.xg1 <= 0.0)
        {
            errwin("can't set graph to log-log, X minimum <= 0.0");
            g[gno].type = XY;
        }
        else if (g[gno].w.yg1 <= 0.0)
        {
            errwin("can't set graph type to log-log, Y minimum <= 0.0");
            g[gno].type = XY;
        }
        break;
    }
}

void defaultx(int gno, int setno)
{
    int i, first = 1;
    double xgmin, xgmax, xmax, xmin, tmp;

    xgmin = xgmax = 0.0;
    if (setno < 0)
    {
        for (i = 0; i < g[gno].maxplot; i++)
        {
            if (isactive_set(gno, i))
            {
                getsetminmax(gno, i, &xmin, &xmax, &tmp, &tmp);
                if (first)
                {
                    xgmin = xmin;
                    xgmax = xmax;
                    first = 0;
                }
                else
                {
                    xgmin = (xmin < xgmin) ? xmin : xgmin;
                    xgmax = (xmax > xgmax) ? xmax : xgmax;
                }
            }
        }
    }
    else
    {
        if (isactive_set(gno, setno))
        {
            getsetminmax(gno, setno, &xgmin, &xgmax, &tmp, &tmp);
        }
        else
        {
            return;
        }
    }
    if (xgmin != xgmax)
    {
        g[gno].w.xg2 = xgmax;
        g[gno].w.xg1 = xgmin;
    }
    else
    {
        if ((xgmin == 0.0) && (xgmax == 0.0))
        {
            xgmin = 1.0;
        }
        g[gno].w.xg1 = xgmin - 0.1 * fabs(xgmin);
        g[gno].w.xg2 = xgmin + 0.1 * fabs(xgmin);
    }
    switch (g[gno].type)
    {
    case BAR:
    case STACKEDBAR:
        g[gno].w.xg1 -= 0.5;
        g[gno].w.xg2 += 0.5;
        if (g[gno].w.yg1 > 0.0)
        {
            g[gno].w.yg1 = 0.0;
        }
        break;
    case HBAR:
    case STACKEDHBAR:
        if (g[gno].w.xg1 > 0.0)
        {
            g[gno].w.xg1 = 0.0;
        }
        break;
    case LOGX:
        if (g[gno].w.xg1 <= 0.0)
        {
            errwin("can't set graph type to log-linear, X minimum = 0.0");
            g[gno].type = XY;
        }
        break;
    case LOGY:
        if (g[gno].w.yg1 <= 0.0)
        {
            errwin("can't set graph type to linear-log, Y minimum = 0.0");
            g[gno].type = XY;
        }
        break;
    case LOGXY:
        if (g[gno].w.xg1 <= 0.0)
        {
            errwin("can't set graph to log-log, X minimum <= 0.0");
            g[gno].type = XY;
        }
        else if (g[gno].w.yg1 <= 0.0)
        {
            errwin("can't set graph type to log-log, Y minimum <= 0.0");
            g[gno].type = XY;
        }
        break;
    }
}

void defaulty(int gno, int setno)
{
    int i, first = 1;
    double ygmax, ygmin, ymin, ymax, tmp;

    ygmin = ygmax = 0.0;
    if (setno < 0)
    {
        for (i = 0; i < g[gno].maxplot; i++)
        {
            if (isactive_set(gno, i))
            {
                if (g[gno].type == STACKEDBAR)
                {
                }
                else
                {
                    getsetminmax(gno, i, &tmp, &tmp, &ymin, &ymax);
                    if (first)
                    {
                        ygmin = ymin;
                        ygmax = ymax;
                        first = 0;
                    }
                    else
                    {
                        ygmin = (ymin < ygmin) ? ymin : ygmin;
                        ygmax = (ymax > ygmax) ? ymax : ygmax;
                    }
                }
            }
        }
    }
    else
    {
        if (isactive_set(gno, setno))
        {
            getsetminmax(gno, setno, &tmp, &tmp, &ygmin, &ygmax);
        }
        else
        {
            return;
        }
    }
    if (ygmin != ygmax)
    {
        g[gno].w.yg2 = ygmax;
        g[gno].w.yg1 = ygmin;
    }
    else
    {
        if ((ygmin == 0.0) && (ygmax == 0.0))
        {
            ygmin = 1.0;
        }
        g[gno].w.yg1 = ygmin - 0.1 * fabs(ygmin);
        g[gno].w.yg2 = ygmin + 0.1 * fabs(ygmin);
    }
    switch (g[gno].type)
    {
    case BAR:
    case STACKEDBAR:
        if (g[gno].w.yg1 > 0.0)
        {
            g[gno].w.yg1 = 0.0;
        }
        break;
    case HBAR:
    case STACKEDHBAR:
        g[gno].w.yg1 -= 0.5;
        g[gno].w.yg2 += 0.5;
        if (g[gno].w.xg1 > 0.0)
        {
            g[gno].w.xg1 = 0.0;
        }
        break;
    case LOGX:
        if (g[gno].w.xg1 <= 0.0)
        {
            errwin("can't set graph type to log-linear, X minimum = 0.0");
            g[gno].type = XY;
        }
        break;
    case LOGY:
        if (g[gno].w.yg1 <= 0.0)
        {
            errwin("can't set graph type to linear-log, Y minimum = 0.0");
            g[gno].type = XY;
        }
        break;
    case LOGXY:
        if (g[gno].w.xg1 <= 0.0)
        {
            errwin("can't set graph to log-log, X minimum <= 0.0");
            g[gno].type = XY;
        }
        else if (g[gno].w.yg1 <= 0.0)
        {
            errwin("can't set graph type to log-log, Y minimum <= 0.0");
            g[gno].type = XY;
        }
        break;
    }
    if (g[gno].type == BAR || g[gno].type == STACKEDBAR)
    {
        if (g[gno].w.yg1 > 0.0)
        {
            g[gno].w.yg1 = 0.0;
        }
    }
}

/*
 * label: test program to demonstrate nice graph axis labeling
 *
 * Paul Heckbert, 2 Dec 88
 */

static double nicenum(double x, int round);

void default_ticks(int gno, int axis, double *gmin, double *gmax)
{
    tickmarks t;
    double range, d, tmpmax = *gmax, tmpmin = *gmin;

    get_graph_tickmarks(gno, &t, axis);
    if (axis % 2 && (g[gno].type == LOGY || g[gno].type == LOGXY))
    {
        tmpmax = ceil(log10(tmpmax));
        tmpmin = floor(log10(tmpmin));
    }
    else if ((axis % 2 == 0) && (g[gno].type == LOGX || g[gno].type == LOGXY))
    {
        tmpmax = ceil(log10(tmpmax));
        tmpmin = floor(log10(tmpmin));
    }
    range = nicenum(tmpmax - tmpmin, 0);
    d = nicenum(range / (t.t_num - 1), 1);
    tmpmin = floor(tmpmin / d) * d;
    tmpmax = ceil(tmpmax / d) * d;
    if (axis % 2 && (g[gno].type == LOGY || g[gno].type == LOGXY))
    {
        *gmax = pow(10.0, tmpmax);
        *gmin = pow(10.0, tmpmin);
        t.tmajor = (int)d;
        if (t.tmajor == 0.0)
        {
            t.tmajor = 1.0;
        }
        if ((int)t.tmajor < 2)
        {
            t.tminor = 1.0;
        }
        else
        {
            t.tminor = 0.0;
        }
        if (fabs(tmpmax) > 6.0 || fabs(tmpmin) > 6.0)
        {
            t.tl_format = POWER;
            t.tl_prec = 0;
        }
        else
        {
            t.tl_format = DECIMAL;
            t.tl_prec = 0;
        }
    }
    else if ((axis % 2 == 0) && (g[gno].type == LOGX || g[gno].type == LOGXY))
    {
        *gmax = pow(10.0, tmpmax);
        *gmin = pow(10.0, tmpmin);
        t.tmajor = (int)d;
        if (t.tmajor == 0.0)
        {
            t.tmajor = 1.0;
        }
        if (fabs(tmpmax) > 6.0 || fabs(tmpmin) > 6.0)
        {
            t.tl_format = POWER;
            t.tl_prec = 0;
        }
        else
        {
            t.tl_format = DECIMAL;
            t.tl_prec = 0;
        }
        if ((int)t.tmajor < 2)
        {
            t.tminor = 1.0;
        }
        else
        {
            t.tminor = 0.0;
        }
    }
    else
    {
        *gmax = tmpmax;
        *gmin = tmpmin;
        t.tmajor = d;
        t.tminor = d * 0.5;
    }
    set_graph_tickmarks(gno, &t, axis);
}

/*
 * nicenum: find a "nice" number approximately equal to x
 * round if round=1, ceil if round=0
 */

static double nicenum(double x, int round)
{
    int exp;
    double f, y;

    exp = (int)floor(log10(x));
    f = x / pow(10., (double)exp); /* fraction between 1 and 10 */
    if (round)
        if (f < 1.5)
            y = 1.;
        else if (f < 3.)
            y = 2.;
        else if (f < 7.)
            y = 5.;
        else
            y = 10.;
    else if (f <= 1.)
        y = 1.;
    else if (f <= 2.)
        y = 2.;
    else if (f <= 5.)
        y = 5.;
    else
        y = 10.;
    return y * pow(10., (double)exp);
}

void defaultsetgraph(int gno, int setno)
{
    double xmax, xmin, ymax, ymin;

    getsetminmax(gno, setno, &xmin, &xmax, &ymin, &ymax);
    if (xmin != xmax)
    {
        g[gno].w.xg2 = xmax;
        g[gno].w.xg1 = xmin;
    }
    else
    {
        if ((xmin == 0.0) && (xmax == 0.0))
        {
            xmin = 1.0;
        }
        g[gno].w.xg1 = xmin - 0.1 * fabs(xmin);
        g[gno].w.xg2 = xmin + 0.1 * fabs(xmin);
    }
    if (ymin != ymax)
    {
        g[gno].w.yg2 = ymax;
        g[gno].w.yg1 = ymin;
    }
    else
    {
        if ((ymin == 0.0) && (ymax == 0.0))
        {
            ymin = 1.0;
        }
        g[gno].w.yg1 = ymin - 0.1 * fabs(ymin);
        g[gno].w.yg2 = ymin + 0.1 * fabs(ymin);
    }
}

void default_axis(int gno, int method, int axis)
{
    int cx;
    tickmarks t;
    world w;
    double llim, ulim;

    get_graph_tickmarks(gno, &t, axis);
    get_graph_world(gno, &w);
    if (axis % 2)
    {
        llim = w.yg1;
        ulim = w.yg2;
    }
    else
    {
        llim = w.xg1;
        ulim = w.xg2;
    }
    t.tmajor = (ulim - llim) / t.t_num;
    t.tminor = t.tmajor / 2.0;
    cx = (int)(log10(t.tmajor));
    t.tl_prec = ((cx < 0) ? -cx + 1 : 1);
    if (t.tl_prec > 9)
    {
        t.tl_prec = 2;
        t.tl_format = EXPONENTIAL;
    }
    set_graph_tickmarks(gno, &t, axis);
    if (method == AUTO)
    {
        default_ticks(gno, axis, &llim, &ulim);
        if (axis % 2)
        {
            w.yg1 = llim;
            w.yg2 = ulim;
        }
        else
        {
            w.xg1 = llim;
            w.xg2 = ulim;
        }
        set_graph_world(gno, &w);
    }
    if (g[gno].type == XYFIXED)
    {
        setfixedscale(g[gno].v.xv1, g[gno].v.yv1, g[gno].v.xv2, g[gno].v.yv2,
                      &g[gno].w.xg1, &g[gno].w.yg1, &g[gno].w.xg2, &g[gno].w.yg2);
    }
}

void autoscale_graph(int gno, int axis)
{
    if (activeset(gno))
    {
        switch (axis)
        {
        case -3:
            defaultgraph(gno);
            default_axis(gno, g[gno].auto_type, X_AXIS);
            default_axis(gno, g[gno].auto_type, ZX_AXIS);
            default_axis(gno, g[gno].auto_type, Y_AXIS);
            default_axis(gno, g[gno].auto_type, ZY_AXIS);
            break;
        case -2:
            defaultx(gno, -1);
            default_axis(gno, g[gno].auto_type, X_AXIS);
            default_axis(gno, g[gno].auto_type, ZX_AXIS);
            break;
        case -1:
            defaulty(gno, -1);
            default_axis(gno, g[gno].auto_type, Y_AXIS);
            default_axis(gno, g[gno].auto_type, ZY_AXIS);
            break;
        default:
            if (axis % 2)
            {
                defaulty(gno, -1);
            }
            else
            {
                defaultx(gno, -1);
            }
            default_axis(gno, g[gno].auto_type, axis);
            break;
        }
        update_all(gno);
    }
}

void do_autoscale_set(int gno, int setno)
{
    if (isactive_set(gno, setno))
    {
        defaultsetgraph(gno, setno);
        default_axis(gno, g[gno].auto_type, X_AXIS);
        default_axis(gno, g[gno].auto_type, ZX_AXIS);
        default_axis(gno, g[gno].auto_type, Y_AXIS);
        default_axis(gno, g[gno].auto_type, ZY_AXIS);
        update_all(gno);
        drawgraph();
    }
}

void autoscale_set(int gno, int setno, int axis)
{
    if (isactive(gno, setno))
    {
        switch (axis)
        {
        case -3:
            defaultsetgraph(gno, setno);
            default_axis(gno, g[gno].auto_type, X_AXIS);
            default_axis(gno, g[gno].auto_type, ZX_AXIS);
            default_axis(gno, g[gno].auto_type, Y_AXIS);
            default_axis(gno, g[gno].auto_type, ZY_AXIS);
            break;
        case -2:
            defaultx(gno, setno);
            default_axis(gno, g[gno].auto_type, X_AXIS);
            default_axis(gno, g[gno].auto_type, ZX_AXIS);
            break;
        case -1:
            defaulty(gno, setno);
            default_axis(gno, g[gno].auto_type, Y_AXIS);
            default_axis(gno, g[gno].auto_type, ZY_AXIS);
            break;
        default:
            if (axis % 2)
            {
                defaulty(gno, setno);
            }
            else
            {
                defaultx(gno, setno);
            }
            default_axis(gno, g[gno].auto_type, axis);
            break;
        }
        update_all(gno);
    }
    else
    {
        errwin("Set not active!");
    }
}

/*
 * for zooms, lz => linked to all active graphs
 *
 */
void newworld(int gno, int lz, int axes, double wx1, double wy1, double wx2, double wy2)
{
    int i, ming, maxg;
    if (lz)
    {
        ming = 0;
        maxg = maxgraph;
    }
    else
    {
        ming = gno;
        maxg = gno;
    }
    for (i = ming; i <= maxg; i++)
    {
        if (isactive_graph(i))
        {
            switch (axes)
            {
            case -1:
                g[i].w.xg1 = wx1;
                g[i].w.xg2 = wx2;
                g[i].w.yg1 = wy1;
                g[i].w.yg2 = wy2;
                default_axis(i, g[i].auto_type, X_AXIS);
                default_axis(i, g[i].auto_type, ZX_AXIS);
                default_axis(i, g[i].auto_type, Y_AXIS);
                default_axis(i, g[i].auto_type, ZY_AXIS);
                break;
            case 0:
                g[i].w.xg1 = wx1;
                g[i].w.xg2 = wx2;
                default_axis(i, g[i].auto_type, X_AXIS);
                default_axis(i, g[i].auto_type, ZX_AXIS);
                break;
            case 1:
                g[i].w.yg1 = wy1;
                g[i].w.yg2 = wy2;
                default_axis(i, g[i].auto_type, Y_AXIS);
                default_axis(i, g[i].auto_type, ZY_AXIS);
                break;
            }
        }
    }
}

void wipeout(int ask)
{
    if (ask && !yesno("Kill all graphs, sets, and annotation?", NULL, NULL, NULL))
    {
        return;
    }
    kill_graph(maxgraph);
    do_clear_lines();
    do_clear_boxes();
    do_clear_text();
    cg = 0;
    drawgraph();
}

void update_all(int gno)
{
    update_set_lists(gno);
    update_world(gno);
    update_view(gno);
    update_status(gno, cur_statusitem, -1);
    update_ticks(gno);
    update_autos(gno);
    updatelegends(gno);
    updatesymbols(gno, -1);
    update_label_proc();
    update_locator_items(gno);
    update_draw();
    update_frame_items(gno);
    update_graph_items();
    update_hotlinks();
}
