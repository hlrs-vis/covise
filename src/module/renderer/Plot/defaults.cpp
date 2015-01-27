/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: defaults.c,v 1.5 1994/10/30 07:26:14 pturner Exp pturner $
 *
 * set defaults - changes to the types in defines.h
 * will require changes in here also
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "globals.h"

extern "C" {
extern void cfree(void *);
}

extern void do_select_region(void);
extern void add_point(int gno, int setno, double px, double py, double tx, double ty, int type);
extern int inregion(int regno, double x, double y);
extern void update_set_status(int gno, int setno);
extern int getncols(int gno, int setno);
extern void setlength(int gno, int i, int length);
extern void scanner(char *s, double *x, double *y, int len, double *a, double *b, double *c, double *d, int lenscr, int i, int setno, int *errpos);
extern int setcolor(int col);
extern int setlinestyle(int style);
extern int setlinewidth(int wid);
extern void draw_arrow(double x1, double y1, double x2, double y2, int end, double asize, int type);
extern void my_move2(double x, double y);
extern void my_draw2(double x, double y);
extern int world2view(double x, double y, double *vx, double *vy);
extern void view2world(double vx, double vy, double *x, double *y);
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
extern void update_all(int gno);
extern void set_action(int act);
extern void set_stack_message(void);
extern int getFormat_index(int f);
extern void autoscale_set(int gno, int setno, int axis);
extern void autoscale_graph(int gno, int axis);

#include "noxprotos.h"

static defaults d_d = { 1, 1, 1, 1.0, 4, 0, 1.0 };

/* defaults layout
    int color;
    int lines;
    int linew;
    double charsize;
    int font;
    int fontsrc;
    double symsize;
*/

static world d_w = { 0.0, 1.0, 0.0, 1.0 };

static view d_v = { 0.15, 0.85, 0.15, 0.85 };

void set_program_defaults(void)
{
    int i;
    grdefaults = d_d;
    g = (graph *)calloc(maxgraph, sizeof(graph));
    for (i = 0; i < maxgraph; i++)
    {
        g[i].p = (plotarr *)calloc(maxplot, sizeof(plotarr));
        set_default_graph(i);
    }
    for (i = 0; i < MAXREGION; i++)
    {
        set_region_defaults(i);
    }
    set_default_annotation();
    set_default_string(&timestamp);
    alloc_blockdata(MAXPLOT);
    timestamp.x = 0.03;
    timestamp.y = 0.03;
    if (init_scratch_arrays(maxarr))
    {
        errwin("Couldn't allocate memory for scratch arrays, don't use them");
    }
}

void set_region_defaults(int rno)
{
    int j;
    rg[rno].active = OFF;
    rg[rno].type = 0;
    rg[rno].color = grdefaults.color;
    rg[rno].lines = grdefaults.lines;
    rg[rno].linew = grdefaults.linew;
    for (j = 0; j < MAXGRAPH; j++)
    {
        rg[rno].linkto[j] = -1;
    }
    rg[rno].n = 0;
    rg[rno].x = rg[rno].y = (double *)NULL;
    rg[rno].x1 = rg[rno].y1 = rg[rno].x2 = rg[rno].y2 = 0.0;
}

void set_default_framep(framep *f)
{
    f->active = ON; /* frame on or off */
    f->type = 0; /* frame type */
    f->lines = grdefaults.lines;
    f->linew = grdefaults.linew;
    f->color = grdefaults.color;
    f->fillbg = OFF; /* fill background */
    f->bgcolor = 0; /* background color inside frame */
}

void set_default_world(world *w)
{
    memcpy(w, &d_w, sizeof(world));
}

void set_default_view(view *v)
{
    memcpy(v, &d_v, sizeof(view));
}

void set_default_string(plotstr *s)
{
    s->active = OFF;
    s->loctype = VIEW;
    s->gno = -1;
    s->x = s->y = 0.0;
    s->lines = grdefaults.lines;
    s->linew = grdefaults.linew;
    s->color = grdefaults.color;
    s->rot = 0;
    s->font = grdefaults.font;
    s->just = 0;
    s->charsize = grdefaults.charsize;
    s->s = (char *)malloc(sizeof(char));
    s->s[0] = 0;
}

void set_default_line(linetype *l)
{
    l->active = OFF;
    l->loctype = VIEW;
    l->gno = -1;
    l->x1 = l->y1 = l->x2 = l->y2 = 0.0;
    l->lines = grdefaults.lines;
    l->linew = grdefaults.linew;
    l->color = grdefaults.color;
    l->arrow = 0;
    l->atype = 0;
    l->asize = 1.0;
}

void set_default_box(boxtype *b)
{
    b->active = OFF;
    b->loctype = VIEW;
    b->gno = -1;
    b->x1 = b->y1 = b->x2 = b->y2 = 0.0;
    b->lines = grdefaults.lines;
    b->linew = grdefaults.linew;
    b->color = grdefaults.color;
    b->fill = OFF;
    b->fillcolor = 1;
    b->fillpattern = 1;
}

void set_default_legend(legend *l)
{
    int i;

    l->active = OFF;
    l->loctype = VIEW;
    l->layout = 0;
    l->vgap = 2;
    l->hgap = 1;
    l->len = 4;
    l->legx = 0.8;
    l->legy = 0.8;
    l->font = grdefaults.font;
    l->charsize = 1.0;
    l->color = grdefaults.color;
    l->linew = grdefaults.linew;
    l->lines = grdefaults.lines;
    l->box = OFF;
    l->boxfill = OFF;
    l->boxfillusing = COLOR;
    l->boxfillcolor = 0;
    l->boxfillpat = 1;
    l->boxlcolor = grdefaults.color; /* color for symbol */
    l->boxlinew = grdefaults.linew; /* set plot sym line width */
    l->boxlines = grdefaults.lines; /* set plot sym line style */
    for (i = 0; i < MAXPLOT; i++)
    {
        set_default_string(&(l->str[i]));
    }
}

void set_default_plotarr(plotarr *p)
{
    int i;

    if (p->len > 0)
    {
        for (i = 0; i < MAX_SET_COLS; i++)
        {
            if (p->ex[i] != NULL)
            {
                cfree(p->ex[i]);
            }
            p->ex[i] = NULL;
        }
    }
    p->active = OFF; /* active flag */
    p->type = XY; /* dataset type */
    p->deact = 0; /* deactivated set */
    p->hotlink = 0; /* hot linked set */
    p->hotfile[0] = 0; /* hot linked file name */
    p->len = 0; /* set length */
    p->missing = DATASET_MISSING; /* value for missing data */
    p->s = (char **)NULL; /* pointer to strings */
    p->xmin = p->xmax = p->ymin = p->ymax = 0.0;

    p->sym = 0; /* set plot symbol */
    p->symchar = 0; /* character for symbol */
    p->symskip = 0; /* How many symbols to skip */
    p->symfill = 0; /* Symbol fill type */
    p->symdot = 0; /* Symbol dot in center */
    p->symlines = grdefaults.lines; /* set plot sym line style */
    p->symlinew = grdefaults.linew; /* set plot sym line width */
    p->symcolor = grdefaults.color; /* color for symbol */
    p->symsize = 1.0; /* size of symbols */

    p->font = grdefaults.font; /* font for strings */
    p->format = DECIMAL; /* format for drawing values */
    p->prec = 1; /* precision for drawing values */
    p->just = LEFT; /* justification for drawing values */
    p->where = RIGHT; /* where to draw values */
    p->valsize = 1.0; /* char size for drawing values */

    p->lines = grdefaults.lines;
    p->linew = grdefaults.linew;
    p->color = grdefaults.color;
    p->lineskip = 0; /* How many points to skip when drawing lines */

    p->fill = 0; /* fill type */
    p->fillusing = COLOR; /* fill using color or pattern */
    p->fillcolor = 1; /* fill color */
    p->fillpattern = 0; /* fill pattern */

    p->errbar = -1;
    /* if type is _DX, _DY, _DXDY and errbar =
    * TRUE */
    p->errbarxy = BOTH; /* type of error bar */
    p->errbar_lines = grdefaults.lines; /* error bar line width */
    p->errbar_linew = grdefaults.linew; /* error bar line style */
    p->errbar_riser = ON; /* connecting line between error limits */
    p->errbar_riser_linew = 1;
    /* connecting line between error limits line
    * width */
    p->errbar_riser_lines = 1;
    /* connecting line between error limits line
    * style */

    p->errbarper = 1.0; /* length of error bar */
    p->hilowper = 1.0; /* length of hi-low */

    p->density_plot = 0; /* if type is XYZ then density_plot  = 1 */
    p->zmin = p->zmax = 0.0; /* min max for density plots */

    p->comments[0] = 0; /* how did this set originate */
    p->name[0] = 0; /* set name alias */

    for (i = 0; i < MAX_SET_COLS; i++)
    {
        p->emin[i] = 0.0; /* min for each column */
        p->emax[i] = 0.0; /* max for each column */
        p->imin[i] = 0; /* min loc for each column */
        p->imax[i] = 0; /* max loc for each column */
    }
    p->ep = NULL; /* EditPoints pointer */
}

void set_default_velocityp(velocityp *vp)
{
    vp->active = OFF;
    vp->type = 0;
    vp->loctype = VIEW;
    vp->velx = 0.8;
    vp->vely = 0.7;
    vp->lines = grdefaults.lines;
    vp->linew = grdefaults.linew;
    vp->color = grdefaults.color;
    set_default_string(&(vp->vstr));
    vp->arrowtype = 0;
    vp->vscale = 1.0;
    vp->units = 0;
    vp->userlength = 1.0;
}

void set_default_graph(int gno)
{
    int i;

    g[gno].active = OFF;
    g[gno].hidden = FALSE;
    g[gno].label = OFF;
    g[gno].type = XY;
    g[gno].auto_type = AUTO;
    g[gno].revx = FALSE;
    g[gno].revy = FALSE;
    g[gno].ws_top = 0;
    g[gno].maxplot = maxplot;
    g[gno].dsx = g[gno].dsy = 0.0; /* locator props */
    g[gno].pointset = FALSE;
    g[gno].pt_type = 0;
    g[gno].fx = GENERAL;
    g[gno].fy = GENERAL;
    g[gno].px = 6;
    g[gno].py = 6;
    set_default_ticks(&g[gno].t[0], X_AXIS);
    set_default_ticks(&g[gno].t[1], Y_AXIS);
    set_default_ticks(&g[gno].t[2], ZX_AXIS);
    set_default_ticks(&g[gno].t[3], ZY_AXIS);
    set_default_ticks(&g[gno].t[4], XA_AXIS);
    set_default_ticks(&g[gno].t[5], YA_AXIS);
    set_default_framep(&g[gno].f);
    set_default_world(&g[gno].w);
    set_default_view(&g[gno].v);
    set_default_legend(&g[gno].l);
    set_default_string(&g[gno].labs.title);
    g[gno].labs.title.charsize = 1.5;
    set_default_string(&g[gno].labs.stitle);
    g[gno].labs.stitle.charsize = 1.0;
    for (i = 0; i < maxplot; i++)
    {
        set_default_plotarr(&g[gno].p[i]);
    }
    set_default_velocityp(&g[gno].vp);
}

void realloc_plots(int maxplot)
{
    int i, j;
    for (i = 0; i < maxgraph; i++)
    {
        g[i].p = (plotarr *)realloc(g[i].p, maxplot * sizeof(plotarr));
        for (j = g[i].maxplot; j < maxplot; j++)
        {
            g[i].p[j].len = 0;
            set_default_plotarr(&g[i].p[j]);
        }
        g[i].maxplot = maxplot;
    }
}

void realloc_graph_plots(int gno, int maxplot)
{
    int j;
    g[gno].p = (plotarr *)realloc(g[gno].p, maxplot * sizeof(plotarr));
    for (j = g[gno].maxplot; j < maxplot; j++)
    {
        g[gno].p[j].len = 0;
        set_default_plotarr(&g[gno].p[j]);
    }
    g[gno].maxplot = maxplot;
}

void realloc_graphs(void)
{
    int j;

    g = (graph *)realloc(g, maxgraph * sizeof(graph));
    for (j = MAXGRAPH; j < maxgraph; j++)
    {
        g[j].p = (plotarr *)calloc(maxplot, sizeof(plotarr));
        set_default_graph(j);
    }
}

void set_default_annotation(void)
{
    int i;

    boxes = (boxtype *)malloc(maxboxes * sizeof(boxtype));
    lines = (linetype *)malloc(maxlines * sizeof(linetype));
    pstr = (plotstr *)malloc(maxstr * sizeof(plotstr));

    for (i = 0; i < maxboxes; i++)
    {
        set_default_box(&boxes[i]);
    }
    for (i = 0; i < maxlines; i++)
    {
        set_default_line(&lines[i]);
    }
    for (i = 0; i < maxstr; i++)
    {
        set_default_string(&pstr[i]);
    }
}

void set_default_ticks(tickmarks *t, int a)
{
    int i;

    t->axis = a;
    switch (a)
    {
    case X_AXIS:
    case Y_AXIS:
        t->active = ON;
        t->alt = OFF;
        t->tl_flag = ON;
        t->t_flag = ON;
        break;
    case XA_AXIS:
    case YA_AXIS:
        t->active = ON;
        t->alt = OFF;
        t->tl_flag = OFF;
        t->t_flag = OFF;
        break;
    case ZX_AXIS:
    case ZY_AXIS:
        t->active = ON;
        t->alt = OFF;
        t->tl_flag = OFF;
        t->t_flag = OFF;
        break;
    }
    set_default_string(&t->label);
    t->tmin = 0.0;
    t->tmax = 1.0;
    t->tmajor = 0.5;
    t->tminor = 0.25;
    t->offsx = 0.0;
    t->offsy = 0.0;
    t->label_layout = PARA;
    t->label_place = AUTO;
    t->tl_type = AUTO;
    t->tl_layout = HORIZONTAL;
    t->tl_sign = NORMAL;
    t->tl_prec = 1;
    t->tl_format = DECIMAL;
    t->tl_angle = 0;
    t->tl_just = (a % 2) ? RIGHT : CENTER;
    t->tl_skip = 0;
    t->tl_staggered = 0;
    t->tl_starttype = AUTO;
    t->tl_stoptype = AUTO;
    t->tl_start = 0.0;
    t->tl_stop = 0.0;
    t->tl_op = (a % 2) ? LEFT : BOTTOM;
    t->tl_vgap = 1.0;
    t->tl_hgap = 1.0;
    t->tl_font = grdefaults.font;
    t->tl_charsize = 1.0;
    t->tl_linew = grdefaults.linew;
    t->tl_color = grdefaults.color;
    t->tl_appstr[0] = 0;
    t->tl_prestr[0] = 0;
    t->t_type = AUTO;
    t->t_mflag = ON;
    t->t_integer = OFF;
    t->t_num = 6;
    t->t_inout = IN;
    t->t_log = OFF;
    t->t_op = BOTH;
    t->t_size = 1.0;
    t->t_msize = 0.5;
    t->t_drawbar = OFF;
    t->t_drawbarcolor = grdefaults.color;
    t->t_drawbarlines = grdefaults.lines;
    t->t_drawbarlinew = grdefaults.linew;
    t->t_gridflag = OFF;
    t->t_mgridflag = OFF;
    t->t_color = grdefaults.color;
    t->t_lines = grdefaults.lines;
    t->t_linew = grdefaults.linew;
    t->t_mcolor = grdefaults.color;
    t->t_mlines = grdefaults.lines;
    t->t_mlinew = grdefaults.linew;
    t->t_spec = 0;
    for (i = 0; i < MAX_TICK_LABELS; i++)
    {
        t->t_specloc[i] = 0.0;
        t->t_speclab[i].s = NULL;
    }
}
