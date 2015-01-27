/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: compute.c,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 * perform math between sets
 *
 */

#include <stdio.h>
#include <string.h>
#include "globals.h"
#include "noxprotos.h"

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

void loadset(int gno, int selset, int toval, double startno, double stepno)
{
    int i, lenset;
    double *ltmp;
    double *xtmp, *ytmp;

    if ((lenset = getsetlength(gno, selset)) <= 0)
    {
        char stmp[60];

        sprintf(stmp, "Length of set %d <= 0", selset);
        errwin(stmp);
        return;
    }
    xtmp = getx(gno, selset);
    ytmp = gety(gno, selset);
    switch (toval)
    {
    case 1:
        ltmp = xtmp;
        break;
    case 2:
        ltmp = ytmp;
        break;
    case 3:
        ltmp = ax;
        break;
    case 4:
        ltmp = bx;
        break;
    case 5:
        ltmp = cx;
        break;
    case 6:
        ltmp = dx;
        break;
    default:
        return;
    }
    for (i = 0; i < lenset; i++)
    {
        *ltmp++ = startno + i * stepno;
    }
    updatesetminmax(gno, selset);
    update_set_status(gno, selset);
}

/*
 * evaluate the expression in sscanstr and place the result in selset
 */
int formula(int gno, int selset, char *sscanstr)
{
    char stmp[64], tmpstr[512];
    int i = 0, errpos, lenset;
    double *xtmp, *ytmp;

    if ((lenset = getsetlength(gno, selset)) <= 0)
    {
        sprintf(stmp, "Length of set %d = 0", selset);
        errwin(stmp);
        return (0);
    }
    xtmp = getx(gno, selset);
    ytmp = gety(gno, selset);
    strcpy(tmpstr, sscanstr);
    fixupstr(tmpstr);
    scanner(tmpstr, xtmp, ytmp, lenset, ax, bx, cx, dx, MAXARR, i, selset, &errpos);
    updatesetminmax(gno, selset);
    update_set_status(gno, selset);
    return (errpos);
}
