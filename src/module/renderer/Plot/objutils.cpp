/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: objutils.c,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 * operations on objects (strings, lines, and boxes)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
extern int argmatch(const char *s1, const char *s2, int atleast);
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

/*
 * find the nearest object to (x,y)
 */
void find_item(int gno, double x, double y, int *type, int *numb)
{
    int i;
    double tmp, xtmp1, ytmp1, xtmp2, ytmp2, m = 1e307;
    // double dx, dy;
    boxtype box;
    linetype line;
    plotstr str;

    x = xconv(x);
    y = yconv(y);
    *type = -1;
    for (i = 0; i < MAXBOXES; i++)
    {
        get_graph_box(i, &box);
        if (isactive_box(i))
        {
            if (box.loctype == VIEW)
            {
                xtmp1 = box.x1;
                ytmp1 = box.y1;
                xtmp2 = box.x2;
                ytmp2 = box.y2;
            }
            else
            {
                if (gno == box.gno)
                {
                    xtmp1 = xconv(box.x1);
                    ytmp1 = yconv(box.y1);
                    xtmp2 = xconv(box.x2);
                    ytmp2 = yconv(box.y2);
                }
                else
                {
                    continue;
                }
            }
            tmp = my_hypot((x - xtmp1), (y - ytmp1));
            if (m > tmp)
            {
                *type = BOX;
                *numb = i;
                m = tmp;
            }
            tmp = my_hypot((x - xtmp1), (y - ytmp2));
            if (m > tmp)
            {
                *type = BOX;
                *numb = i;
                m = tmp;
            }
            tmp = my_hypot((x - xtmp2), (y - ytmp1));
            if (m > tmp)
            {
                *type = BOX;
                *numb = i;
                m = tmp;
            }
            tmp = my_hypot((x - xtmp2), (y - ytmp2));
            if (m > tmp)
            {
                *type = BOX;
                *numb = i;
                m = tmp;
            }
        }
    }
    for (i = 0; i < MAXLINES; i++)
    {
        get_graph_line(i, &line);
        if (isactive_line(i))
        {
            if (line.loctype == VIEW)
            {
                xtmp1 = line.x1;
                ytmp1 = line.y1;
                xtmp2 = line.x2;
                ytmp2 = line.y2;
            }
            else
            {
                if (gno == line.gno)
                {
                    xtmp1 = xconv(line.x1);
                    ytmp1 = yconv(line.y1);
                    xtmp2 = xconv(line.x2);
                    ytmp2 = yconv(line.y2);
                }
                else
                {
                    continue;
                }
            }
            tmp = my_hypot((x - xtmp1), (y - ytmp1));
            if (m > tmp)
            {
                *type = LINE;
                *numb = i;
                m = tmp;
            }
            tmp = my_hypot((x - xtmp2), (y - ytmp2));
            if (m > tmp)
            {
                *type = LINE;
                *numb = i;
                m = tmp;
            }
        }
    }
    for (i = 0; i < MAXSTR; i++)
    {
        get_graph_string(i, &str);
        if (isactive_string(i))
        {
            if (str.loctype == VIEW)
            {
                xtmp1 = str.x;
                ytmp1 = str.y;
            }
            else
            {
                if (gno == str.gno)
                {
                    xtmp1 = xconv(str.x);
                    ytmp1 = yconv(str.y);
                }
                else
                {
                    continue;
                }
            }
            tmp = my_hypot((x - xtmp1), (y - ytmp1));
            if (m > tmp)
            {
                *type = PLOT_STRING;
                *numb = i;
                m = tmp;
            }
        }
    }
}

int isactive_line(int lineno)
{
    if (0 <= lineno && lineno < MAXLINES)
        return (lines[lineno].active == ON);
    return (0);
}

int isactive_box(int boxno)
{
    if (0 <= boxno && boxno < MAXBOXES)
        return (boxes[boxno].active == ON);
    return (0);
}

int isactive_string(int strno)
{
    if (0 <= strno && strno < MAXSTR)
        return (pstr[strno].s[0]);
    return (0);
}

int next_line(void)
{
    int i;

    for (i = 0; i < MAXLINES; i++)
    {
        if (!isactive_line(i))
        {
            lines[i].active = ON;
            return (i);
        }
    }
    errwin("Error - no lines available");
    return (-1);
}

int next_box(void)
{
    int i;

    for (i = 0; i < MAXBOXES; i++)
    {
        if (!isactive_box(i))
        {
            boxes[i].active = ON;
            return (i);
        }
    }
    errwin("Error - no boxes available");
    return (-1);
}

int next_string(void)
{
    int i;

    for (i = 0; i < MAXSTR; i++)
    {
        if (!isactive_string(i))
        {
            return (i);
        }
    }
    errwin("Error - no strings available");
    return (-1);
}

void copy_object(int type, int from, int to)
{
    char *tmpbuf;
    switch (type)
    {
    case BOX:
        boxes[to] = boxes[from];
        break;
    case LINE:
        lines[to] = lines[from];
        break;
    case PLOT_STRING:
        kill_string(to);
        free(pstr[to].s);
        tmpbuf = (char *)malloc((strlen(pstr[from].s) + 1) * sizeof(char));
        pstr[to] = pstr[from];
        pstr[to].s = tmpbuf;
        strcpy(pstr[to].s, pstr[from].s);
        break;
    }
}

void kill_box(int boxno)
{
    boxes[boxno].active = OFF;
}

void kill_line(int lineno)
{
    lines[lineno].active = OFF;
}

void kill_string(int stringno)
{
    if (pstr[stringno].s != NULL)
    {
        free(pstr[stringno].s);
    }
    pstr[stringno].s = (char *)malloc(sizeof(char));
    pstr[stringno].s[0] = 0;
    pstr[stringno].active = OFF;
}

void set_plotstr_string(plotstr *pstr, char *buf)
{
    if (pstr->s != NULL)
    {
        free(pstr->s);
    }
    pstr->s = NULL;
    if (buf != NULL)
    {
        pstr->s = (char *)malloc(sizeof(char) * (strlen(buf) + 1));
        strcpy(pstr->s, buf);
    }
    else
    {
        pstr->s = (char *)malloc(sizeof(char));
        pstr->s[0] = 0;
    }
}

int define_string(char *s, double wx, double wy)
{
    int i;

    i = next_string();
    if (i >= 0)
    {
        if (s != NULL)
        {
            free(pstr[i].s);
        }
        if (s != NULL)
        {
            pstr[i].s = (char *)malloc(sizeof(char) * (strlen(s) + 1));
            strcpy(pstr[i].s, s);
        }
        else
        {
            pstr[i].s = (char *)malloc(sizeof(char));
            pstr[i].s[0] = 0;
        }
        pstr[i].font = string_font;
        pstr[i].color = string_color;
        pstr[i].linew = string_linew;
        pstr[i].rot = string_rot;
        pstr[i].charsize = string_size;
        pstr[i].loctype = string_loctype;
        pstr[i].just = string_just;
        pstr[i].active = ON;
        if (string_loctype == VIEW)
        {
            pstr[i].x = xconv(wx);
            pstr[i].y = yconv(wy);
            pstr[i].gno = -1;
        }
        else
        {
            pstr[i].x = wx;
            pstr[i].y = wy;
            pstr[i].gno = cg;
        }
        return i;
    }
    return -1;
}

void do_clear_lines(void)
{
    int i;

    for (i = 0; i < MAXLINES; i++)
    {
        kill_line(i);
    }
    if (inwin)
    {
        drawgraph();
    }
}

void do_clear_boxes(void)
{
    int i;

    for (i = 0; i < MAXBOXES; i++)
    {
        kill_box(i);
    }
    if (inwin)
    {
        drawgraph();
    }
}

void do_clear_text(void)
{
    int i;

    for (i = 0; i < MAXSTR; i++)
    {
        kill_string(i);
    }
    if (inwin)
    {
        drawgraph();
    }
}

void realloc_lines(int n)
{
    int i;
    if (n > maxlines)
    {
        lines = (linetype *)realloc(lines, n * sizeof(linetype));
        for (i = maxlines; i < n; i++)
        {
            set_default_line(&lines[i]);
        }
        maxlines = n;
    }
}

void realloc_boxes(int n)
{
    int i;
    if (n > maxboxes)
    {
        boxes = (boxtype *)realloc(boxes, n * sizeof(boxtype));
        for (i = maxboxes; i < n; i++)
        {
            set_default_box(&boxes[i]);
        }
        maxboxes = n;
    }
}

void realloc_strings(int n)
{
    int i;
    if (n > maxstr)
    {
        pstr = (plotstr *)realloc(pstr, n * sizeof(plotstr));
        for (i = maxstr; i < n; i++)
        {
            set_default_string(&pstr[i]);
        }
        maxstr = n;
    }
}
