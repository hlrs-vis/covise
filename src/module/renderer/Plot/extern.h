/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <net/covise_connect.h>
#include <net/message.h>
#include <net/message_types.h>
#include <covise/covise_process.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>

#include "defines.h"
#include "noxprotos.h"
#include "PlotCommunication.h"

int GetXVText(double scale, char *s);
int GetYVText(double scale, char *s);

extern PlotCommunication *cm;
void update_ledit_items(int gno);
void set_menus(int sens);
int get_pagelayout(int p);
void stufftext(char *s, int sp);
void free(void *ptr);
void lmfit(char *f, int n, double *x, double *y, double *yf, int np, double *a, double tol, int *info);
void fixupstr(char *val);
void extractsets_region(int gfrom, int gto, int rno);
void deletesets_region(int gno, int rno);
void set_wait_cursor();
void unset_wait_cursor();
void drawimage();
void set_cursor(int c);
void set_write_mode(int m);
void dispstrxlib(int x, int y, int rot, char *s, int just, int fudge);
void get_default_canvas_size(int *w, int *h);
void set_canvas_size(int w, int h, int o);
void findpoint_inset(int gno, int setno, double x, double y, int *loc);
void do_breakset(int gno, int setno, int ind);
void set_graph_viewport(int gno, double vx1, double vy1, double vx2, double vy2);
// void DefineSetSelectorFilter(SetChoiceItem * s);
// void do_regress_proc(Widget w, XtPointer client_data, XtPointer call_data);
void do_regress(int setno, int ideg, int iresid, int rno, int invr);
void activateset(int gno, int setno);
double my_hypot(double x, double y);
void stasum(double *x, int n, double *xbar, double *sd, int flag);
int next_box(void);
int next_line(void);
int next_string(void);
// void set_plotstr_string(plotstr *pstr, char *buf);
void set_prop(int gno, ...);
void set_axis_prop(int whichgraph, int naxis, int prop, double val);
void do_activateset(int gno, int setno, int len);
void droppoints(int gno, int setno, int startno, int endno, int dist);
void do_moveset(int gfrom, int j1, int gto, int j2);
void softkillset(int gno, int setno);
void kill_graph(int gno);
void wipeout(int ask);
double do_int(int setno, int itype);
void defaultsetgraph(int gno, int setno);
void draw_focus(int gno);
void read_image(char *fname);
void write_image(char *fname);
void outputset(int gno, int setno, char *fname, char *dformat);
void push_world(void);
void pop_world(void);
void cycle_world_stack(void);
void show_world_stack(int n);
void add_world(int gno, double x1, double x2, double y1, double y2,
               double t1, double t2, double u1, double u2);
void clear_world_stack(void);

// void do_regress_proc(Widget w, XtPointer client_data, XtPointer call_data);
void do_running_command(int type, int setno, int rlen);
void do_fourier_command(int ftype, int setno, int ltype);
// void do_spline_proc(Widget w, XtPointer client_data, XtPointer call_data);
void do_spline(int set, double start, double stop, int n);
void do_histo_command(int fromset, int toset, int tograph,
                      double minb, double binw, int nbins);
// void do_differ_proc(Widget w, XtPointer client_data, XtPointer call_data);
void do_differ(int setno, int itype);
void gwindup_proc(void);
void gwinddown_proc(void);
void gwindright_proc(void);
void gwindleft_proc(void);
void gwindshrink_proc(void);
void gwindexpand_proc(void);
void scroll_proc(int value);
void scrollinout_proc(int value);
void my_doublebuffer(int mode);
void my_frontbuffer(int mode);
void my_backbuffer(int mode);
void my_swapbuffer(void);
void putparms(int gno, FILE *pp, int imbed);
void set_hotlink(int gno, int setno, int onoroff, char *fname, int src);
int set_pagelayout(int layout);
void set_toolbars(int bar, int onoff);
void flush_pending(void);
void drawpolysym(double *x, double *y, int len, int sym, int skip, int fill, double size);
void lowtoupper(char *s);
void log_results(const char *buf);
int follow(int expect, int ifyes, int ifno);
int checkon(int prop, int old_val, int new_val);
void expand_tilde(char *buf);
void set_title(char *ts);
void set_left_footer(const char *s);
void setbgcolor(int c);
void xlibsetcmap(int i, int r, int g, int b);
void scrunch_points(int *x, int *y, int *n);
void stripspecial(char *s, char *cs);
int setfont(int f);
int isoneof(int c, char *s);
void puthersh(int xpos, int ypos, double scale, int dir, int just, int color, void (*vector)(int, int, int), char *s);
extern "C" {
void free(void *);
double lgamma(double x);
}

void do_select_region(void);
void add_point(int gno, int setno, double px, double py, double tx, double ty, int type);
int inregion(int regno, double x, double y);
void update_set_status(int gno, int setno);
int getncols(int gno, int setno);
void setlength(int gno, int i, int length);
void scanner(char *s, double *x, double *y, int len, double *a, double *b, double *c, double *d, int lenscr, int i, int setno, int *errpos);
int setcolor(int col);
int setlinestyle(int style);
int setlinewidth(int wid);
void draw_arrow(double x1, double y1, double x2, double y2, int end, double asize, int type);
void my_move2(double x, double y);
void my_draw2(double x, double y);
int world2view(double x, double y, double *vx, double *vy);
void view2world(double vx, double vy, double *x, double *y);
void initialize_screen(int *argc, char **argv);
void set_program_defaults(void);
void initialize_cms_data(void);
int getparms(int gno, char *plfile);
// void set_printer_proc(Widget w, XtPointer client_data, XtPointer call_data);
void set_graph_active(int gno);
int argmatch(const char *s1, const char *s2, int atleast);
void realloc_plots(int maxplot);
int alloc_blockdata(int ncols);
int alloc_blockdata(int ncols);
void realloc_graph_plots(int gno, int maxplot);
void realloc_graphs(void);
void realloc_lines(int n);
void realloc_boxes(int n);
void realloc_strings(int n);
void read_param(char *pbuf);
int getdata(int gno, char *fn, int src, int type);
int getdata_step(int gno, char *fn, int src, int type);
void create_set_fromblock(int gno, int type, char *cols);
int fexists(char *to);
void do_writesets(int gno, int setno, int imbed, char *fn, char *format);
void do_writesets_binary(int gno, int setno, char *fn);
int isdir(char *f);
int activeset(int gno);
void defaulty(int gno, int setno);
void default_axis(int gno, int method, int axis);
void defaultx(int gno, int setno);
void set_printer(int device, char *prstr);
void defaultgraph(int gno);
// void set_plotstr_string(plotstr *pstr, char *buf);
void arrange_graphs(int grows, int gcols);
void hselectfont(int f);
void defineworld(double x1, double y1, double x2, double y2, int mapx, int mapy);
int islogx(int gno);
int islogy(int gno);
void viewport(double x1, double y1, double x2, double y2);
void runbatch(char *bfile);
void do_hardcopy(void);
void do_main_loop(void);
int yesno(const char *msg1, const char *s1, const char *s2, const char *helptext);
void errwin(const char *s);
void killset(int gno, int setno);
void set_default_graph(int gno);
void do_copyset(int gfrom, int j1, int gto, int j2);
void drawgraph(void);
void fswap(double *x, double *y);
void iswap(int *x, int *y);
void updatesetminmax(int gno, int setno);
void getsetminmax(int gno, int setno, double *x1, double *x2, double *y1, double *y2);
void setfixedscale(double xv1, double yv1, double xv2, double yv2,
                   double *xg1, double *yg1, double *xg2, double *yg2);
void do_clear_lines(void);
void do_clear_boxes(void);
void do_clear_text(void);
void update_set_lists(int gno);
void update_world(int gno);
void update_view(int gno);
void update_status(int gno, int itemtype, int itemno);
void update_ticks(int gno);
void update_autos(int gno);
void updatelegends(int gno);
void updatesymbols(int gno, int value);
void update_label_proc(void);
void update_locator_items(int gno);
void update_draw(void);
void update_frame_items(int gno);
void update_graph_items(void);
void update_hotlinks(void);
void update_all(int gno);
void set_action(int act);
void set_stack_message(void);
int getFormat_index(int f);
void autoscale_set(int gno, int setno, int axis);
void autoscale_graph(int gno, int axis);
