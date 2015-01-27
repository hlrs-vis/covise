/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

double setcharsize(double size);
double xconv(double x);
double yconv(double y);
int clipt(double d, double n, double *te, double *tl);
int initgraphics(int device);
int lengthpoly(double *x, double *y, int n);
void set_coordmap(int mapx, int mapy);
int setcolor(int col);
int setfont(int f);
int setlinestyle(int style);
int setlinewidth(int wid);
int setpattern(int k);
int symok(double x, double y);
int world2device(double wx, double wy, int *x, int *y);
int world2deviceabs(double wx, double wy, int *x, int *y);
int world2view(double x, double y, double *vx, double *vy);
void defineworld(double x1, double y1, double x2, double y2, int mapx, int mapy);
void device2view(int x, int y, double *vx, double *vy);
void device2world(int x, int y, double *wx, double *wy);
void draw_arrow(double x1, double y1, double x2, double y2, int end, double asize, int type);
void draw_head(int ix1, int iy1, int ix2, int iy2, int sa, int type);
void drawcircle(double xc, double yc, double s, int f);
void drawgrid(int dir, double start, double end, double y1, double y2, double step, int cy, int ly, int wy);
void drawpoly(double *x, double *y, int n);
void drawpolyseg(double *x, double *y, int n);
void drawpolysym(double *x, double *y, int len, int sym, int skip, int fill, double size);
void drawsym(int x, int y, int sym, double size, int fill);
void drawtic(double x, double y, int dir, int axis);
void drawtitle(char *title, int which);
void errorbar(double x, double y, double ebarlen, int xy);
void boxplotsym(double x, double med, double il, double iu, double ol, double ou);
void fillcolor(int n, double *px, double *py);
void fillpattern(int n, double *px, double *py);
void fillrectcolor(double x1, double y1, double x2, double y2);
void fillrectpat(double x1, double y1, double x2, double y2);
void leavegraphics(void);
void my_backbuffer(int mode);
void my_circle(double xc, double yc, double s);
void my_doublebuffer(int mode);
void my_draw2(double x2, double y2);
void my_filledcircle(double xc, double yc, double s);
void my_frontbuffer(int mode);
void my_move2(double x, double y);
void my_swapbuffer(void);
void openclose(double x, double y1, double y2, double ebarlen, int xy);
void putlegend(int i, /* which set */
               int d,
               /* flag, 1 = no draw, just compute min/max
 * for bounding box */
               int xlen, /* length of legend */
               int ylen, /* distance between entries */
               double size, /* symbol size */
               double x, /* location x */
               double y, /* location y */
               int sy, /* symbol */
               int ly, /* line style */
               int cy, /* line color */
               int wy, /* line width */
               char *s, /* legend string */
               int fill, /* symbol fill */
               int sc, /* symbol color */
               int sw, /* symbol linewidth */
               int sl); /* symbol linestyle */
void putbarlegend(int i, /* which set */
                  int d,
                  /* flag, 1 = no draw, just compute min/max
 * for bounding box */
                  int xlen, /* length of legend */
                  int ylen, /* distance between entries */
                  double size, /* symbol size */
                  double x, /* location x */
                  double y, /* location y */
                  int sy, /* symbol */
                  int ly, /* line style */
                  int cy, /* line color */
                  int wy, /* line width */
                  char *s, /* legend string */
                  int fill, /* symbol fill */
                  int fu, /* fill using pattern or color */
                  int fc, /* fill color */
                  int fp); /* fill pattern */
void putlegendrect(int fill, int fillusing, int fillcolor, int fillpat, int cy, int wy, int ly);
void rect(double x1, double y1, double x2, double y2);
void setclipping(int fl);
void setfixedscale(double xv1, double yv1, double xv2, double yv2, double *xg1, double *yg1, double *xg2, double *yg2);
void setticksize(double sizex, double sizey);
void symcircle(int x, int y, double s, int f);
void symdiamond(int x, int y, double s, int f);
void symplus(int x, int y, double s, int f);
void symsplat(int x, int y, double s, int f);
void symsquare(int x, int y, double s, int f);
void symstar(int x, int y, double s, int f);
void symtriangle1(int x, int y, double s, int f);
void symtriangle2(int x, int y, double s, int f);
void symtriangle3(int x, int y, double s, int f);
void symtriangle4(int x, int y, double s, int f);
void symx(int x, int y, double s, int f);
void velplt(double xx, double yy, double u, double v, double vscale, int type);
void view2world(double vx, double vy, double *x, double *y);
void viewport(double x1, double y1, double x2, double y2);
void writestr(double x, double y, int dir, int just, char *s);
