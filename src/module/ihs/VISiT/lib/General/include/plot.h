#ifndef PLOT_INCLUDED
#define PLOT_INCLUDED

FILE *InitPlotfile(char *fn);
void setrange(const char *axis, float min, float max, int intvl, FILE *fp);
void setlabel(const char *axis, const char *text, FILE *fp);
void setplot(const char *dat, struct Isofield *isof, FILE *fp);
void ClosePlotfile(FILE *fp);
void PlotIsofield(struct Isofield *isof);
#endif                                            // PLOT_INCLUDED
