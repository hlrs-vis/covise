#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/fatal.h"
#include "include/iso.h"
#include "include/plot.h"

#define RANGE_LINE   "set %sr[%6.1f:%6.1f]"
#define TICS_LINE "set %stics %7.2f,%7.2f,%7.2f"
#define LABEL_LINE   "set %slabel \"%s\""
#define PLOT_LINE1   " '%s' index %d u 1:2 "
#define PLOT_LINE2   " '%s' index %d u 1:3 "
#define TITLE_LINE   "tit\"iso = %6.2f\" "
#define LINEPOINTS   "w lp lt %d lw 1 pt 3 "
#define LINE      "w l lt %d lw 1 "

FILE *InitPlotfile(char *fn)
{
   char buf[256+50];
   FILE *fp;

   if ((fp = fopen(fn, "w")) == NULL)
   {
      sprintf(buf, "Error opening plot file %s\n", fn);
      fatal(buf);
   }
   fprintf(fp, "reset\n");
   fprintf(fp, "set grid xtics ytics\n");
   fprintf(fp, "set key right outside\n");
   return fp;
}


void setrange(const char *axis, float min, float max, int intvl, FILE *fp)
{
   char range[100], tics[100];
   float inkr;

   sprintf(range, RANGE_LINE, axis, min, max);
   fprintf(fp, "%s\n", range);
   if (intvl)
   {
      inkr = (max - min)/(intvl - 1);
      sprintf(tics, TICS_LINE, axis, min, inkr, max);
      fprintf(fp, "%s\n", tics);
   }
   return;
}


void setlabel(const char *axis, const char *text, FILE *fp)
{
   char label[200];

   sprintf(label, LABEL_LINE, axis, text);
   fprintf(fp, "%s\n", label);
   return;
}


void setplot(const char *dat, struct Isofield *isof, FILE *fp)
{
   int i;
   char plot[100], line[200];

   for (i = 0; i < isof->num; i++)
   {
      if (!i)
         fprintf(fp, "pl ");
      sprintf(plot, PLOT_LINE1, dat, i);
      strcpy(line, plot);
      sprintf(plot, TITLE_LINE, isof->ic[i]->isoval);
      strcat(line, plot);
      if (isof->ic[i]->measured)
         sprintf(plot, LINEPOINTS, isof->ic[i]->calc+1);
      else
         sprintf(plot, LINE, isof->ic[i]->calc+1);
      strcat(line, plot);
      fprintf(fp, "%s", line);
      if (i != isof->num-1)
         fprintf(fp, ",\\\n");
      else
         fprintf(fp, "\n\npause -1\n");
   }
   for (i = 0; i < isof->num; i++)
   {
      if (!i)
         fprintf(fp, "pl ");
      sprintf(plot, PLOT_LINE2, dat, i);
      strcpy(line, plot);
      sprintf(plot, TITLE_LINE, isof->ic[i]->isoval);
      strcat(line, plot);
      if (isof->ic[i]->measured)
         sprintf(plot, LINEPOINTS, isof->ic[i]->calc+1);
      else
         sprintf(plot, LINE, isof->ic[i]->calc+1);
      strcat(line, plot);
      fprintf(fp, "%s", line);
      if (i != isof->num-1)
         fprintf(fp, ",\\\n");
      else
         fprintf(fp, "\n");
   }
   return;
}


void ClosePlotfile(FILE *fp)
{
   if (fp)
   {
      fprintf(fp, "pause -1\n");
      fclose(fp);
   }
   else
      fatal("fclose on NULL pointer\n");
   return;
}


void PlotIsofield(struct Isofield *isof)
{
   int i;
   const char *fdat = "iso.dat";
   const char *fplt = "iso.gnu";
   char buf[200];
   FILE *fp_dat, *fp_plt;
   const char syscall[200] = "gnuplot ";

   // write plot data file
   if ((fp_dat = fopen (fdat, "w")) == NULL)
   {
      sprintf(buf, "Error writing plot data file %s\n", fdat);
      fatal(buf);
   }
   for (i = 0; i < isof->num; i++)
   {
#ifdef DEBUG
      DumpIsocurve(isof->ic[i], fp_dat);
      fprintf(fp_dat, "\n\n");
      printf("\n");
#endif
   }
   fclose(fp_dat);

   // write plot file
   fp_plt = InitPlotfile((char *)fplt);
   // min/max berechnen

   // ranges setzen

   // labeltext einfuegen
   setlabel("x", "x label text", fp_plt);
   setlabel("y", "y label text", fp_plt);

   // plotspezifikationen
   setplot(fdat, isof, fp_plt);
   ClosePlotfile(fp_plt);

   // systemaufruf
   strcat((char *)syscall, (char *)fplt);
   int res = system(syscall);
   static_cast<void>(res); // unused
   return;
}
