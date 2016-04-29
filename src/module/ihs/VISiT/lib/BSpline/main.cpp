#include <stdio.h>
#include <stdlib.h>
#include "../General/include/fatal.h"
#include "../General/include/flist.h"
#include "../General/include/points.h"
#include "../General/include/cfg.h"
#include "include/bspline.h"

#define BSP1      "[spline1]"
#define BSP_DEG   "degree"
#define BSP_NOP   "number of points"
#define BSP_POINT "point%d"

int main(int argc, char **argv)
{
   int i, degree, num;
   struct Point *poly, *spline;
   struct Flist *knot;
   float t0, dt;
   float x[3];
   FILE *fp;
   char *cfg, *buf, key[100];
   const char *data = "bsp.dat";

   // check args and open data file
   if (argc < 2)
   {
      printf ("Enter config file.\n");
      printf ("Usage: spline <cfg file>\n");
      exit(1);
   }
   cfg = argv[1];
   if ((fp = fopen(cfg, "r")) == NULL)
   {
      fprintf (stderr, "Error reading file %s\n", cfg);
      exit(1);
   }
   fclose(fp);
   if ((fp = fopen(data, "w")) == NULL)
   {
      fprintf(stderr, "Error opening file %s\n", data);
      exit(1);
   }

   // read cfg file
   printf ("Reading data from file %s\n", cfg);
   if ((buf = IHS_GetCFGValue(cfg, BSP1, BSP_DEG)) != NULL)
   {
      int iret = sscanf(buf, "%d", &degree);
      if(iret != 1)
	fprintf(stderr, "Error reading degree \n");
      free(buf);
   }
   if ((buf = IHS_GetCFGValue(cfg, BSP1, BSP_NOP)) != NULL)
   {
      int iret = sscanf(buf, "%d", &num);
      if(iret != 1)
	fprintf(stderr, "Error reading num \n");
      free(buf);
   }
   poly = AllocPointStruct();
   for (i = 0; i < num; i++)
   {
      sprintf(key, BSP_POINT, i+1);
      if ((buf = IHS_GetCFGValue(cfg, BSP1, key)) != NULL)
      {
         int iret = sscanf(buf, "%f, %f, %f", &x[0], &x[1], &x[2]);
         if(iret != 3)
	    fprintf(stderr, "Error reading x1 x2 x3 \n");
         AddPoint(poly, x[0], x[1], x[2]);
         free(buf);
      }
   }

   // write polygon coordinates to file
   for (i = 0; i < poly->nump; i++)
      fprintf(fp, "%10.6f  %10.6f  %10.6f\n", poly->x[i], poly->y[i], poly->z[i]);
   fprintf(fp, "\n\n");

   // compute spline knots and coordinates for parameter values t0
   knot = BSplineKnot(poly, degree);
#ifdef DEBUG
   DumpPoints(poly, NULL);
   DumpFlist(knot);
#endif
   dt = 0.01f;
   t0 = 0;
   spline = AllocPointStruct();
   while (t0 <= 1.0)
   {
      BSplinePoint(degree, poly, knot, t0, &x[0]);
      AddPoint(spline, x[0], x[1], x[2]);
      t0 += dt;
   }
   BSplinePoint(degree, poly, knot, 1.0, &x[0]);
   AddPoint(spline, x[0], x[1], x[2]);
   // write spline points to file
   for (i = 0; i < spline->nump; i++)
      fprintf(fp, "%10.6f  %10.6f  %10.6f\n", spline->x[i], spline->y[i], spline->z[i]);
   fclose(fp);

   FreePointStruct(poly);
   FreePointStruct(spline);
   FreeFlistStruct(knot);

   return(0);
}
