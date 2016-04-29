#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#define  DECLA
#include <General/include/cfg.h>
#include <General/include/geo.h>
#include <General/include/log.h>
#include <General/include/common.h>

#ifdef   DRAFT_TUBE
#include <DraftTube/include/tube.h>
#include <DraftTube/include/tgrid.h>
#endif

#ifdef   AXIAL_RUNNER
#include <AxialRunner/include/axial.h>
#include <AxialRunner/include/ar2cov.h>
#endif

#ifdef   DIAGONAL_RUNNER
#include <RadialRunner/include/diagonal.h>
#include <RadialRunner/include/rr2cov.h>
#endif

#ifdef   RADIAL_RUNNER
#include <RadialRunner/include/radial.h>
#include <RadialRunner/include/rr2cov.h>
#endif

#ifdef   GATE
#include <Gate/include/gate.h>
#include <Gate/include/ga2cov.h>
#endif

#define  GENERAL  "[general]"
#define  TYPE  "type"
#define MINMAX  "min-max scaling"

static int ReadGeneral(struct geometry *g, const char *fn);

#ifdef   NICHT_RAUS
struct covise_info *CreateGeometry(char *fn)
{
   struct geometry g;
   struct covise_info *ci;
#ifdef   PROGGEO
   char buf[255];
   struct tgrid *tg;
#endif

   ReadGeneral(&g, fn);

   switch (g.type)
   {
      case GT_TUBE:
#ifdef   DRAFT_TUBE
         g.tu = ReadTube(fn);
#endif
         break;
      case GT_RRUNNER:
#ifdef   RADIAL_RUNNER
         ReadRadialRunner(&g, fn);
#endif
         break;
      case GT_DRUNNER:
#ifdef  DIAGONAL_RUNNER
         ReadRadialRunner(&g, fn);
#endif
         break;
      case GT_ARUNNER:
#ifdef   AXIAL_RUNNER
         ReadAxialRunner(&g, fn);
#endif
         break;
      case GT_GATE:
#ifdef   GATE
         ReadGate(&g, fn);
#endif
         break;
      default:
         break;
   }
   //ci = CreateGeometry4Covise(&g);
#ifdef   PROGGEO
   switch (g.type)
   {
      case GT_TUBE:
#ifdef   DRAFT_TUBE
         //Tube2CoviseDump(ci);
         tg = CreateTGrid(g.tu);
         WriteTGrid(tg, "tgrid");
         WriteTBoundaryConditions(tg, "tgrid");
         DumpTGrid(tg);
#endif                                      // DRAFT_TUBE
         break;
      case GT_RRUNNER:
         break;
      case GT_DRUNNER:
         break;
      case GT_ARUNNER:
         break;
      case GT_GATE:
         break;
      default:
         break;
   }
   strcat(strcpy(buf, fn), ".new");
   WriteGeometry(&g, buf);
#endif                                         // PROGGEO
   return ci;
}
#endif                                            // NICHT_RAUS

struct geometry *ReadGeometry(const char *fn)
{
   struct geometry *g;

   if ((g = (struct geometry *)calloc(1, sizeof(struct geometry))) != NULL)
   {
     if(ReadGeneral(g, fn)) return NULL;

      switch (g->type)
      {
         case GT_TUBE:
#ifdef   DRAFT_TUBE
            g->tu = ReadTube(fn);
#endif
            break;
         case GT_RRUNNER:
#ifdef   RADIAL_RUNNER
            ReadRadialRunner(g, fn);
#endif
            break;
         case GT_DRUNNER:
#ifdef   DIAGONAL_RUNNER
            ReadRadialRunner(g, fn);
#endif
            break;
         case GT_ARUNNER:
#ifdef  AXIAL_RUNNER
            ReadAxialRunner(g, fn);
#endif
            break;
         case GT_GATE:
#ifdef   GATE
            ReadGate(g, fn);
#endif
            break;
         default:
            break;
      }
   }
   return g;
}


static int ReadGeneral(struct geometry *g, const char *fn)
{
   char *p;
   char *tmp;
   int i;

   if ((tmp = IHS_GetCFGValue(fn, GENERAL, TYPE)) != NULL)
   {
      p = tmp;
      while ((*p = tolower(*p)) != 0)
         p++;
      for (i = 0; i < Num_GT_Type(); i++)
      {
         dprintf(5, "ReadGeneral(): tmp=%s,GT_Type(%d)=%s\n",
            tmp, i, GT_Type(i));
         if (!strcmp(GT_Type(i), tmp))
         {
            g->type = i;
            dprintf(1, "Type of geometry: %s (%d)", GT_Type(i), i);
         }
      }
      free(tmp);
   }
   else
      dprintf(0, "IHS_GetCFGValue(%s, %s, %s) failed !\n", fn,
         GENERAL, TYPE);
   if ((tmp = IHS_GetCFGValue(fn, GENERAL, MINMAX)) != NULL)
   {
      int iread = sscanf(tmp,"%f, %f",&g->minmax[0], &g->minmax[1]);
      if(iread != 2)
      {
         dprintf(0, "Can't read minmax");
      }
   }
   else
   {
      g->minmax[0] = 1.0; g->minmax[1] = 1.0;
   }
   if (!g->type)
   {
      dprintf(0, "Missing TYPE in section [GENERAL]\n");
      return 1 ;
   }
   return 0;
}
