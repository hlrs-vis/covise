#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <General/include/cfg.h>
#include <General/include/geo.h>
#include <General/include/log.h>
#include <General/include/common.h>

#ifdef   DRAFT_TUBE
#include <DraftTube/include/tube.h>
#endif

#ifdef   AXIAL_RUNNER
#include <AxialRunner/include/axial.h>
#endif

#ifdef   RADIAL_RUNNER
#include <RadialRunner/include/radial.h>
#endif

#define  GENERAL  "[general]"
#define  TYPE  "type"
#define MINMAX  "min-max scaling"

int WriteGeometry(struct geometry *g, const char *fn)
{
   int res = 0;
   FILE *fp;

   if (g && (fp = fopen(fn, "w")) != NULL)
   {

      fprintf(fp, "%30c\n", '#');
      fprintf(fp, "# Date      : %s\n","unknown");
      fprintf(fp, "# Host      : %s\n","unknwon");
      fprintf(fp, "%30c\n", '#');
      fprintf(fp, "\n%s\n", GENERAL);
      fprintf(fp, "%45s = %s\n", TYPE, GT_Type(g->type));
      fprintf(fp, "%45s = %.5f, %.5f\n",MINMAX,g->minmax[0],
         g->minmax[1]);

      switch (g->type)
      {
         case GT_TUBE:
#ifdef   DRAFT_TUBE
            res = WriteTube(g->tu, fp);
#endif
            break;
         case GT_RRUNNER:
#ifdef   RADIAL_RUNNER
            res = WriteRadialRunner(g->rr, fp);
#endif
            break;
         case GT_ARUNNER:
#ifdef   AXIAL_RUNNER
            res = WriteAxialRunner(g->ar, fp);
#endif
            break;
         default:
            res = 0;
            break;
      }
      fprintf(fp, "# End of parameters ############################\n");
      fclose(fp);
   }
   else
   {
      res = 0;
   }
   return res;
}
