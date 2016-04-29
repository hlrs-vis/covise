#include <stdio.h>
#include <stdlib.h>
#include "../General/include/geo.h"
#include "../General/include/cov.h"
#include "../General/include/log.h"
#include "include/mesh.h"
#include "include/rr_grid.h"

#ifdef RADIAL_RUNNER
#include "../RadialRunner/include/radial.h"
#endif

#ifdef DIAGONAL_RUNNER
#include "../RadialRunner/include/diagonal.h"
#endif

int main(int argc, char **argv)
{

   char *fn;
   FILE *fp;

   struct geometry *g;
#if defined(RADIAL_RUNNER) || defined(DIAGONAL_RUNNER)
   struct rr_grid *grid;
#endif

   //	SetLogLevel(LOG_NOTHING);
   if (argc != 2)
   {
      fprintf(stderr, "usage: geometry inputfile\n");
      exit(1);
   }

   fn = argv[1];
   /* stupid check version .. (but easy)	*/
   if ((fp = fopen(fn, "r")) == NULL)
   {
      fprintf(stderr, "File '%s' couldn't be opened\n", fn);
      exit(1);
   }
   fclose(fp);

   //SetLogLevel(LOG_ALL);
   g = ReadGeometry(fn);

#ifdef RADIAL_RUNNER
   grid = CreateRR_Mesh(g->rr);
#endif
#ifdef DIAGONAL_RUNNER
   grid = CreateRR_Mesh(g->rr);
#endif

#if defined(RADIAL_RUNNER) || defined(DIAGONAL_RUNNER)
   FreeRRGridMesh(grid);
   free(grid);
#endif

   free(g);

   return 0;
}
