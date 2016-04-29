#include <stdio.h>
#include <stdlib.h>
#include <../General/include/cfg.h>
#include <../General/include/geo.h>
#include <../General/include/log.h>
#include <../General/include/common.h>

#include "../RunnerGridGen/include/mesh.h"

#ifdef AXIAL_RUNNER
#include "include/axial.h"
#include "../RunnerGridGen/include/rr_grid.h"
#endif

int main(int argc, char **argv)
{

   char *fn;
   FILE *fp;

   struct geometry *g;
   struct rr_grid *grid;

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

   g = ReadGeometry(fn);

#ifdef AXIAL_RUNNER
   grid = CreateAR_Mesh(g->ar);
#endif

#ifdef RADIAL_RUNNER
   grid = CreateRR_Mesh(g->rr);
#endif
#ifdef DIAGONAL_RUNNER
   grid = CreateRR_Mesh(g->rr);
#endif

   FreeRRGridMesh(grid);
   free(grid);
   free(g);

   return 0;
}
