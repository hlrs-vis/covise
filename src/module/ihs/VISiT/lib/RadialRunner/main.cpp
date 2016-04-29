#include <stdio.h>
#include <stdlib.h>
#include "../General/include/geo.h"
#include "../General/include/cov.h"
//#include "../DraftTube/include/tgrid.h"
#include "../General/include/log.h"

int main(int argc, char **argv)
{
   FILE *fp;
   char * fn;

   if (argc != 2)
   {
      fprintf(stderr, "usage: geometry inputfile\n");
      exit(1);
   }

   fn = argv[1];
   /* stupid check version .. (but easy)	*/
   if ((fp = fopen(fn, "r")) == NULL)
   {
      fprintf(stderr, "File %s couldn't be opened\n", fn);
      exit(1);
   }
   fclose(fp);

   //CreateGeometry(fn);
   ReadGeometry(fn);

   return 0;
}
