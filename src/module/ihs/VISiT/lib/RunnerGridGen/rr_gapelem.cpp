#ifdef GAP
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/ilist.h"
#include "../General/include/points.h"
#include "../General/include/nodes.h"
#include "../General/include/elements.h"
#include "../General/include/fatal.h"

#include "include/rr_grid.h"

#ifndef ABS
#define ABS(a)    ( (a) >= (0) ? (a) : -(a) )
#endif
#ifndef SIGN
#define SIGN(a)    ( (a) >= (0) ? (1) : -(1) )
#endif

int   CreateRR_GapElements(struct region **reg, struct region **next_reg,
struct Element *e, int *psteperiodic, int *ssteperiodic,
struct Element *pseblade, struct Element *sseblade,
struct Element *pseteperiodic, struct Element *sseteperiodic,
struct Element *ewall, struct Element *shroud, int reg_num,
int gpreg_num, int offset, int itip, int ishroud
#ifdef DEBUG_GAPELEM
, struct node **n
#endif
)
{
   int i, j, k;
   int maxreg_num, first, dfine;
   int elem[8], blelem[8], perielem[8];
   int pericount, new_elems, inew_sselems;

   int *nod0, *nod, *nod0_next, *nod_next;

   struct Ilist  *nodes0;
   struct region *tmpreg;

#ifdef DEBUG_GAPELEM
   int **elm;
   char fn[111];
   FILE *fp;
   static int count = 0;

   sprintf(fn,"rr_gapelem_%02d.txt", count++);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   fprintf(fp," CreateRR_GapElements %d\n",count);
   fprintf(fp," offset = %d\n",offset);
   fprintf(fp," itip = %d, ishroud = %d\n\n",itip, ishroud);
   fprintf(fp," Elem. ID  node IDs\n\n");
#endif

   new_elems = e->nume;
#ifdef DEBUG_GAPELEM
   fprintf(fp," new_elems = %d\n",new_elems);
#endif

#ifdef DEBUG
   if(itip == 0) fprintf(stderr,"\n");
   fprintf(stderr,"CreateRR_GapElements: creating tip clearance elements ... ");fflush(stderr);
#endif
   // **************************************************
   // create elements in tip clearance region
   maxreg_num = reg_num + gpreg_num;
   perielem[5] = perielem[6] = perielem[7] = -1;

   for(i = reg_num; i < maxreg_num; i++)
   {
      if(i == reg_num+1) inew_sselems = e->nume;
      tmpreg = reg[i];
      nodes0 = tmpreg->nodes[0];
      nod0   = tmpreg->nodes[tmpreg->numl]->list;
      nod0_next = next_reg[i]->nodes[next_reg[i]->numl]->list;
      // le element
      first = nodes0->list[0] - 1;                // index of first chord node
      nod    = nod0;
      nod_next = nod0_next;
      elem[0] = *(nod++);
      elem[1] = *(nod++);
      elem[2] = *(nod++);
      elem[3] = *nod;
      elem[4] = *(nod_next++);
      elem[5] = *(nod_next++);
      elem[6] = *(nod_next++);
      elem[7] = *nod_next;
      AddElement(e, elem);
#ifdef DEBUG_GAPELEM
      fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
         e->nume-1, elem[0],elem[1],elem[2],elem[3],
         elem[4],elem[5],elem[6],elem[7]);
#endif
      for(j = 1; j < nodes0->num-1; j++)
      {
         first += nodes0->list[j-1];
         nod = nod0 + first;
         nod_next = nod0_next + first;
         dfine = nodes0->list[j+1] - nodes0->list[j];
         // discretization is constant
         if(dfine == 0)
         {
#ifdef DEBUG_GAPELEM
            fprintf(fp,"dfine = %d\n",dfine);
#endif
            for(k = 0; k < nodes0->list[j]-1; k++)
            {
               elem[0] = *nod;
               nod += nodes0->list[j];
               elem[1] = *nod;
               elem[2] = *(++nod);
               nod -= nodes0->list[j];
               elem[3] = *nod;
               elem[4] = *nod_next;
               nod_next += nodes0->list[j];
               elem[5] = *nod_next;
               elem[6] = *(++nod_next);
               nod_next -= nodes0->list[j];
               elem[7] = *nod_next;
               AddElement(e, elem);
#ifdef DEBUG_GAPELEM
               fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
                  e->nume-1, elem[0],elem[1],elem[2],elem[3],
                  elem[4],elem[5],elem[6],elem[7]);
#endif
            }
         }                                        // dfine == 0

         // discretization changes by +/- 2 nodes
         else if (dfine == 2)
         {
#ifdef DEBUG_GAPELEM
            fprintf(fp,"dfine = %d\n",dfine);
#endif
            // first element
            elem[0] = *(nod - nodes0->list[j]);
            nod += nodes0->list[j];
            elem[1] = *nod;
            elem[2] = *(++nod);
            nod -= nodes0->list[j]+1;
            elem[3] = *nod;
            elem[4] = *(nod_next - nodes0->list[j]);
            nod_next += nodes0->list[j];
            elem[5] = *nod_next;
            elem[6] = *(++nod_next);
            nod_next -= nodes0->list[j]+1;
            elem[7] = *nod_next;
            AddElement(e, elem);
#ifdef DEBUG_GAPELEM
            fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
               e->nume-1, elem[0],elem[1],elem[2],elem[3],
               elem[4],elem[5],elem[6],elem[7]);
#endif
            // other elems
            for(k = 0; k < nodes0->list[j]-1; k++)
            {
               elem[0] = *nod;
               nod += nodes0->list[j]+1;
               elem[1] = *nod;
               elem[2] = *(++nod);
               nod -= nodes0->list[j]+1;
               elem[3] = *nod;
               elem[4] = *nod_next;
               nod_next += nodes0->list[j]+1;
               elem[5] = *nod_next;
               elem[6] = *(++nod_next);
               nod_next -= nodes0->list[j]+1;
               elem[7] = *nod_next;
               AddElement(e, elem);
#ifdef DEBUG_GAPELEM
               fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
                  e->nume-1, elem[0],elem[1],elem[2],elem[3],
                  elem[4],elem[5],elem[6],elem[7]);
#endif
            }
            // last element
            elem[0] = *nod;
            nod += nodes0->list[j]+1;
            elem[1] = *nod;
            elem[2] = *(++nod);
            elem[3] = *(nod - nodes0->list[j+1] - nodes0->list[j]);
            elem[4] = *nod_next;
            nod_next += nodes0->list[j]+1;
            elem[5] = *nod_next;
            elem[6] = *(++nod_next);
            elem[7] = *(nod_next - nodes0->list[j+1] - nodes0->list[j]);
            AddElement(e, elem);
#ifdef DEBUG_GAPELEM
            fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
               e->nume-1, elem[0],elem[1],elem[2],elem[3],
               elem[4],elem[5],elem[6],elem[7]);
#endif
         }                                        // dfine == 2

         else if (dfine == -2)
         {
#ifdef DEBUG_GAPELEM
            fprintf(fp,"dfine = %d\n",dfine);
#endif
            // first elem.
            elem[0] = *nod;
            elem[1] = *(nod + nodes0->list[j] + nodes0->list[j+1]);
            elem[2] = *(nod + nodes0->list[j]);
            elem[3] = *(++nod);
            elem[4] = *nod_next;
            elem[5] = *(nod_next + nodes0->list[j] + nodes0->list[j+1]);
            elem[6] = *(nod_next + nodes0->list[j]);
            elem[7] = *(++nod_next);
            AddElement(e, elem);
#ifdef DEBUG_GAPELEM
            fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
               e->nume-1, elem[0],elem[1],elem[2],elem[3],
               elem[4],elem[5],elem[6],elem[7]);
#endif
            // others
            for(k = 1; k < nodes0->list[j]-2; k++)
            {
               elem[0] = *nod;
               nod += nodes0->list[j]-1;
               elem[1] = *nod;
               elem[2] = *(++nod);
               nod -= nodes0->list[j]-1;
               elem[3] = *nod;
               elem[4] = *nod_next;
               nod_next += nodes0->list[j]-1;
               elem[5] = *nod_next;
               elem[6] = *(++nod_next);
               nod_next -= nodes0->list[j]-1;
               elem[7] = *nod_next;
               AddElement(e, elem);
#ifdef DEBUG_GAPELEM
               fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
                  e->nume-1, elem[0],elem[1],elem[2],elem[3],
                  elem[4],elem[5],elem[6],elem[7]);
#endif
            }
            // last elem.
            elem[0] = *nod;
            elem[1] = *(nod + nodes0->list[j]-1);
            elem[2] = *(nod + nodes0->list[j]-1 + nodes0->list[j+1]);
            elem[3] = *(++nod);
            elem[4] = *nod_next;
            elem[5] = *(nod_next + nodes0->list[j]-1);
            elem[6] = *(nod_next + nodes0->list[j]-1 + nodes0->list[j+1]);
            elem[7] = *(++nod_next);
            AddElement(e, elem);
#ifdef DEBUG_GAPELEM
            fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
               e->nume-1, elem[0],elem[1],elem[2],elem[3],
               elem[4],elem[5],elem[6],elem[7]);
#endif
         }                                        // dfine == -2
         // switch from odd to even (reduction by 1)
         else if (dfine == -1)
         {
#ifdef DEBUG_GAPELEM
            fprintf(fp,"dfine = %d\n",dfine);
#endif
            for(k = 0; k < nodes0->list[j+1]-1; k++)
            {
               elem[0] = *nod;
               nod += nodes0->list[j];
               elem[1] = *nod;
               elem[2] = *(++nod);
               nod -= nodes0->list[j];
               elem[3] = *nod;
               elem[4] = *nod_next;
               nod_next += nodes0->list[j];
               elem[5] = *nod_next;
               elem[6] = *(++nod_next);
               nod_next -= nodes0->list[j];
               elem[7] = *nod_next;
               AddElement(e, elem);
#ifdef DEBUG_GAPELEM
               fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
                  e->nume-1, elem[0],elem[1],elem[2],elem[3],
                  elem[4],elem[5],elem[6],elem[7]);
#endif
            }
            elem[0] = *nod;
            nod    += nodes0->list[j];
            elem[1] = *nod;
            nod    += nodes0->list[j+1];
            elem[2] = *nod;
            elem[3] = *(nod - nodes0->list[j+1] - nodes0->list[j] + 1);
            elem[4] = *nod_next;
            nod_next    += nodes0->list[j];
            elem[5] = *nod_next;
            nod_next    += nodes0->list[j+1];
            elem[6] = *nod_next;
            elem[7] = *(nod_next - nodes0->list[j+1] - nodes0->list[j] + 1);
            AddElement(e, elem);
#ifdef DEBUG_GAPELEM
            fprintf(fp," %8d %8d %8d %8d %8d %8d %8d %8d %8d\n",
               e->nume-1, elem[0],elem[1],elem[2],elem[3],
               elem[4],elem[5],elem[6],elem[7]);
#endif
         }                                        // dfine == -1
         // other cases are not supported yet!
         else
         {
#ifdef DEBUG_GAPELEM
            fprintf(fp,"dfine = %d\n",dfine);
#endif
            fprintf(stderr,"\n invalid change in number of nodes (%d)\n",dfine);
            fprintf(stderr," on chord no. %d of region no. %d!\n",j+1,i+1);
            fprintf(stderr," src: %s, line: %d\n\n",__FILE__,__LINE__);
            exit(-1);
         }
      }                                           // end j (number of chords)
#ifdef DEBUG_GAPELEM
      fprintf(fp,"\n\n# ps\n");
#endif
   }                                              // end i (tip regions)
   // **************************************************
   // determine boundary elements and set boundary conditions
   new_elems = e->nume - new_elems;
   fprintf(fp,"\n#\n new_elems = %d, inew_sselems = %d\n",new_elems, inew_sselems);
   // get blade surface elems. and shroud elems.
   blelem[5] = blelem[6] = blelem[7] = -1;
   if(itip == 0)
   {
      for(i = e->nume - new_elems; i < e->nume; i++)
      {
         blelem[0] = i;
         blelem[1] = e->e[i][0];
         blelem[2] = e->e[i][1];
         blelem[4] = e->e[i][2];
         blelem[3] = e->e[i][3];
         if(i < inew_sselems) AddElement(sseblade, blelem);
         else AddElement(pseblade, blelem);
         AddElement(ewall, blelem);
      }
   }
   if(ishroud == 0)
   {
      for(i = e->nume - new_elems; i < e->nume; i++)
      {
         blelem[0] = i;
         blelem[1] = e->e[i][4];
         blelem[2] = e->e[i][5];
         blelem[4] = e->e[i][6];
         blelem[3] = e->e[i][7];
         AddElement(shroud, blelem);
      }
   }
   // periodic boundaries
   for(i = e->nume - new_elems; i < e->nume; i++)
   {
      blelem[0] = i;
      pericount = 0;
      for(j = 0; j < 8; j++)
      {
         if(psteperiodic[e->e[i][j]])
         {
            blelem[++pericount] = e->e[i][j];
            if(pericount == 4)
            {
               AddElement(pseteperiodic, blelem);
               break;
            }
         }
         if(ssteperiodic[e->e[i][j]])
         {
            blelem[++pericount] = e->e[i][j];
            if(pericount == 4)
            {
               AddElement(sseteperiodic, blelem);
               break;
            }
         }
         continue;
      }
   }                                              // end i

   // **************************************************
#ifdef DEBUG_GAPELEM
   fprintf(fp,"\n\n");
   elm = e->e + e->nume-new_elems;
   fprintf(fp,"# element no. %7d (%8d%8d%8d%8d)\n",
      e->nume-new_elems+i+1, (*elm)[0]+1,
      (*elm)[1]+1, (*elm)[2]+1, (*elm)[3]+1);
   for(j = 0; j < 4; j++)
   {
      fprintf(fp,"%8d %16.6f %16.6f %16.6f\n",(*elm)[j]+1, n[(*elm)[j]]->x,
         n[(*elm)[j]]->y, n[(*elm)[j]]->z);
   }
   fprintf(fp,"%8d %16.6f %16.6f %16.6f\n\n",(*elm)[0]+1, n[(*elm)[0]]->x,
      n[(*elm)[0]]->y, n[(*elm)[0]]->z);
   elm++;

   for(i = 1; i < new_elems; i++)
   {
      fprintf(fp,"# element no. %7d (%8d%8d%8d%8d)\n",
         e->nume-new_elems+i+1, (*elm)[0]+1,
         (*elm)[1]+1, (*elm)[2]+1, (*elm)[3]+1);
      for(j = 0; j < 4; j++)
      {
         fprintf(fp,"%8d %16.6f %16.6f %16.6f\n",(*elm)[j]+1, n[(*elm)[j]]->x,
            n[(*elm)[j]]->y, n[(*elm)[j]]->z);
      }
      fprintf(fp,"\n");
      elm++;
   }
   if(ishroud == 0)
   {
      fprintf(fp,"\n\n");
      elm = e->e + e->nume-new_elems;
      fprintf(fp,"# element no. %7d (%8d%8d%8d%8d)\n",
         e->nume-new_elems+i+1, (*elm)[0]+1,
         (*elm)[1]+1, (*elm)[2]+1, (*elm)[3]+1);
      for(j = 4; j < 8; j++)
      {
         fprintf(fp,"%8d %16.6f %16.6f %16.6f\n",(*elm)[j]+1, n[(*elm)[j]]->x,
            n[(*elm)[j]]->y, n[(*elm)[j]]->z);
      }
      fprintf(fp,"%8d %16.6f %16.6f %16.6f\n\n",(*elm)[4]+1, n[(*elm)[4]]->x,
         n[(*elm)[4]]->y, n[(*elm)[4]]->z);
      elm++;

      for(i = 1; i < new_elems; i++)
      {
         fprintf(fp,"# element no. %7d (%8d%8d%8d%8d)\n",
            e->nume-new_elems+i+1, (*elm)[0]+1,
            (*elm)[1]+1, (*elm)[2]+1, (*elm)[3]+1);
         for(j = 4; j < 8; j++)
         {
            fprintf(fp,"%8d %16.6f %16.6f %16.6f\n",(*elm)[j]+1, n[(*elm)[j]]->x,
               n[(*elm)[j]]->y, n[(*elm)[j]]->z);
         }
         fprintf(fp,"\n");
         elm++;
      }
   }

   fclose(fp);
#endif
#ifdef DEBUG
   fprintf(stderr,"done!\n");
#endif

   return (0);
}
#endif                                            // GAP
