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

#ifdef DEBUG_ELEMENTS
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int CreateRR_Elements(struct region **reg, struct Element *e, int *psblade,
int *ssblade, int *psleperiodic, int *ssleperiodic,
int *psteperiodic, int *ssteperiodic, int *inlet,
int *outlet, struct Element *ewall,
struct Element *pseblade, struct Element *sseblade,
struct Element *pseleperiodic,
struct Element *sseleperiodic,
struct Element *pseteperiodic,
struct Element *sseteperiodic, struct Element *einlet,
struct Element *eoutlet,  int reg_num, int offset)
{
   int i, j, k, l;
   int elem[8];
   int blelem[8];
   int pelem[8];
   int inelem[8];
   int outelem[8];
   int blcount, pcount, incount, outcount;

   int *nod0, *nod;

#ifdef DEBUG_ELEMENTS
   char fn[111];
   FILE *fp;
   static int count = 0;

   sprintf(fn,"rr_debugelem_%02d.txt", count++);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   fprintf(fp," CreateRR_Elements %d\n",count);
   fprintf(fp," offset = %d\n\n",offset);
   fprintf(fp," Elem. ID  node IDs\n\n");
#endif

#ifdef DEBUG
   fprintf(stderr,"CreateRR_Elements: creating elements ... ");fflush(stderr);
#endif

   blelem[5]  = blelem[6]  = blelem[7]  = -1;
   pelem[5]   = pelem[6]   = pelem[7]   = -1;
   inelem[5]  = inelem[6]  = inelem[7]  = -1;
   outelem[5] = outelem[6] = outelem[7] = -1;
   for(i = 0; i < reg_num; i++)
   {
      nod0 = &reg[i]->nodes[reg[i]->numl]->list[0];
      for(j = 0; j < reg[i]->nodes[0]->num-1; j++)
      {
         nod = nod0 + j;
         for(k = 0; k < reg[i]->nodes[1]->num-1; k++)
         {
            elem[0] = *nod;
            elem[1] = *(++nod);
            nod += reg[i]->nodes[0]->num;
            elem[2] = *nod;
            elem[3] = *(--nod);
            elem[4] = elem[0] + offset;
            elem[5] = elem[1] + offset;
            elem[6] = elem[2] + offset;
            elem[7] = elem[3] + offset;
            AddElement(e, elem);

#ifdef DEBUG_ELEMENTS
            fprintf(fp," %8d   %8d   %8d   %8d   %8d   %8d   %8d   %8d   %8d\n",
               e->nume, e->e[e->nume-1][0]+1, e->e[e->nume-1][1]+1,
               e->e[e->nume-1][2]+1, e->e[e->nume-1][3]+1, e->e[e->nume-1][4]+1,
               e->e[e->nume-1][5]+1, e->e[e->nume-1][6]+1, e->e[e->nume-1][7]+1);
#endif

            // get boundary elements
            if(j == 0 || j == reg[i]->nodes[0]->num-2 || k == 0 || k == reg[i]->nodes[1]->num-2)
            {
               blcount = pcount = incount = outcount = 0;
               blelem[blcount]   = e->nume-1;
               pelem[pcount]     = e->nume-1;
               inelem[incount]   = e->nume-1;
               outelem[outcount] = e->nume-1;
               for(l = 0; l < 8; l++)
               {
                  if(psleperiodic[elem[l]] == 1)
                  {
                     pelem[++pcount] = elem[l];
                     if(pcount == 4)
                     {
                        AddElement(pseleperiodic,pelem);
                     }
                  }
                  if(ssleperiodic[elem[l]] == 1)
                  {
                     pelem[++pcount] = elem[l];
                     if(pcount == 4)
                     {
                        AddElement(sseleperiodic,pelem);
                     }
                  }
                  if(psteperiodic[elem[l]] == 1)
                  {
                     pelem[++pcount] = elem[l];
                     if(pcount == 4)
                     {
                        AddElement(pseteperiodic,pelem);
                     }
                  }
                  if(ssteperiodic[elem[l]] == 1)
                  {
                     pelem[++pcount] = elem[l];
                     if(pcount == 4)
                     {
                        AddElement(sseteperiodic,pelem);
                     }
                  }
                  if(psblade[elem[l]] == 1)
                  {
                     blelem[++blcount] = elem[l];
                     if(blcount == 4)
                     {
                        AddElement(pseblade,blelem);
                        AddElement(ewall,blelem);
                     }
                  }
                  if(ssblade[elem[l]] == 1)
                  {
                     blelem[++blcount] = elem[l];
                     if(blcount == 4)
                     {
                        AddElement(sseblade,blelem);
                        AddElement(ewall,blelem);
                     }
                  }
                  if(inlet[elem[l]] == 1)
                  {
                     inelem[++incount] = elem[l];
                     if(incount == 4)
                     {
                        AddElement(einlet,inelem);
                     }
                  }
                  if(outlet[elem[l]] == 1)
                  {
                     outelem[++outcount] = elem[l];
                     if(outcount == 4)
                     {
                        AddElement(eoutlet,outelem);
                     }
                  }

               }                                  // end l
            }                                     // end j == 0

         }                                        // end k ... loop over line no.2
      }                                           // end j ... loop over line no.1
   }                                              // end i ... loop over regions

#ifdef DEBUG_ELEMENTS
   fclose(fp);
#endif
#ifdef DEBUG
   fprintf(stderr,"done!\n");
#endif

   return(0);
}


int GetHubElements(struct Element *e, struct Element *wall, struct Element *frictionless,
struct Element *shroudext, struct Nodelist *n, int offset,
float linlet, float lhub)
{
   int i;

   float midlen;

   int elem[8];
   int **tmpelem;

#ifdef DEBUG_BC
   int j;

   char *fn1 = "rr_hubelems.txt";
   char *fn2 = "rr_flesselems.txt";
   FILE *fp1;
   FILE *fp2;

   if( (fp1 = fopen(fn1,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn1);
      exit(-1);
   }
   if( (fp2 = fopen(fn2,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn2);
      exit(-1);
   }
#endif

#ifdef DEBUG_BC
   fprintf(stderr,"GetHubElements: offset = %d, lhub = %f\n",offset, lhub);
#endif

   elem[5] = elem[6] = elem[7] = -1;
   for(i = 0; i < offset; i++)
   {
      tmpelem = e->e + i;
      elem[0] = i;
      elem[1] = (*tmpelem)[0];
      elem[2] = (*tmpelem)[1];
      elem[4] = (*tmpelem)[2];
      elem[3] = (*tmpelem)[3];
      midlen  = (n->n[elem[1]]->l + n->n[elem[2]]->l
         + n->n[elem[3]]->l + n->n[elem[4]]->l) / 4.0;
      if(midlen < linlet)
      {
         AddElement(shroudext, elem);
#ifdef DEBUG_BC
         fprintf(fp1,"# %8d\n",elem[0]+1);
         for(j = 1; j < 5; j++)
         {
            fprintf(fp1,"%8d %16.6f %16.6f %16.6f\n",elem[j]+1, n->n[elem[j]]->x,
               n->n[elem[j]]->y, n->n[elem[j]]->z);
         }
         fprintf(fp1,"%8d %16.6f %16.6f %16.6f\n\n\n",elem[1]+1, n->n[elem[1]]->x,
            n->n[elem[1]]->y, n->n[elem[1]]->z);
#endif
      }
      else if(midlen < lhub)
      {
         AddElement(wall, elem);
#ifdef DEBUG_BC
         fprintf(fp1,"# %8d\n",elem[0]+1);
         for(j = 1; j < 5; j++)
         {
            fprintf(fp1,"%8d %16.6f %16.6f %16.6f\n",elem[j]+1, n->n[elem[j]]->x,
               n->n[elem[j]]->y, n->n[elem[j]]->z);
         }
         fprintf(fp1,"%8d %16.6f %16.6f %16.6f\n\n\n",elem[1]+1, n->n[elem[1]]->x,
            n->n[elem[1]]->y, n->n[elem[1]]->z);
#endif
      }
      else
      {
         AddElement(frictionless, elem);
#ifdef DEBUG_BC
         fprintf(fp2,"# %8d\n",elem[0]+1);
         for(j = 1; j < 5; j++)
         {
            fprintf(fp2,"%8d %16.6f %16.6f %16.6f\n",elem[j]+1, n->n[elem[j]]->x,
               n->n[elem[j]]->y, n->n[elem[j]]->z);
         }
         fprintf(fp2,"%8d %16.6f %16.6f %16.6f\n\n\n",elem[1]+1, n->n[elem[1]]->x,
            n->n[elem[1]]->y, n->n[elem[1]]->z);
#endif
      }
   }

#ifdef DEBUG_BC
   fclose(fp1);
   fclose(fp2);
#endif
   return(0);
}


int   GetShroudElements(struct Element *e, struct Element *shroud, struct Element *shroudext,
struct Nodelist *n, int offset, int ge_num, float linlet, float lhub)
{
   int i;
   int elem[8];
   int **tmpelem;

   float midlen;

#ifdef DEBUG_BC
   fprintf(stderr,"GetShroudElements: offset  = %d, ge_num = %d\n",offset, ge_num);
#endif

   elem[5]  = elem[6] = elem[7] = -1;
   elem[0]  = offset * (ge_num-2) - 1;
   tmpelem  = e->e + elem[0];
#ifdef DEBUG_BC
   fprintf(stderr,"GetShroudElements: elem[0]    = %d (first element)\n",elem[0]+1);
#endif
   for(i = 0; i < offset; i++)
   {
      tmpelem++;
      elem[0]++;
      elem[1] = (*tmpelem)[4];
      elem[2] = (*tmpelem)[5];
      elem[4] = (*tmpelem)[6];
      elem[3] = (*tmpelem)[7];
      midlen  = (n->n[elem[1]]->l + n->n[elem[2]]->l
         + n->n[elem[3]]->l + n->n[elem[4]]->l) / 4.0;
      if(midlen < linlet)
      {
         AddElement(shroudext, elem);
      }
      else if(midlen < lhub)
      {
         AddElement(shroud, elem);
      }
      else
      {
         AddElement(shroudext, elem);
      }
   }
#ifdef DEBUG_BC
   fprintf(stderr,"GetShroudElements: elem[0] = %d (last element)\n",elem[0]);
#endif
   return(0);
}

int GetAllHubElements(struct Element *e, struct Element *hubAll, struct Nodelist *n, int offset)
{
   int i;

//   float midlen;

   int elem[8];
   int **tmpelem;


   elem[5] = elem[6] = elem[7] = -1;
   for(i = 0; i < offset; i++)
   {
      tmpelem = e->e + i;
      elem[0] = i;
      elem[1] = (*tmpelem)[0];
      elem[2] = (*tmpelem)[1];
      elem[4] = (*tmpelem)[2];
      elem[3] = (*tmpelem)[3];

      AddElement(hubAll, elem);
   }
   return(0);
}

int   GetAllShroudElements(struct Element *e, struct Element *shroudAll,
struct Nodelist *n, int offset, int ge_num)
{
   int i;
   int elem[8];
   int **tmpelem;
   
   elem[5]  = elem[6] = elem[7] = -1;
   elem[0]  = offset * (ge_num-2) - 1;
   tmpelem  = e->e + elem[0];

   for(i = 0; i < offset; i++)
   {
      tmpelem++;
      elem[0]++;
      elem[1] = (*tmpelem)[4];
      elem[2] = (*tmpelem)[5];
      elem[4] = (*tmpelem)[6];
      elem[3] = (*tmpelem)[7];
//      midlen  = (n->n[elem[1]]->l + n->n[elem[2]]->l
//         + n->n[elem[3]]->l + n->n[elem[4]]->l) / 4.0;
      //if(midlen < linlet)
      //{
      //   AddElement(shroudext, elem);
      //}
      //else if(midlen < lhub)
      //{
      //   AddElement(shroud, elem);
      //}
      //else
      //{
         AddElement(shroudAll, elem);
      //}
   }
   return(0);
}

#ifdef RR_IONODES
int GetRRIONodes(struct Element *e, struct Nodelist *n, struct ge **ge, int ge_num,
struct Element *in, struct Element *out, struct cgrid **cge, int npoin_ext)
{

   int i, j, k, l;
   int eoff, ege_num, estep[3], elnum[3];
   int elem[8], num0, ix, count, max, portion;
#ifdef RROUTLET_NOGAP
   int next_elem, dj, sig, *jelem, *step;
#endif

   int **tmp;
   int  *belem;
   float *elen, len, avlen;
#ifdef PATRAN_SES_OUT
   const char *fnpat = "RUNNER_IOelements.ses";
   FILE *fppat;
#endif

#ifdef DEBUG_BC
   char *fn = "rr_ionodes.txt";
   FILE *fp;

   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"file '%s'!\n",fn);
      exit(-1);
   }

   fprintf(fp,"ge_num = %d\n", ge_num);
#endif

   elem[5] = elem[6] = elem[7] = -1;
   eoff    = e->nume/(ge_num-1);
   ege_num = ge_num-1;

   // ****** Runner inlet elements
   elnum[0]   = cge[0]->reg[0]->line[0]->nump-1;
   estep[0]   = cge[0]->reg[0]->line[1]->nump-1;
#ifdef DEBUG_BC
   fprintf(fp,"eoff = %d, ege_num = %d\n",eoff, ege_num);
   fprintf(fp,"inlet:\n elnum[0] = %d, estep[0] = %d\n",elnum[0], estep[0]);
#endif
   num0 = 0;
   for(i = 0; i < ege_num; i++)
   {
      tmp  = e->e + num0;
      for(j = 0; j < elnum[0]; j++)
      {
         elem[0] = num0 + j*estep[0];
         elem[1] = (*tmp)[0];
         elem[2] = (*tmp)[1];
         elem[3] = (*tmp)[4];
         elem[4] = (*tmp)[5];
         AddElement(in, elem);
         tmp += estep[0];
      }
      num0 += eoff;
   }                                              // end i, ge_num

   // **************************************************
   // horizontal balance plane, complicated and not very reliable.
   // ****** Runner outlet elements
   ix = ge[0]->ml->p->nump - npoin_ext;
   num0  = (cge[0]->reg[0]->line[0]->nump-1) * (cge[0]->reg[0]->line[1]->nump-1);
   num0 += (cge[0]->reg[1]->line[0]->nump-1) * (cge[0]->reg[1]->line[1]->nump-1);
   num0 += (cge[0]->reg[2]->line[0]->nump-1) * (cge[0]->reg[2]->line[1]->nump-1);
   num0 += (cge[0]->reg[3]->line[0]->nump-1) * (cge[0]->reg[3]->line[1]->nump-1);
   elnum[0] = cge[0]->reg[4]->line[0]->nump-1;
   elnum[1] = cge[0]->reg[5]->line[0]->nump-1;
   elnum[2] = cge[0]->reg[6]->line[0]->nump-1;
   estep[0] = cge[0]->reg[4]->line[1]->nump-1;
   estep[1] = cge[0]->reg[5]->line[1]->nump-1;
   estep[2] = cge[0]->reg[6]->line[1]->nump-1;
#ifdef DEBUG_BC
   fprintf(fp,"\noutlet:\n elnum[0], elnum[1], elnum[2]: %d, %d, %d\n",
      elnum[0], elnum[1], elnum[2]);
   fprintf(fp," estep[0], estep[1], estep[2]: %d, %d, %d\n",
      estep[0], estep[1], estep[2]);
   fprintf(fp," num0 = %d, ix = %d\n",num0, ix);
#endif
   portion = elnum[0]+elnum[1]+elnum[2];
   max     = portion;
   if( (belem = (int*)calloc(portion, sizeof(int))) == NULL)
   {
      fatal("memory for int*!");
      exit(-1);
   }
   if( (elen = (float*)calloc(portion, sizeof(float))) == NULL)
   {
      fatal("memory for float*!");
      exit(-1);
   }
#ifdef RROUTLET_NOGAP
   if( (jelem = (int*)calloc(portion, sizeof(int))) == NULL)
   {
      fatal("memory for float*!");
      exit(-1);
   }
   if( (step = (int*)calloc(portion, sizeof(int))) == NULL)
   {
      fatal("memory for float*!");
      exit(-1);
   }
#endif

   // find runner outlet elements, hub
   count = 0;
   len = ge[0]->ml->len[ix];
   // avrg. length for first elements in region
   for(l = 0; l < 3; l++)
   {
      elem[0] = num0;
      tmp     = e->e + num0;
#ifdef DEBUG_BC
      fprintf(fp," l: %d: num0 = %d\n",l, num0);
#endif
      for(j = 0; j < elnum[l]; j++)
      {
         avlen = 0;
         for(k = 0; k < 8; k++) avlen += n->n[(*tmp)[k]]->l;
         avlen *= 0.125;
         elen[j] = avlen;
#ifdef DEBUG_BC
         fprintf(fp," elen[%02d] = %f\n",j, elen[j]);
#endif
         tmp += estep[l];
         elem[0] += estep[l];
      }
      tmp     = e->e + num0;
      elem[0] = num0;
      for(i = 0; i < elnum[l]; i++)
      {
         elem[0] = num0 + i * estep[l];
         tmp     = e->e + elem[0];
         for(j = 0; j < estep[l]; j++)
         {
            avlen = 0;
            for(k = 0; k < 8; k++) avlen += n->n[(*tmp)[k]]->l;
            avlen *= 0.125;
            if(avlen > len && elen[i] < len)      // elem. found
            {
#ifdef DEBUG_BC
               fprintf(fp," i, j: %d, %d: len = %f, avlen = %f, count = %d\n",
                  i,j,len,avlen, count+1);
               fprintf(fp," %d",elem[0]+1);
               for(k = 0; k < 8; k++) fprintf(fp,"   %d",(*tmp)[k]+1);
               fprintf(fp,"\n");
#endif
#ifdef RROUTLET_NOGAP
               jelem[count] = j;
               step[count] = estep[l];
#endif
               belem[count++] = elem[0];
               elen[i] = avlen;
               break;
            }
            tmp++;
            elem[0]++;
         }
         continue;
      }                                           // end i
      num0 += elnum[l] * estep[l];
   }                                              // end l, regions 5-7

   for(i = 0; i < ege_num; i++)
   {
      tmp = e->e + belem[0];
      elem[0] = belem[0];
      elem[1] = (*tmp)[0];
      elem[2] = (*tmp)[1];
      elem[3] = (*tmp)[4];
      elem[4] = (*tmp)[5];
      AddElement(out, elem);
      if(i != ege_num-1) belem[0] += eoff;
      for(j = 1; j < count; j++)
      {
#ifdef DEBUG_BC
         fprintf(fp," belem[%03d] = %d\n", j, belem[j]);
#endif
         tmp = e->e + belem[j];
         elem[0] = belem[j];
         elem[1] = (*tmp)[0];
         elem[2] = (*tmp)[1];
         elem[3] = (*tmp)[4];
         elem[4] = (*tmp)[5];
         AddElement(out, elem);
#ifdef RROUTLET_NOGAP
         // avoid gaps
         if( ((dj = jelem[j] - jelem[j-1]) != 0) && (step[j] == step[j-1]))
         {
            sig = SIGN(dj);
            dj  = ABS(dj);
#ifdef DEBUG_BC
            fprintf(fp,"dj = %d, sig = %d\n", dj, sig);
#endif
            for(k = 0; k < dj; k++)
            {
               if(sig < 0)
               {
                  next_elem = (belem[j]) + k;
                  tmp = e->e + next_elem;
                  elem[0] = next_elem;
                  elem[1] = (*tmp)[0];
                  elem[2] = (*tmp)[3];
                  elem[3] = (*tmp)[4];
                  elem[4] = (*tmp)[7];
                  AddElement(out, elem);
               }
               else
               {
                  if(i != ege_num-1) next_elem = (belem[j-1]-eoff) + k;
                  else next_elem = (belem[j-1]) + k;
                  tmp = e->e + next_elem;
                  elem[0] = next_elem;
                  elem[1] = (*tmp)[1];
                  elem[2] = (*tmp)[2];
                  elem[3] = (*tmp)[5];
                  elem[4] = (*tmp)[6];
                  AddElement(out, elem);
               }
#ifdef DEBUG_BC
               fprintf(fp," i, j, k: %d, %d, %d: belem(j-1) = %d, next_elem = %d\n",
                  i, j, k, belem[j], next_elem);
#endif

            }

         }
#endif                                   // RROUTLET_NOGAP
         if(i != ege_num-1) belem[j] += eoff;

      }                                           // end j

   }                                              // end i

   free(belem);
   free(elen);
#ifdef RROUTLET_NOGAP
   free(jelem);
   free(step);
#endif

#ifdef DEBUG_BC
   fclose(fp);
   fprintf(stderr,"GetRRIONodes done!\n");
#endif
#ifdef PATRAN_SES_OUT
   if( (fppat = fopen(fnpat,"w+")) == NULL)
   {
      fprintf(stderr,"Could not open file '%s'!\n", fnpat);
      exit(-1);
   }
   fprintf(fppat,"uil_list_a.clear()\n");
   for(i = 0; i < in->nume; i++)
   {
      fprintf(fppat,"list_create_target_list(\"lista\",\"elm %d\")\n",
         1+in->e[i][0]);
      for(j = 1; j <= 4; j++)
      {
         fprintf(fppat,"list_create_target_list(\"lista\",\"node %d\")\n", in->e[i][j]+1);
      }
   }
   for(i = 0; i < out->nume; i++)
   {
      fprintf(fppat,"list_create_target_list(\"lista\",\"elm %d\")\n",
         1+out->e[i][0]);
      for(j = 1; j <= 4; j++)
      {
         fprintf(fppat,"list_create_target_list(\"lista\",\"node %d\")\n", out->e[i][j]+1);
      }
   }
   fprintf(fppat,"list_save_group(\"lista\",\"%s\",FALSE)\n", "runnerIO");
   fprintf(fppat,"uil_viewport_post_groups.posted_groups(\"default_viewport\",1,[\"%s\"])\n","runnerIO");
   fprintf(fppat,"gu_fit_view()\n");

   fclose(fppat);
#endif
   return(1);
}
#endif

#ifdef DEBUG_BC
int DumpBCElements(struct Element *bcelem, struct Nodelist *n, char *name)
{
   int i, j;

   char fn[111];
   FILE *fp;

   sprintf(fn,"rr_%s.txt",name);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   for(i = 0; i < bcelem->nume; i++)
   {
      fprintf(fp,"# %8d\n",bcelem->e[i][0]+1);
      for(j = 1; j < 3; j++)
      {
         fprintf(fp,"%8d %16.6f %16.6f %16.6f\n",bcelem->e[i][j]+1, n->n[bcelem->e[i][j]]->x,
            n->n[bcelem->e[i][j]]->y, n->n[bcelem->e[i][j]]->z);
      }
      for(j = 4; j >= 3 ; j--)
      {
         fprintf(fp,"%8d %16.6f %16.6f %16.6f\n",bcelem->e[i][j]+1, n->n[bcelem->e[i][j]]->x,
            n->n[bcelem->e[i][j]]->y, n->n[bcelem->e[i][j]]->z);
      }
      fprintf(fp,"%8d %16.6f %16.6f %16.6f\n\n\n",bcelem->e[i][1]+1, n->n[bcelem->e[i][1]]->x,
         n->n[bcelem->e[i][1]]->y, n->n[bcelem->e[i][1]]->z);
   }
   fclose(fp);

   return(0);
}
#endif

#ifdef DEBUG_ELEMENTS
int DumpElements(struct Nodelist *n, struct Element *e, int ge_num)
{
   int i, j, k;
   int **elem;
   int offset;

   char fn[111];
   FILE *fp;

   offset = e->nume/(ge_num-1);

   for(i = 0; i < ge_num-1; i++)
   {
      sprintf(fn,"rr_elems_%02d.txt", i);
      if( (fp = fopen(fn,"w+")) == NULL)
      {
         fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
         exit(-1);
      }
      fprintf(fp,"##  number of elements: %d (total), %d (per plane)\n\n", e->nume,offset);
      elem = e->e + i*offset;
      for(j = 0; j < offset; j++)
      {
         for(k = 0; k < 4; k++)
         {
            fprintf(fp,"  %8d   %10.6f   %10.6f   %10.6f\n",
               n->n[(*elem)[k]]->id, n->n[(*elem)[k]]->x,
               n->n[(*elem)[k]]->y,  n->n[(*elem)[k]]->z);
         }
         fprintf(fp,"  %8d   %10.6f   %10.6f   %10.6f\n",
            n->n[(*elem)[0]]->id, n->n[(*elem)[0]]->x,
            n->n[(*elem)[0]]->y,  n->n[(*elem)[0]]->z);
         for(k = 4; k < 8; k++)
         {
            fprintf(fp,"  %8d   %10.6f   %10.6f   %10.6f\n",
               n->n[(*elem)[k]]->id, n->n[(*elem)[k]]->x,
               n->n[(*elem)[k]]->y,  n->n[(*elem)[k]]->z);
         }
         fprintf(fp,"  %8d   %10.6f   %10.6f   %10.6f\n",
            n->n[(*elem)[4]]->id, n->n[(*elem)[4]]->x,
            n->n[(*elem)[4]]->y,  n->n[(*elem)[4]]->z);
         for(k = 1; k < 4; k++)
         {
            fprintf(fp,"\n  %8d   %10.6f   %10.6f   %10.6f\n",
               n->n[(*elem)[k]]->id, n->n[(*elem)[k]]->x,
               n->n[(*elem)[k]]->y,  n->n[(*elem)[k]]->z);
            fprintf(fp,"  %8d   %10.6f   %10.6f   %10.6f\n",
               n->n[(*elem)[k+4]]->id, n->n[(*elem)[k+4]]->x,
               n->n[(*elem)[k+4]]->y,  n->n[(*elem)[k+4]]->z);
         }
         elem++;
         fprintf(fp,"\n\n");
      }
      fclose(fp);
   }
   return 0;
}
#endif

#ifdef MESH_2DMERIDIAN_OUT
int Write_2DMeridianMesh(struct Nodelist *n, struct Element *e, int ge_num)
{
   static int put2DElements(struct Nodelist *n, int **elem,
      int eoffset, int k0, FILE *fp);
   int i, ii;
   int eoffset;

   int **elem;

   char fn[123];
   FILE *fp;

   // **************************************************
   // to reduce size of file
   ii = 1;
   if(ge_num > 10) ii = ge_num/2;

   eoffset = e->nume / (ge_num-1);

   // **************************************************
   for(i = 0; i < ge_num; i+=ii)
   {
      sprintf(fn,"rr_mesh2Dmeridian_%02d.dat",i);
      if( (fp = fopen(fn,"w+")) == NULL)
      {
         fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
         exit(-1);
      }
      if(i == ge_num-1)
      {
         elem = e->e + (i-1)*eoffset;
         put2DElements(n,elem,eoffset,4,fp);
      }
      else
      {
         elem = e->e + i*eoffset;
         put2DElements(n,elem,eoffset,0,fp);
      }
      fclose(fp);
   }
   return 0;
}


static int put2DElements(struct Nodelist *n, int **elem,
int eoffset, int k0, FILE *fp)
{
   int j, k, kk;
   for(j = 0; j < eoffset; j++)
   {
      fprintf(fp,"# %5d\n",j+1);
      for(k = 0; k <= 4; k++)
      {
         kk = k%4+k0;
         fprintf(fp," %6d %12.4e %12.4e\n",
            n->n[(*elem)[kk]]->id,
            n->n[(*elem)[kk]]->arc,
            n->n[(*elem)[kk]]->l);
      }
      fprintf(fp,"\n\n");
      elem++;
   }
   return 0;
}
#endif
