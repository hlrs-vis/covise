#ifdef ADD_DT
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/points.h"
#include "../General/include/nodes.h"
#include "../General/include/fatal.h"
#include "../General/include/bias.h"

#include "include/rr_grid.h"

// creates nodes and Elements in Draft tube extension:
// first the part behind the hub will be meshed (same z-coord. as outlet)
// then these nodes will be kept as a reference. For the two cross sections
// (opening angle 6->8 deg and dt-outlet) in the dt. the radii will be scaled
// with the reference nodes' radius. Meshing is along a vector from cross-sect.
// to the next cross section. The elements will be created as soon as the nodes
// are available, as well as the boundary conditions.
int CreateRR_DTNodsElms(struct Nodelist *n, struct Element *e, struct Ilist *outnodes,
struct Element *outelms, struct Element *wallelms,
struct Element *shroudelms, struct Element *ssperielms,
struct Element *psperielms, int ge_num, float delta_z)
{

   int i;
   int nnum_new, elnum_new, elem[8], i_mid;
   int dis[2], type[2], npr, npcs, numn_in;

   float bias_val[2], alpha[2], z[2], r_out;
   float r_min, r_max, z_out, r_scale;
   float p1[3], p2[3], p[3], v[3], r[3];

   struct Point *wall;
   struct Flist *bias;
   struct Ilist *nodes;

   struct Point *cs[3];

#ifdef DEBUG_ADDDT
   char *fn = "debugDTadd.txt";
   FILE *fp = NULL;

   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"\n couldn't open file '%s'!\n\n",fn);
      exit(-1);
   }
   fprintf(stderr,"\nCreateRR_DTNodsElms() ... ");fflush(stderr);
#endif

   // **************************************************
   // init discretization and geometry values
   dis [0]     =  50;
   type[0]     =  1;
   bias_val[0] = -10.0;
   dis [1]     =  20;
   type[1]     =  1;
   bias_val[1] = -2.0;

   z[0]        =  1.832-delta_z;
   z[1]        =  3.664-delta_z;
   alpha[0]    =  6.0  * M_PI/180;
   alpha[1]    =  8.47 * M_PI/180;

   p[2] = p1[2] = p2[2] = v[2] = 0.0;

   // get min. and max. radius
   r_min = n->n[outnodes->list[0]]->r;            // first outlet node
                                                  // last
   r_max = n->n[outnodes->list[outnodes->num-1]]->r;
   z_out = n->n[outnodes->list[0]]->z;
   r[0]  = r_max;
   r[1]  = r_max + z[0]*tan(alpha[0]);
   r[2]  = 1.531/2;
   nnum_new  = n->num;
   elnum_new = e->nume;
   npr       = outnodes->num/ge_num;              // nodes per radius (= meridian)
#ifdef DEBUG_ADDDT
   fprintf(fp,"# ge_num = %d, outnodes->num = %d\n",ge_num,outnodes->num);
   fprintf(fp,"# npr = %d (nodes per radius (= meridian)\n", npr);
   fprintf(fp,"# z_out = %10.6f, delta_z = %10.6f\n", z_out, delta_z);
   fprintf(fp,"# r_min, r_max: %10.6f, %10.6f\n",r_min, r_max);
#endif
   // **************************************************
   // get nodes at outlet for 0.0 < r < r_min (numn_in).
   nodes = CopyIlistStruct(outnodes);
   // nodes in reference plane, outlet
   cs[0] = AllocPointStruct();
   for(i = outnodes->num-1; i >= 0; i--)
   {
      AddPoint(cs[0], n->n[outnodes->list[i]]->r,
         n->n[outnodes->list[i]]->phi,
         n->n[outnodes->list[i]]->z);
   }
   i_mid = (int)(npr/2);
   p1[0] = r_min;

   // **************************************************
   // first part of draft tube, 6 deg opening angle
   //
   bias  = CalcBladeElementBias(dis[0], 0.0, 1.0, type[0], bias_val[0]);
   wall  = AllocPointStruct();
   p1[0] = r_max;
   p1[1] = z_out;
   p2[0] = p1[0] + z[0]*tan(alpha[0]);
   p2[1] = p1[1] - z[0];
   v[0]  = p2[0] - p1[0];
   v[1]  = p2[1] - p1[1];
#ifdef DEBUG_ADDDT
   fprintf(fp,"# dt 1.: p1 = [%10.6f, %10.6f, %10.6f]\n",
      p1[0], p1[1], p1[2]);
   fprintf(fp,"# dt 1.: p2 = [%10.6f, %10.6f, %10.6f]\n",
      p2[0], p2[1], p2[2]);
   fprintf(fp,"# dt 1.: v  = [%10.6f, %10.6f, %10.6f]\n",
      v[0], v[1], v[2]);
   fprintf(fp,"\n\n %10.6f  %10.6f  %10.6f\n",p1[0], p1[1], p1[2]);
#endif
   for(i = 1; i < bias->num; i++)
   {
      p[0] = p1[0] + bias->list[i] * v[0];
      p[1] = p1[1] + bias->list[i] * v[1];
      fprintf(fp," %10.6f  %10.6f  %10.6f\n",p[0], p[1], p[2]);
      AddVPoint(wall,p);
   }
   FreeFlistStruct(bias);
   // **************************************************
   // second part of draft tube, 6 deg opening angle
   bias  = CalcBladeElementBias(dis[1], 0.0, 1.0, type[1], bias_val[1]);
   wall  = AllocPointStruct();
   p1[0] = p2[0];
   p1[1] = p2[1];
   p2[0] = p1[0] + (z[1]-z[0])*tan(alpha[1]);
   p2[1] = p1[1] - (z[1]-z[0]);
   v[0]  = p2[0] - p1[0];
   v[1]  = p2[1] - p1[1];
#ifdef DEBUG_ADDDT
   fprintf(fp,"# dt 2.: p1 = [%10.6f, %10.6f, %10.6f]\n",
      p1[0], p1[1], p1[2]);
   fprintf(fp,"# dt 2.: p2 = [%10.6f, %10.6f, %10.6f]\n",
      p2[0], p2[1], p2[2]);
   fprintf(fp,"# dt 2.: v  = [%10.6f, %10.6f, %10.6f]\n",
      v[0], v[1], v[2]);
   fprintf(fp,"\n\n %10.6f  %10.6f  %10.6f\n",p1[0], p1[1], p1[2]);
#endif
   for(i = 1; i < bias->num; i++)
   {
      p[0] = p1[0] + bias->list[i] * v[0];
      p[1] = p1[1] + bias->list[i] * v[1];
#ifdef DEBUG_ADDDT
      fprintf(fp," %10.6f  %10.6f  %10.6f\n",p[0], p[1], p[2]);
#endif
      AddVPoint(wall,p);
   }
   FreeFlistStruct(bias);

#ifdef DEBUG_ADDDT
   fprintf(stderr,"done!\n");
   fclose(fp);
#endif

   FreePointStruct(wall);
   FreePointStruct(cs[0]);
   //FreePointStruct(cs[1]);
   //FreePointStruct(cs[2]);

   CalcNodeCoords(&n->n[nnum_new], n->num - nnum_new);
   nnum_new  = n->num - nnum_new;
   elnum_new = e->nume - elnum_new;
   fprintf(stdout,"\nAdditional draft tube nodes/elements: %d / %d\n",
      nnum_new, elnum_new);
   fprintf(stdout,"New total number of nodes/elements: %d / %d\n",
      n->num, e->nume);

   return 0;
}
#endif                                            // ADD_DT
