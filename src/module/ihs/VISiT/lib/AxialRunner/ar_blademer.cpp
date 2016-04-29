#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../General/include/geo.h"
#include "../General/include/log.h"
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/curve.h"
#include "../General/include/plane_geo.h"
#include "include/axial.h"
#include "include/ar_blademer.h"
#include "include/ar_contours.h"

int BladeMeridianIntersection(struct curve *bl, struct axial *ar, struct Point *intersection)
{
   int i, j, k;
   int found = 0;                                 // flag if segment is found
   int loopstop = 0;
   int next_edge_segment = 0;                     // flag if segment has no intersection
   int jstart = 1;                                // meridian curve to start next search
   int kfound = 0;                                // number of last found segment
   int depth = 5;                                 // search depth in one direction
   int upper, lower;                              // upper and lower search start bounds
   float b[3], bvec[3];                           // edge segment start point and vector
   float bmin, bmax;                              // min/max y coord of edge segment
   float m[3], mvec[3];                           // meridian segment start point and vector
   float mmin, mmax;                              // min/max y coord of meridian segment
   float inter[3];                                // segment intersection
   float dvec1[3], dvec2[3];                      // edge start point to meridian start point vectors
   float vprod1, vprod2;                          // vector product z coordinate
   float par;                                     // parameter value of found intersection
   float tolerance = 1.0e-6f;                     // shift tolerance in case bmin == bmax
#ifdef DEBUG_BLMER_INTERSECTION
   FILE *fp=NULL, *fm=NULL;
   char fname[255];
   int nedge = 0;
   static int ncall = 0;
   char *fn;

   inter[0] = inter[1] = inter[2] = 0.0;
   mvec[0] = mvec[1] = mvec[2] = 0.0;
   sprintf(fname, "ar_blmerint_%02d.txt", ncall++);
   fn = DebugFilename(fname);
   if (!fn || !*fn || (fm = fopen(fn, "w")) == NULL)
      dprintf(0, "cannot open file %s\n", fname);
#endif                                         // DEBUG_BLMER_INTERSECTION

   // search all blade edge contour points
   if (ar->mod->inl)
   {
      kfound = NPOIN_SPLN_INLET + NPOIN_SPLN_BEND + (int)(floor(0.1 * NPOIN_SPLN_CORE));
   }
   else
   {
      kfound = (int)(floor(0.1 * NPOIN_SPLN_CORE));
   }

   for (i = 0; i < bl->p->nump - 1; i++)
   {
      loopstop = 0;
#ifdef DEBUG_BLMER_INTERSECTION
      sprintf(fname, "ar_intersect%02d_%02d.txt", nedge++, (ncall-1));
      fn = DebugFilename(fname);
      if (!fn || !*fn || (fp = fopen(fn, "w")) == NULL)
         dprintf(0, "cannot open file %s\n", fname);
      if (fp)  fprintf(fp, "\nBLADE LINE SEGMENT %d\n", i);
#endif                                      // DEBUG_BLMER_INTERSECTION
      found = 0;
      next_edge_segment = 0;

      // blade edge segment vector
      b[0]    = bl->p->x[i];
      b[1]    = bl->p->y[i];
      b[2]    = bmin = bmax = bl->p->z[i];
      bvec[0] = bl->p->x[i+1] - b[0];
      bvec[1] = bl->p->y[i+1] - b[1];
      bvec[2] = bl->p->z[i+1] - b[2];
      ((b[2] + bvec[2]) > bmin) ? (bmax = b[2] + bvec[2]) : (bmin = b[2] + bvec[2]);
      bmin -= tolerance;
      bmax += tolerance;

      //search all meridian contour curves
      for (j = jstart; j < ar->be_num-1; j++)
      {
#ifdef DEBUG_BLMER_INTERSECTION
         if (fp) fprintf(fp, "\nMERIDIAN CURVE %d\n", j);
         if (fp) fprintf(fp, "jstart = %d\n", jstart);
#endif                                   // DEBUG_BLMER_INTERSECTION
         found = 0;
         upper = lower = kfound;

         // search meridian contour curve until
         //  a) intersection is found, or
         //  b) sign change of vector product (no intersection)
         while (!next_edge_segment && !found)
         {

            // search ascending
            for (k = upper; k < (upper+depth); k++)
            {
               if (k >= ar->me[j]->ml->p->nump-1)
               {
                  dprintf(0, "WARNING: k >= ar->me[j]->ml->p->nump-1, loop stopped\n");
                  if(++loopstop == 10)
                     return BLADE_MERID_ERR;
                  break;
               }

#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "##################################################\n");
                  fprintf(fp, "# ascending search:\n");
                  fprintf(fp, "#  kfound = %d\t", kfound);
                  fprintf(fp, "kstart/kend = %d/%d\t", upper, (upper+depth));
                  fprintf(fp, "k = %d\n", k);
                  fprintf(fp, "##################################################\n\n");
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               // meridian segment vector
               m[0]    = ar->me[j]->ml->p->x[k];
               m[1]    = ar->me[j]->ml->p->y[k];
               m[2]    = mmin = mmax = ar->me[j]->ml->p->z[k];
               dprintf(10, "i=%d, j=%d, k=%d, ar->be_num=%d, ar->me[%d]->ml->p->nump=%d\n", i, j, k, ar->be_num, j, ar->me[j]->ml->p->nump);
               if (k >= ar->me[j]->ml->p->nump)
               {
                  dprintf(0, "k=%d >= (upper+depth)=%d\n", k, upper+depth);
                  exit(1);
               }
               dprintf(10, "ar->me[%d]->ml->p->x[%d+1]=%f, m[0]=%f\n", j, k, ar->me[j]->ml->p->x[k+1], m[0]);
               mvec[0] = ar->me[j]->ml->p->x[k+1] - m[0];
               mvec[1] = ar->me[j]->ml->p->y[k+1] - m[1];
               mvec[2] = ar->me[j]->ml->p->z[k+1] - m[2];
               ((m[2] + mvec[2]) > mmin) ? (mmax = m[2] + mvec[2]) : (mmin = m[2] + mvec[2]);
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "edge segment i=%d\n", i);
                  fprintf(fp, " b(i)      : %8.6f, %8.6f, %8.6f\n", b[0], b[1], b[2]);
                  fprintf(fp, " b(i+1)    : %8.6f, %8.6f, %8.6f\n", b[0]+bvec[0], b[1]+bvec[1], b[2]+bvec[2]);
                  fprintf(fp, " bmin, bmax: %8.6f, %8.6f\n", bmin, bmax);
                  fprintf(fp, "meridian segment k=%d\n", k);
                  fprintf(fp, " m(k)      : %8.6f, %8.6f, %8.6f\n", m[0], m[1], m[2]);
                  fprintf(fp, " m(k+1)    : %8.6f, %8.6f, %8.6f\n", m[0]+mvec[0], m[1]+mvec[1], m[2]+mvec[2]);
                  fprintf(fp, " mmin, mmax: %8.6f, %8.6f\n", mmin, mmax);
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               // intersection and location check on segments
               LineIntersectXZ(b, bvec, m, mvec, inter);
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "intersection\n");
                  fprintf(fp, " inter     : %8.6f, %8.6f, %8.6f\n", inter[0], inter[1], inter[2]);
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               if ((inter[2] > bmin) && (inter[2] <= bmax) &&
                  (inter[2] > mmin) && (inter[2] <= mmax))
               {
                  found  = 1;
                  AddVPoint(intersection, inter);
                  par  = (ar->me[j]->ml->par[k+1] - ar->me[j]->ml->par[k]);
                  par /= (ar->me[j]->ml->p->z[k+1] - ar->me[j]->ml->p->z[k]);
                  par *= (inter[2] - ar->me[j]->ml->p->z[k]);
                  par += ar->me[j]->ml->par[k];
                  upper = lower = kfound = k;
                  jstart++;
#ifdef DEBUG_BLMER_INTERSECTION
                  if (fp)  fprintf(fm, "%8.6f %8.6f %8.6f %8.6f\n", bl->p->x[j], bl->p->y[j], bl->p->z[j], par);
                  if (fp)  fprintf(fp, " par = %8.6f\n", par);
#else
                  break;
#endif                          // DEBUG_BLMER_INTERSECTION
               }
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp) fprintf(fp, " FOUND = %d\n\n", found);
               if (found) break;
#endif                             // DEBUG_BLMER_INTERSECTION
               // vectors to meridian points from blade segment start point
               dvec1[0] = m[0] - b[0];
               dvec1[1] = m[1] - b[1];
               dvec1[2] = m[2] - b[2];
               dvec2[0] = m[0] + mvec[0] - b[0];
               dvec2[1] = m[1] + mvec[1] - b[1];
               dvec2[2] = m[2] + mvec[2] - b[2];
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "edge meridian vectors:\n");
                  fprintf(fp, " dvec1: %8.6f %8.6f %8.6f\n", dvec1[0], dvec1[1], dvec1[2]);
                  fprintf(fp, " dvec2: %8.6f %8.6f %8.6f\n", dvec2[0], dvec2[1], dvec2[2]);
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               // vector products with blade segment vector
               vprod1   = bvec[2] * dvec1[0] - bvec[0] * dvec1[2];
               vprod2   = bvec[2] * dvec2[0] - bvec[0] * dvec2[2];
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "vector product (y coord):\n");
                  fprintf(fp, " vprod1 = %8.6f\n vprod2 = %8.6f\n", vprod1, vprod2);
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               if ((vprod1 * vprod2) < 0.0)
               {
                  next_edge_segment = 1;
#ifndef DEBUG_BLMER_INTERSECTION
                  break;
#endif
               }
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp) fprintf(fp, " NEXT_EDGE_SEGMENT = %d\n\n", next_edge_segment);
#endif                             // DEBUG_BLMER_INTERSECTION
               if (next_edge_segment) break;
            }
            if (next_edge_segment || found) break;

#ifdef DEBUG_BLMER_INTERSECTION
            if (fp)  fprintf(fp, "\n\n");
#endif                                // DEBUG_BLMER_INTERSECTION
            // search descending
            for (k = (lower-1); k > (lower-1-depth); k--)
            {
               if (k < 0)
               {
                  //dprintf(0, "WARNING: k < 0, loop stopped\n");
                  if(++loopstop == 10)
                     return BLADE_MERID_ERR;

                  break;
               }
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "##################################################\n");
                  fprintf(fp, "# descending search:\n");
                  fprintf(fp, "#  kfound = %d\t", kfound);
                  fprintf(fp, "kstart/kend = %d/%d\t", (lower-1), (lower-1-depth));
                  fprintf(fp, "k = %d\n", k);
                  fprintf(fp, "##################################################\n\n");
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               // meridian segment vector
               m[0]    = ar->me[j]->ml->p->x[k];
               m[1]    = ar->me[j]->ml->p->y[k];
               m[2]    = mmin = mmax = ar->me[j]->ml->p->z[k];
               mvec[0] = ar->me[j]->ml->p->x[k+1] - m[0];
               mvec[1] = ar->me[j]->ml->p->y[k+1] - m[1];
               mvec[2] = ar->me[j]->ml->p->z[k+1] - m[2];
               ((m[2] + mvec[2]) > mmin) ? (mmax = m[2] + mvec[2]) : (mmin = m[2] + mvec[2]);
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "edge segment i=%d\n", i);
                  fprintf(fp, " b(i)      : %8.6f, %8.6f, %8.6f\n", b[0], b[1], b[2]);
                  fprintf(fp, " b(i+1)    : %8.6f, %8.6f, %8.6f\n", b[0]+bvec[0], b[1]+bvec[1], b[2]+bvec[2]);
                  fprintf(fp, " bmin, bmax: %8.6f, %8.6f\n", bmin, bmax);
                  fprintf(fp, "meridian segment k=%d\n", k);
                  fprintf(fp, " m(k)      : %8.6f, %8.6f, %8.6f\n", m[0], m[1], m[2]);
                  fprintf(fp, " m(k+1)    : %8.6f, %8.6f, %8.6f\n", m[0]+mvec[0], m[1]+mvec[1], m[2]+mvec[2]);
                  fprintf(fp, " mmin, mmax: %8.6f, %8.6f\n", mmin, mmax);
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               // intersection and location check on segments
               LineIntersectXZ(&b[0], &bvec[0], &m[0], &mvec[0], &inter[0]);
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "intersection\n");
                  fprintf(fp, " inter     : %8.6f, %8.6f, %8.6f\n", inter[0], inter[1], inter[2]);
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               if ((inter[2] > bmin) && (inter[2] < bmax) &&
                  (inter[2] > mmin) && (inter[2] < mmax))
               {
                  found  = 1;
                  AddVPoint(intersection, inter);
                  par  = (ar->me[j]->ml->par[k+1] - ar->me[j]->ml->par[k]);
                  par /= (ar->me[j]->ml->p->z[k+1] - ar->me[j]->ml->p->z[k]);
                  par *= (inter[2] - ar->me[j]->ml->p->z[k]);
                  par += ar->me[j]->ml->par[k];
                  upper = lower = kfound = k;
                  jstart++;
#ifdef DEBUG_BLMER_INTERSECTION
                  if (fm)  fprintf(fm, "%8.6f %8.6f %8.6f\n", bl->p->x[j], bl->p->y[j], bl->p->z[j]);
                  if (fp)  fprintf(fp, " par = %8.6f\n", par);
#else
                  break;
#endif                          // DEBUG_BLMER_INTERSECTION
               }
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp) fprintf(fp, " FOUND = %d\n\n", found);
               if (found) break;
#endif                             // DEBUG_BLMER_INTERSECTION
               // vectors to meridian points from blade segment start point
               dvec1[0] = m[0] - b[0];
               dvec1[1] = m[1] - b[1];
               dvec1[2] = m[2] - b[2];
               dvec2[0] = m[0] + mvec[0] - b[0];
               dvec2[1] = m[1] + mvec[1] - b[1];
               dvec2[2] = m[2] + mvec[2] - b[2];
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "edge meridian vectors:\n");
                  fprintf(fp, " dvec1: %8.6f %8.6f %8.6f\n", dvec1[0], dvec1[1], dvec1[2]);
                  fprintf(fp, " dvec2: %8.6f %8.6f %8.6f\n", dvec2[0], dvec2[1], dvec2[2]);
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               // vector products with blade segment vector
               vprod1   = bvec[2] * dvec1[0] - bvec[0] * dvec1[2];
               vprod2   = bvec[2] * dvec2[0] - bvec[0] * dvec2[2];
#ifdef DEBUG_BLMER_INTERSECTION
               if (fp)
               {
                  fprintf(fp, "vector product (y coord):\n");
                  fprintf(fp, " vprod1 = %8.6f\n vprod2 = %8.6f\n", vprod1, vprod2);
               }
#endif                             // DEBUG_BLMER_INTERSECTION
               if ((vprod1 * vprod2) < 0.0)
               {
                  next_edge_segment = 1;
#ifdef DEBUG_BLMER_INTERSECTION
                  if (fp)  fprintf(fp, " NEXT_EDGE_SEGMENT = %d\n\n", next_edge_segment);
#endif                          // DEBUG_BLMER_INTERSECTION
                  break;
               }
            }
            if (next_edge_segment)  break;
            // restart search with shifted upper/lower start point, same i,j
            if (!found)
            {
               upper += depth;
               lower -= depth;
            }
         }
         if (next_edge_segment)
            break;
      }
#ifdef DEBUG_BLMER_INTERSECTION
      if (fp)  fclose(fp);
#endif                                      // DEBUG_BLMER_INTERSECTION
   }
#ifdef DEBUG_BLMER_INTERSECTION
   if (fm) fclose(fm);
#endif                                         // DEBUG_BLMER_INTERSECTION
   return 0;
}


// new version with double vector product check!
int BladeMeridianIntersection2(struct curve *bl, struct axial *ar,
struct Point *intersection)
{
   int i, j, k;
   int found = 0;                                 // flag if segment is found
   int loopstop = 0;
   int next_edge_segment = 0;                     // flag if segment has no intersection
   int jstart = 1;                                // meridian curve to start next search
   int kfound = 0;                                // number of last found segment
   int depth = 5;                                 // search depth in one direction
   int upper, lower;                              // upper and lower search start bounds
   float b[3], bvec[3];                           // edge segment start point and vector
   float m[3], mvec[3];                           // meridian segment start point and vector
   float mmin, mmax;                              // min/max y coord of meridian segment
   float inter[3];                                // segment intersection
   float dvec1[3], dvec2[3];                      // edge start to meridian start vectors
   float vprod1, vprod2;                          // vector product z coordinate
   inter[0] = inter[1] = inter[2] = 0.0;
   mvec[0] = mvec[1] = mvec[2] = 0.0;

   // search all blade edge contour points
   if (ar->mod->inl)
   {
      kfound = NPOIN_SPLN_INLET + NPOIN_SPLN_BEND
         + (int)(floor(0.1 * NPOIN_SPLN_CORE));
   }
   else
   {
      kfound = (int)(floor(0.1 * NPOIN_SPLN_CORE));
   }

   for (i = 0; i < bl->p->nump - 1; i++)
   {
      loopstop = 0;
      found = 0;
      next_edge_segment = 0;

      // blade edge segment vector
      b[0]    = bl->p->x[i];
      b[1]    = bl->p->y[i];
      b[2]    = bl->p->z[i];
      bvec[0] = bl->p->x[i+1] - b[0];
      bvec[1] = bl->p->y[i+1] - b[1];
      bvec[2] = bl->p->z[i+1] - b[2];

      //search all meridian contour curves
      for (j = jstart; j < ar->be_num-1; j++)
      {
         found = 0;
         upper = lower = kfound;

         while (!found)
         {

            // search ascending
            for (k = upper; k < (upper+depth); k++)
            {
               if (k >= ar->me[j]->ml->p->nump-1)
               {
                  dprintf(0, "WARNING: k >= ar->me[j]->ml->p->nump-1, loop stopped\n");
                  if(++loopstop == 10)
                     return BLADE_MERID_ERR;
                  break;
               }

               // meridian segment vector
               m[0]    = ar->me[j]->ml->p->x[k];
               m[1]    = ar->me[j]->ml->p->y[k];
               m[2]    = mmin = mmax = ar->me[j]->ml->p->z[k];
               mvec[0] = ar->me[j]->ml->p->x[k+1] - m[0];
               mvec[1] = ar->me[j]->ml->p->y[k+1] - m[1];
               mvec[2] = ar->me[j]->ml->p->z[k+1] - m[2];
               // vectors to meridian points from blade segment start point
               dvec1[0] = m[0] - b[0];
               dvec1[1] = m[1] - b[1];
               dvec1[2] = m[2] - b[2];
               dvec2[0] = m[0] + mvec[0] - b[0];
               dvec2[1] = m[1] + mvec[1] - b[1];
               dvec2[2] = m[2] + mvec[2] - b[2];
               // vector products with blade segment vector
               vprod1   = bvec[2] * dvec1[0] - bvec[0] * dvec1[2];
               vprod2   = bvec[2] * dvec2[0] - bvec[0] * dvec2[2];
               if ((vprod1 * vprod2) > 0.0) continue;
               else
               {
                  // vector prod for meridian section vector
                  dvec1[0] *= -1.0;
                  dvec1[1] *= -1.0;
                  dvec1[2] *= -1.0;
                  dvec2[0] = b[0] + bvec[0] - m[0];
                  dvec2[1] = b[1] + bvec[1] - m[1];
                  dvec2[2] = b[2] + bvec[2] - m[2];
                  vprod1   = mvec[2] * dvec1[0] - mvec[0] * dvec1[2];
                  vprod2   = mvec[2] * dvec2[0] - mvec[0] * dvec2[2];
                  if((vprod1*vprod2) > 0.0) next_edge_segment = 1;
                  else
                  {
                     found = 1;
                     LineIntersectXZ(b, bvec, m, mvec, inter);
                     AddVPoint(intersection, inter);
                     upper = lower = kfound = k;
                     jstart++;
                  }
               }
               if(next_edge_segment) break;
            }                                     // k
            if (next_edge_segment || found) break;

            // search descending
            for (k = (lower-1); k > (lower-1-depth); k--)
            {
               if (k < 0)
               {
                  //dprintf(0, "WARNING: k < 0, loop stopped\n");
                  if(++loopstop == 10)
                     return BLADE_MERID_ERR;

                  break;
               }
               // meridian segment vector
               m[0]    = ar->me[j]->ml->p->x[k];
               m[1]    = ar->me[j]->ml->p->y[k];
               m[2]    = mmin = mmax = ar->me[j]->ml->p->z[k];
               mvec[0] = ar->me[j]->ml->p->x[k+1] - m[0];
               mvec[1] = ar->me[j]->ml->p->y[k+1] - m[1];
               mvec[2] = ar->me[j]->ml->p->z[k+1] - m[2];
               // vectors to meridian points from blade segment start point
               dvec1[0] = m[0] - b[0];
               dvec1[1] = m[1] - b[1];
               dvec1[2] = m[2] - b[2];
               dvec2[0] = m[0] + mvec[0] - b[0];
               dvec2[1] = m[1] + mvec[1] - b[1];
               dvec2[2] = m[2] + mvec[2] - b[2];
               // vector products with blade segment vector
               vprod1   = bvec[2] * dvec1[0] - bvec[0] * dvec1[2];
               vprod2   = bvec[2] * dvec2[0] - bvec[0] * dvec2[2];
               if ((vprod1 * vprod2) > 0.0) continue;
               else
               {
                  // vector prod for meridian section vector
                  dvec1[0] *= -1.0;
                  dvec1[1] *= -1.0;
                  dvec1[2] *= -1.0;
                  dvec2[0] = b[0] + bvec[0] - m[0];
                  dvec2[1] = b[1] + bvec[1] - m[1];
                  dvec2[2] = b[2] + bvec[2] - m[2];
                  vprod1   = mvec[2] * dvec1[0] - mvec[0] * dvec1[2];
                  vprod2   = mvec[2] * dvec2[0] - mvec[0] * dvec2[2];
                  if((vprod1*vprod2) > 0.0) next_edge_segment = 1;
                  else
                  {
                     found = 1;
                     LineIntersectXZ(b, bvec, m, mvec, inter);
                     AddVPoint(intersection, inter);
                     upper = lower = kfound = k;
                     jstart++;
                  }
               }
               if(next_edge_segment) break;
            }                                     // k
            // restart search with shifted upper/lower start point, same i,j
            if (!found)
            {
               upper += depth;
               lower -= depth;
            }
         }                                        // while !found
         if (next_edge_segment) break;
      }                                           // j
   }                                              // i
   return 0;
}
