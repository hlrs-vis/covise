#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../General/include/geo.h"

#include "include/radial.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/flist.h"
#include "../General/include/plane_geo.h"
#include "../General/include/parameter.h"
#include "../General/include/profile.h"
#include "../General/include/bias.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#define SHROUD_EXT      0.4                       // extension factor of shroud height
#define SHROUD_R_SCAL   0.5                       // scale factor, r-coord., shroud ext. end point
#define HUB_EXT_RAD     0.05f                      // radius at hub extension end
#define IN_EXT_H        0.05                      // height factor for inlet ext.
#define IN_EXT_R        0.2                       // radius factor for inlet ext.
#define BSPLN_DEGREE 3                            // bspline degree

#ifdef DEBUG_MERIDIANS
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

// **************************************************
int CreateDR_MeridianContours(struct radial *rr)
{
   int i, j;
   float rref, rsph[2], z_trans, sec;
   float alpha, dalpha, beta, dbeta, delta, phi;
   float p[3], p1[3], p2[3], p3[3];
   float u[3], u1[3], u3[3], s_end[3], s_ext[3], h_ext[3];
   float v[3], v1[3], v2[3], v3[3], v4[3];
   float cp[3], s1[3], s2[3], s3[3], s4[3];
   float part[2];
   v[0]=v[1]=v[2]=0.0f;

   struct Point *h_poly = NULL;
   struct Flist *h_knot = NULL;
   struct Point *s_poly = NULL;
   struct Flist *s_knot = NULL;

#ifndef NO_INLET_EXT
   int p_max, be_max;
   float t;
#endif

#ifdef DEBUG_MERIDIANS
   char fn[119];
   FILE *fp;

   sprintf(fn,"dr_debugmerids.txt");
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"file '%s'!\n",fn);
      exit(-1);
   }
#endif

   // **************************************************
   // spline polygon partition pars.
   part[0] = part[1] = 0.5;
   // reference RADIUS!
   rref = rr->ref * 0.5;

   // free + allocate memory
   if (rr->le->c)
   {
      FreeCurveStruct(rr->le->c);
      rr->le->c = NULL;
   }
   rr->le->c = AllocCurveStruct();
   if (rr->te->c)
   {
      FreeCurveStruct(rr->te->c);
      rr->te->c = NULL;
   }
   rr->te->c = AllocCurveStruct();

   // **************************************************
   // sphere start and end points, contour later
   p[1] = s1[1] = s2[1] = s3[1] = s4[1] = 0.0;
   // shroud end point p2 & start point p1
   s2[0] = rref * rr->sphdiam[1];
   s2[1] = rr->ref * rr->ospheight;
   s1[0] = rref * rr->sphdiam[0];
   s1[1] = s2[1] +  rr->ref * rr->spheight;

   // find center of sphere on axis, r = 0
   cp[1] = p[1] = v[1] = 0.0;
   v[0]  =  s1[0] - s2[0];
   v[1]  =  s1[1] - s2[1];
   // center pt. of chord s2-s1
   p[0]  =  s2[0] + 0.5 * v[0];
   p[1]  =  s2[1] + 0.5 * v[1];
   // center pt. of sphere on y-axis
   cp[0] = 0.0;
   cp[1] = p[1] + (p[0]/v[1]) * v[0];

   v[1] = v1[1] = v2[1] = v3[1] = v4[1] = 0.0;
   v1[0] = cp[0] - s1[0];
   v1[1] = cp[1] - s1[1];
   v2[0] = cp[0] - s2[0];
   v2[1] = cp[1] - s2[1];
   rsph[1]  = V_Len(v1);                          // sphere radius, shroud
   V_MultScal(v1,1/rsph[1]);
   V_MultScal(v2,1/rsph[1]);

   // sphere hub points, p3, p4
   if(rr->stpara[0])
   {
      p[0]   =  s1[0] + rr->ref * rr->sphcond * v1[0];
      p[1]   =  s1[1] + rr->ref * rr->sphcond * v1[1];
      v[0]   =  p[0] - cp[0];
      v[1]   =  p[1] - cp[1];
      delta  =  acos(V_Angle(v1,v2));
      dalpha =  delta * rr->stpara[0];
      v3[0]  =  v[0] * cos(dalpha) - v[1] * sin(dalpha);
      v3[1]  =  v[0] * sin(dalpha) + v[1] * cos(dalpha);
      s3[0]  =  cp[0] + v3[0];
      s3[1]  =  cp[1] + v3[1];
      v3[0] *= -1.0;
      v3[1] *= -1.0;
   }
   else
   {
      s3[0] = s1[0] + rr->ref * rr->sphcond * v1[0];
      s3[1] = s1[1] + rr->ref * rr->sphcond * v1[1];
      v3[0] = cp[0] - s3[0];
      v3[1] = cp[1] - s3[1];
   }
   if(rr->stpara[1])
   {
      p[0]   =  s2[0] + rr->ref * rr->sphcond * v2[0];
      p[1]   =  s2[1] + rr->ref * rr->sphcond * v2[1];
      v[0]   =  p[0] - cp[0];
      v[1]   =  p[1] - cp[1];
      delta  =  acos(V_Angle(v1,v2));
      dalpha =  delta * rr->stpara[1];
      v4[0]  =  v[0] * cos(dalpha) + v[1] * sin(dalpha);
      v4[1]  = -v[0] * sin(dalpha) + v[1] * cos(dalpha);
      s4[0]  =  cp[0] + v4[0];
      s4[1]  =  cp[1] + v4[1];
      v4[0] *= -1.0;
      v4[1] *= -1.0;
   }
   else
   {
      s4[0] = s2[0] + rr->ref * rr->sphcond * v2[0];
      s4[1] = s2[1] + rr->ref * rr->sphcond * v2[1];
      v4[0] = cp[0] - s4[0];
      v4[1] = cp[1] - s4[1];
   }
   rsph[0]  = V_Len(v3);                          // sphere radius, shroud
   V_MultScal(v3,1/rsph[0]);
   V_MultScal(v4,1/rsph[0]);

   // **************************************************
   // first spline section, inlet -> sphere
   u[1]  = u1[1] = u3[1] = 0.0;
   p1[1] = p2[1] = p3[1] = 0.0;

   // shroud, last point & vector
   // free + allocate memory
   if (s_poly)
   {
      FreePointStruct(s_poly);
      FreeFlistStruct(s_knot);
      s_poly = NULL;
      s_knot = NULL;
   }
   // inlet
   p1[0] =  rref * rr->diam[0];
   p1[1] =  rr->ref * rr->height;
   u1[0] = -sin(rr->angle[0]);
   u1[1] =  cos(rr->angle[0]);
   // outlet point ( = sphere inlet point s1)
   u3[0] =  v1[1];
   u3[1] = -v1[0];
   LineIntersect(s1,u3, p1,u1, p2);
   s_poly = CurvePolygon(p1, p2, s1, part[0], part[1]);

   // hub
   if (h_poly)
   {
      FreePointStruct(h_poly);
      FreeFlistStruct(h_knot);
      h_poly = NULL;
      h_knot = NULL;
   }
   // normal to shroud
   u[0]   = -sin(rr->angle[0] - rr->iop_angle[1] - 0.5 * M_PI);
   u[1]   =  cos(rr->angle[0] - rr->iop_angle[1] - 0.5 * M_PI);
   // inlet point
   p1[0] += rr->ref * rr->cond[0] * u[0];
   p1[1] += rr->ref * rr->cond[0] * u[1];
   u1[0]  = -sin(rr->angle[0] - (rr->iop_angle[0] + rr->iop_angle[1]));
   u1[1]  =  cos(rr->angle[0] - (rr->iop_angle[0] + rr->iop_angle[1]));
   // outlet point ( = s3, u3 unchanged!)
   u3[0] =  v3[1];
   u3[1] = -v3[0];
   LineIntersect(s3,u3, p1,u1, p2);
   h_poly = CurvePolygon(p1, p2, s3, part[0], part[1]);

   // translate points
   z_trans = 0.5 * (h_poly->y[0] + s_poly->y[0]);
   for (i = 0; i < s_poly->nump; i++)
      s_poly->y[i] -= z_trans;
   for (i = 0; i < h_poly->nump; i++)
      h_poly->y[i] -= z_trans;
   cp[1] -= z_trans;
   s1[1] -= z_trans;
   s2[1] -= z_trans;
   s3[1] -= z_trans;
   s4[1] -= z_trans;

   s_knot = BSplineKnot(s_poly, BSPLN_DEGREE);
   h_knot = BSplineKnot(h_poly, BSPLN_DEGREE);

#ifdef DEBUG_MERIDIANS
   VPRINTF(s1,fp);
   VPRINTF(s2,fp);
   VPRINTF(s3,fp);
   VPRINTF(s4,fp);
   VPRINTF(cp,fp);
   fprintf(fp,"rsph = %f   %f\n",rsph[0],rsph[1]);
#endif

   // contour points
   for(i = 0; i < NPOIN_SPLINE-1; i++)
   {
      sec = (float)i/(float)(NPOIN_SPLINE - 1);
      BSplinePoint(BSPLN_DEGREE, h_poly, h_knot, sec, p1);
      BSplinePoint(BSPLN_DEGREE, s_poly, s_knot, sec, p2);
      u[0] = p2[0] - p1[0];
      u[1] = p2[1] - p1[1];
      u[2] = p2[2] - p1[2];
      for(j = 0; j < rr->be_num; j++)
      {
         if (i == 0)                              // delete previous, allocate new
         {
            if (rr->be[j]->ml)
            {
               FreeCurveStruct(rr->be[j]->ml);
               rr->be[j]->ml = NULL;
            }
            rr->be[j]->ml = AllocCurveStruct();
#ifndef NO_INLET_EXT
            if(rr->be[j]->ml->p->portion < NPOIN_EXT)
            {
               rr->be[j]->ml->p->portion = NPOIN_EXT + 1;
            }
            // init point mem. and leave space for prefix ext.
            AddCurvePoint(rr->be[j]->ml, 0.0, 0.0, 0.0, 0.0, 0.0);
            rr->be[j]->ml->p->nump = NPOIN_EXT-1;
#endif
         }
         p[0] = p1[0] + rr->be[j]->para * u[0];
         p[1] = p1[1] + rr->be[j]->para * u[1];
         p[2] = p1[2] + rr->be[j]->para * u[2];
         AddCurvePoint(rr->be[j]->ml, p[0], p[1], p[2], 0.0, rr->be[j]->para);
      }
#ifdef GAP
      if(i == 0)
      {
         if(rr->gp->ml)
         {
            FreeCurveStruct(rr->gp->ml);
            rr->gp->ml = NULL;
         }
         rr->gp->ml = AllocCurveStruct();
#ifndef NO_INLET_EXT
         if(rr->gp->ml->p->portion < NPOIN_EXT)
         {
            rr->gp->ml->p->portion = NPOIN_EXT + 1;
         }
         // init point mem. and leave space for pre-ext.
         AddCurvePoint(rr->gp->ml, 0.0, 0.0, 0.0, 0.0, 0.0);
         rr->gp->ml->p->nump = NPOIN_EXT-1;
#endif                                   // NO_INLET_EXT
      }
      p[0] = p1[0] + rr->gp->para * u[0];
      p[1] = p1[1] + rr->gp->para * u[1];
      p[2] = p1[2] + rr->gp->para * u[2];
      AddCurvePoint(rr->gp->ml, p[0], p[1], p[2], 0.0, rr->gp->para);
#endif                                      // GAP
   }
   FreePointStruct(s_poly);
   FreePointStruct(h_poly);
   FreeFlistStruct(s_knot);
   FreeFlistStruct(h_knot);

   // **************************************************
   // sphere contour points
   delta  = acos(V_Angle(v1,v2));
   dalpha = delta/(float)(NPOIN_SPHERE - 1);
   alpha  = 0.0;
   phi    = acos(V_Angle(v3,v4));
   dbeta  = phi/(float)(NPOIN_SPHERE - 1);
   beta   = 0.0;
#ifdef DEBUG_MERIDIANS
   fprintf(fp, " atan(V_Angle(v1,v2)) = %f, dalpha = %f\n",
      180/M_PI*atan(V_Angle(v1,v2)),180/M_PI*dalpha);
#endif
   for(i = 0; i < NPOIN_SPHERE; i++)
   {
      v[0] =  cos(beta) * v3[0] + sin(beta) * v3[1];
      v[1] = -sin(beta) * v3[0] + cos(beta) * v3[1];
      beta += dbeta;
      p1[0] = cp[0] - v[0] * rsph[0];
      p1[1] = cp[1] - v[1] * rsph[0];
      v[0] =  cos(alpha) * v1[0] + sin(alpha) * v1[1];
      v[1] = -sin(alpha) * v1[0] + cos(alpha) * v1[1];
      alpha += dalpha;
      p2[0] = cp[0] - v[0] * rsph[1];
      p2[1] = cp[1] - v[1] * rsph[1];
      u[0] = p2[0] - p1[0];
      u[1] = p2[1] - p1[1];
      u[2] = p2[2] - p1[2];
      for(j = 0; j < rr->be_num; j++)
      {
         p[0] = p1[0] + rr->be[j]->para * u[0];
         p[1] = p1[1] + rr->be[j]->para * u[1];
         p[2] = p1[2] + rr->be[j]->para * u[2];
         AddCurvePoint(rr->be[j]->ml, p[0], p[1], p[2], 0.0, rr->be[j]->para);
      }
#ifdef GAP
      p[0] = p1[0] + rr->gp->para * u[0];
      p[1] = p1[1] + rr->gp->para * u[1];
      p[2] = p1[2] + rr->gp->para * u[2];
      AddCurvePoint(rr->gp->ml, p[0], p[1], p[2], 0.0, rr->gp->para);
#endif
   }

   // le points, hub
   alpha = phi * rr->le->para[0];
   v[0]  =  cos(alpha) * v3[0] + sin(alpha) * v3[1];
   v[1]  = -sin(alpha) * v3[0] + cos(alpha) * v3[1];
   p1[0] = cp[0] - v[0] * rsph[0];
   p1[1] = cp[1] - v[1] * rsph[0];
   AddCurvePoint(rr->le->c, p1[0], p1[1], p1[2], 0.0, 0.0);
   rr->le->h_norm[0] = -v[0];
   rr->le->h_norm[1] = -v[1];
   rr->le->h_norm[2] = -v[2];
   // shroud
   alpha = delta * rr->le->para[1];
   v[0]  =  cos(alpha) * v1[0] + sin(alpha) * v1[1];
   v[1]  = -sin(alpha) * v1[0] + cos(alpha) * v1[1];
   p1[0] = cp[0] - v[0] * rsph[1];
   p1[1] = cp[1] - v[1] * rsph[1];
   AddCurvePoint(rr->le->c, p1[0], p1[1], p1[2], 0.0, 0.0);
   rr->le->s_norm[0] = -v[0];
   rr->le->s_norm[1] = -v[1];
   rr->le->s_norm[2] = -v[2];

   // te points, hub
   alpha = phi * rr->te->para[0];
   v[0]  =  cos(alpha) * v3[0] + sin(alpha) * v3[1];
   v[1]  = -sin(alpha) * v3[0] + cos(alpha) * v3[1];
   p1[0] = cp[0] - v[0] * rsph[0];
   p1[1] = cp[1] - v[1] * rsph[0];
   AddCurvePoint(rr->te->c, p1[0], p1[1], p1[2], 0.0, 0.0);
   rr->te->h_norm[0] = -v[0];
   rr->te->h_norm[1] = -v[1];
   rr->te->h_norm[2] = -v[2];
   // shroud
   alpha = delta * rr->te->para[1];
   v[0]  =  cos(alpha) * v1[0] + sin(alpha) * v1[1];
   v[1]  = -sin(alpha) * v1[0] + cos(alpha) * v1[1];
   p1[0] = cp[0] - v[0] * rsph[1];
   p1[1] = cp[1] - v[1] * rsph[1];
   AddCurvePoint(rr->te->c, p1[0], p1[1], p1[2], 0.0, 0.0);
   rr->te->s_norm[0] = -v[0];
   rr->te->s_norm[1] = -v[1];
   rr->te->s_norm[2] = -v[2];

   // **************************************************
   // second spline section
   // shroud
   s2[1] += z_trans;
   u1[0]  = -v2[1];
   u1[1]  =  v2[0];
   p3[0]  =  rref * rr->diam[1];
   p3[1]  =  0.0;
   u3[0]  = -sin(rr->angle[1]);
   u3[1]  =  cos(rr->angle[1]);
   LineIntersect(s2,u1, p3,u3, p2);
   s_poly = CurvePolygon(s2, p2, p3, part[0], part[1]);

   // hub
   s4[1] += z_trans;
   u1[0]  = -v4[1];
   u1[1]  =  v4[0];
   u[0]   = -sin(rr->angle[1] - rr->oop_angle[1] + 0.5 * M_PI);
   u[1]   =  cos(rr->angle[1] - rr->oop_angle[1] + 0.5 * M_PI);
   p3[0] +=  rr->ref * rr->cond[1] * u[0];
   p3[1] +=  rr->ref * rr->cond[1] * u[1];
   u3[0]  = -sin(rr->angle[1] - (rr->oop_angle[0] + rr->oop_angle[1]));
   u3[1]  =  cos(rr->angle[1] - (rr->oop_angle[0] + rr->oop_angle[1]));
   LineIntersect(s4,u1, p3, u3, p2);
   h_poly = CurvePolygon(s4, p2, p3, part[0], part[1]);

   // translate polygon
   for (i = 0; i < s_poly->nump; i++)
      s_poly->y[i] -= z_trans;
   for (i = 0; i < h_poly->nump; i++)
      h_poly->y[i] -= z_trans;
   s2[1] -= z_trans;
   s4[1] -= z_trans;

   s_knot = BSplineKnot(s_poly, BSPLN_DEGREE);
   h_knot = BSplineKnot(h_poly, BSPLN_DEGREE);

   // contour points
   for(i = 1; i < NPOIN_SPLINE; i++)
   {
      sec = (float)i/(float)(NPOIN_SPLINE - 1);
      BSplinePoint(BSPLN_DEGREE, h_poly, h_knot, sec, p1);
      BSplinePoint(BSPLN_DEGREE, s_poly, s_knot, sec, p2);
      u[0] = p2[0] - p1[0];
      u[1] = p2[1] - p1[1];
      u[2] = p2[2] - p1[2];
      for(j = 0; j < rr->be_num; j++)
      {
         p[0] = p1[0] + rr->be[j]->para * u[0];
         p[1] = p1[1] + rr->be[j]->para * u[1];
         p[2] = p1[2] + rr->be[j]->para * u[2];
         AddCurvePoint(rr->be[j]->ml, p[0], p[1], p[2], 0.0, rr->be[j]->para);
      }
#ifdef GAP
      p[0] = p1[0] + rr->gp->para * u[0];
      p[1] = p1[1] + rr->gp->para * u[1];
      p[2] = p1[2] + rr->gp->para * u[2];
      AddCurvePoint(rr->gp->ml, p[0], p[1], p[2], 0.0, rr->gp->para);
#endif
   }

   FreePointStruct(s_poly);
   FreePointStruct(h_poly);
   FreeFlistStruct(s_knot);
   FreeFlistStruct(h_knot);

   // **************************************************
   // shroud contour end point and extension vector
   be_max = rr->be_num - 1;
   p_max  = rr->be[be_max]->ml->p->nump - 1;
   s_ext[0]  = s_end[0] = rr->be[be_max]->ml->p->x[p_max];
   s_ext[0] -= rr->be[be_max]->ml->p->x[p_max-1];
   s_ext[1]  = s_end[1] = rr->be[be_max]->ml->p->y[p_max];
   s_ext[1] -= rr->be[be_max]->ml->p->y[p_max-1];
   s_ext[2]  = s_end[2] = rr->be[be_max]->ml->p->z[p_max];
   s_ext[2] -= rr->be[be_max]->ml->p->z[p_max-1];
   t             = (SHROUD_EXT * rr->height * rr->ref)/fabs(s_ext[1]);
   p3[0] = s_end[0] + t * s_ext[0] * SHROUD_R_SCAL;
   p3[1] = s_end[1] + t * s_ext[1];
   p3[2] = s_end[2] + t * s_ext[2];
   v[0]  = 0.0;
   v[1]  = 0.0;
   v[2]  = 1.0;
   LineIntersect(s_end, s_ext, p3, v, p2);
   s_poly = CurvePolygon(s_end, p2, p3, 0.5, 0.5);
   s_knot = BSplineKnot(s_poly, BSPLN_DEGREE);
#ifdef DEBUG_MERIDIANS
   VPRINTF(s_end, fp);
   VPRINTF(p2, fp);
   VPRINTF(p3, fp);
#endif
   // hub contour end point and extension vector
   p_max  = rr->be[0]->ml->p->nump - 1;
   h_ext[0]  = p1[0] = rr->be[0]->ml->p->x[p_max];
   h_ext[0] -= rr->be[0]->ml->p->x[p_max-1];
   h_ext[1]  = p1[1] = rr->be[0]->ml->p->y[p_max];
   h_ext[1] -= rr->be[0]->ml->p->y[p_max-1];
   h_ext[2]  = p1[2] = rr->be[0]->ml->p->z[p_max];
   h_ext[2] -= rr->be[0]->ml->p->z[p_max-1];
   // hub extension end point and end vector
   p3[0] = HUB_EXT_RAD;
   v[0]  = 0.0;
   v[1]  = 0.0;
   v[2]  = 1.0;
   LineIntersect(&p1[0], &h_ext[0],&p3[0], &v[0], &p2[0]);
   h_poly = CurvePolygon(&p1[0], &p2[0], &p3[0], 0.5f, 0.4f);
   h_knot = BSplineKnot(h_poly, BSPLN_DEGREE);
   // runner meridian contours extension
   for (i = 1; i < NPOIN_EXT; i++)
   {
      sec = (float)i/(float)(NPOIN_EXT - 1);
      BSplinePoint(BSPLN_DEGREE, h_poly, h_knot, sec, &p1[0]);
      BSplinePoint(BSPLN_DEGREE, s_poly, s_knot, sec, &p2[0]);
      u[0] = p2[0] - p1[0];
      u[1] = p2[1] - p1[1];
      u[2] = p2[2] - p1[2];
#ifdef DEBUG_MERIDIANS
      VPRINTF(p2, fp);
      VPRINTF(u, fp);
#endif
      for(j = 0; j < rr->be_num; j++)
      {
         p[0] = p1[0] + rr->be[j]->para * u[0];
         p[1] = p1[1] + rr->be[j]->para * u[1];
         p[2] = p1[2] + rr->be[j]->para * u[2];
         AddCurvePoint(rr->be[j]->ml, p[0], p[1], p[2], 0.0, rr->be[j]->para);
      }
#ifdef GAP
      p[0] = p1[0] + rr->gp->para * u[0];
      p[1] = p1[1] + rr->gp->para * u[1];
      p[2] = p1[2] + rr->gp->para * u[2];
      AddCurvePoint(rr->gp->ml, p[0], p[1], p[2], 0.0, rr->gp->para);
#endif
   }
   FreePointStruct(h_poly);
   FreeFlistStruct(h_knot);
   FreePointStruct(s_poly);
   FreeFlistStruct(s_knot);

   // **************************************************
#ifndef NO_INLET_EXT
   // create prefix, inlet extension, shroud
   // end point, beginning of runner part
   p_max  = NPOIN_EXT-1;
   be_max = rr->be_num-1;
   p3[0] = rr->be[be_max]->ml->p->x[p_max];
   p3[1] = rr->be[be_max]->ml->p->y[p_max];
   v[0]  =  sin(rr->angle[0]);
   v[1]  = -cos(rr->angle[0]);
   // start point
   p1[0] = (1.0 + IN_EXT_R) * rr->be[be_max]->ml->p->x[p_max];
   p1[1] = p1[0]/v[0] * v[1] * IN_EXT_H + rr->be[be_max]->ml->p->y[p_max];
   if(rr->be[be_max]->ml->p->x[p_max] <= rr->be[be_max]->ml->p->x[p_max+NPOIN_MERIDIAN])
   {
      u[0]  = -1.0;
      u[1]  = -1.0;
      delta =  atan(u[1]/u[0]);
   }
   else
   {
      u[0]  = -1.0;
      u[1]  =  0.0;
      delta =  0.0;
   }
   LineIntersect(p3,v, p1,u, p2);
#ifdef DEBUG_MERIDIANS
   fprintf(stderr,"p_max = %d\n",p_max);
   fprintf(stderr,"rr->be[be_max]->ml->p->nump = %d\n",rr->be[be_max]->ml->p->nump);
   fprintf(stderr,"rr->be[be_max]->ml->p->x[p_max] = %f\n",rr->be[be_max]->ml->p->x[p_max]);
   fprintf(stderr,"p1 = %f  %f  %f\n",p1[0], p1[1], p1[2]);
   fprintf(stderr,"p2 = %f  %f  %f\n",p2[0], p2[1], p2[2]);
   fprintf(stderr,"p3 = %f  %f  %f\n",p3[0], p3[1], p3[2]);
#endif
   s_poly = CurvePolygon(p1,p2,p3, 0.5, 0.5);
   s_knot = BSplineKnot(s_poly, BSPLN_DEGREE);

   // hub
   // last point, beginning runner part
   p3[0] = rr->be[0]->ml->p->x[p_max];
   p3[1] = rr->be[0]->ml->p->y[p_max];
   v[0]    =  sin(rr->angle[0] - (rr->iop_angle[0] + rr->iop_angle[1]));
   v[1]    = -cos(rr->angle[0] - (rr->iop_angle[0] + rr->iop_angle[1]));
   // starting point.
   if(delta)
   {
      p1[0] -= rr->ref * rr->cond[0] * cos(delta);
      p1[1] += rr->ref * rr->cond[0] * sin(delta);
   }
   else
   {
      p1[1] += rr->ref * rr->cond[0];
   }
   LineIntersect(p3,v, p1,u, p2);
   h_poly = CurvePolygon(p1,p2,p3, 0.5, 0.5);
   h_knot = BSplineKnot(h_poly, BSPLN_DEGREE);

   p_max = rr->be[0]->ml->p->nump;
#ifdef DEBUG_MERIDIANS
   fprintf(stderr,"p_max = %d\n",p_max);
#endif
   for(i = 0; i < NPOIN_EXT-1; i++)
   {
      sec = (float)i/(float)(NPOIN_EXT - 1);
      BSplinePoint(BSPLN_DEGREE, h_poly, h_knot, sec, p1);
      BSplinePoint(BSPLN_DEGREE, s_poly, s_knot, sec, p2);
      u[0] = p2[0] - p1[0];
      u[1] = p2[1] - p1[1];
      u[2] = p2[2] - p1[2];
      for(j = 0; j < rr->be_num; j++)
      {
         // shift back pointers
         if(i == 0)
         {
            rr->be[j]->ml->p->nump = 0;
         }
         p[0] = p1[0] + rr->be[j]->para * u[0];
         p[1] = p1[1] + rr->be[j]->para * u[1];
         p[2] = p1[2] + rr->be[j]->para * u[2];
         AddCurvePoint(rr->be[j]->ml, p[0], p[1], p[2], 0.0, rr->be[j]->para);
      }
#ifdef GAP
      if(i == 0)
      {
         rr->gp->ml->p->nump = 0;
      }
      p[0] = p1[0] + rr->gp->para * u[0];
      p[1] = p1[1] + rr->gp->para * u[1];
      p[2] = p1[2] + rr->gp->para * u[2];
      AddCurvePoint(rr->gp->ml, p[0], p[1], p[2], 0.0, rr->gp->para);
#endif
   }

   FreePointStruct(s_poly);
   FreePointStruct(h_poly);
   FreeFlistStruct(s_knot);
   FreeFlistStruct(h_knot);
#endif                                         // !NO_INLET_EXT

   // **************************************************
   // calculate curve arc lengths
   for (j = 0; j < rr->be_num; j++)
   {
      rr->be[j]->ml->p->nump = p_max;
      CalcCurveArclen(rr->be[j]->ml);
   }
#ifdef GAP
   rr->gp->ml->p->nump = p_max;
   CalcCurveArclen(rr->gp->ml);
#endif

#ifdef DEBUG_MERIDIANS
   fclose(fp);
#endif

#ifdef DEBUG_MERIDIANS
   for (j = 0; j < rr->be_num; j++)
   {
      sprintf(fn, "dr_meridian_%02d.txt", j);
      if ((fp = fopen(fn, "w")) == NULL)
      {
         fprintf(stderr, "error writing meridian file '%s' *yakk*\n", fn);
         exit(-1);
      }
      DumpCurve(rr->be[j]->ml, fp);
      fclose(fp);
   }
#ifdef GAP
   sprintf(fn, "dr_meridian_gap.txt");
   if ((fp = fopen(fn, "w")) == NULL)
   {
      fprintf(stderr, "error writing meridian file '%s' *yakk*\n", fn);
      exit(-1);
   }
   DumpCurve(rr->gp->ml, fp);
   fclose(fp);
#endif
   // sphere points
   sprintf(fn, "dr_sphere.txt");
   if ((fp = fopen(fn, "w")) == NULL)
   {
      fprintf(stderr, "error writing meridian file '%s' *yakk*\n", fn);
      exit(-1);
   }
   fprintf(fp,"  %f  %f  %f\n",s1[0], s1[1], s1[2]);
   fprintf(fp,"  %f  %f  %f\n",s2[0], s2[1], s2[2]);
   fprintf(fp,"\n\n");
   fprintf(fp,"  %f  %f  %f\n",s3[0], s3[1], s3[2]);
   fprintf(fp,"  %f  %f  %f\n",s4[0], s4[1], s4[2]);
   fprintf(fp,"\n\n");
   fprintf(fp,"  %f  %f  %f\n",s1[0], s1[1], s1[2]);
   fprintf(fp,"  %f  %f  %f\n",cp[0], cp[1], cp[2]);
   fprintf(fp,"\n\n");
   fprintf(fp,"  %f  %f  %f\n",s2[0], s2[1], s2[2]);
   fprintf(fp,"  %f  %f  %f\n",cp[0], cp[1], cp[2]);
   fprintf(fp,"\n\n");
   fprintf(fp,"  %f  %f  %f\n",s1[0], s1[1], s1[2]);
   fprintf(fp,"  %f  %f  %f\n",s1[0]+v1[1], s1[1]-v1[0], s1[2]+v1[2]);
   fprintf(fp,"\n\n");
   fprintf(fp,"  %f  %f  %f\n",s2[0], s2[1], s2[2]);
   fprintf(fp,"  %f  %f  %f\n",s2[0]-v2[1], s2[1]+v2[0], s2[2]+v2[2]);

   fclose(fp);
#endif

   return(1);
}
