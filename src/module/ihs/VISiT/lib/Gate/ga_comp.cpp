#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <strings.h>
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include "../General/include/cfg.h"
#include "../General/include/geo.h"
#include "include/gate.h"
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/curve.h"
#include "../General/include/profile.h"
#include "../General/include/parameter.h"
#include "../General/include/bias.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#define INIT_PORTION 25
#define BSPLN_DEGREE 3                            // bspline degree

int CreateGA_Contours(struct gate *ga)
{
   int i;
   float par, x[3];
   struct Flist *knot;

   ga->chub = AllocCurveStruct();
   knot     = BSplineKnot(ga->phub, BSPLN_DEGREE);
   for (i = 0; i <= ga->num_hub_arc+2; i++)
   {
      par = (float)i/(float)(ga->num_hub_arc+2);
      BSplinePoint(BSPLN_DEGREE, ga->phub, knot, par, &x[0]);
      AddCurvePoint(ga->chub, x[0], x[1], x[2], 0.0, par);
   }
   FreeFlistStruct(knot);
   CalcCurveArclen(ga->chub);
#ifdef CONTOUR_WIREFRAME
   CreateGAContourWireframe(ga->chub);
#endif                                         // CONTOUR_WIREFRAME

   ga->cshroud = AllocCurveStruct();
   knot        = BSplineKnot(ga->pshroud, BSPLN_DEGREE);
   for (i = 0; i <= NPOIN_SHROUD_AB+2; i++)
   {
      par = (float)i/(float)(NPOIN_SHROUD_AB+2);
      BSplinePoint(BSPLN_DEGREE, ga->pshroud, knot, par, &x[0]);
      AddCurvePoint(ga->cshroud, x[0], x[1], x[2], 0.0, par);
   }
   FreeFlistStruct(knot);
   CalcCurveArclen(ga->cshroud);
#ifdef CONTOUR_WIREFRAME
   CreateGAContourWireframe(ga->cshroud);
#endif                                         // CONTOUR_WIREFRAME

   return(1);
}


#ifdef CONTOUR_WIREFRAME
void CreateGAContourWireframe(struct curve *c)
{

   int i, j;
   static int ncall = 0;
   const int nsec = 36;
   float x, y, z;
   float angle, roma[2][2];
   const float rot = 2 * M_PI/nsec;
   FILE *fp_3d, *fp_2d;
   char fname[255];

   sprintf(fname, "ga_contour2d_%02d.txt", ncall);
   fp_2d = fopen(fname, "w");
   sprintf(fname, "ga_contour3d_%02d.txt", ncall++);
   fp_3d = fopen(fname, "w");
   for (i = 0; i <= nsec; i++)
   {
      angle      = i * rot;
      roma[0][0] =  cos(angle);
      roma[0][1] = -sin(angle);
      roma[1][0] =  sin(angle);
      roma[1][1] =  cos(angle);
      for (j = 0; j < c->p->nump; j++)
      {
         x = c->p->x[j] * roma[0][0] + c->p->y[j] * roma[0][1];
         y = c->p->x[j] * roma[1][0] + c->p->y[j] * roma[1][1];
         z = c->p->z[j];
         fprintf(fp_3d, "%10.8f %10.8f %10.8f\n", x, y, z);
      }
      fprintf(fp_3d, "\n");
   }
   for (i = 0; i < c->p->nump; i++)
      fprintf(fp_2d, "%10.8f %10.8f %10.8f\n", c->p->x[i], c->p->y[i], c->p->z[i]);
   fclose(fp_2d);
   fclose(fp_3d);
}
#endif                                            // CONTOUR_WIREFRAME

int CreateGA_BladeElements(struct gate *ga)
{

   // covise on-the-fly modifications...
   fprintf(stderr, "CreateGA_BladeElements(): after COVISE modification:\n");

#ifdef DEBUG
   DumpGA(ga);
#endif

   fprintf(stderr, "CreateGA_BladeElements(): entering SurfacesGA_BladeElements()\n");
   // calculate blade surfaces and centre line
   SurfacesGA_BladeElement(ga);

#ifdef GNUPLOT
   //WriteGNU_GA(ga);
#endif                                         // GNUPLOT

   fprintf(stderr, "CreateGA_BladeElements()...  end \n");
   return(1);
}


int InitGA_BladeElements(struct gate *ga)         // called from ReadGate in ga_io.cpp
{

   ga->bp   = AllocBladeProfile();
   Parameter2Profile(ga->prof, ga->bp);
#ifdef DEBUG_PARAFIELDS
   fprintf(stderr, "\nblade element camber:\n");
   DumpParameterSet(ga->camb);
   fprintf(stderr, "\nblade profile:\n");
   DumpBladeProfile(ga->bp);
#endif                                         // DEBUG_PARAFIELDS

   return(0);
}


int SurfacesGA_BladeElement(struct gate *ga)
{
   int j, t_sec;

   float piv, cl_sec, scale_t, te, t, tmax, angle;
   double x1, y1, x2, y2, ux, uy, q;

   float p[3];

   struct Point *cl_poly = NULL;
   struct Flist *cl_knot = NULL;

   struct Point *profile = NULL;
   struct Flist *profknot = NULL;

   struct Point *profile2 = NULL;
   struct Point *profnorm2 = NULL;

#ifdef DEBUG_SURFACES
   FILE *fp;
   char fname[255];
   static int nbe = 0;

   sprintf(fname, "ga_beplane_%02d.txt", nbe++);
   fp = fopen(fname, "w");
#endif                                         // DEBUG_SURFACES

   // delete previous data and allocate new
   if (cl_poly)
   {
      FreePointStruct(cl_poly);
      cl_poly = NULL;
   }
   if (cl_knot)
   {
      FreeFlistStruct(cl_knot);
      cl_knot = NULL;
   }

   FreePointStruct(ga->cl);                       // centre line
   FreePointStruct(ga->clg);
   FreePointStruct(ga->ps);                       // pressure side
   FreePointStruct(ga->ss);                       // suction side

   // geometry created here (by module Gate)
   if (ga->geofromfile == 0)
   {
      // calculate centre line knots from camber set
      cl_poly = AllocPointStruct();
      p[0] = p[1] = p[2] = 0.0;
      for (j = 0; j < ga->camb->num; j++)
      {
         p[0] = (1 - ga->camb->loc[j]) * ga->chord;
         p[1] = ga->camb->val[j] * ga->maxcamb * -1.0f;
         AddVPoint(cl_poly, p);
      }
      cl_knot = BSplineKnot(cl_poly, BSPLN_DEGREE);

      // calculate centre line and gradient
      ga->cl  = AllocPointStruct();
      ga->clg = AllocPointStruct();
      // centre line points and gradient
      for (j = 0; j < ga->bp->num; j++)
      {
         cl_sec = float(pow((float)ga->bp->c[j], (float)ga->bp_shift));
         BSplinePoint(BSPLN_DEGREE, cl_poly, cl_knot, cl_sec, &p[0]);
         AddVPoint(ga->cl, p);
         BSplineNormal(BSPLN_DEGREE, cl_poly, cl_knot, cl_sec, &p[0]);
         AddVPoint(ga->clg, p);
      }

      // scale cl to machine size, calculate ps/ss
      // scale_t ... tot. profile thickness scale up
      ga->ss  = AllocPointStruct();
      ga->ps  = AllocPointStruct();
      t_sec   = ga->bp->t_sec;
      tmax    = ga->bp->t[t_sec] * ga->chord;
      scale_t = ga->p_thick / tmax;

      FreeBladeProfile(ga->bp);
      InitGA_BladeElements(ga);

      for (j = 0; j < ga->bp->num; j++)
      {
         // pressure side (-) and suction side (+), machine size
         te   = 0.0;                              // 0.5 * ga->bp->c[j] * ga->te_thick;
         t    = 0.5f * ga->chord * ga->bp->t[j];
         p[0] = ga->cl->x[j] - ga->clg->x[j] * (scale_t * t + te);
         p[1] = ga->cl->y[j] - ga->clg->y[j] * (scale_t * t + te);
         p[2] = ga->cl->z[j] - ga->clg->z[j] * (scale_t * t + te);
         AddVPoint(ga->ps, p);
         p[0] = ga->cl->x[j] + ga->clg->x[j] * (scale_t * t + te);
         p[1] = ga->cl->y[j] + ga->clg->y[j] * (scale_t * t + te);
         p[2] = ga->cl->z[j] + ga->clg->z[j] * (scale_t * t + te);
         AddVPoint(ga->ss, p);
      }

      // calculate pivot coords
      piv = ga->pivot / ga->chord;
      piv = 1 - piv;
      BSplinePoint(BSPLN_DEGREE, cl_poly, cl_knot, piv, &p[0]);

      ga->bp->num = ga->ss->nump;

   }                                              // end ga->geofromfile == 0

   // geometry from file
   if (ga->geofromfile == 1)
   {
      if (ReadProfileFromFile(ga, ga->cfgfile))
      {
         printf("ReadProfileFromFile suceeded\n");

         ga->bp->num = ga->ss->nump;
         if (  (ga->ps->nump != ga->ss->nump) || (ga->cl->nump != ga->ss->nump)  )
            fprintf(stderr,"blade profile error: need same point numbers for \
                ss, ps and cl in configuration file!\n");
      }
      p[0] = ga->p_pivot->x[0];
      p[1] = ga->p_pivot->y[0];
      p[2] = ga->p_pivot->z[0];
   }

   // translate profile: pivot in (0|0|0) for rotation

   printf("ga->bp->num = %d\n", ga->bp->num);
   for (j = 0; j < ga->bp->num; j++)
   {
      ga->cl->x[j] -= p[0];
      ga->cl->y[j] -= p[1];
      ga->cl->z[j] -= p[2];
      ga->ps->x[j] -= p[0];
      ga->ps->y[j] -= p[1];
      ga->ps->z[j] -= p[2];
      ga->ss->x[j] -= p[0];
      ga->ss->y[j] -= p[1];
      ga->ss->z[j] -= p[2];
   }

   // calculate a new spline with defined bias through these splines
   // delete previous data and allocate new
   // bias for blade
   struct Flist *bladebias_temp = NULL;
   if (bladebias_temp)
   {
      FreeFlistStruct(bladebias_temp);
      bladebias_temp = NULL;
   }
   bladebias_temp = AllocFlistStruct(ga->bp->num);
   bladebias_temp = CalcBladeElementBias(ga->bp->num, 0.0, 1.0, 1, 10.0);

   struct Flist *bladebias = NULL;
   if (bladebias)
   {
      FreeFlistStruct(bladebias);
      bladebias = NULL;
   }
   bladebias = AllocFlistStruct(2 * ga->bp->num - 1);

   // pressure side and suction side together!
   if (profile)
   {
      FreePointStruct(profile);
      profile = NULL;
   }
   if (profknot)
   {
      FreeFlistStruct(profknot);
      profknot = NULL;
   }
   profile = AllocPointStruct();

   //profknot = AllocFlistStruct(2 * ga->bp->num -1);
   for (j = 0; j <ga->bp->num; j++)
   {
      p[0] = ga->ps->x[ga->bp->num-1 - j];
      p[1] = ga->ps->y[ga->bp->num-1 - j];
      AddVPoint(profile, p);
   }
   for (j = 1; j <ga->bp->num; j++)
   {
      p[0] = ga->ss->x[j];
      p[1] = ga->ss->y[j];
      AddVPoint(profile, p);
   }
   profknot = BSplineKnot(profile, BSPLN_DEGREE);

   // calculate length of pressure side and suction side (for bias)
   float len_ps = 0.0;
   float len_ss = 0.0;
   for (j = 1; j < ga->bp->num; j++)
   {
      len_ps += float(pow ( (float)pow((float)(ga->ps->x[j]-ga->ps->x[j-1]),(float)2.) + pow((float)(ga->ps->y[j]-ga->ps->y[j-1]),(float)2.) , (float)0.5));
      len_ss += float(pow ( (float)pow((float)(ga->ss->x[j]-ga->ss->x[j-1]),(float)2.) + pow((float)(ga->ss->y[j]-ga->ss->y[j-1]),(float)2.) , (float)0.5));
   }

   // calculate biaslist for profile spline knot information
   for (j = 0; j < ga->bp->num; j++)
   {
      Add2Flist(bladebias, bladebias_temp->list[j] * len_ps / (len_ps + len_ss));
   }
   for (j = 1 ; j < ga->bp->num; j++)
   {
      Add2Flist(bladebias, ( len_ps / (len_ss + len_ps) ) + (1 - bladebias_temp->list[ga->bp->num-1-j]) * ( len_ss / (len_ss + len_ps) ) );
   }
   //DumpFlist(bladebias);
   FreeFlistStruct(bladebias_temp);

   if (profile2)
   {
      FreePointStruct(profile2);
      profile2 = NULL;
   }
   if (profnorm2)
   {
      FreePointStruct(profnorm2);
      profnorm2 = NULL;
   }
   profile2 = AllocPointStruct();
   profnorm2 = AllocPointStruct();

   // calculate profile with defined bias
   for (j = 0; j < 2 * ga->bp->num - 1; j++)
   {
      BSplinePoint(BSPLN_DEGREE, profile, profknot, bladebias->list[j], &p[0]);
      AddVPoint(profile2 , p);
      BSplineNormal(BSPLN_DEGREE, profile, profknot, bladebias->list[j], &p[0]);
      AddVPoint(profnorm2 , p);
   }

   // sort arrays for ps and ss
   for (j = 0; j < ga->bp->num; j++)
   {
      ga->ps->x[ga->bp->num-1 - j] = profile2->x[j];
      ga->ps->y[ga->bp->num-1 - j] = profile2->y[j];
      ga->ps->z[ga->bp->num-1 - j] = profile2->z[j];
   }
   for (j = ga->bp->num - 1; j < 2 * ga->bp->num - 1; j++)
   {
      ga->ss->x[j - ga->bp->num +1] = profile2->x[j];
      ga->ss->y[j - ga->bp->num +1] = profile2->y[j];
      ga->ss->z[j - ga->bp->num +1] = profile2->z[j];
   }

#ifdef DEBUG_SURFACES
   for (j = 0; j < ga->bp->num; j++)
   {
      fprintf(fp, "%8.6f %8.6f %8.6f\n", ga->cl->x[j], ga->cl->y[j], ga->cl->z[j]);
   }
   for (j = 0; j < ga->bp->num; j++)
   {
      fprintf(fp, "%8.6f %8.6f %8.6f\n", ga->ss->x[j], ga->ss->y[j], ga->ss->z[j]);
   }
   for (j = 0; j < ga->bp->num; j++)
   {
      fprintf(fp, "%8.6f %8.6f %8.6f\n", ga->ps->x[j], ga->ps->y[j], ga->ps->z[j]);
   }

   // pivot point = 0.0 0.0 0.0 +
   j = ga->ss->nump-1;
   fprintf(fp, "0.0 0.0 0.0  %8.6f %8.6f %8.6f ", ga->ps->x[j], ga->ps->y[j], ga->ps->z[j]);

   fclose(fp);
#endif                                         // DEBUG_SURFACES

   // gate: transform to machine coordinates (in x-direction)
   if (ga->geofromfile==0)
   {
      for (j = 0; j < ga->bp->num; j++)
      {
         ga->cl->x[j] += ga->pivot_rad;
         ga->ps->x[j] += ga->pivot_rad;
         ga->ss->x[j] += ga->pivot_rad;
      }
   }
   else                                           //(geofromfile == 1)
   {
      for (j = 0; j < ga->bp->num; j++)
      {
         ga->cl->x[j] += ga->p_pivot->x[0];
         ga->cl->y[j] += ga->p_pivot->y[0];
         ga->cl->z[j] += ga->p_pivot->z[0];

         ga->ps->x[j] += ga->p_pivot->x[0];
         ga->ps->y[j] += ga->p_pivot->y[0];
         ga->ps->z[j] += ga->p_pivot->z[0];

         ga->ss->x[j] += ga->p_pivot->x[0];
         ga->ss->y[j] += ga->p_pivot->y[0];
         ga->ss->z[j] += ga->p_pivot->z[0];
      }
   }

   // calculate new axis coordinates (bx|by) for (0|0), trailing edge and pivot on one line
   double xp, yp;
   if (ga->geofromfile==0)
   {
      xp = ga->pivot_rad;
      yp = 0.;
   }
   else
   {
      xp = ga->p_pivot->x[0];
      yp = ga->p_pivot->y[0];
      ga->pivot_rad = float(pow ( (float)pow((float)ga->p_pivot->x[0],(float)2.)+pow((float)ga->p_pivot->y[0],(float)2.) , (float)0.5 ));
   }

   j = ga->bp->num-2;

   x1 = ga->cl->x[j];
   y1 = ga->cl->y[j];
   x2 = ga->cl->x[j+1];
   y2 = ga->cl->y[j+1];

   ux = x2 - xp;
   uy = y2 - yp;

   double s = pow ( (float)pow((float)ux,(float)2.)+pow((float)uy,(float)2.) , (float)0.5 );

   ux /= s;
   uy /= s;

   q = ga->pivot_rad - s;

   double bx, by;
   bx = x2 + q * ux;
   by = y2 + q * uy;

   // transform to new axis in (0|0)
   for (j = 0; j < ga->bp->num; j++)
   {
      ga->cl->x[j] -= float(bx);
      ga->cl->y[j] -= float(by);

      ga->ps->x[j] -= float(bx);
      ga->ps->y[j] -= float(by);

      ga->ss->x[j] -= float(bx);
      ga->ss->y[j] -= float(by);
   }
   xp -= bx;
   yp -= by;
   //printf("******* Abweichung = %5.2lf\n", ga->pivot_rad - pow (pow(xp,2)+pow(yp,2),0.5) );

   // transform around z-axis (trailing edge and pivot will be on x-axis)
   // pivot coordinates: (xp|yp) = (ga->pivot_rad|0)
   angle = float(asin(yp / ga->pivot_rad));

   if (xp<0)
   {
      angle = float(M_PI - angle);
   }
   angle *= -1;
   RotateBlade(ga, angle, 0.0, 0.0);

   // calculate actual beta
   j = ga->bp->num-2;
   x1 = ga->cl->x[j+1];
   y1 = ga->cl->y[j+1];
   x2 = ga->cl->x[j];
   y2 = ga->cl->y[j];
   ga->beta = float(acos ( (x2*x2-x1*x2-y1*y2+y2*y2) / ( sqrt (    (x2*x2+y2*y2)*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)       ) ) ) ));
   //printf("******* beta vorher = %5.2lf\n", ga->beta * 180. / M_PI);

   // ******** trailing edge and pivot are now on x-axis) **********

   // gewuenschtes beta2 fuer aktuellen Bestpunkt aus Eulergleichung berechnen
   // gilt bisher nur fuer gegen den Uhrzeigersinn rotierende Maschinen (von oben betrachtet)
   double beta_soll = atan2 ( (float)(60 * 9.81 * ga->H * ga->in_height) , (float)(ga->nopt * ga->Qopt));
   //printf("******* beta_soll = %5.2lf\n", beta_soll * 180. / M_PI);

   // rotate blade element around pivot
   // rotate blade to bestpoint (beta2)
   double r = ga->pivot_rad;
   // Vorzeichen haengt von Drehsinn ab!
   // hier bestimmen der Drehrichtung mit Determinante
   double xv, yv;
   xv = ga->cl->x[0];
   yv = ga->cl->y[0];
   double det = x1*yv-y1*xv;
   int sign;
   if (det>=0)
   {
      //printf("******* math. neg. drehende Maschine\n");
      sign=1;
   }
   else
   {
      //printf("******* math. pos. drehende Maschine\n");
      sign=-1;
   }
   angle = float(sign * ga->beta - sign * acos  ( s / r * sin(beta_soll) * sin(beta_soll) + cos(beta_soll) * sqrt(  1-(s*s/r*r)*sin(beta_soll)*sin(beta_soll))   ) ) ;
   //printf("******* angle = %5.2lf\n", angle * 180. / M_PI);
   RotateBlade(ga, angle, ga->pivot_rad, 0.0);
   //RotateBlade(ga, angle, xp, yp);

   // **********  blade is now in bestpoint, but pivot still on x-axis ************

   // beta_opt aus der Skeltettlinie des Profils berechnen
   j = ga->bp->num-2;
   x1 = ga->cl->x[j+1];
   y1 = ga->cl->y[j+1];
   x2 = ga->cl->x[j];
   y2 = ga->cl->y[j];

   ga->beta = float(acos ( (x2*x2-x1*x2-y1*y2+y2*y2) / ( pow (  (float)((x2*x2+y2*y2)*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)  )) , 0.5f) ) ));
   //printf("******* beta nachher = %5.2lf\n", ga->beta * 180. / M_PI);

   // lichte Weite a0opt bestimmen
   // Pivot drehen
   double a0opt;
   a0opt = get_a0(ga, (float)sign);

   int i;

   // array a0(beta)

   angle= float(sign*(-100./180.)*M_PI);
   for (i=0; i<20; i++)
   {
      if(i!=0) {angle = float(sign*(10./180.)*M_PI);}
      RotateBlade(ga, angle, ga->pivot_rad, 0.0);
      ga->a0_beta[i] = float(get_a0(ga, (float)sign)/a0opt);

   }
   RotateBlade(ga, float(sign*(-90./180.)*M_PI), ga->pivot_rad, 0.0);

   /*
      // print dependency beta - a0/a0opt in file
      FILE *stream;
      char datei_steuer[200];
      strcpy(datei_steuer,"beta_a0.dat");
      if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
      {
         printf( "Kann '%s' nicht lesen!\n", datei_steuer);
      }

      else
   {
   for (i=0; i<20; i++)
   {
   fprintf(stream, "%8.5lf %8.5lf\n", (float)(-100+10*i), ga->a0_beta[i]);
   }
   fclose(stream);
   }
   */

   // turn blade around pivot to running point (blade angle)

   angle = sign * ga->bangle;
   RotateBlade(ga, angle, ga->pivot_rad, 0.0);

   // beta_ende aus der Skeltettlinie des Profils berechnen
   j = ga->bp->num-2;
   x1 = ga->cl->x[j+1];
   y1 = ga->cl->y[j+1];
   x2 = ga->cl->x[j];
   y2 = ga->cl->y[j];
   ga->beta = float(acos ( (x2*x2-x1*x2-y1*y2+y2*y2) / ( pow (  float((x2*x2+y2*y2)*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)  )) , 0.5f) ) ));
   printf("******* Austrittswinkel beta = %5.2lf\n", ga->beta * 180. / M_PI);

   // lichte Weite ao bestimmen
   double a0;
   a0 = get_a0(ga, float(sign));
   ga->Q = float(a0 / a0opt * ga->Qopt);

   return(1);
}


int ReadProfileFromFile(struct gate *ga, const char *fn)
{

   // alocate and fill structs for pressure side, suction side and center line
   ga->cl  = AllocPointStruct();
   ga->clg  = AllocPointStruct();
   ga->ps  = AllocPointStruct();
   ga->ss  = AllocPointStruct();
   ReadPointStruct(ga->cl, "[blade center line]", fn);
   ReadPointStruct(ga->ps, "[blade pressure side]", fn);
   ReadPointStruct(ga->ss, "[blade suction side]", fn);

   /*
   FILE *fp;
   char fname[255];
   static int nbe = 0;
   int j;

   sprintf(fname, "ga_beplane_%02d.txt", nbe++);
   fp = fopen(fname, "w");

   fprintf(fp, "Anzahl Punkte: %d\n", ga->bp->num);

   for (j = 0; j < ga->bp->num; j++) {
   fprintf(fp, "%8.6f %8.6f %8.6f\n", ga->cl->x[j], ga->cl->y[j], ga->cl->z[j]);
   }
   for (j = 0; j < ga->bp->num; j++) {
   fprintf(fp, "%8.6f %8.6f %8.6f\n", ga->ss->x[j], ga->ss->y[j], ga->ss->z[j]);
   }
   for (j = 0; j < ga->bp->num; j++) {
   fprintf(fp, "%8.6f %8.6f %8.6f\n", ga->ps->x[j], ga->ps->y[j], ga->ps->z[j]);
   }

   fclose(fp);

   */
   // read pivot point
   ga->p_pivot = AllocPointStruct();
   ReadPointStruct(ga->p_pivot, "[blade pivot]", fn);

   return(1);
}


int CreateShell(struct gate *ga, int n_circles, int steps, float *xpl, float *ypl)
{

   // this function creates linestrips for the plot modul to plot the shell-diagram

   int i, j, n;
   float r_fac;
   float rx = float(ga->out_rad2 * ga->nopt / pow ((float)ga->H,0.5f));
   float ry = float(ga->Qopt / ( pow ((float)ga->H,(float)0.5) * pow((float)ga->out_rad2,(float)2.) ));
   float d_cross_x, d_cross_y;

   // generate eta-isolines
   for (j=0; j<n_circles; j++)
   {
      r_fac = float(1./n_circles*(j+1)-0.5/n_circles);
      n = j*2*steps;
      xpl[n+0]=rx + r_fac*rx;
      ypl[n+0]=ry;
      xpl[n+2*steps-1]=xpl[n+0];
      ypl[n+2*steps-1]=ypl[n+0];
      for (i=0; i<steps-1; i++)
      {
         xpl[n+2*i+1] = float(rx + r_fac*rx*cos(i*2*M_PI/steps));
         ypl[n+2*i+1] = float(ry + r_fac*ry*sin(i*2*M_PI/steps));

         xpl[n+2*i+2] = xpl[n+2*i+1];
         ypl[n+2*i+2] = ypl[n+2*i+1];
      }
   }

   // marker for actual position
   n = 2*n_circles*steps;
   d_cross_x = 0.66f * 0.1f * rx;
   d_cross_y = 0.1f * ry;

   xpl[n] = float(ga->out_rad2 / pow ((float)ga->H,(float)0.5) * ga->n - d_cross_x);
   ypl[n] = float(1. / ( pow ((float)ga->H,(float)0.5) * pow((float)ga->out_rad2,(float)2.) ) * ga->Q);
   xpl[n+1] = float(ga->out_rad2 / pow ((float)ga->H,(float)0.5)* ga->n + d_cross_x);
   ypl[n+1] = float(1. / ( pow ((float)ga->H,(float)0.5) * pow((float)ga->out_rad2,(float)2.) ) * ga->Q);

   xpl[n+2] = float(ga->out_rad2 / pow ((float)ga->H,(float)0.5) * ga->n);
   ypl[n+2] = float(1. / ( pow ((float)ga->H,(float)0.5) * pow((float)ga->out_rad2,(float)2.) ) * ga->Q - d_cross_y);
   xpl[n+3] = float(ga->out_rad2 / pow ((float)ga->H,(float)0.5) * ga->n);
   ypl[n+3] = float(1. / ( pow ((float)ga->H,(float)0.5) * pow((float)ga->out_rad2,(float)2.) ) * ga->Q + d_cross_y);

   return(1);
}


int CreateIsoAngleLines(struct gate *ga, int pos, int n_isolines, float xwmin, float xwmax, float *xpl, float *ypl)
{
   (void) xwmin;
   // draw isolines of blade angle in shell-diagram
   int i, j;
   float a0_a0opt;
   float beta;

   float f0, f1, f2, f_0, f_2;
   float a, b, c, d, e, m, b0;
   float x = 0;
   float res[3];
   int num = 0;

   // calculate beta for a0max (quadratic interpolation)

   i=10;
   while (ga->a0_beta[i+1] >  ga->a0_beta[i])
   {
      i++;
   }

   f0 = ga->a0_beta[i-1];
   f1 = ga->a0_beta[i];
   f2 = ga->a0_beta[i+1];
   f_0 = ( f1 - ga->a0_beta[i-2] ) / 2.0f;
   f_2 = ( ga->a0_beta[i+2] - f1 ) / 2.0f;
   /*printf("f0 = %5.2lf\n", f0);
   printf("f1 = %5.2lf\n", f1);
   printf("f2 = %5.2lf\n", f2);
   printf("f_0 = %5.2lf\n", f_0);
   printf("f_2 = %5.2lf\n", f_2);*/

   e = f0;
   d = f_0;
   c = 0.25f * ( -5.0f*f2 +16.0f*f1 +2.0f*f_2 -8.0f*f_0 -11.0f*f0 );
   b = 0.25f * ( 7.0f*f2 -16.0f*f1 -3.0f*f_2 +5.0f*f_0 +9.0f*f0 );
   a = f1 - b - c - d - e;

   cubic_equation(4*a, 3*b, 2*c, 1*d, &num, res);
   /*	printf("******* y = %5.3lfx4+%5.3lfx³+%5.3lfx²+%5.3lfx+%5.3lf=0\n", a, b, c, d, e);
   printf("******* Die Gleichung y' = %5.3lfx³+%5.3lfx²+%5.3lfx+%5.3lf=0\n", 4*a, 3*b, 2*c, 1*d);
   printf("******* hat %d Loesung(en)\n", num);
   for (i=0; i<num; i++)
   {
      printf("******* x = %5.2lf\n", i, res[i]);
   } */

   if (num==1)
   {
      x = res[0];
   }
   if (num==2)
   {
      if ( (res[0]>=0)&&(res[0]<=1) )
      {
         x = res[0];
      }
      else
      {
         x = res[1];
      }
   }
   if (num==3)
   {
      if ( (res[0]>=0)&&(res[0]<=1) )
      {
         x = res[0];
      }
      if ( (res[1]>=0)&&(res[1]<=1) )
      {
         x = res[1];
      }
      if ( (res[2]>=0)&&(res[2]<=1) )
      {
         x = res[2];
      }
   }

   if ( (x>1.)&&(x<0.) )
   {
      printf("Error in Routine CreateIsoAngleLines\n");
   }

   ga->beta_max = (i-11)*10 + x*10;
   ga->a0_max = a*x*x*x*x + b*x*x*x + c*x*x + d*x + e;
   ga->qmax = ga->Qopt * ga->a0_max;

   // calculate beta_min for a0min

   i=10;
   while (ga->a0_beta[i-1] < ga->a0_beta[i])
   {
      i--;
   }

   printf("******** closing a0 = %5.2lf\n", ga->a0_beta[i]);

   if (ga->a0_beta[i]<0.08)
   {
      ga->close=1;
   }
   else
   {
      ga->close=0;
   }

   m = (ga->a0_beta[i+2]-ga->a0_beta[i+1])/10.0f;
   b0 = ga->a0_beta[i+1]-m*(i+1-10)*10.0f;

   ga->beta_min = -b0/m;

   if (ga->close==1)
   {
      for (i=1; i<=n_isolines; i++)
      {

         beta = ga->beta_min + (ga->beta_max - ga->beta_min) * i / n_isolines;
         if (beta>0)
         {
#ifdef WIN32
            j = (int) ((beta/10.)+0.5) + 10;
#else
            j = (int) trunc(beta/10.) + 10;
#endif
         }
         else
         {
#ifdef WIN32
            j = (int) (((beta-10)/10.)+0.5) + 10;
#else
            j = (int) trunc((beta-10)/10.) + 10;
#endif
         }

         if (i==n_isolines)
         {
            a0_a0opt = ga->qmax / ga->Qopt;
         }
         else
         {
            a0_a0opt = ga->a0_beta[j] + ( ga->a0_beta[j+1] - ga->a0_beta[j]) * (beta-10*(j-10)) / 10.0f;
         }

         xpl[pos + 2*(i-1)] = 0.;
         ypl[pos + 2*(i-1)] = float(ga->Qopt * a0_a0opt / ( pow ((float)ga->H,(float)0.5) * pow((float)ga->out_rad2,(float)2.)));

         xpl[pos + 2*(i-1)+1 ] = xwmax;
         ypl[pos + 2*(i-1)+1] = float(ga->Qopt * a0_a0opt / ( pow ((float)ga->H,(float)0.5) * pow((float)ga->out_rad2,(float)2.)));
      }
   }

   return(1);
}


int RotateBlade(struct gate *ga, float angle, float x_piv, float y_piv)
{
   float roma[2][2];
   float p[3];
   int j;

   // translate back for rotation
   if (x_piv!=0)
   {
      for (j = 0; j < ga->bp->num; j++)
      {
         ga->cl->x[j] -= x_piv;
         ga->ps->x[j] -= x_piv;
         ga->ss->x[j] -= x_piv;
      }
   }
   if (y_piv!=0)
   {
      for (j = 0; j < ga->bp->num; j++)
      {
         ga->cl->y[j] -= y_piv;
         ga->ps->y[j] -= y_piv;
         ga->ss->y[j] -= y_piv;
      }
   }

   // rotates blade around (0|0)
   roma[0][0] = float(cos(angle));
   roma[0][1] = float(-sin(angle));
   roma[1][0] = float(sin(angle));
   roma[1][1] = float(cos(angle));
   for (j = 0; j < ga->bp->num; j++)
   {
      p[0] = roma[0][0] * ga->cl->x[j] + roma[0][1] * ga->cl->y[j];
      p[1] = roma[1][0] * ga->cl->x[j] + roma[1][1] * ga->cl->y[j];
      ga->cl->x[j] = p[0];
      ga->cl->y[j] = p[1];
      p[0] = roma[0][0] * ga->ps->x[j] + roma[0][1] * ga->ps->y[j];
      p[1] = roma[1][0] * ga->ps->x[j] + roma[1][1] * ga->ps->y[j];
      ga->ps->x[j] = p[0];
      ga->ps->y[j] = p[1];
      p[0] = roma[0][0] * ga->ss->x[j] + roma[0][1] * ga->ss->y[j];
      p[1] = roma[1][0] * ga->ss->x[j] + roma[1][1] * ga->ss->y[j];
      ga->ss->x[j] = p[0];
      ga->ss->y[j] = p[1];
   }

   // translate to original place
   if (x_piv!=0)
   {
      for (j = 0; j < ga->bp->num; j++)
      {
         ga->cl->x[j] += x_piv;
         ga->ps->x[j] += x_piv;
         ga->ss->x[j] += x_piv;
      }
   }
   if (y_piv!=0)
   {
      for (j = 0; j < ga->bp->num; j++)
      {
         ga->cl->y[j] += y_piv;
         ga->ps->y[j] += y_piv;
         ga->ss->y[j] += y_piv;
      }
   }

   return(1);

}


float get_a0(struct gate *ga, float sign)
{
   int i, j;

   double a0 = 10 * ga->pivot_rad;
   double dist;

   float angle;
   float hk_lw_x;
   float hk_lw_y;
   float roma[2][2];
   float p[3];

   angle = float(- sign * 2 * M_PI / ga->nob);
   roma[0][0] = float(cos(angle));
   roma[0][1] = float(-sin(angle));
   roma[1][0] = float(sin(angle));
   roma[1][1] = float(cos(angle));

   //exact calculation
   if (sign==1)
   {
      for (i = 0; i < ga->bp->num; i++)
      {
         p[0] = roma[0][0] * ga->ps->x[i] + roma[0][1] * ga->ps->y[i];
         p[1] = roma[1][0] * ga->ps->x[i] + roma[1][1] * ga->ps->y[i];
         hk_lw_x = p[0];
         hk_lw_y = p[1];
         for (j = 0; j < ga->bp->num; j++)
         {
            dist = sqrt ( (ga->ss->x[j]-hk_lw_x)*(ga->ss->x[j]-hk_lw_x) + (ga->ss->y[j]-hk_lw_y)*(ga->ss->y[j]-hk_lw_y) );
            if (dist < a0) { a0 = dist; }
         }
      }
   }
   else
   {
      for (i = 0; i < ga->bp->num; i++)
      {
         p[0] = roma[0][0] * ga->ps->x[i] + roma[0][1] * ga->ps->y[i];
         p[1] = roma[1][0] * ga->ps->x[i] + roma[1][1] * ga->ps->y[i];
         hk_lw_x = p[0];
         hk_lw_y = p[1];
         for (j = 0; j < ga->bp->num; j++)
         {
            dist = sqrt ( (ga->ps->x[j]-hk_lw_x)*(ga->ps->x[j]-hk_lw_x) + (ga->ps->y[j]-hk_lw_y)*(ga->ps->y[j]-hk_lw_y) );
            if (dist < a0) { a0 = dist; }
         }
      }
   }

   return(float(a0));

}


int cubic_equation(float a, float b, float c, float d, int *num, float res[3])
{

   // solves cubic equations
   /*
   Algorithmus
   Die kubische Gleichung
   ax³ + bx² + cx + d = 0
   hat entweder eine reelle und zwei komplexe Lösungen oder drei reelle Lösungen.
   Bringe sie zuerst durch die lineare Transformation
   x = (y - b)/(3a)
   auf die reduzierte kubische Gleichung
   y³ + 3py + q = 0
   die kein quadratisches Glied mehr enthält. Dabei ist
   p = 3ac - b²
   q = 2b³ - 9abc + 27a²d
   Berechne dann deren Diskriminante
   D = q² + 4p³.
   Falls D > 0, so gibt es eine reelle und zwei konjugiert komplexe Lösungen:
   u = 1/2 cubrt[ -4q + 4sqrt[D] ]
   v = 1/2 cubrt[ -4q - 4sqrt[D] ]
   y1 = u+v
   y2 = -(u+v)/2 + (u-v)/2 sqrt[3] i
   y3 = -(u+v)/2 - (u-v)/2 sqrt[3] i
   Falls D = 0, so gibt es drei reelle Lösungen, von denen mindestens zwei gleich sind.
   Man kann sie wie beim Fall D > 0 berechnen, nur sind y2 und y3 jetzt reell, da ihre
   Imaginärteile null werden.
   Falls D < 0, so gibt es drei verschiedene reelle Lösungen:
   cos(phi) = -q / (2sqrt[-p³])
   y1 = 2 sqrt[-p] cos( phi/3 )
   y2 = 2 sqrt[-p] cos( phi/3 + 120° )
   y3 = 2 sqrt[-p] cos( phi/3 + 240° )
   Die Lösungen der allgemeinen kubischen Gleichung lauten in jedem Fall
   x1 = (y1 - b) / (3a)
   x2 = (y2 - b) / (3a)
   x3 = (y3 - b) / (3a)
   Die Natur ihrer Lösungen ist dieselbe wie bei der reduzierten Gleichung.
   */

   double p, q;
   double D;

   float y[3] = { 0.0, 0.0, 0.0 };

   p = 3*a*c - b*b;
   q = 2*b*b*b - 9*a*b*c + 27*a*a*d;

   D = q*q + 4*p*p*p;

   if ((D>0) || (D==0))
   {
      float u, v;

      *num = 1;

      if ((-4*q + 4*sqrt(D))>0)
      {
         u = float(0.5 * pow((float)(-4*q + 4*sqrt(D)), (float)(1./3.) ));
      }
      else
      {
         u = float(-0.5 * pow((float)(4*q - 4*sqrt(D)), (float)(1./3.) ));
      }

      if ((-4*q - 4*sqrt(D))>0)
      {
         v = float(0.5 * pow((float)(-4*q - 4*sqrt(D)), (float)(1./3.) ));
      }
      else
      {
         v = float(-0.5 * pow((float)(4*q + 4*sqrt(D)), (float)(1./3.) ));
      }

      y[0] = u+v;

   }

   if (D==0)
   {
      float u, v;

      *num = 2;

      u = float(0.5 * pow((float)(-4*q), (float)(1./3.)));
      v = float(0.5 * pow((float)(-4*q), (float)(1./3.)));

      y[1] = -(u+v)/2;
      y[2] = -(u+v)/2;

   }

   if (D<0)
   {
      float phi;

      *num = 3;

      phi = float(acos ( -q / ( 2*sqrt(-p*p*p) )));

      y[0] = float(2 * sqrt(-p) * cos( phi/3 ));
      y[1] = float(2 * sqrt(-p) * cos( phi/3 + (2./3.) * M_PI ));
      y[2] = float(2 * sqrt(-p) * cos( phi/3 + (4./3.) * M_PI ));

   }

   res[0] = (y[0] - b) / (3*a);
   if (*num>1)
   {
      res[1] = (y[1] - b) / (3*a);
      if (*num==3) {res[2] = (y[2] - b) / (3*a);}
   }

   return(1);

}
