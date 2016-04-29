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
#include <Gate/include/gate.h>
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/curve.h"
#include "../General/include/profile.h"
#include "../General/include/parameter.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#define POINT               "point%d"
#define GA              "[gate data]"
#define GA_Q            "Q"
#define GA_H            "H"
#define GA_N            "n"
#define GA_QOPT            "Q opt"
#define GA_NOPT            "n opt"
#define GA_NOB          "number of blades"
#define GA_BANGLE       "blade angle"
#define BL_PIVOTRAD        "pivot radius"
#define BL              "[blade data]"
#define BL_CHORD        "chord length"
#define BL_PIVOT        "pivot location"
#define BL_ANGLE        "chord angle"
#define BL_PTHICK       "blade thickness"
#define BL_CAMBER       "maximum centre line camber"
#define BL_SHIFT        "blade profile shift"
#define CO              "[contour data]"
#define CO_INHEIGHT        "inlet height"
#define CO_INRAD        "inlet radius"
#define CO_INZ          "inlet z"
#define CO_OUTRAD1         "outlet inner diameter"
#define CO_OUTRAD2         "outlet outer diameter"
#define CO_OUTZ            "outlet z"
#define CO_SHROUDA         "shroud a"
#define CO_SHROUDB         "shroud b"
#define CO_HUBA            "hub a"
#define CO_HUBB            "hub b"
#define CO_NPHUBARC        "number hub arc points"
#define CAMBER          "[centre line camber distribution]"
#define PROFILE            "[blade profile]"
#define GA_HCHEIGHT        "cap height"
#define GR              "[grid data]"
#define GR_SAVE_GRID    "save grid"
#define GR_GEO_FROM_FILE   "read geometry from file"
#define GR_EDGE_PS         "blade ps area border"
#define GR_EDGE_SS         "blade ss area border"
#define GR_BOUND        "boundary layer thickness"
#define GR_N_RAD        "n points radial"
#define GR_N_BOUND         "n points boundary layer"
#define GR_N_OUT        "n points outlet"
#define GR_N_IN            "n points inlet"
#define GR_N_PS_BACK    "n points ps back"
#define GR_N_PS_FRONT      "n points ps front"
#define GR_N_SS_BACK    "n points ss back"
#define GR_N_SS_FRONT      "n points ss front"
#define GR_LEN_OUT_HUB     "start length outlet area hub"
#define GR_LEN_OUT_SHROUD  "start length outlet area shroud"
#define GR_LEN_EXPAND_IN   "length inlet expansion"
#define GR_LEN_EXPAND_OUT  "length outlet expansion"
#define GR_COMP_PS_BACK    "compression ps back"
#define GR_COMP_PS_FRONT   "compression ps front"
#define GR_COMP_SS_BACK    "compression ss back"
#define GR_COMP_SS_FRONT   "compression ss front"
#define GR_COMP_TRAIL      "compressoin trail"
#define GR_COMP_OUT        "compression outlet"
#define GR_COMP_IN         "compression inlet"
#define GR_COMP_BOUND      "compression boundary layer"
#define GR_COMP_MIDDLE     "compression middle"
#define GR_COMP_RAD        "compression radial"
#define GR_SHIFT_OUT    "shift outlet"
#define HC              "[hub contour]"
#define SC              "[shroud contour]"
#define BP              "[blade pivot]"
#define BSS             "[blade suction side]"
#define BPS             "[blade pressure side]"
#define BCL             "[blade center line]"

#define INIT_PORTION 25

struct gate *AllocGate(void)
{
   struct gate *ga;

   if ((ga = (struct gate *)calloc(1, sizeof(struct gate))) == NULL)
      fatal("memory for (struct gate *)");

   if ((ga->gr = (struct ggrid_para *)calloc(1, sizeof(struct ggrid_para))) == NULL)
      fatal("memory for (struct ggrid_para *)");

   return ga;
}


int ReadGate(struct geometry *g, const char *fn)
{

   fprintf(stderr, "entering ReadGate ...\n");

   char *tmp;
   
   g->ga = (struct gate *)AllocGate();

   // gate data

   // flow rate
   if ((tmp = IHS_GetCFGValue(fn, GA, GA_Q)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->Q);
      free(tmp);
   }
   // head
   if ((tmp = IHS_GetCFGValue(fn, GA, GA_H)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->H);
      free(tmp);
   }
   // rotation speed
   if ((tmp = IHS_GetCFGValue(fn, GA, GA_N)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->n);
      free(tmp);
   }
   // flow rate in bestpoint
   if ((tmp = IHS_GetCFGValue(fn, GA, GA_QOPT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->Qopt);
      free(tmp);
   }
   // rotation speed in bestpoint
   if ((tmp = IHS_GetCFGValue(fn, GA, GA_NOPT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->nopt);
      free(tmp);
   }
   // number of blades
   if ((tmp = IHS_GetCFGValue(fn, GA, GA_NOB)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->nob);
      free(tmp);
   }
   // blade angle
   if ((tmp = IHS_GetCFGValue(fn, GA, GA_BANGLE)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->bangle);
      free(tmp);
   }
   // pivot radius
   if ((tmp = IHS_GetCFGValue(fn, GA, BL_PIVOTRAD)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->pivot_rad);
      free(tmp);
   }
   // chord length
   if ((tmp = IHS_GetCFGValue(fn, BL, BL_CHORD)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->chord);
      free(tmp);
   }
   // pivot
   if ((tmp = IHS_GetCFGValue(fn, BL, BL_PIVOT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->pivot);
      free(tmp);
   }
   // angle
   if ((tmp = IHS_GetCFGValue(fn, BL, BL_ANGLE)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->angle);
      free(tmp);
   }
   // p_thick
   if ((tmp = IHS_GetCFGValue(fn, BL, BL_PTHICK)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->p_thick);
      free(tmp);
   }
   // maxcamb (maximum center line camber)
   if ((tmp = IHS_GetCFGValue(fn, BL, BL_CAMBER)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->maxcamb);
      free(tmp);
   }
   // bp_shift
   if ((tmp = IHS_GetCFGValue(fn, BL, BL_SHIFT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->bp_shift);
      free(tmp);
   }

   // contour data
   // in_height
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_INHEIGHT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->in_height);
      free(tmp);
   }
   // in_rad
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_INRAD)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->in_rad);
      free(tmp);
   }
   // in_z
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_INZ)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->in_z);
      free(tmp);
   }
   // out_rad1
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_OUTRAD1)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->out_rad1);
      free(tmp);
   }
   // out_rad2
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_OUTRAD2)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->out_rad2);
      free(tmp);
   }
   // out_z
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_OUTZ)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->out_z);
      free(tmp);
   }
   // shroud_a
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_SHROUDA)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->shroud_ab[0]);
      free(tmp);
   }
   // shroud_b
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_SHROUDB)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->shroud_ab[1]);
      free(tmp);
   }
   // hub_a
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_HUBA)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->hub_ab[0]);
      free(tmp);
   }
   // hub_b
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_HUBB)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->hub_ab[1]);
      free(tmp);
   }
   // num_hub_arc
   if ((tmp = IHS_GetCFGValue(fn, CO, CO_NPHUBARC)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->num_hub_arc);
      free(tmp);
   }

   // parameter sets: camber, profile
   g->ga->camb = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ga->camb, CAMBER, fn);

   g->ga->prof = AllocParameterStruct(0);
   ReadParameterSet(g->ga->prof, PROFILE, fn);

   g->ga->phub    = AllocPointStruct();
   g->ga->pshroud = AllocPointStruct();

   g->ga->phub_n    = AllocPointStruct();

   // grid parameters

   // save grid (geo, rb)
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_SAVE_GRID)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->savegrid);
      free(tmp);
   }

   // read geomatry from file (hub, shroud, profile)
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_GEO_FROM_FILE)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->geofromfile);
      free(tmp);
   }

   // edge_ps
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_EDGE_PS)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->edge_ps);
      free(tmp);
   }
   // edge_ss
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_EDGE_SS)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->edge_ss);
      free(tmp);
   }
   // bound_layer
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_BOUND)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->bound_layer);
      free(tmp);
   }
   // n points rad
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_N_RAD)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->n_rad);
      free(tmp);
   }
   // n points boundary layer
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_N_BOUND)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->n_bound);
      free(tmp);
   }
   // n points outlet
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_N_OUT)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->n_out);
      free(tmp);
   }
   // n points inlet
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_N_IN)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->n_in);
      free(tmp);
   }
   // n points blade ps back
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_N_PS_BACK)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->n_blade_ps_back);
      free(tmp);
   }
   // n points blade ps front
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_N_PS_FRONT)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->n_blade_ps_front);
      free(tmp);
   }
   // n points blade ss back
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_N_SS_BACK)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->n_blade_ss_back);
      free(tmp);
   }
   // n points blade ss front
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_N_SS_FRONT)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->n_blade_ss_front);
      free(tmp);
   }
   // length at start outlet area hub
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_LEN_OUT_HUB)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->len_start_out_hub);
      free(tmp);
   }
   // length at start outlet area shroud
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_LEN_OUT_SHROUD)) != NULL)
   {
      sscanf(tmp, "%d", &g->ga->gr->len_start_out_shroud);
      free(tmp);
   }
   // length of inlet expansion
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_LEN_EXPAND_IN)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->len_expand_in);
      free(tmp);
   }
   // length of outlet expansion
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_LEN_EXPAND_OUT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->len_expand_out);
      free(tmp);
   }
   // compression ps back
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_COMP_PS_BACK)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_ps_back);
      free(tmp);
   }
   // compression ps front
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_COMP_PS_FRONT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_ps_front);
      free(tmp);
   }
   // compression ss back
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_COMP_SS_BACK)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_ss_back);
      free(tmp);
   }
   // compression ss front
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_COMP_SS_FRONT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_ss_front);
      free(tmp);
   }
   // compression trail
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_COMP_TRAIL)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_trail);
      free(tmp);
   }
   // compression outlet
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_COMP_OUT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_out);
      free(tmp);
   }
   // compression inlet
   if ((tmp = IHS_GetCFGValue(fn, GR,GR_COMP_IN)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_in);
      free(tmp);
   }
   // compression boundary layer
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_COMP_BOUND)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_bound);
      free(tmp);
   }
   // compression middle
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_COMP_MIDDLE)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_middle);
      free(tmp);
   }
   // compression radial
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_COMP_RAD)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->comp_rad);
      free(tmp);
   }
   // shift outlet
   if ((tmp = IHS_GetCFGValue(fn, GR, GR_SHIFT_OUT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ga->gr->shift_out);
      free(tmp);
   }

   //
   // END OF READ PARAMETERS
   //

   // hub and shroud contours
   //CreateGA_Contours(g->ga);		//in ga_comp.c, only necessary if contour is spline-based

   // memory and first assignment of blade elements
   InitGA_BladeElements(g->ga);                   //in ga_comp.c

#ifdef DEBUG
   DumpGA(g->ga);
#endif                                         // DEBUG

   return(0);
}


void DumpGA(struct gate *ga)
{
   static int fcount = 0;
   char fname[255];
   FILE *ferr;
   int j;

   sprintf(fname, "ga_struct_%02d.txt", fcount++);
   ferr = fopen(fname, "w");
   if (ferr)
   {
      fprintf(ferr, " gate data:\n");
      fprintf(ferr, "ga->nob     = %d\n", ga->nob);
      fprintf(ferr, "ga->bangle  = %7.4f", ga->bangle);
      fprintf(ferr, "(%6.2f)\n", ga->bangle * 180.0/M_PI);
      fprintf(ferr, "ga->pivot_rad  = %7.4f\n", ga->pivot_rad);
      fprintf(ferr, "ga->chord   = %7.4f\n", ga->chord);
      fprintf(ferr, "ga->pivot   = %7.4f\n", ga->pivot);
      fprintf(ferr, "ga->angle   = %7.4f", ga->angle);
      fprintf(ferr, " (%6.2f)\n", ga->angle * 180.0/M_PI);
      fprintf(ferr, "ga->p_thick = %7.4f\n", ga->p_thick);
      fprintf(ferr, "ga->maxcamb = %7.4f\n", ga->maxcamb);
      fprintf(ferr, "ga->bp_shift = %7.4f\n", ga->bp_shift);

      //fprintf(ferr, "ga->hc_height = %7.4f\n", ga->hc_height);

      fprintf(ferr, "centre line camber distribution:\n");
      if (ga->camb)
      {
         for (j = 0; j < ga->camb->num; j++)
            fprintf(ferr, "loc[%3d] = %7.4f   val[%3d] = %7.4f\n", j, ga->camb->loc[j], j, ga->camb->val[j]);
      }
      fprintf(ferr, "profile:\n");
      if (ga->prof)
      {
         for (j = 0; j < ga->prof->num; j++)
            fprintf(ferr, "loc[%3d] = %7.4f   val[%3d] = %7.4f\n", j, ga->prof->loc[j], j, ga->prof->val[j]);
      }

      fprintf(ferr, "\n");
      fclose(ferr);
   }
}


#ifdef GNUPLOT
void WriteGNU_GA(struct gate *ga)
{
   int j;
   static int ncall = 0;
   float x, y, z;
   FILE *fp;
   char fname[255];

   sprintf(fname, "ga_blade3d_%02d.txt", ncall++);
   fp = fopen(fname, "w");
#ifdef CENTRE_LINE
   fprintf(fp, "# cetre line\n");
   for (i = 0; i < ga->be_num; i++)
   {
      for (j = 0; j < ga->be[i]->bp->num; j++)
      {
         x = ga->be[i]->cl->x[j];
         y = ga->be[i]->cl->y[j];
         z = ga->be[i]->cl->z[j];
         fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
      }
      fprintf(fp, "\n");
   }
   fprintf(fp, "\n\n");
#endif                                         // CENTRE_LINE

   fprintf(fp, "# pressure side\n");
   for (j = 0; j < ga->bp->num; j++)
   {
      x = ga->ps->x[j];
      y = ga->ps->y[j];
      z = ga->ps->z[j];
      fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
   }
   fprintf(fp, "\n\n");

   fprintf(fp, "# suction side\n");
   for (j = 0; j < ga->bp->num; j++)
   {
      x = ga->ss->x[j];
      y = ga->ss->y[j];
      z = ga->ss->z[j];
      fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
   }
   fprintf(fp, "\n\n");

#ifdef EXTENSION
   fprintf(fp, "# hub extension:\n");
#ifdef PRESSURE_SIDE
   for (i = 0; i < ga->mhub->ps_end->nump; i++)
   {
      x = ga->mhub->ps_end->x[i];
      y = ga->mhub->ps_end->y[i];
      z = ga->mhub->ps_end->z[i];
      fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
   }
   fprintf(fp, "\n");
#endif                                         // PRESSURE_SIDE
   for (i = 0; i < ga->mhub->ss_end->nump; i++)
   {
      x = ga->mhub->ss_end->x[i];
      y = ga->mhub->ss_end->y[i];
      z = ga->mhub->ss_end->z[i];
      fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
   }
   fprintf(fp, "\n\n");

#ifdef SHROUD_EXT
   fprintf(fp, "# shroud extension:\n");
   for (i = 0; i < ga->mshroud->ps_end->nump; i++)
   {
      x = ga->mshroud->ps_end->x[i];
      y = ga->mshroud->ps_end->y[i];
      z = ga->mshroud->ps_end->z[i];
      fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
   }
   fprintf(fp, "\n");
   for (i = 0; i < ga->mshroud->ss_end->nump; i++)
   {
      x = ga->mshroud->ss_end->x[i];
      y = ga->mshroud->ss_end->y[i];
      z = ga->mshroud->ss_end->z[i];
      fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
   }
   fprintf(fp, "\n");
#endif                                         // SHROUD_EXT
#endif                                         // EXTENSION
}
#endif                                            // GNUPLOT

#define P_LEN  31
int WriteGate(struct gate *ga, FILE *fp)
{
   int i;
   char section[100];

   sprintf(section, GA);
   fprintf(fp, "\n%s\n", section);
   fprintf(fp, "%*s = %6.2f\n", P_LEN, GA_Q, ga->Q);
   fprintf(fp, "%*s = %6.2f\n", P_LEN, GA_H, ga->H);
   fprintf(fp, "%*s = %6.2f\n", P_LEN, GA_N, ga->n);
   fprintf(fp, "%*s = %6.2f\n", P_LEN, GA_QOPT, ga->Qopt);
   fprintf(fp, "%*s = %6.2f\n", P_LEN, GA_NOPT, ga->nopt);
   fprintf(fp, "%*s = %d\n", P_LEN, GA_NOB, ga->nob);
   fprintf(fp, "%*s = %6.2f\n", P_LEN, GA_BANGLE, ga->bangle*180/M_PI);
   fprintf(fp, "%*s = %6.2f\n", P_LEN, BL_PIVOTRAD, ga->pivot_rad);

   sprintf(section, BL);
   fprintf(fp, "\n%s\n", section);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, BL_CHORD, ga->chord);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, BL_PIVOT, ga->pivot);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, BL_ANGLE, ga->angle*180./M_PI);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, BL_PTHICK, ga->p_thick);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, BL_CAMBER, ga->maxcamb);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, BL_SHIFT, ga->bp_shift);

   sprintf(section, CO);
   fprintf(fp, "\n%s\n", section);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_INHEIGHT, ga->in_height);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_INRAD, ga->in_rad);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_INZ, ga->in_z);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_OUTRAD1, ga->out_rad1);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_OUTRAD2, ga->out_rad2);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_OUTZ, ga->out_z);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_SHROUDA, ga->shroud_ab[0]);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_SHROUDB, ga->shroud_ab[1]);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_HUBA, ga->hub_ab[0]);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, CO_HUBB, ga->hub_ab[1]);
   fprintf(fp, "%*s = %5d\n", P_LEN, CO_NPHUBARC, ga->num_hub_arc);

   sprintf(section, CAMBER);
   fprintf(fp, "\n%s\n", section);
   for (i = 0; i < ga->camb->num; i++)
   {
      fprintf(fp, "%*s%d = %6.3f, %6.3lf\n", P_LEN, "stat", i, ga->camb->loc[i], ga->camb->val[i]);
   }

   sprintf(section, PROFILE);
   fprintf(fp, "\n%s\n", section);
   for (i = 0; i < ga->prof->num; i++)
   {
      fprintf(fp, "%*s%d = %6.3f, %6.3lf\n", P_LEN, "stat", i, ga->prof->loc[i], ga->prof->val[i]);
   }

   sprintf(section, GR);
   fprintf(fp, "\n%s\n", section);

   fprintf(fp, "%*s = %d\n", P_LEN, GR_SAVE_GRID, ga->gr->savegrid);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_GEO_FROM_FILE, ga->geofromfile);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_EDGE_PS, ga->gr->edge_ps);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_EDGE_SS, ga->gr->edge_ss);
   fprintf(fp, "%*s = %7.4f\n", P_LEN, GR_BOUND, ga->gr->bound_layer);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_N_RAD, ga->gr->n_rad);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_N_BOUND, ga->gr->n_bound);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_N_OUT, ga->gr->n_out);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_N_IN, ga->gr->n_in);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_N_PS_BACK, ga->gr->n_blade_ps_back);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_N_PS_FRONT, ga->gr->n_blade_ps_front);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_N_SS_BACK, ga->gr->n_blade_ss_back);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_N_SS_FRONT, ga->gr->n_blade_ss_front);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_LEN_OUT_HUB, ga->gr->len_start_out_hub);
   fprintf(fp, "%*s = %d\n", P_LEN, GR_LEN_OUT_SHROUD, ga->gr->len_start_out_shroud);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_LEN_EXPAND_IN, ga->gr->len_expand_in);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_LEN_EXPAND_OUT, ga->gr->len_expand_out);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_PS_BACK   , ga->gr->comp_ps_back);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_PS_FRONT, ga->gr->comp_ps_front);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_SS_BACK   , ga->gr->comp_ss_back);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_SS_FRONT, ga->gr->comp_ss_front);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_TRAIL, ga->gr->comp_trail);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_OUT , ga->gr->comp_out);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_IN, ga->gr->comp_in);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_BOUND, ga->gr->comp_bound);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_MIDDLE , ga->gr->comp_middle);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_COMP_RAD, ga->gr->comp_rad);
   fprintf(fp, "%*s = %5.2f\n", P_LEN, GR_SHIFT_OUT, ga->gr->shift_out);

   fprintf(fp, "\n");
   fprintf(fp, "#these parameters are only used if 'read geometry from file = 1'\n");
   sprintf(section, HC);
   fprintf(fp, "%s\n", section);

   sprintf(section, SC);
   fprintf(fp, "\n%s\n", section);

   sprintf(section, BP);
   fprintf(fp, "\n%s\n", section);

   sprintf(section, BSS);
   fprintf(fp, "\n%s\n", section);

   sprintf(section, BPS);
   fprintf(fp, "\n%s\n", section);

   sprintf(section, BCL);
   fprintf(fp, "\n%s\n", section);

   return 1;
}


int ReadPointStruct(struct Point *p, const char *sec, const char *fn)
{
   int i, num;
   float x, y, z;
   char *tmp;
   char key[127];

   num = 0;
   for (i = 0; ; i++)
   {
      sprintf(key, POINT, i);
      if ((tmp = IHS_GetCFGValue(fn, sec, key)) != NULL)
      {
         sscanf(tmp, "%f, %f, %f", &x, &y, &z);
         free(tmp);
         num = AddPoint(p, x, y, z);
      }
      else
         break;
   }
   return(num);
}
