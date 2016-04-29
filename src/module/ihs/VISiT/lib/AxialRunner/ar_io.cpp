#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <strings.h>
#endif
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include "../General/include/cfg.h"
#include "../General/include/geo.h"
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/curve.h"
#include "../General/include/profile.h"
#include "../General/include/parameter.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"
#include "include/axial.h"
#include "include/ar_contours.h"
#include "include/ar_meridiancont.h"
#include "include/ar_condarea.h"
#include "include/ar_arclenbe.h"
#include "include/ar_updatemer.h"
#include "include/ar_createbe.h"
#include "include/ar_initbe.h"

#define AR        "[runner data]"
#define AR_NOB    "number of blades"
#define AR_ENLACE   "blade enlacement"
#define AR_PIVOT  "pivot location"
#define AR_LEPART "leading edge polygon partition"
#define AR_TEPART "trailing edge polygon partition"
#define AR_BANGLE   "blade angle"

#define AR_DIM      "[machine dimensions]"
#define AR_SDIAM  "shroud diameter (reference)"
#define AR_HDIAM    "hub diameter"
#define AR_IEXTH    "inlet extension height"
#define AR_IEXTDIAM "inlet extension diameter"
#define AR_IARB_ANGLE "inlet pitch angle"
#define AR_IARB_PART  "inlet spline partitions"
#define AR_HCORN    "semi-vertical/horizontal axis hub bend"
#define AR_HRUN     "runner height below inlet"
#define AR_HSDIAM   "hub sphere diameter"
#define AR_SSDIAM   "shroud sphere diameter"
#define AR_SRAD     "shroud bend radius start"
#define AR_SANG     "shroud bend start arc angle"
#define AR_ERAD     "shroud bend radius end"
#define AR_HNOS     "number of sections hub bend"
#define AR_SHEMI    "shroud hemisphere"
#define AR_SCOUNT   "shroud sphere counter arc"
#define AR_SNOS     "number of sections counter arc"
#define AR_PCAP     "hub cap point%d diameter/height"
#define AR_HDRAFT   "draft tube inlet height below runner"
#define AR_DDRAFT   "draft tube inlet diameter"
#define AR_ANGDRAFT "draft tube opening angle"

#define AR_MODEL    "[geometry model]"
#define AR_FORCECAMB    "force camber"
#define AR_EULER  "euler equation"
#define AR_MINL      "inlet extension"
#define AR_MBEND  "bend region"
#define AR_MOUTL  "outlet extension"
#define AR_MARBITRARY  "inlet arbitrary"

#define AR_DES    "[design data]"
#define DD_DIS    "machine discharge"
#define DD_HEAD      "machine head"
#define DD_REVS      "machine revolutions"

#define AR_LEDAT  "[leading edge data]"
#define AR_TEDAT  "[trailing edge data]"
#define AR_ICON      "hub constriction"
#define AR_OCON      "shroud constriction"
#define AR_ZCON      "zero constriction"

#define AR_IANGLE "[inlet angle]"
#define AR_MIANGLE   "[inlet angle modification]"
#define AR_OANGLE "[outlet angle]"
#define AR_MOANGLE   "[outlet angle modification]"
#define AR_PTHICK "[blade thickness]"
#define AR_TETHICK   "[trailing edge thickness]"
#define AR_CAMB   "[centre line camber]"
#define AR_CAMBPOS   "[centre line camber position]"
#define AR_PROF      "[blade profile]"
#define BP_SHIFT  "[blade profile shift]"

#define AR_BE     "[blade element bias]"
#define AR_BENUM  "number of elements"
#define AR_BIAS     "bias factor"
#define AR_BTYPE    "bias type"
#define AR_EXTRA  "extrapolation"

#define HUB_EXT
#define SHROUD_EXT

#define STAT "stat%d"
#define L_LEN 45

struct axial *AllocAxialRunner(void)
{
   struct axial *ar;

   if ((ar = (struct axial *)calloc(1, sizeof(struct axial))) != NULL)
   {
      if ((ar->mod = (struct model *)calloc(1, sizeof(struct model))) == NULL)
         fatal("memory for (struct model *)");
      if ((ar->des = (struct design *)calloc(1, sizeof(struct design))) == NULL)
         fatal("memory for (struct design *)");
      if ((ar->le = (struct edge *)calloc(1, sizeof(struct edge))) == NULL)
         fatal("memory for (struct edge *)");
      if ((ar->te = (struct edge *)calloc(1, sizeof(struct edge))) == NULL)
         fatal("memory for (struct edge *)");
   }
   else
   {
      fatal("memory for (struct axial *)");
   }
   return ar;
}


int ReadAxialRunner(struct geometry *g, const char *fn)
{
   int i;
   float x, y, z;
   char key[128];
   char *tmp;

   g->ar = AllocAxialRunner();

   // runner data
   if ((tmp = IHS_GetCFGValue(fn, AR, AR_NOB)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->nob);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR, AR_BANGLE)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->bangle);
      free(tmp);
      //g->ar->bangle *= M_PI/180.0;
   }
   if ((tmp = IHS_GetCFGValue(fn, AR, AR_ENLACE)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->enlace);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR, AR_PIVOT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->piv);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR, AR_LEPART)) != NULL)
   {
      sscanf(tmp, "%f, %f, %f", &g->ar->le_part[0],
         &g->ar->le_part[1], &g->ar->le_part[2]);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR, AR_TEPART)) != NULL)
   {
      sscanf(tmp, "%f, %f", &g->ar->te_part[0], &g->ar->te_part[1]);
      free(tmp);
   }

   // geometry model
   if ((tmp = IHS_GetCFGValue(fn, AR_MODEL, AR_EULER)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->euler);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_MODEL, AR_FORCECAMB)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->clspline);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_MODEL, AR_MINL)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->mod->inl);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_MODEL, AR_MBEND)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->mod->bend);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_MODEL, AR_MOUTL)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->mod->outl);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_MODEL, AR_MARBITRARY)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->mod->arbitrary);
      free(tmp);
   }
   if (g->ar->mod->inl && !g->ar->mod->bend)
      g->ar->mod->bend = 1;

   // machine dimensions
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_HDIAM)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->diam[0]);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_SDIAM)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->ref);
      free(tmp);
      g->ar->diam[1] = 1.0;
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_IEXTH)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->h_inl_ext);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_IEXTDIAM)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->d_inl_ext);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_IARB_ANGLE)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->arb_angle);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_IARB_PART)) != NULL)
   {
      sscanf(tmp, "%f, %f", &g->ar->arb_part[0],&g->ar->arb_part[1]);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_SRAD)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->r_shroud[0]);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_ERAD)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->r_shroud[1]);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_SANG)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->ang_shroud);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_HCORN)) != NULL)
   {
      sscanf(tmp, "%f, %f", &g->ar->a_hub, &g->ar->b_hub);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_HNOS)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->hub_nos);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_HRUN)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->h_run);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_HSDIAM)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->d_hub_sphere);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_SSDIAM)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->d_shroud_sphere);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_SHEMI)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->shroud_hemi);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_SCOUNT)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->shroud_counter_rad);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_SNOS)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->counter_nos);
      free(tmp);
   }
   g->ar->p_hubcap = AllocPointStruct();
   for (i = 0; ; i++)
   {
      sprintf(key, AR_PCAP, i);
      x = y = z = 0.0;
      if ((tmp = IHS_GetCFGValue(fn, AR_DIM, key)) != NULL)
      {
         sscanf(tmp, "%f, %f", &x, &z);
         free(tmp);
         AddPoint(g->ar->p_hubcap, x, y, z);
         dprintf(6, "ar->hubcap[]: i=%2d: x=%f, y=%f, z=%f\n", i, x, y, z);
      }
      else
      {
         if (i == 0)
         {
            fprintf(stderr, "\nERROR: keyword %s missing in section %s in file %s\n", key, AR_DIM,fn);
            exit(1);
         }
         break;
      }
   }
   g->ar->cap_nop = g->ar->p_hubcap->nump;
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_HDRAFT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->h_draft);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_DDRAFT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->d_draft);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DIM, AR_ANGDRAFT)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->ang_draft);
      free(tmp);
   }

   // design data
   if ((tmp = IHS_GetCFGValue(fn, AR_DES, DD_DIS)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->des->dis);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DES, DD_HEAD)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->des->head);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_DES, DD_REVS)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->des->revs);
      free(tmp);
   }

   // leading edge data
   if ((tmp = IHS_GetCFGValue(fn, AR_LEDAT, AR_ICON)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->le->con[0]);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_LEDAT, AR_OCON)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->le->con[1]);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_LEDAT, AR_ZCON)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->le->nocon);
      free(tmp);
   }

   // trailing edge data
   if ((tmp = IHS_GetCFGValue(fn, AR_TEDAT, AR_ICON)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->te->con[0]);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_TEDAT, AR_OCON)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->te->con[1]);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_TEDAT, AR_ZCON)) != NULL)
   {
      sscanf(tmp, "%f", &g->ar->te->nocon);
      free(tmp);
   }
   if (CheckAR_EdgeConstriction(g->ar))
      fatal("runner edge constriction too large!");

   // parameter sets
   g->ar->iang = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ar->iang, AR_IANGLE, fn);
   //Parameter2Radians(g->ar->iang);

   g->ar->oang = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ar->oang, AR_OANGLE, fn);
   //Parameter2Radians(g->ar->oang);

   g->ar->t = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ar->t, AR_PTHICK, fn);

   g->ar->tet = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ar->tet, AR_TETHICK, fn);

   g->ar->camb = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ar->camb, AR_CAMB, fn);

   g->ar->camb_pos = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ar->camb_pos, AR_CAMBPOS, fn);

   g->ar->bps = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ar->bps, BP_SHIFT, fn);

   g->ar->mod_iang = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ar->mod_iang,AR_MIANGLE,fn);
   //Parameter2Radians(g->ar->mod_iang);

   g->ar->mod_oang = AllocParameterStruct(INIT_PORTION);
   ReadParameterSet(g->ar->mod_oang,AR_MOANGLE,fn);
   //Parameter2Radians(g->ar->mod_oang);

   dprintf(5, "\ninlet angle data:\n");
   DumpParameterSet(g->ar->iang);
   dprintf(5, "\ninlet angle modification data:\n");
   DumpParameterSet(g->ar->mod_iang);
   dprintf(5, "\noutlet angle:\n");
   DumpParameterSet(g->ar->oang);
   dprintf(5, "\noutlet angle modification data:\n");
   DumpParameterSet(g->ar->mod_oang);
   dprintf(5, "\nblade thickness data:\n");
   DumpParameterSet(g->ar->t);
   dprintf(5, "\ntrailing edge thicknes data:\n");
   DumpParameterSet(g->ar->tet);
   dprintf(5, "\ncentre line camber data:\n");
   DumpParameterSet(g->ar->camb);
   dprintf(5, "\ncentre line camber position data:\n");
   DumpParameterSet(g->ar->camb_pos);
   dprintf(5, "\nprofile shift data:\n");
   DumpParameterSet(g->ar->bps);

   // blade element bias
   if ((tmp = IHS_GetCFGValue(fn, AR_BE, AR_BENUM)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->be_num);
      free(tmp);
      if (g->ar->be_num < 2)
         fatal("number of blade elements less than 2!");
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_BE, AR_BIAS)) != NULL )
   {
      sscanf(tmp, "%f", &g->ar->be_bias);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_BE, AR_BTYPE)) != NULL )
   {
      sscanf(tmp, "%d", &g->ar->be_type);
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, AR_BE, AR_EXTRA)) != NULL)
   {
      sscanf(tmp, "%d", &g->ar->extrapol);
      free(tmp);
   }

   // blade profile data
   g->ar->bp = AllocBladeProfile();
   ReadBladeProfile(g->ar->bp, AR_PROF, fn);
   dprintf(5, "\nblade profile data:\n");
   DumpBladeProfile(g->ar->bp);

   //
   // END OF READ PARAMETERS,
   // GENERATION OF GEOMETRY
   //

   InitAR_BladeElements(g->ar);

#ifndef COVISE_MODULE
   CreateAR_Contours(g->ar);
   CreateAR_MeridianContours(g->ar);
   CreateAR_ConduitAreas(g->ar);
   ArclenAR_BladeElements(g->ar);
   CreateAR_BladeElements(g->ar);
   UpdateAR_Meridians(g->ar);

#ifdef PLOT_BLADE_EDGES
   PlotAR_BladeEdges(g->ar);
#endif                                         // PLOT_BLADE_EDGES
#endif                                         // COVISE_MODULE

   DumpAR(g->ar);

   return(0);
}


int CheckAR_EdgeConstriction(struct axial *ar)
{
   int err = 0;
   float totcon[2];
   float tol = 0.001f;

   totcon[0] = ar->le->con[0] + ar->te->con[0];
   totcon[1] = ar->le->con[1] + ar->te->con[1];
   if ((totcon[0] - 1.0) >= tol)
   {
      err = 1;
   }
   else if ((totcon[1] - 1.0) >= tol)
   {
      err = 1;
   }
   return(err);
}


void DumpAR(struct axial *ar)
{
   static int fcount = 0;
   char fname[255];
   FILE *ferr=NULL;
   char *fn;
   int i;

   if (!ar)
      return;
   sprintf(fname, "ar_struct_%02d.txt", fcount++);
   fn = DebugFilename(fname);
   if(fn)
   ferr = fopen(fn, "w");
   if (ferr)
   {
      fprintf(ferr, " [RUNNER DATA]\n");
      fprintf(ferr, "               ar->nob = %d\n", ar->nob);
      fprintf(ferr, "            ar->bangle = %7.4f", ar->bangle);
      fprintf(ferr, " (%6.4f)\n", RAD(ar->bangle));
      fprintf(ferr, "            ar->enlace = %7.4f\n", ar->enlace);
      fprintf(ferr, "               ar->piv = %7.4f\n", ar->piv);
      fprintf(ferr, "           ar->le_part = %7.4f, %7.4f\n", ar->le_part[0],ar->le_part[1]);
      fprintf(ferr, "           ar->te_part = %7.4f, %7.4f\n", ar->te_part[0],ar->te_part[1]);
      fprintf(ferr, "\n [MACHINE DIMENSIONS]\n");
      fprintf(ferr, "               ar->ref = %7.4f\n", ar->ref);
      fprintf(ferr, "           ar->diam[0] = %7.4f\n", ar->diam[0]);
      fprintf(ferr, "           ar->diam[1] = %7.4f\n", ar->diam[1]);
      fprintf(ferr, "         ar->h_inl_ext = %7.4f\n", ar->h_inl_ext);
      fprintf(ferr, "         ar->d_inl_ext = %7.4f\n", ar->d_inl_ext);
      fprintf(ferr, "       ar->r_shroud[0] = %7.4f\n", ar->r_shroud[0]);
      fprintf(ferr, "        ar->ang_shroud = %7.4f", ar->ang_shroud);
      fprintf(ferr, " (%6.4f)\n", RAD(ar->ang_shroud));
      fprintf(ferr, "       ar->r_shroud[1] = %7.4f\n", ar->r_shroud[1]);
      fprintf(ferr, "             ar->a_hub = %7.4f\n", ar->a_hub);
      fprintf(ferr, "             ar->b_hub = %7.4f\n", ar->b_hub);
      fprintf(ferr, "           ar->hub_nos = %d\n", ar->hub_nos);
      fprintf(ferr, "             ar->h_run = %7.4f\n", ar->h_run);
      fprintf(ferr, "      ar->d_hub_sphere = %7.4f\n", ar->d_hub_sphere);
      fprintf(ferr, "           ar->cap_nop = %d\n", ar->cap_nop);
      for (i = 0; i < ar->hub_nos; i++)
      {
         fprintf(ferr, "        hub cap point%d = %5.3f, %5.3f, %5.3f\n", i, ar->p_hubcap->x[i], ar->p_hubcap->y[i], ar->p_hubcap->z[i]);
      }
      fprintf(ferr, "   ar->d_shroud_sphere = %7.4f\n", ar->d_shroud_sphere);
      fprintf(ferr, "       ar->shroud_hemi = %d\n", ar->shroud_hemi);
      fprintf(ferr, "ar->shroud_counter_rad = %d\n", ar->shroud_counter_rad);
      fprintf(ferr, "           ar->h_draft = %7.4f\n", ar->h_draft);
      fprintf(ferr, "           ar->d_draft = %7.4f\n", ar->d_draft);
      fprintf(ferr, "         ar->ang_draft = %7.4f", ar->ang_draft);
      fprintf(ferr, " (%6.4f)\n", RAD(ar->ang_draft));
      fprintf(ferr, "\n [GEOMETRY MODEL]\n");
      fprintf(ferr, "         ar->mod->inl = %d\n", ar->mod->inl);
      fprintf(ferr, "        ar->mod->bend = %d\n", ar->mod->bend);
      fprintf(ferr, "        ar->mod->outl = %d\n", ar->mod->bend);
      fprintf(ferr, "\n [BLADE ELEMENT BIAS]\n");
      fprintf(ferr, "            ar->be_num = %d\n", ar->be_num);
      fprintf(ferr, "           ar->be_bias = %7.4f ", ar->be_bias);
      fprintf(ferr, "(%d)\n", ar->be_type);
      fprintf(ferr, "          ar->extrapol = %7.4f\n", ar->be_bias);
      if (ar->be != NULL)
      {
         fprintf(ferr, "\n [BLADE ELEMENTS]\n");
         for(i = 0; i < ar->be_num; i++)
         {
            fprintf(ferr, "        ar->be[%02d]->para = %7.4f\n", i, ar->be[i]->para);
            fprintf(ferr, "    ar->be[%02d]->angle[0] = %7.4f", i, ar->be[i]->angle[0]);
            fprintf(ferr, " (%6.4f)\n", RAD(ar->be[i]->angle[0]));
            fprintf(ferr, "    ar->be[%02d]->angle[1] = %7.4f", i, ar->be[i]->angle[1]);
            fprintf(ferr, " (%6.4f)\n", RAD(ar->be[i]->angle[1]));
            fprintf(ferr, "ar->be[%02d]->mod_angle[0] = %7.4f", i, ar->be[i]->mod_angle[0]);
            fprintf(ferr, " (%6.4f)\n", RAD(ar->be[i]->mod_angle[0]));
            fprintf(ferr, "ar->be[%02d]->mod_angle[1] = %7.4f", i, ar->be[i]->mod_angle[1]);
            fprintf(ferr, " (%6.4f)\n", RAD(ar->be[i]->mod_angle[1]));
            fprintf(ferr, " ar->be[%02d]->con_area[0] = %7.4f\n", i, ar->be[i]->con_area[0]);
            fprintf(ferr, " ar->be[%02d]->con_area[1] = %7.4f\n", i, ar->be[i]->con_area[1]);
            fprintf(ferr, "     ar->be[%02d]->p_thick = %7.4f\n", i, ar->be[i]->p_thick);
            fprintf(ferr, "    ar->be[%02d]->te_thick = %7.4f\n", i, ar->be[i]->te_thick);
            fprintf(ferr, "        ar->be[%02d]->camb = %7.4f\n", i, ar->be[i]->camb);
            fprintf(ferr, "    ar->be[%02d]->camb_pos = %7.4f\n", i, ar->be[i]->camb_pos);
            fprintf(ferr, "     ar->be[%02d]->te_wrap = %7.4f", i, ar->be[i]->te_wrap);
            fprintf(ferr, " (%6.4f)\n", GRAD(ar->be[i]->te_wrap));
            fprintf(ferr, "     ar->be[%02d]->bl_wrap = %7.4f", i, ar->be[i]->bl_wrap);
            fprintf(ferr, " (%6.4f)\n", GRAD(ar->be[i]->bl_wrap));
            fprintf(ferr, "    ar->be[%02d]->bp_shift = %7.4f\n", i, ar->be[i]->bp_shift);
            fprintf(ferr, "      ar->be[%02d]->cl_len = %7.4f\n", i, ar->be[i]->cl_len);
            fprintf(ferr, "         ar->be[%02d]->rad = %7.4f\n", i, ar->be[i]->rad);
            fprintf(ferr, "       ar->be[%02d]->pivot = %7.4f\n", i, ar->be[i]->pivot);
            fprintf(ferr, "\n");
            fprintf(ferr, "        ar->me[%02d]->para = %7.4f\n", i, ar->me[i]->para);
            fprintf(ferr, "\n");
         }
      }
      fclose(ferr);
   }
}


void WriteGNU_AR(struct axial *ar)
{
   int i,j;
   static int ncall = 0;
   float x, y, z;
   FILE *fp=NULL;
   char fname[255];
   char *fn;

   sprintf(fname, "ar_blade3d_%02d.txt", ncall++);
   fn = DebugFilename(fname);
   if (fn && (fp = fopen(fn, "w")) != NULL)
   {
      fprintf(fp, "# centre line\n");
      for (i = 0; i < ar->be_num; i++)
      {
         for (j = 0; j < ar->be[i]->cl_cart->nump; j++)
         {
            x = ar->be[i]->cl_cart->x[j];
            y = ar->be[i]->cl_cart->y[j];
            z = ar->be[i]->cl_cart->z[j];
            fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
         }
         fprintf(fp, "\n");
      }
      fprintf(fp, "\n\n");

      fprintf(fp, "# pressure side\n");
      for (i = 0; i < ar->be_num; i++)
      {
         for (j = 0; j < ar->be[i]->ps_cart->nump; j++)
         {
            x = ar->be[i]->ps_cart->x[j];
            y = ar->be[i]->ps_cart->y[j];
            z = ar->be[i]->ps_cart->z[j];
            fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
         }
         fprintf(fp, "\n");
      }
      fprintf(fp, "\n\n");

      fprintf(fp, "# suction side\n");
      for (i = 0; i < ar->be_num; i++)
      {
         for (j = 0; j < ar->be[i]->ss_cart->nump; j++)
         {
            x = ar->be[i]->ss_cart->x[j];
            y = ar->be[i]->ss_cart->y[j];
            z = ar->be[i]->ss_cart->z[j];
            fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
         }
         fprintf(fp, "\n");
      }
      fprintf(fp, "\n\n");

#ifdef HUB_EXT
      fprintf(fp, "# hub extension:\n");
      for (i = 0; i < ar->mhub->ps_int->nump; i++)
      {
         x = ar->mhub->ps_int->x[i];
         y = ar->mhub->ps_int->y[i];
         z = ar->mhub->ps_int->z[i];
         fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
      }
      fprintf(fp, "\n");
      for (i = 0; i < ar->mhub->ss_int->nump; i++)
      {
         x = ar->mhub->ss_int->x[i];
         y = ar->mhub->ss_int->y[i];
         z = ar->mhub->ss_int->z[i];
         fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
      }
      fprintf(fp, "\n\n");
#endif                                      // HUB_EXT

#ifdef SHROUD_EXT
      fprintf(fp, "# shroud extension:\n");
      for (i = 0; i < ar->mshroud->ps_int->nump; i++)
      {
         x = ar->mshroud->ps_int->x[i];
         y = ar->mshroud->ps_int->y[i];
         z = ar->mshroud->ps_int->z[i];
         fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
      }
      fprintf(fp, "\n");
      for (i = 0; i < ar->mshroud->ss_int->nump; i++)
      {
         x = ar->mshroud->ss_int->x[i];
         y = ar->mshroud->ss_int->y[i];
         z = ar->mshroud->ss_int->z[i];
         fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
      }
      fprintf(fp, "\n");
#endif                                      // SHROUD_EXT
      fclose(fp);
   }
}


int WriteAxialRunner(struct axial *ar, FILE *fp)
{
   int i;
   char buf[200];

   dprintf(1,"Entering WriteAxialRunner() ... \n");

   // write data to file
   fprintf(fp, "\n%s\n",AR);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_NOB, ar->nob);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_ENLACE, ar->enlace);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_PIVOT, ar->piv);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_BANGLE, ar->bangle);
   fprintf(fp, "%*s = %f, %f, %f\n", L_LEN, AR_LEPART, ar->le_part[0],
      ar->le_part[1],ar->le_part[2]);
   fprintf(fp, "%*s = %f, %f\n", L_LEN, AR_TEPART, ar->te_part[0], ar->te_part[1]);
   fprintf(fp, "\n%s\n",AR_DIM);
   fprintf(fp, "# reference diameter [m]\n");
   fprintf(fp, "%*s = %f\n", L_LEN, AR_SDIAM, ar->ref);
   fprintf(fp, "# relative quantities [-]\n");
   fprintf(fp, "%*s = %f\n", L_LEN, AR_HDIAM, ar->diam[0]);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_IEXTH, ar->h_inl_ext);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_IEXTDIAM, ar->d_inl_ext);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_IARB_ANGLE, ar->arb_angle);
   fprintf(fp, "%*s = %f, %f\n", L_LEN, AR_IARB_PART,
      ar->arb_part[0],ar->arb_part[1]);
   fprintf(fp, "%*s = %f, %f\n", L_LEN, AR_HCORN, ar->a_hub, ar->b_hub);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_HRUN, ar->h_run);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_HSDIAM, ar->d_hub_sphere);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_SSDIAM, ar->d_shroud_sphere);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_SRAD, ar->r_shroud[0]);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_ERAD, ar->r_shroud[1]);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_SANG, ar->ang_shroud);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_HNOS, ar->hub_nos);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_SHEMI, ar->shroud_hemi);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_SCOUNT, ar->shroud_counter_rad);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_SNOS, ar->counter_nos);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_HDRAFT, ar->h_draft);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_DDRAFT, ar->d_draft);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_ANGDRAFT, ar->ang_draft);
   dprintf(6," WriteAxialRunner(): %s \n",AR_PCAP);
   for(i = 0; i < ar->p_hubcap->nump; i++)
   {
      dprintf(6," WriteAxialRunner(): %d (%d)\n",
         i, ar->p_hubcap->nump);
      sprintf(buf, AR_PCAP, i);
      dprintf(6," WriteAxialRunner(): buf  = %s\n",buf);
      dprintf(6," WriteAxialRunner(): x[i] = %f\n",
         ar->p_hubcap->x[i]);
      dprintf(6," WriteAxialRunner(): z[i] = %f\n",
         ar->p_hubcap->z[i]);
      fprintf(fp,"%*s = %f, %f\n", L_LEN, buf,
         ar->p_hubcap->x[i], ar->p_hubcap->z[i]);
   }
   dprintf(6," WriteAxialRunner(): %s \n",AR_MODEL);
   fprintf(fp, "\n%s\n",AR_MODEL);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_EULER, ar->euler);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_FORCECAMB, ar->clspline);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_MINL, ar->mod->inl);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_MBEND, ar->mod->bend);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_MOUTL, ar->mod->outl);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_MARBITRARY, ar->mod->arbitrary);
   fprintf(fp, "\n%s\n",AR_DES);
   fprintf(fp, "%*s = %f\n", L_LEN, DD_DIS, ar->des->dis);
   fprintf(fp, "%*s = %f\n", L_LEN, DD_HEAD, ar->des->head);
   fprintf(fp, "%*s = %f\n", L_LEN, DD_REVS,ar->des->revs);
   fprintf(fp, "\n%s\n",AR_LEDAT);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_ICON, ar->le->con[0]);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_OCON, ar->le->con[1]);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_ZCON, ar->le->nocon);
   fprintf(fp, "\n%s\n",AR_TEDAT);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_ICON, ar->te->con[0]);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_OCON, ar->te->con[1]);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_ZCON, ar->te->nocon);

   dprintf(6," WriteAxialRunner(): %s \n",AR_IANGLE);
   fprintf(fp, "\n%s\n",AR_IANGLE);
   for(i = 0; i < ar->be_num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
         ar->be[i]->para, ar->be[i]->angle[0]);
   }
   fprintf(fp, "\n%s\n",AR_OANGLE);
   for(i = 0; i < ar->be_num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
         ar->be[i]->para, ar->be[i]->angle[1]);
   }
   fprintf(fp, "\n%s\n",AR_MIANGLE);
   for(i = 0; i < ar->be_num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
         ar->be[i]->para, ar->be[i]->mod_angle[0]);
   }
   fprintf(fp, "\n%s\n",AR_MOANGLE);
   for(i = 0; i < ar->be_num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
         ar->be[i]->para, ar->be[i]->mod_angle[1]);
   }
   fprintf(fp, "\n%s\n",AR_PTHICK);
   for(i = 0; i < ar->be_num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
         ar->be[i]->para, ar->be[i]->p_thick);
   }
   fprintf(fp, "\n%s\n",AR_TETHICK);
   for(i = 0; i < ar->be_num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
         ar->be[i]->para, ar->be[i]->te_thick);
   }
   fprintf(fp, "\n%s\n",AR_CAMB);
   for(i = 0; i < ar->be_num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
         ar->be[i]->para, ar->be[i]->camb);
   }
   fprintf(fp, "\n%s\n",AR_CAMBPOS);
   for(i = 0; i < ar->be_num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
         ar->be[i]->para, ar->be[i]->camb_pos);
   }
   fprintf(fp, "\n%s\n",BP_SHIFT);
   for(i = 0; i < ar->be_num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
         ar->be[i]->para, ar->be[i]->bp_shift);
   }

   fprintf(fp, "\n%s\n",AR_PROF);
   fprintf(fp, "%*s = %d\n", L_LEN, "naca style", ar->bp->naca);
   for(i = 0; i < ar->bp->num; i++)
   {
      sprintf(buf, STAT, i);
      fprintf(fp, "%*s = %13.8f, %14.8f\n", L_LEN, buf,
         ar->bp->c[i], ar->bp->t[i]);
   }

   fprintf(fp, "\n%s\n",AR_BE);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_BENUM, ar->be_num);
   fprintf(fp, "%*s = %f\n", L_LEN, AR_BIAS, ar->be_bias);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_BTYPE, ar->be_type);
   fprintf(fp, "%*s = %d\n", L_LEN, AR_EXTRA, ar->extrapol);

   dprintf(1,"Leaving WriteAxialRunner() ... \n");
   return 1;
}


#ifdef PLOT_BLADE_EDGES
#define NPOIN_CIRCLE 100

void PlotAR_BladeEdges(struct axial *ar)
{
   int i, j, odd;
   static int ncall = 0;
   float theta_blade, theta[2], radius;
   float rad_hub, rad_shroud;
   FILE *fp=NULL, *fgnu=NULL;
   char fname[255], fname_gnu[255];
   char *fn;

   rad_hub    = 0.5 * ar->ref * ar->diam[0];
   rad_shroud = 0.5 * ar->ref;

   sprintf(fname, "ar_edges_%02d.txt", ncall++);
   fn = DebugFilename(fname);
   if(fn)
   fp = fopen(fn, "w");

   sprintf(fname_gnu, "ar_edges.gnu");
   fn = DebugFilename(fname);
   if(fn)
   if ((fgnu = fopen(fn, "w")) != NULL)
   {
      fprintf(fgnu, "reset\n");
      fprintf(fgnu, "set polar\n");
      fprintf(fgnu, "set grid polar\n");
      fprintf(fgnu, "set nokey\n");
      fprintf(fgnu, "set size ratio 1.0\n");
      fprintf(fgnu, "set xr[-%d:%d]\n", (int)(rad_shroud+0.5), (int)(rad_shroud+0.5));
      fprintf(fgnu, "set yr[-%d:%d]\n\n", (int)(rad_shroud+0.5), (int)(rad_shroud+0.5));
      fprintf(fgnu, "pl '%s' ", fname);
   }

   for (j = 0; j < ar->nob; j++)
   {
      theta_blade = 2.0 * M_PI / ar->nob;
      odd = j % 2;
      if (j)
      {
         if (fgnu)
         {
            fprintf(fgnu, " '' index %d u 1:3 w l lt 1 lw 2, \\\n", j);
            fprintf(fgnu, " '' index %d u 2:3 w l lt 2 lw 2, \\\n", j);
         }
      }
      else
      {
         if (fgnu)
         {
            fprintf(fgnu, " index %d u 1:3 w l lt 1 lw 2, \\\n", j);
            fprintf(fgnu, " '' index %d u 2:3 w l lt 2 lw 2, \\\n", j);
         }
      }
      for (i = 0; i < ar->be_num; i++)
      {
         radius    = 0.5 * ar->ref * (ar->diam[0] + (1.0 - ar->diam[0]) * ar->be[i]->para);
         theta[0]  = ar->be[i]->arc * ar->be[i]->pivot;
         theta[1]  = theta[0] - ar->be[i]->arc;
         theta[0] += (j * theta_blade);
         theta[1] += (j * theta_blade);
         if (fp)
            fprintf(fp, "%8.6f   %8.6f   %8.6f\n", theta[0], theta[1], radius);
      }
      if (fp)
         fprintf(fp, "\n\n");
   }
   if (fgnu)
   {
      fprintf(fgnu, " '' index %d u 1:2 w l lt -1, \\\n", j);
      fprintf(fgnu, " '' index %d u 1:3 w l lt -1", j);
      fclose(fgnu);
   }

   if (fp)
   {
      for (i = 0; i <= NPOIN_CIRCLE; i++)
      {
         theta[0] = i * 2.0 * M_PI / NPOIN_CIRCLE;
         fprintf(fp, "%8.6f   %8.6f   %8.6f\n", theta[0], rad_hub, rad_shroud);
      }
      fclose(fp);
   }
}
#endif                                            // PLOT_BLADE_EDGES
