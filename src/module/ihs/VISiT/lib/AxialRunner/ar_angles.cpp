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

#include "../General/include/fatal.h"
#include "../General/include/geo.h"
#include "../General/include/log.h"
#include "../General/include/plane_geo.h"
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/curve.h"
#include "../General/include/curvepoly.h"
#include "../General/include/profile.h"
#include "../General/include/parameter.h"
#include "../General/include/bias.h"
#include "../General/include/coordtrans.h"
#include "../General/include/common.h"
#include "../General/include/v.h"
#include "AxialRunner/include/axial.h"

#define SHIFTANGLE(alpha) ( (alpha)<(0.0) ? (alpha)+(M_PI) : (alpha) )
static int CalcAR_InletMerVel(struct axial *ar);
static int CalcAR_Angles(struct axial *ar);

int CalcAR_BladeAngles(struct axial *ar)
{
   int err = 0;
   if(!ar->des)
   {
      fatal("missing design data in input file!");
      return DESIGN_DATA_ERR;
   }
   if(ar->des->revs == 0 || ar->des->dis == 0 || ar->des->head == 0)
   {
      fatal("incomplete design data set!");
      return DESIGN_DATA_ERR;
   }
   CalcAR_InletMerVel(ar);
   CalcAR_Angles(ar);

   return err;
}


static int CalcAR_InletMerVel(struct axial *ar)
{
   int i;
   float v0;

   if(ar->des->vratio <= 0.0 || !ar->vratio_flag) return 0;

   // only proper if be_num odd!!
   v0 = ar->des->dis/(ar->me[ar->be_num/2]->area->list[0]*
      (1.0+(ar->des->vratio-1.0)/3.0));
   for(i = 0; i < ar->be_num; i++)
   {
      ar->be[i]->mer_vel[0] = v0*((ar->des->vratio-1.0)*
         pow(ar->be[i]->para,2)+1);
   }

   return 0;
}


// basic calculation of runner inlet/outlet blade angles
// since le and te position depends on the pivot location and the blade angles
// the start and end z-coord. of the runner core region will be used
// for a approximation of area
static int CalcAR_Angles(struct axial *ar)
{
   int i, err = 0;
   float con_area;

   con_area = M_PI *
      (pow(ar->be[ar->be_num-1]->rad,2)-pow(ar->be[0]->rad,2));
   dprintf(5,"\ncon_area = %f, ref = %f, xs = %f, xh = %f\n",
      con_area, ar->ref, ar->be[ar->be_num-1]->rad,
      ar->be[0]->rad);
   // calc. outlet angle for each blade element
   for(i = 0; i < ar->be_num; i++)
   {
      // trailing edge first!
      // conduit area for current blade element
      ar->be[i]->con_area[1] = con_area;
      // meridional, circumferential vel. and rotational
      // part of absolute vel.
      ar->be[i]->mer_vel[1] = ar->des->dis/ar->be[i]->con_area[1];
      ar->be[i]->cir_vel[1] = ar->be[i]->rad *
         M_PI*ar->des->revs/30.0;
      // outlet blade angle
      ar->be[i]->angle[1]  = atan( ar->be[i]->mer_vel[1] /
         (ar->be[i]->cir_vel[1]-
         ar->be[i]->rot_abs[1]) );
      ar->be[i]->angle[1]  = SHIFTANGLE(ar->be[i]->angle[1]);
      ar->be[i]->angle[1] *= 180.f/(float)M_PI;
      ar->be[i]->angle[1] -= ar->be[i]->mod_angle[1];

      // now do the leading edge
      ar->be[i]->con_area[0] = con_area;
      // meridional, circumferential vel. and rotational
      // part of absolute vel.
      if(ar->des->vratio <= 0.0 || !ar->vratio_flag)
         ar->be[i]->mer_vel[0] =
            ar->des->dis/ar->be[i]->con_area[0];
      ar->be[i]->cir_vel[0] = ar->be[i]->rad *
         M_PI*ar->des->revs/30.0;
      ar->be[i]->rot_abs[0] = (9.81*ar->des->head +
         ar->be[i]->cir_vel[1]*
         ar->be[i]->rot_abs[1]) /
         (ar->be[i]->cir_vel[0]);
      // inlet blade angle
      ar->be[i]->angle[0]  = atan( ar->be[i]->mer_vel[0] /
         ( ar->be[i]->cir_vel[0]-
         ar->be[i]->rot_abs[0]));
      ar->be[i]->angle[0]  = SHIFTANGLE(ar->be[i]->angle[0]);
      ar->be[i]->angle[0] *= 180.f/(float)M_PI;
      ar->be[i]->angle[0] += ar->be[i]->mod_angle[0];

      dprintf(5,"r = %f: angles: %f, %f,\n mer_vel: %f, %f\n",
         ar->be[i]->rad,ar->be[i]->angle[0],ar->be[i]->angle[1],
         ar->be[i]->mer_vel[0],ar->be[i]->mer_vel[1]);
      dprintf(5,"cir_vel: %f, %f, rot_abs: %f, %f\n",
         ar->be[i]->cir_vel[0],ar->be[i]->cir_vel[1],
         ar->be[i]->rot_abs[0],ar->be[i]->rot_abs[1]);
      if((ar->be[i]->angle[0]<ar->be[i]->angle[1]) ||
         (ar->be[i]->angle[0] > 90.0))
         return EULER_ANGLE_ERR;
   }                                              // end i, blade elements

   return err;
}
