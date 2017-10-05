#include <api/coFeedback.h>
#include "AxialRunner.h"
#include "../lib/General/include/log.h"
#include <do/coDoPolygons.h>

void AxialRunner::CreateGeo(void)
{

	coDoPolygons *poly;
	struct covise_info *ci;

   if ((ci = CreateGeometry4Covise(geo))) {
      sendInfo(" Specific speed: %.4f",geo->ar->des->spec_revs);
      dprintf(3, "AxialRunner::compute(const char *): Geometry created\n");
      BladeElements2CtrlPanel();
      BladeElements2Reduced();
      poly = new coDoPolygons(blade->getObjName(),
         ci->p->nump,
         ci->p->x, ci->p->y, ci->p->z,
         ci->vx->num,  ci->vx->list,
         ci->pol->num, ci->pol->list);
      poly->addAttribute("MATERIAL","metal metal.34");
      //poly->addAttribute("DEPTH_ONLY","X");
      poly->addAttribute("vertexOrder","1");

      coFeedback feedback("AxialARPlugin");
      feedback.addPara(p_BladeAngle);
      feedback.apply(poly);

      blade->setCurrentObject(poly);

      poly = new coDoPolygons(hub->getObjName(),
         ci->p->nump,
         ci->p->x, ci->p->y, ci->p->z,
         ci->lvx->num,  ci->lvx->list,
         ci->lpol->num, ci->lpol->list);
      poly->addAttribute("MATERIAL","metal metal.34");
      //poly->addAttribute("DEPTH_ONLY","X");
      poly->addAttribute("vertexOrder","1");
      hub->setCurrentObject(poly);

      poly = new coDoPolygons(shroud->getObjName(),
         ci->p->nump,
         ci->p->x, ci->p->y, ci->p->z,
         ci->cvx->num,  ci->cvx->list,
         ci->cpol->num, ci->cpol->list);
      poly->addAttribute("MATERIAL","metal metal.34");
      //poly->addAttribute("DEPTH_ONLY","X");
      poly->addAttribute("vertexOrder","1");
      shroud->setCurrentObject(poly);

      if(p_WriteBladeData->getValue())
      {
         if(PutBladeData(geo->ar))
            sendError("%s", GetLastErr());
      }
   }
   else {
      dprintf(0, "Error in CreateGeometry4Covise\n");
      sendError("%s", GetLastErr());
   }

}
