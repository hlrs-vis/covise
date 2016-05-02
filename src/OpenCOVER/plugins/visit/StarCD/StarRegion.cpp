// Mon Nov  5 19:49:08 MET 2001
// Sa abend eingecheckt
// Fr abend eingecheckt
// Di abend eingecheckt
#include <config/CoviseConfig.h>
#include "StarRegion.h"
#include <cover/RenderObject.h>
#include <cover/coInteractor.h>

#include <iostream>

StarRegion::StarRegion(const char *name, int index, coInteractor *inter, int debug)
{
   this->debug = debug;
   if (debug)
      fprintf(stderr,"\nStarRegion::StarRegion [%s]\n", name);

   int i, j;
   osg::Vec3f pos;
   osg::ref_ptr<osg::Node> sensableNode;
   float *val;
   int numVal;
   int iScalar =0;                                // index of scalar var
   regionName = new char[strlen(name)+1];
   strcpy(regionName, name);
   regionIndex = index;
   coviseObjName = new char[strlen(inter->getObject()->getName())+1];
   strcpy(coviseObjName,inter->getObject()->getName());

   feedback=inter;
   enabled = false;

   interactorList = 0;
   interactorSensorList = 0;

//    highLight = new pfHighlight();
//    highLight->setMode(PFHL_LINES);
//    highLight->setColor(PFHL_FGCOLOR, 0.0, 0.0, 0.0);
//    highLight->setLineWidth(3.0);

   // get the number of parameters
   numParams = inter->getNumParam();
   numInteractors = numParams-1;                  // first parameter local is not an interactive parameter

   // create interactor list and interactor sensor list
   interactorList = new StarRegionInteractor*[numInteractors];

   interactorSensorList = new InteractorSensor*[numInteractors];

   // compute the position and normal of the region
   geoset = 0;
   center.set(0,0,0);
   normal.set(-1, 0, 0);

   // get the "local" parameter which is a vector
   inter->getFloatVectorParam(0, numVal, val);
   local.set(val[0], val[1], val[2]);

   fprintf(stderr,"local:[%f %f %f]\n", local[0], local[1], local[2]);
   // create all interactors
   for (j = 1; j < numParams; j++)
   {
      i=j-1;                                      // andreas hat vorne eine neuen Parameter eingefuegt, der
      // nicht randbedingung ist

      if (debug)
         fprintf(stderr,"\tparameter [%s]:\n", inter->getParaName(j));

      float size = coCoviseConfig::getFloat("COVER.Plugin.StarCD.ArrowSize",50);

      if (strcmp(inter->getParaType(j), "FLOVEC") == 0)
      {

         inter->getFloatVectorParam(j, numVal, val);
         interactorList[i] = new StarRegionVectorInteractor(inter->getParaName(j), center, normal,  local, size, true, val[0], val[1], val[2]);
         sensableNode = interactorList[i]->getSensableNode();
         interactorSensorList[i] = new InteractorSensor(sensableNode.get(), interactorList[i]);
         interactorSensorList[i]->disable();
      }
      if (strcmp(inter->getParaType(j), "FLOSLI") == 0)
      {

         float mi, ma, v;
         inter->getFloatSliderParam(j, mi, ma, v);
         ////float a = 360.0/(numParams-1);
         ////osg::Matrix m;
         ////m.makeRot(a*(i-1), normal[0], normal[1], normal[2]);
         ////pos.set(0, 0, -radius);
         ////pos.xformVec(pos, m);
         ////pos += center;
         pos = center;
         //fprintf(stderr,"\ncreating Scalar Interactor:\n");
         interactorList[i] = new StarRegionScalarInteractor(inter->getParaName(j), center, normal, local, size, true, mi, ma, v, 0.0, 135.0, radius, iScalar);
         sensableNode = interactorList[i]->getSensableNode();
         interactorSensorList[i] = new InteractorSensor(sensableNode.get(), interactorList[i]);
         interactorSensorList[i]->disable();

         cover->addSliderButton(inter->getParaName(j), regionName, mi, ma, v, sliderButtonCallback, this);
         iScalar++;
      }
   }

}


void StarRegion::update(coInteractor *inter)
{
   float *val;
   int numVal;
   float mi, ma, v;
   int i, j;

   //if (debug)
   //   inter->print(stderr);

   delete[] coviseObjName;
   coviseObjName = new char[strlen(inter->getObject()->getName())+1];
   strcpy(coviseObjName,inter->getObject()->getName());
   // get the "local" parameter which is a vector
   inter->getFloatVectorParam(0, numVal, val);
   local.set(val[0], val[1], val[2]);
   fprintf(stderr,"local:[%f %f %f]\n", local[0], local[1], local[2]);

   // update all interactors
   for (j = 1; j < numParams; j++)
   {
      i=j-1;                                      // andreas hat vorne eine neuen Parameter eingefuegt, der
      // nicht randbedingung ist

      if (debug)
         fprintf(stderr,"\tparameter [%s]:\n", inter->getParaName(j));

      if (strcmp(inter->getParaType(j), "FLOVEC") == 0)
      {
         inter->getFloatVectorParam(j, numVal, val);
         interactorList[i]->setValue(val[0], val[1], val[2]);
         interactorList[i]->setLocal(local);
      }
      if (strcmp(inter->getParaType(j), "FLOSLI") == 0)
      {
         inter->getFloatSliderParam(j, mi, ma, v);
         interactorList[i]->setValue(mi, ma, v);
         interactorList[i]->setLocal(local);
         cover->setSliderValue(inter->getParaName(j), v);
      }
   }
}


StarRegion::~StarRegion()
{
   int i, j;
   if (debug)
      fprintf(stderr,"StarRegion::~StarRegion\n");
   for (j = 1; j < numParams; j++)
   {
      i=j-1;
      delete interactorList[i];
      delete interactorSensorList[i];
      cover->removeButton(interactorList[i]->getParamName().c_str(), regionName);
   }
   delete []interactorList;
   delete []interactorSensorList;
   delete[] coviseObjName;

}


void StarRegion::sliderButtonCallback(void* reg, buttonSpecCell* spec)
{

   StarRegion *r;
   int i;
   r = (StarRegion*)reg;

   if (r->feedback)
   {
      for (i = 0; i < r->numInteractors; i++)
      {
         if ( strcmp(r->interactorList[i]->getParamName().c_str(), spec->name)==0 )
         {
            float mi, ma, v;
            r->interactorList[i]->getValue(&mi, &ma, &v);
            fprintf(stderr,"current slider value:[%f %f %f]\n", mi, ma, v);
            fprintf(stderr,"setting feedback of slider [%s] to [%f]\n", spec->name, spec->state);
            r->feedback->setSliderParam(spec->name, mi, ma, spec->state);
         }
      }
   }

}


char * StarRegion::getName()
{
   return regionName;
}


char *StarRegion::getCoviseObjectName()
{
   return coviseObjName;
}


void StarRegion::computeCenterAndNormal()
{
   osg::BoundingSphere sphere;
   osg::BoundingBox bbox;
   // get the region center and region normal
   // this works only for flat regions
   sphere = patchesGeode->computeBound();

   center = sphere.center();
   //radius = sphere.radius;
   if (patchesGeode->getNumDrawables())
   {
      cerr << "StarRegion::computeCenterAndNormal info: getNumDrawables = " << patchesGeode->getNumDrawables() << endl;
      bbox = patchesGeode->getDrawable(0)->getBound();
   }
   float r0=bbox._max[0]-bbox._min[0];
   float r1=bbox._max[1]-bbox._min[1];
   float r2=bbox._max[2]-bbox._min[2];
   fprintf(stderr,"bbox=[%f %f %f]\n", r0, r1, r2);
   if ( (r0>=r1) && (r0>=r2) )
      radius=0.5*r0;
   else if ( (r1>=r0) && (r1>=r2) )
      radius=0.5*r1;
   else
      radius = 0.5*r2;

   // test
   radius = 0.5*r2;

   //fprintf(stderr,"**** region=[%s]\n", regionName);
   fprintf(stderr,"**** center=[%f %f %f]\n", center[0], center[1], center[2]);
   fprintf(stderr,"**** radius=[%f %f]\n", radius, sphere.radius());
   //fprintf(stderr,"**** normal=[%f %f %f]\n", normal[0], normal[1], normal[2]);
}


void StarRegion::setPatchesGeode(osg::Geode *geode)
{
   patchesGeode = geode;
   int i, j;
   if (patchesGeode.get())
   {
      if (patchesGeode->getNumDrawables())
         geoset = patchesGeode->getDrawable(0);
      computeCenterAndNormal();
      if (enabled)
         enableHighLight();

      // update all interactors
      for (j = 1; j < numParams; j++)
      {
         i=j-1;                                   // andreas hat vorne eine neuen Parameter eingefuegt, der
         // nicht randbedingung ist
         interactorList[i]->setCenter(center);
         if (interactorList[i]->getType() == StarRegionInteractor::Scalar)
            ((StarRegionScalarInteractor*)interactorList[i])->setRadius(radius);
      }

   }

}


osg::Geode* StarRegion::getPatchesGeode()
{
   return patchesGeode.get();
}


void StarRegion::enableHighLight()
{
   enabled=true;
   std::cerr << "StarRegion::enableHighLight fixme: stub" << std::endl;
//    if (geoset)
//       geoset->setHlight(highLight);
}


void
StarRegion::disableHighLight()
{
   enabled=false;
   std::cerr << "StarRegion::disableHighLight fixme: stub" << std::endl;
//    if (geoset)
//       geoset->setHlight(PFHL_OFF);
}
