#include "Flowmid.h"
#include "coInterpolatorFactory.h"
#include "DistributedBC.h"
#include <string>
#include <string.h>
#include <sstream>

/*int main(int argc, char *argv[])
{

   Flowmid *application = new Flowmid();

   application->start(argc,argv);

   return argc;

}*/


Flowmid::Flowmid(int argc, char *argv[])
:coModule(argc, argv,"IHS - Coupling-Modul")                 
// description in the module setup window
{
   // input ports ******************************************************************************************************

   //input ports for down- and upstream coupling
   outletcellUSInPort = addInputPort("outletcellUSInPort","Polygons","Upstream Outlet Cells");
   inletcellDSInPort = addInputPort("inletcellDSInPort","Polygons","Downstream Inlet Cells");

   //input ports for downstream-coupling
                                                  // From DomainDecomposition Runner!
   distbocoDSInPort = addInputPort("distbocoDSInPort","USR_DistFenflossBoco","Distributed Downstream Boundary Conditions");
                                                  // From Fenfloss Gate!
   velocityUSInPort = addInputPort("velocityUSInPort","Vec3","Upstream Velocity");

   //input ports for upstream-coupling
   distbocoUSInPort = addInputPort("distbocoUSInPort","USR_DistFenflossBoco","Distributed Upstream Boundary Conditions");
   pressureDSInPort = addInputPort("pressureDSInPort","Float","Downstream Pressure");

   // output ports ******************************************************************************************************

   //output ports for downstream-coupling
                                                  // For Fenfloss Runner!
   distbocoDSOutPort = addOutputPort("distbocoDSOutPort","USR_DistFenflossBoco","Distributed Downstream Boundary Conditions");

   //output ports for upstream-coupling
                                                  // For Fenfloss Gate!
   distbocoUSOutPort = addOutputPort("distbocoUSOutPort","USR_DistFenflossBoco","Distributed Upstream Boundary Conditions");

   distbocoDSInPort->setRequired(0);
   outletcellUSInPort->setRequired(0);
   velocityUSInPort->setRequired(0);
   distbocoUSInPort->setRequired(0);
   inletcellDSInPort->setRequired(0);
   pressureDSInPort->setRequired(0);

   // parameters ******************************************************************************************************
   interpolationTypeChoice = addChoiceParam("Type of Interpolation","interpolationTypeChoice");

   choiceMap = coInterpolatorFactory::getInterpolatorMap();
   for(map<string,int>::const_iterator mci = choiceMap.begin(); mci != choiceMap.end(); mci++)
      sortedChoiceMap[mci->second] = mci->first;

   interp_choice = new char*[sortedChoiceMap.size()];

   for(int i(0); i<sortedChoiceMap.size(); i++)
      interp_choice[i] = strdup((sortedChoiceMap[i]).c_str());

   interpolationTypeChoice->setValue(sortedChoiceMap.size(),interp_choice,0);

   downStream = addBooleanParam("Downstream Coupling","downstream_coupling");
   downStream->setValue(1);
   upStream = addBooleanParam("Upstream Coupling","upstream_coupling");
   upStream->setValue(1);
}


int Flowmid::compute(const char *)
{
   bool ds_flag(false);
   bool us_flag(false);

   const char *distbocoDSname;
   const char *distbocoUSname;

   int interpMethod;
   string interpolationMethod;

                                                  //TODO: Output ueberarbeiten
   sendInfo("Executing Flowmid Version 13.07.2004.16:21.");

   coDistributedObject *outletcellUSobj = outletcellUSInPort->getCurrentObject();
   coDistributedObject *inletcellDSobj = inletcellDSInPort->getCurrentObject();

   coDistributedObject *distbocoDSobj = distbocoDSInPort->getCurrentObject();
   coDistributedObject *velocityUSobj = velocityUSInPort->getCurrentObject();

   coDistributedObject *distbocoUSobj = distbocoUSInPort->getCurrentObject();
   coDistributedObject *pressureDSobj = pressureDSInPort->getCurrentObject();

   if(downStream->getValue())
   {

      ds_flag = true;
      //Check Input-Ports for Objects
      if(!distbocoDSobj)
      {
         ds_flag = false;
         sendError("No Object received at port %s.",distbocoDSInPort->getName());
      }
      if(!outletcellUSobj)
      {
         ds_flag = false;
         sendError("No Object received at port %s.",outletcellUSInPort->getName());
      }
      if(!velocityUSobj)
      {
         ds_flag = false;
         sendError("No Object received at port %s.",velocityUSInPort->getName());
      }
      if(!inletcellDSobj)
      {
         ds_flag = false;
         sendError("No Object received at port %s.",inletcellDSInPort->getName());
      }
      //Check Object Types
      if(ds_flag)
      {
         if(!distbocoDSobj->isType("SETELE"))
         {
            ds_flag = false;
            sendError("Expected Object of Type SETELE, received Type %s at Port %s.",distbocoDSobj->getType(),distbocoDSInPort->getName());
         }
         if(!outletcellUSobj->isType("POLYGN"))
         {
            ds_flag = false;
            sendError("Expected Object of Type POLYGN, received Type %s at Port %s.",outletcellUSobj->getType(),outletcellUSInPort->getName());
         }
         if(!velocityUSobj->isType("USTVDT"))
         {
            ds_flag = false;
            sendError("Expected Object of Type USTVDT, received Type %s at Port %s.",velocityUSobj->getType(),velocityUSInPort->getName());
         }
         if(!inletcellDSobj->isType("POLYGN"))
         {
            ds_flag = false;
            sendError("Expected Object of Type POLYGN, received Type %s at Port %s.",inletcellDSobj->getType(),inletcellDSInPort->getName());
         }
      }

      if(!ds_flag)
         sendInfo("No Downstream Coupling possible.");
   }

   if(upStream->getValue())
   {

      us_flag = true;

      if(!distbocoUSobj)
      {
         us_flag = false;
         sendError("No Object received at port %s.",distbocoUSInPort->getName());
      }
      if(!inletcellDSobj)
      {
         us_flag = false;
         sendError("No Object received at port %s.",inletcellDSInPort->getName());
      }
      if(!pressureDSobj)
      {
         us_flag = false;
         sendError("No Object received at port %s.",pressureDSInPort->getName());
      }
      if(!outletcellUSobj)
      {
         us_flag = false;
         sendError("No Object received at port %s.",outletcellUSInPort->getName());
      }

      if(us_flag)
      {
         if(!distbocoUSobj->isType("SETELE"))
         {
            us_flag = false;
            sendError("Expected Object of Type SETELE, received Type %s at Port %s.",distbocoUSobj->getType(),distbocoUSInPort->getName());
         }
         if(!inletcellDSobj->isType("POLYGN"))
         {
            us_flag = false;
            sendError("Expected Object of Type POLYGN, received Type %s at Port %s.",inletcellDSobj->getType(),inletcellDSInPort->getName());
         }
         if(!pressureDSobj->isType("USTSDT"))
         {
            us_flag = false;
            sendError("Expected Object of Type USTSDT, received Type %s at Port %s.",pressureDSobj->getType(),pressureDSInPort->getName());
         }
         if(!outletcellUSobj->isType("POLYGN"))
         {
            us_flag = false;
            sendError("Expected Object of Type POLYGN, received Type %s at Port %s.",outletcellUSobj->getType(),outletcellUSInPort->getName());
         }
      }

      if(!us_flag)
         sendInfo("No Upstream Coupling possible.");
   }

   //get interpolationMethod!
   interpMethod = interpolationTypeChoice->getValue();
   interpolationMethod = sortedChoiceMap[interpMethod];

   // Downstream-Interpolation ******************************************************************************************************

   if(ds_flag)
   {

      sendInfo("Creating Downstream Interpolator Object ...");
      coInterpolator *dsinterpolator = coInterpolatorFactory::getInterpolator(outletcellUSobj,velocityUSobj,interpolationMethod);

      if(dsinterpolator != NULL)
      {

         sendInfo("Creating Downstream Boundary Condition Object...");
         DistributedBC *dsboco = new DistributedBC((coDoSet *)distbocoDSobj);

         sendInfo("Setting Boundary Condition in Downstream-BC Object ...");
         dsboco->setBoundaryCondition(dsinterpolator,(coDoPolygons *)inletcellDSobj);

         sendInfo("Setting Downstream-BC to Output Port ...");
         distbocoDSname = distbocoDSOutPort->getObjName();
         distbocoDSOutPort->setCurrentObject(new coDoSet((char *)distbocoDSname,dsboco->getPortObj()));

         dsinterpolator->writeInfo("dsinfo.out");
      }
      else
         sendInfo("Could not create Interpolator Object!");
   }

   // Upstream-Interpolation ******************************************************************************************************

   if(us_flag)
   {

      sendInfo("Creating Upstream Interpolator Object ...");
      coInterpolator *usinterpolator = coInterpolatorFactory::getInterpolator(inletcellDSobj,pressureDSobj,interpolationMethod);

      if(usinterpolator != NULL)
      {

         sendInfo("Creating Upstream Boundary Condition Object ...");
         DistributedBC *usboco = new DistributedBC((coDoSet *)distbocoUSobj);

         sendInfo("Setting Boundary Condition in Upstream-BC Object ...");
         usboco->setBoundaryCondition(usinterpolator,(coDoPolygons *)outletcellUSobj);

         sendInfo("Setting Upstream-BC to Output Port ...");
         distbocoUSname = distbocoUSOutPort->getObjName();
         distbocoUSOutPort->setCurrentObject(new coDoSet((char *)distbocoUSname,usboco->getPortObj()));

         //usinterpolator->writeInfo("usinfo.out");
      }
      else
         sendInfo("Could not create Interpolator Object!");
   }

   return CONTINUE_PIPELINE;
}
MODULE_MAIN(Flowmid)
