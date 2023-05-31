#include "EKU.h"


#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <string>
#include <numeric>
#include <array>
#include <cstdlib>

using namespace opencover;

EKU *EKU::plugin = NULL;
void EKU::preFrame()
{
    sensorList.update();
    //Test if button is pressed
    int state = cover->getPointerButton()->getState();
    if (myinteraction->isRunning()) //when interacting the Sphere will be moved
    {
        static osg::Matrix invStartHand;
        static osg::Matrix startPos;
        if (!interActing)
        {
            //remember invStartHand-Matrix, when interaction started and mouse button was pressed
            invStartHand.invert(cover->getPointerMat() * cover->getInvBaseMat());
            startPos = mymtf->getMatrix(); //remember position of sphere, when interaction started
            interActing = true; //register interaction
        }
        else
        {
            //calc the tranformation matrix when interacting is running and mouse button was pressed
            osg::Matrix transMat = startPos * invStartHand * (cover->getPointerMat() * cover->getInvBaseMat());
            mymtf->setMatrix(transMat);
        }
    }
    if (myinteraction->wasStopped() && state == false)
    {
        interActing = false; //unregister interaction
    }
}


EKU::EKU()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("EKUPlugin", cover->ui)

{

    plugin = this;
    fprintf(stderr, "EKUplugin::EKUplugin\n");

    // read file
    scene = osgDB::readNodeFile("/home/AD.EKUPD.COM/matthias.epple/data/osgt/EKU_Box_large.osgt");
    if (!scene.valid())
      {
          osg::notify( osg::FATAL ) << "Unable to load data file. Exiting." << std::endl;
          //return( 1 );
      }

    //all Points to observe from file
    std::vector<osg::Vec3> truckPos;
    FindNamedNode *fnnPoints= new FindNamedNode( "px",&truckPos);
    scene->accept(*fnnPoints );
    delete fnnPoints;
    for(const auto& x : truckPos)
        trucks.push_back(new Truck(x));


    osg::Vec3Array* obsPoints = new osg::Vec3Array; //Note: Remove this unecessary
    for(auto x:trucks)
        obsPoints->push_back( x->pos);


    //all possible camera locations from file
    std::vector<osg::Vec3> camPos;
    FindNamedNode *fnnCam= new FindNamedNode( "cx",&camPos);
    scene->accept(*fnnCam );
    delete fnnCam;
    {   // for each location create a cam with different alpha and beta angles
        std::vector<osg::Vec2> camRots;
        const int userParam =4;//stepsize = PI/userParam
        const int n_alpha = 2*userParam;
        const int n_beta = userParam/2;//+1;
        double alpha =0;
        double beta =0;
        for(int cnt = 0; cnt<n_alpha; cnt++){
            for(int cnt2 = 0; cnt2<n_beta; cnt2++){//stepsize ok?
                osg::Vec2 vec(alpha*M_PI/180, beta*M_PI/180);
                camRots.push_back(vec);
                beta+=180/userParam;
            }
            beta=0;
            alpha+=180/userParam;
        }

        const std::string myString="Cam";
        size_t cnt=0;
        for(const auto& c: camPos)
        {
            for(const auto& x:camRots)
            {
               cnt++;
               cameras.push_back(new Cam(c,x,*obsPoints,myString+std::to_string(cnt)));

            }
        }
    }

    //Create UI
    EKUMenu  = new ui::Menu("EKU", this);

    //Add Truck
    AddTruck = new ui::Action(EKUMenu , "addTruck");
    AddTruck->setCallback([this](){
        doAddTruck();
    });

    //Add Cam
    AddCam = new ui::Action(EKUMenu , "addCam");
    AddCam->setCallback([this](){
        doAddCam();
    });

    //Remove Truck
    RmvTruck = new ui::Action(EKUMenu , "removeTruck");
    RmvTruck->setCallback([this](){
            doRemoveTruck();
    });

    //FOV
    FOVRegulator = new ui::Slider(EKUMenu , "Slider1");
    FOVRegulator->setText("FOV");
    FOVRegulator->setBounds(30., 120.);
    FOVRegulator->setValue(60.);
    FOVRegulator->setCallback([this,obsPoints](double value, bool released){
        for(auto x :finalCams)
        {
          x->updateFOV(value);
          x->cam->calcVisMat(*obsPoints);
        }
    });

    //Camera visibility
    VisibilityRegulator = new ui::Slider(EKUMenu , "Slider2");
    VisibilityRegulator->setText("Visibility");
    VisibilityRegulator->setBounds(10., 60.);
    VisibilityRegulator->setValue(30.0);
    VisibilityRegulator->setCallback([this,obsPoints](double value, bool released){
        for(auto x :finalCams)
        {
          x->updateVisibility(value);
          x->cam->calcVisMat(*obsPoints);
        }
    });

    //Make Cameras invisible
    MakeCamsInvisible = new ui::Button(EKUMenu , "CamerasVisible");
    MakeCamsInvisible->setText("CamerasVisible");
    MakeCamsInvisible->setState(true);
    MakeCamsInvisible->setCallback([this](bool state){
        if(!state)
        {
            for(auto x :finalCams)
            {
              x->disactivate();
            }
           // MakeCamsInvisible->setState(false);
        }
        else
        {
            for(auto x :finalCams)
            {
              x->activate();
            }
        }
    });


     // Start GA algorithm and plot final cams

       {

         const size_t pointsToObserve = trucks.size();
         ga =new GA(cameras,pointsToObserve);
         auto finalCamIndex = ga->getfinalCamPos();

         size_t cnt2=0;
         const std::string camName="Cam";
         for(auto& x:finalCamIndex)
         {
             if(x==1){
                //finalCams.push_back(new CamDrawable(cameras.at(cnt2)->pos,cameras.at(cnt2)->rot,camName+std::to_string(cnt2)));
                    finalCams.push_back(new CamDrawable(cameras.at(cnt2)));
                }
            cnt2++;
         }
     }

    //Draw final Scene
    finalScene = new osg::Group;
    finalScene->setName("finalScene");
    finalScene->addChild(scene.get());
    myinteraction = new vrui::coTrackerButtonInteraction(vrui::coInteraction::ButtonA, "MoveMode", vrui::coInteraction::Medium);
    interActing = false;
    for(const auto& x:finalCams)
    {
        finalScene->addChild(x->getCamDrawable().get());
        //add User interaction to each final camera
        userInteraction.push_back(new mySensor(x->getCamGeode(), x->cam->getName(), myinteraction,x,&trucks,&finalCams));
    }
    int cntTrucks =0;
    for(const auto& x:trucks)
    {
        finalScene->addChild(x->getTruckDrawable().get());
        //add User interaction to each safety zone
        userInteraction.push_back(new mySensor(x->getTruckDrawable(),cntTrucks, "Truck", myinteraction,x,&finalCams));
        cntTrucks++;
    }
    // add sensors to sensorList
    for(const auto& x : userInteraction)
        sensorList.append(x);

    cover->getObjectsRoot()->addChild(finalScene.get());

    //Write obj file
    osgDB::writeNodeFile(*finalScene, "OpenCOVER/plugins/hlrs/EKU/EKU_result.obj");
}

EKU::~EKU()
{
    fprintf(stderr, "BorePlugin::~BorePlugin\n");

}
bool EKU::init()
{

    return true;
}

void EKU::doAddTruck()
{

    size_t pos = trucks.size();
    trucks.push_back(new Truck(osg::Vec3((pos+1)*2,0,0)));
}


void EKU::doRemoveTruck()
{
    if (!trucks.empty())
       trucks.back()->destroy();

    delete trucks.back();

    if(trucks.size()>0)
        trucks.pop_back();

    /*TOD:
    - delete Trucks from screen
    - when last element is deleted program chrashes (because first is not part of vector?)
    */
}
void EKU::doAddCam()
{

}

osg::Geode* EKU::createPolygon()
{
   // The Drawable geometry is held under Geode objects.
   osg::Geode* geode = new osg::Geode();
   geode->setName("Landscape");
   osg::Geometry* geom = new osg::Geometry();
   osg::StateSet *stateset = geode->getOrCreateStateSet();
   // Associate the Geometry with the Geode.
   geode->addDrawable(geom);
   // Declare an array of vertices to create a simple polygon.
   osg::Vec3Array* verts = new osg::Vec3Array;
   verts->push_back( osg::Vec3( 50.0f,  50.0f,  0.0f) ); // 2 right back
   verts->push_back( osg::Vec3( 50.0f, -50.0f,  0.0f) ); // 1 right front
   verts->push_back( osg::Vec3(-50.0f,  50.0f,  0.0f) ); // 3 left  back
   verts->push_back( osg::Vec3(-50.0f, -50.0f,  0.0f) ); // 0 left  front
   // Associate this set of vertices with the Geometry.
   geom->setVertexArray(verts);
   // Next, create a primitive set and add it to the Geometry as a polygon.
   osg::DrawElementsUInt* face =
      new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
   face->push_back(0);
   face->push_back(1);
   face->push_back(2);
   face->push_back(3);
   geom->addPrimitiveSet(face);
   //Create normal
   osg::Vec3Array* normals = new osg::Vec3Array();
   normals->push_back(osg::Vec3(0.f ,0.f, 1.f));  //left front
   normals->push_back(osg::Vec3(0.f ,0.f, 1.f));
   normals->push_back(osg::Vec3(0.f ,0.f, 1.f));
   normals->push_back(osg::Vec3(0.f ,0.f, 1.f));
   geom->setNormalArray(normals);
   geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    //create Materal
    osg::Material *material = new osg::Material;
    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.2f, 0.2f, 1.0f));
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    material->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
    stateset->setAttributeAndModes(material);
    stateset->setNestRenderBins(false);

   // Create a color for the polygon.
   osg::Vec4Array* colors = new osg::Vec4Array;
   colors->push_back( osg::Vec4(0.0f, 0.5f, 0.0f, 1.0f) ); // dark green
   // The next step is to associate the array of colors with the geometry.
   // Assign the color indices created above to the geometry and set the
   // binding mode to _OVERALL.
   geom->setColorArray(colors);
   geom->setColorBinding(osg::Geometry::BIND_OVERALL);
   // Return the geode as the root of this geometry.
   return geode;
}

COVERPLUGIN(EKU)
