#include "EKU.h"

#include <iostream>       // std::cout, std::endl
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <osg/Material>
using namespace opencover;

EKU *EKU::plugin = NULL;

EKU::EKU(): ui::Owner("EKUPlugin", cover->ui)
{
  /*  osg::Node* d3model;
    d3model = coVRFileManager::instance()->loadIcon("knife");
    d3model->setName("knife");
*/

    plugin = this;
    fprintf(stderr, "EKUplugin::EKUplugin\n");

   // trucks.push_back(new Truck());
    cameras.push_back(new Cam());

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
    FOVRegulator->setBounds(0., 10.);
    FOVRegulator->setValue(1.5);
    FOVRegulator->setCallback([this](double value, bool released){
        for(auto x :cameras)
          x->updateFOV(value);
    });

    //Camera visibility
    VisibilityRegulator = new ui::Slider(EKUMenu , "Slider2");
    VisibilityRegulator->setText("Visibility");
    VisibilityRegulator->setBounds(1.0, 10.);
    VisibilityRegulator->setValue(1.0);
    VisibilityRegulator->setCallback([this](double value, bool released){
        for(auto x :cameras)
          x->updateVisibility(value);
    });


     cover->getObjectsRoot()->addChild(createPolygon());

  /*  //Position of Objects
    moveDown = new osg::PositionAttitudeTransform();
    moveDown->setPosition(osg::Vec3( 0.0f, 0.0f, -0.5f));
    moveToSide = new osg::PositionAttitudeTransform();
    moveToSide->setPosition( osg::Vec3(-4.0f, 0.0f, 0.0f));
    moveUp = new osg::PositionAttitudeTransform();
    moveUp->setPosition(osg::Vec3( 0.0f, 0.0f, 1.0f));

    moveUp->addChild( camera1->createPyramid() );
    moveUp->setUpdateCallback( new RotationCallback() );
*/

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
   verts->push_back( osg::Vec3(-50.0f, -50.0f,  0.0f) ); // 0 left  front
   verts->push_back( osg::Vec3( 50.0f, -50.0f,  0.0f) ); // 1 right front
   verts->push_back( osg::Vec3( 50.0f,  50.0f,  0.0f) ); // 2 right back
   verts->push_back( osg::Vec3(-50.0f,  50.0f,  0.0f) ); // 3 left  back
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
