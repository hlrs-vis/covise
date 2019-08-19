#include "EKU.h"


#include <iostream>       // std::cout, std::endl
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <fstream>
#include <string>
#include <numeric>
#include <stdlib.h>

#include <osg/Material>

using namespace opencover;

EKU *EKU::plugin = NULL;


EKU::EKU(): ui::Owner("EKUPlugin", cover->ui)
{
   // sleep(10);

  /*  osg::Node* d3model;
    d3model = coVRFileManager::instance()->loadIcon("knife");
    d3model->setName("knife");
*/

    plugin = this;
    fprintf(stderr, "EKUplugin::EKUplugin\n");

    trucks.push_back(new Truck(osg::Vec3(20,0,20)));
    trucks.push_back(new Truck(osg::Vec3(-40,0,20)));
    trucks.push_back(new Truck(osg::Vec3(0,20,0)));
    trucks.push_back(new Truck(osg::Vec3(0,40,0)));
  //  trucks.push_back(new Truck(osg::Vec3(-10,0,10)));
  //  trucks.push_back(new Truck(osg::Vec3(10,-10,5)));
  //  trucks.push_back(new Truck(osg::Vec3(-20,-10,5)));
  //  trucks.push_back(new Truck(osg::Vec3(10,-20,5)));

    const osg::Vec2 o{90*M_PI/180,30*M_PI/180};
    const osg::Vec3 p{0,0,20};
    //const osg::Vec2 o1{10,-60};
    //const osg::Vec3 p1{0,0,-100};
    osg::Vec3Array* obsPoints = new osg::Vec3Array;

      for(auto x:trucks)
        obsPoints->push_back( x->pos);

{   // for each location create a cam with different alpha and beta angles

    std::vector<osg::Vec2> camRots;
    const int userParam =4;//stepsize = PI/userParam
    const int n_alpha = 2*userParam;
    const int n_beta = userParam/2;//+1;


   /* for(int alpha = 0; alpha <=n_alpha*180/userParam; alpha+=180/userParam){
        for(int beta = 0; beta <=n_beta*180/userParam; beta+=180/userParam){//stepsize ok?
            osg::Vec2 vec((double)alpha*M_PI/180, (double)beta*M_PI/180);
          // osg::Vec2 vec((double)alpha*M_PI/180, 0);
          //  osg::Vec2 vec(0, (double)beta*M_PI/180);
            camRots.push_back(vec);
        }
    }
    */
    double alpha =0;
    double beta =0;
    for(int cnt = 0; cnt<n_alpha; cnt++){
        for(int cnt2 = 0; cnt2<n_beta; cnt2++){//stepsize ok?
            osg::Vec2 vec(alpha*M_PI/180, beta*M_PI/180);

          //osg::Vec2 vec((double)alpha*M_PI/180, 0);
          //  osg::Vec2 vec(0, (double)beta*M_PI/180);
            camRots.push_back(vec);
            beta+=180/userParam;
        }
        beta=0;
        alpha+=180/userParam;
    }

    const std::string myString="Cam";
    size_t cnt=0;
    for(auto& x:camRots)
    {
       cnt++;
       cameras.push_back(new Cam(p,x,*obsPoints,myString+std::to_string(cnt)));
      // finalCams.push_back(new CamDrawable(p,x,myString+std::to_string(cnt)));
        // finalCams.push_back(new CamDrawable(p1,o1));
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
          x->calcVisMat(*obsPoints);
        }
    });

    //Camera visibility
    VisibilityRegulator = new ui::Slider(EKUMenu , "Slider2");
    VisibilityRegulator->setText("Visibility");
    VisibilityRegulator->setBounds(10., 50.);
    VisibilityRegulator->setValue(30.0);
    VisibilityRegulator->setCallback([this,obsPoints](double value, bool released){
        for(auto x :finalCams)
        {
          x->updateVisibility(value);
          x->calcVisMat(*obsPoints);
        }
    });


     cover->getObjectsRoot()->addChild(createPolygon());
     //cover->getObjectsRoot()->addChild(createPoints());

     /*
       Start GA algorithm and plot final cams
     */
     {

         const size_t pointsToObserve = trucks.size();
         ga =new GA(cameras,pointsToObserve);
         auto finalCamIndex = ga->getfinalCamPos();

         size_t cnt2=0;
         const std::string camName="Cam";
         for(auto& x:finalCamIndex)
         {
             if(x==1)
                finalCams.push_back(new CamDrawable(cameras.at(cnt2)->pos,cameras.at(cnt2)->rot,camName+std::to_string(cnt2)));

            cnt2++;
         }
     }

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

osg::Geode* EKU::createPoints()
// create POINTS
    {
        osg::Geode* geode = new osg::Geode();
        // create Geometry object to store all the vertices and points primitive.
        osg::Geometry* pointsGeom = new osg::Geometry();
        // create a Vec3Array and add to it all my coordinates.
        // Like all the *Array variants (see include/osg/Array) , Vec3Array is derived from both osg::Array
        // and std::vector<>.  osg::Array's are reference counted and hence sharable,
        // which std::vector<> provides all the convenience, flexibility and robustness
        // of the most popular of all STL containers.

        osg::Vec3Array* vertices = new osg::Vec3Array;
        vertices->push_back(osg::Vec3(20,0,10));
        vertices->push_back(osg::Vec3(50,0,0));


        // pass the created vertex array to the points geometry object.
        pointsGeom->setVertexArray(vertices);


        // create the color of the geometry, one single for the whole geometry.
        // for consistency of design even one single color must added as an element
        // in a color array.
        osg::Vec4Array* colors = new osg::Vec4Array;
        // add a white color, colors take the form r,g,b,a with 0.0 off, 1.0 full on.
        colors->push_back(osg::Vec4(1.0f,1.0f,0.0f,1.0f));

        // pass the color array to points geometry, note the binding to tell the geometry
        // that only use one color for the whole object.
        pointsGeom->setColorArray(colors, osg::Array::BIND_OVERALL);


        // Set the normal in the same way as the color.
        // (0,-1,0) points toward the viewer, in the default coordinate
        // setup.  Even for POINTS, the normal specified here
        // is used to determine how the geometry appears under different
        // lighting conditions.
        osg::Vec3Array* normals = new osg::Vec3Array;
        normals->push_back(osg::Vec3(0.0f,-1.0f,0.0f));
        pointsGeom->setNormalArray(normals, osg::Array::BIND_OVERALL);


        // create and add a DrawArray Primitive (see include/osg/Primitive).  The first
        // parameter passed to the DrawArrays constructor is the Primitive::Mode which
        // in this case is POINTS (which has the same value GL_POINTS), the second
        // parameter is the index position into the vertex array of the first point
        // to draw, and the third parameter is the number of points to draw.
        pointsGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,vertices->size()));


        // add the points geometry to the geode.
        geode->addDrawable(pointsGeom);


        return geode;
    }



COVERPLUGIN(EKU)
