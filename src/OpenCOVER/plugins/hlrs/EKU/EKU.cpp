#include "EKU.h"
#include "openGA.hpp"

#include <iostream>       // std::cout, std::endl
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <fstream>
#include <string>
#include <numeric>
#include <stdlib.h>

#include <osg/Material>

using namespace opencover;

struct MySolution
{   // FIXME: Use template to generate std::array with another size
    std::array<int,4> cam;

    std::string to_string() const
    {

        std::string myString;
        int cnt =1;
        for(auto i : cam)
        {
           myString += "cam"+std::to_string(cnt)+":"+std::to_string(i)+" ";
           cnt++;
        }
        return
            std::string("{") + myString + "}";

    }
};

struct MyMiddleCost
{
    // This is where the results of simulation
    // is stored but not yet finalized.
    double objective;
};

typedef EA::Genetic<MySolution,MyMiddleCost> GA_Type;
typedef EA::GenerationType<MySolution,MyMiddleCost> Generation_Type;

int myrandom() {
    if (rand() % 2 == 0)
        return 1;
    else return 0;
}

void init_genes(MySolution& p,const std::function<double(void)> &rnd01)
{
    // rnd01() gives a random number in 0~1
    for(auto& i : p.cam)
        i = 0+1*myrandom();
}

bool eval_solution(const MySolution& p,MyMiddleCost &c)
{
    /*
     meine Lösung. aber wozu brauche ich neue variable überhaupt?
     kann doch direkt Mysolution aufaddieren?
    std::array<int,4> cam;
    for(auto i : p.cam)
    {
        i += p.cam;
    }
 */

 /* generierte Lösung
    const int& cam1=p.cam1;
    const int& cam2=p.cam2;
    const int& cam3=p.cam3;
    const int& cam4=p.cam4;

    c.objective=cam1+cam2+cam3+cam4;
*/
    c.objective = std::accumulate(p.cam.begin(), p.cam.end(),0);

    int visMatrix [3][4] = {{1,0,0,0},
                            {0,1,0,0},
                            {0,0,0,1}};
    // TODO: implement constraint as function not as if statement!
    if((p.cam[0]*visMatrix[0][0]+p.cam[1]*visMatrix[0][1]+p.cam[2]*visMatrix[0][2]+p.cam[3]*visMatrix[0][3])>= 1 &&
       (p.cam[0]*visMatrix[1][0]+p.cam[1]*visMatrix[1][1]+p.cam[2]*visMatrix[1][2]+p.cam[3]*visMatrix[1][3])>= 1 &&
       (p.cam[0]*visMatrix[2][0]+p.cam[1]*visMatrix[2][1]+p.cam[2]*visMatrix[2][2]+p.cam[3]*visMatrix[2][3])>= 1)
       return true; // solution is accepted
    else
       return false;
}

MySolution mutate(const MySolution& X_base,const std::function<double(void)> &rnd01,double shrink_scale)
{
    MySolution X_new;
    bool in_range;
    do{
        in_range=true;
        X_new=X_base;

        for(auto &i : X_new.cam)
        {
            i+=0.2*(rnd01()-rnd01())*shrink_scale;
            in_range=in_range&&(i>=0 && i<1);
        }
      /*  X_new.cam1+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam1>=0 && X_new.cam1<1);
        X_new.cam2+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam2>=0 && X_new.cam2<1);
        X_new.cam3+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam3>=0 && X_new.cam3<1);
        X_new.cam4+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam4>=0 && X_new.cam4<1);
      */
    } while(!in_range);
    return X_new;
}

MySolution crossover(const MySolution& X1, const MySolution& X2,const std::function<double(void)> &rnd01)
{
    MySolution X_new;
    double r;
   for(size_t it =0; it<X1.cam.size(); ++it )
   {
        X_new.cam[it]=r*X1.cam[it]+(1.0-r)*X2.cam[it];
        r=rnd01();
   }

  /*  r=rnd01();
    X_new.cam1=r*X1.cam1+(1.0-r)*X2.cam1;
    r=rnd01();
    X_new.cam2=r*X1.cam2+(1.0-r)*X2.cam2;
    r=rnd01();
    X_new.cam3=r*X1.cam3+(1.0-r)*X2.cam3;
    r=rnd01();
    X_new.cam4=r*X1.cam4+(1.0-r)*X2.cam4;
   */
    return X_new;
}

double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
    // finalize the cost
    double final_cost=0.0;
    final_cost+=X.middle_costs.objective;
    return final_cost;
}

std::ofstream output_file;

void SO_report_generation(int generation_number,const EA::GenerationType<MySolution,MyMiddleCost> &last_generation,const MySolution& best_genes)
{
    std::cout
        <<"Generation ["<<generation_number<<"], "
        <<"Best="<<last_generation.best_total_cost<<", "
        <<"Average="<<last_generation.average_cost<<", "
        <<"Best genes=("<<best_genes.to_string()<<")"<<", "
        <<"Exe_time="<<last_generation.exe_time
        <<std::endl;

    output_file
        <<generation_number<<"\t"
        <<last_generation.average_cost<<"\t"
        <<last_generation.best_total_cost<<"\t"
        <<best_genes.to_string()<<"\n";
}



EKU *EKU::plugin = NULL;

EKU::EKU(): ui::Owner("EKUPlugin", cover->ui)
{
    sleep(10);

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


    // cover->getObjectsRoot()->addChild(createPolygon());

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


     output_file.open("results.txt");
     output_file<<"step"<<"\t"<<"cost_avg"<<"\t"<<"cost_best"<<"\t"<<"solution_best"<<"\n";

     EA::Chronometer timer;
     timer.tic();
     GA_Type ga_obj;

     ga_obj.problem_mode=EA::GA_MODE::SOGA;
     ga_obj.multi_threading=false;
     ga_obj.verbose=false;
     ga_obj.population=1000;
     ga_obj.generation_max=1000;
     ga_obj.calculate_SO_total_fitness=calculate_SO_total_fitness;
     ga_obj.init_genes=init_genes;
     ga_obj.eval_solution=eval_solution;
     ga_obj.mutate=mutate;
     ga_obj.crossover=crossover;
     ga_obj.SO_report_generation=SO_report_generation;
     ga_obj.best_stall_max=10;
     ga_obj.elite_count=10;
     ga_obj.crossover_fraction=0.7;
     ga_obj.mutation_rate=0.2;
     ga_obj.best_stall_max=10;
     ga_obj.elite_count=10;
     ga_obj.solve();
  //   std::cout<<"The problem is optimized in "<<timer.toc()<<" seconds."<<std::endl;

     output_file.close();
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
