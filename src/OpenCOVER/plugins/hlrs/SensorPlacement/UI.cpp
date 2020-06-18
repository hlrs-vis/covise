#include "UI.h"
#include "Helper.h"

#include "DataManager.h"
#include "Factory.h"
#include "GA.h"

using namespace opencover;

bool UI::m_DeleteStatus{false};
bool UI::m_showOrientations{true};

UI::UI() : ui::Owner("SensorPlacementUI", cover->ui)
{
   // Main menu
    m_MainMenu = new ui::Menu("SensorPlacement",this);

    m_AddCamera = new ui::Action(m_MainMenu,"AddCamera");
    m_AddCamera-> setText("Add Camera");
    m_AddCamera-> setCallback([]()
    {
       DataManager::AddSensor(createSensor(SensorType::Camera));
    }
    );

    m_AddSafetyZone = new ui::Action(m_MainMenu,"AddSafetyZoe");
    m_AddSafetyZone-> setText("Add Safety Zone");
    m_AddSafetyZone-> setCallback([]()
    {
       osg::Matrix m;
       m.setTrans(osg::Vec3(20,20,20));
       DataManager::AddZone(createZone(ZoneType::ROIzone));
    }
    );

    m_AddSensorZone = new ui::Action(m_MainMenu,"AddSensorZone");
    m_AddSensorZone-> setText("Add Sensor Zone");
    m_AddSensorZone-> setCallback([]()
    {
       osg::Matrix m;
       m.setTrans(osg::Vec3(20,20,20));
       DataManager::AddZone(createZone(ZoneType::CameraZone));
    }
    );

    m_Delete = new ui::Button(m_MainMenu,"Delete");
    m_Delete->setText("Remove");
    m_Delete->setState(m_DeleteStatus);
    m_Delete->setCallback([this](bool state)
    {
      m_DeleteStatus = state;
      if(m_DeleteStatus)
         std::cout<<"Delete Status"<<m_DeleteStatus<<std::endl;
    });

   // Sensor
   m_SensorProps = new ui::Menu(m_MainMenu,"SensorProps");
   m_SensorProps->setText("Sensor Properties");

   m_ShowOrientations = new ui::Button(m_SensorProps,"ShowOrientations");
   m_ShowOrientations->setText("Show Orientations");
   m_ShowOrientations->setState(m_showOrientations);
   m_ShowOrientations->setCallback([this](bool state)
   {
      m_showOrientations = state;
   });

   // Camera
    m_CameraProps = new ui::Menu(m_SensorProps,"CameraProps");
    m_CameraProps->setText("Camera Properties");

    m_Visibility = new ui::Slider(m_CameraProps,"Visibility");
    m_Visibility->setText("Visibility [ % ]");
    m_Visibility->setBounds(10., 100.);
    m_Visibility->setCallback([](double value, bool released)
    {

    });

   //Optimization
   m_Optimization = new ui::Menu(m_MainMenu, "Optimization");
   m_Optimization->setText("Optimization");
   
   m_MaxCoverage1 = new ui::Action(m_Optimization,"MaxCoverage1");
   m_MaxCoverage1-> setText("MaxCoverage1");
   m_MaxCoverage1-> setCallback([this]()
   {
      this->checkForObstacles();
      this->checkVisibility();
     // auto up(myHelpers::make_unique<GA>(maxCoverage1));
      //TODO only on master
   }
   );

   m_MaxCoverage2 = new ui::Action(m_Optimization,"MaxCoverage2");
   m_MaxCoverage2-> setText("MaxCoverage2");
   m_MaxCoverage2-> setCallback([this]()
   {
      this->checkForObstacles();
      this->checkVisibility();

     // auto up(myHelpers::make_unique<GA>(maxCoverage2));
      //TODO only on master

   }
   );

};

void UI::checkForObstacles()const
{
   for(const auto& sensor : DataManager::GetInstance().GetSensors())
      sensor->checkForObstacles();
}

void UI::checkVisibility()const
{
   //for(const auto& sensor : DataManager::GetInstance().GetSensors())
      //sensor->calcVisibilityMatrix();
}