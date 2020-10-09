
#include "UI.h"
#include "Helper.h"

#include "DataManager.h"
#include "Factory.h"
#include "GA.h"
#include "SensorPlacement.h"
#include "Sensor.h"

using namespace opencover;

bool UI::m_DeleteStatus{false};
bool UI::m_showAverageUDPPositions{false};
bool UI::m_showShortestUDPPositions{true};


UI::UI() : ui::Owner("SensorPlacementUI", cover->ui)
{
   // Main menu-------------------------------------------------------------------------------
    m_MainMenu = new ui::Menu("SensorPlacement",this);

    m_AddCamera = new ui::Action(m_MainMenu,"AddCamera");
    m_AddCamera-> setText("Add Camera");
    m_AddCamera-> setCallback([]()
    {
       DataManager::AddSensor(Factory::createSensor(SensorType::Camera));
    }
    );

    m_AddSafetyZonePrio1 = new ui::Action(m_MainMenu,"AddSafetyZonePRIO1");
    m_AddSafetyZonePrio1-> setText("Add PRIO1 Zone");
    m_AddSafetyZonePrio1-> setCallback([]()
    {
       osg::Matrix m;
       m.setTrans(osg::Vec3(20,20,20));
       DataManager::AddSafetyZone(Factory::createSafetyZone(SafetyZone::Priority::PRIO1));
    }
    );

    m_AddSafetyZonePrio2 = new ui::Action(m_MainMenu,"AddSafetyZonePRIO2");
    m_AddSafetyZonePrio2-> setText("Add PRIO2 Zone");
    m_AddSafetyZonePrio2-> setCallback([]()
    {
       osg::Matrix m;
       m.setTrans(osg::Vec3(20,20,20));
       DataManager::AddSafetyZone(Factory::createSafetyZone(SafetyZone::Priority::PRIO2));
    }
    );

    m_AddSensorZone = new ui::Action(m_MainMenu,"AddSensorZone");
    m_AddSensorZone-> setText("Add Sensor Zone");
    m_AddSensorZone-> setCallback([this]()
    {
       osg::Matrix m;
       m.setTrans(osg::Vec3(20,20,20));
       DataManager:: AddSensorZone(Factory::createSensorZone());
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

   // Sensor Menu-------------------------------------------------------------------------------
   m_SensorProps = new ui::Menu(m_MainMenu,"SensorProps");
   m_SensorProps->setText("Sensor Properties");

   m_Rotx = new ui::Button(m_SensorProps,"Rotate X");
   m_Rotx->setText("Rotate X");
   m_Rotx->setState(SensorWithMultipleOrientations::s_SensorProps.getRotX());
   m_Rotx->setCallback([this](bool state)
   {
      SensorWithMultipleOrientations::s_SensorProps.setRotX(state);
   });

   m_Roty = new ui::Button(m_SensorProps,"Rotate Y");
   m_Roty->setText("Rotate Y");
   m_Roty->setState(SensorWithMultipleOrientations::s_SensorProps.getRotY());
   m_Roty->setCallback([this](bool state)
   {
      SensorWithMultipleOrientations::s_SensorProps.setRotY(state);
   });

   m_Rotz = new ui::Button(m_SensorProps,"Rotate Z");
   m_Rotz->setText("Rotate Z");
   m_Rotz->setState(SensorWithMultipleOrientations::s_SensorProps.getRotZ());
   m_Rotz->setCallback([this](bool state)
   {
      SensorWithMultipleOrientations::s_SensorProps.setRotZ(state);
   });

   m_SliderStepSizeX = new ui::Slider(m_SensorProps,"StepSizeX");
   m_SliderStepSizeX->setIntegral(true);
   m_SliderStepSizeX->setText("Step size X[°]");
   m_SliderStepSizeX->setBounds(1, 90);
   m_SliderStepSizeX->setValue(SensorWithMultipleOrientations::s_SensorProps.getStepSizeX());
   m_SliderStepSizeX->setCallback([](double value, bool released)
   {
      SensorWithMultipleOrientations::s_SensorProps.setStepSizeX(value);
   });

   m_SliderStepSizeY = new ui::Slider(m_SensorProps,"StepSizeY");
   m_SliderStepSizeY->setIntegral(true);
   m_SliderStepSizeY->setText("Step size Y[°]");
   m_SliderStepSizeY->setBounds(1, 90);
   m_SliderStepSizeY->setValue(SensorWithMultipleOrientations::s_SensorProps.getStepSizeY());
   m_SliderStepSizeY->setCallback([](double value, bool released)
   {
      SensorWithMultipleOrientations::s_SensorProps.setStepSizeY(value);
   });

   m_SliderStepSizeZ = new ui::Slider(m_SensorProps,"StepSizeZ");
   m_SliderStepSizeZ->setIntegral(true);
   m_SliderStepSizeZ->setText("Step size Z[°]");
   m_SliderStepSizeZ->setBounds(1, 90);
   m_SliderStepSizeZ->setValue(SensorWithMultipleOrientations::s_SensorProps.getStepSizeZ());
   m_SliderStepSizeZ->setCallback([](double value, bool released)
   {
      SensorWithMultipleOrientations::s_SensorProps.setStepSizeZ(value);
   });


   m_ShowOrientations = new ui::Button(m_SensorProps,"ShowOrientations");
   m_ShowOrientations->setText("Show Orientations");
   m_ShowOrientations->setState(SensorWithMultipleOrientations::s_SensorProps.getVisualizeOrientations());
   m_ShowOrientations->setCallback([this](bool state)
   {
      SensorWithMultipleOrientations::s_SensorProps.setVisualizeOrientations(state);
   });

  

   // Camera Menu------------------------------------------------------------------------------------
    m_CameraProps = new ui::Menu(m_SensorProps,"CameraProps");
    m_CameraProps->setText("Camera Properties");

    m_Visibility = new ui::Slider(m_CameraProps,"Visibility");
    m_Visibility->setText("Visibility [ % ]");
    m_Visibility->setBounds(10., 100.);
    m_Visibility->setCallback([](double value, bool released)
    {

    });

   //Optimization Menu-------------------------------------------------------------------------------
   m_Optimization = new ui::Menu(m_MainMenu, "Optimization");
   m_Optimization->setText("Optimization");
   
   m_MaxCoverage1 = new ui::Action(m_Optimization,"MaxCoverage1");
   m_MaxCoverage1-> setText("MaxCoverage1");
   m_MaxCoverage1-> setCallback([this]()
   {
      optimize(FitnessFunctionType::MaxCoverage1); 
   });
   

   m_MaxCoverage2 = new ui::Action(m_Optimization,"MaxCoverage2");
   m_MaxCoverage2-> setText("MaxCoverage2");
   m_MaxCoverage2-> setCallback([this]()
   {
      optimize(FitnessFunctionType::MaxCoverage2);
      getSensorInSensorZone(0);
      getSensorInSensorZone(1);
      getSensorInSensorZone(2);
      getSensorInSensorZone(3);
      getSensorInSensorZone(4);
      getSensorInSensorZone(5);
      getSensorInSensorZone(6);
      getSensorInSensorZone(7);
      getSensorInSensorZone(8);
      getSensorInSensorZone(9);
      getSensorInSensorZone(10);


   });


    //UDP Menu-------------------------------------------------------------------------------
   m_UDP = new ui::Menu(m_MainMenu, "UDP");
   m_UDP->setText("UDP");


   m_showAverageUDPObjectionPosition = new ui::Button(m_UDP,"Average_positions");
   m_showAverageUDPObjectionPosition->setText("Average positions");
   m_showAverageUDPObjectionPosition->setState(m_showAverageUDPPositions);
   m_showAverageUDPObjectionPosition->setCallback([this](bool state)
   {
      m_showAverageUDPPositions = state;
   });

   m_showShortestUDPObjectionPosition = new ui::Button(m_UDP,"Shortest_positions");
   m_showShortestUDPObjectionPosition->setText("Shortest positions");
   m_showShortestUDPObjectionPosition->setState(m_showShortestUDPPositions);
   m_showShortestUDPObjectionPosition->setCallback([this](bool state)
   {
      m_showShortestUDPPositions = state;
   });

   m_AverageFrames = new ui::Button(m_UDP,"Average_Frames");
   m_AverageFrames->setText("Calc pos wiht average of the last frames");
   m_AverageFrames->setState(DetectedCameraOrObject::s_frameAverage);
   m_AverageFrames->setCallback([this](bool state)
   {
      DetectedCameraOrObject::s_frameAverage = state;
   });



};

