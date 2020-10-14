
#include "UI.h"
#include "Helper.h"

#include "DataManager.h"
#include "Factory.h"
#include "GA.h"
#include "SensorPlacement.h"
#include "Sensor.h"

#include <osgDB/ReadFile>

using namespace opencover;

bool UI::m_DeleteStatus{false};
bool UI::m_showAverageUDPPositions{false};
bool UI::m_showShortestUDPPositions{true};

// void UI::updateOptimizationResults(float total, float prio1, float prio2, fitness)
// {
   // m_TotalCoverage->setText("Total Coverage: "+total+"%");
   // m_Prio2Coverage->setText("Prio2 Coverage: "+prio2+"%");
   // m_Prio1Coverage->setText("Prio1 Coverage: "+prio1+"%");
   // m_Fitness->setText("Fitness: " +);
   // m_NbrCameras->setText("Cameras: " +);
   // m_NbrControlPoints->setText("Control points: " +);
   // m_OptimizationTime->setText("Optimization time: " +);
// }


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
       m.setTrans(osg::Vec3(0,0,0));
       DataManager::AddSafetyZone(Factory::createSafetyZone(SafetyZone::Priority::PRIO1,m,0.20,0.30,0.02));
    }
    );

    m_AddSafetyZonePrio2 = new ui::Action(m_MainMenu,"AddSafetyZonePRIO2");
    m_AddSafetyZonePrio2-> setText("Add PRIO2 Zone");
    m_AddSafetyZonePrio2-> setCallback([]()
    {
       osg::Matrix m;
       m.setTrans(osg::Vec3(0,0,0));
       DataManager::AddSafetyZone(Factory::createSafetyZone(SafetyZone::Priority::PRIO2,m,0.30,0.20,0.02));
    }
    );

    m_AddSensorZone = new ui::Action(m_MainMenu,"AddSensorZone");
    m_AddSensorZone-> setText("Add Sensor Zone");
    m_AddSensorZone-> setCallback([this]()
    {
       osg::Matrix m;
       m.setTrans(osg::Vec3(0,0,0));
       DataManager:: AddSensorZone(Factory::createSensorZone(SensorType::Camera, m, 0.5,0.2,0.02));
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
   m_Visibility->setText("Visibility [m]");
   m_Visibility->setBounds(1.0, 60.0);
   m_Visibility->setValue(Camera::s_CameraProps.m_DepthView);
   m_Visibility->setCallback([](double value, bool released)
   {
      DataManager::updateDoF(value);
   });

   m_FOV = new ui::Slider(m_CameraProps,"FOV");
   m_FOV->setIntegral(true);
   m_FOV->setText("FoV [°]: ");
   m_FOV->setBounds(30, 140);
   m_FOV->setValue(Camera::s_CameraProps.m_FoV);
   m_FOV->setCallback([](double value, bool released)
   {
      DataManager::updateFoV(value);
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
   });

   m_ResetColors = new ui::Action(m_Optimization,"Reset colors");
   m_ResetColors-> setText("Reset colors");
   m_ResetColors-> setCallback([this]()
   {
      DataManager::setOriginalZoneColor();
   });

   m_VisibilityThreshold = new ui::Slider(m_Optimization,"VisibilityThreshold");
   m_VisibilityThreshold->setText("Visibility threshold");
   m_VisibilityThreshold->setBounds(0.0, 1.0);
   m_VisibilityThreshold->setValue(GA::s_VisibiltyThreshold);
   m_VisibilityThreshold->setCallback([](double value, bool released)
   {
      GA::s_VisibiltyThreshold = value;
   });

   m_UseVisibiltyThreshold = new ui::Button(m_Optimization,"UseVisibilityThreshold");
   m_UseVisibiltyThreshold->setText("Use Visibility threshold for orientation comparison");
   m_UseVisibiltyThreshold->setState(GA::s_UseVisibilityThrsholdInOrientationComparison);
   m_UseVisibiltyThreshold->setCallback([this](bool state)
   {
      GA::s_VisibiltyThreshold = state;
   });

   m_OnlyKeepOrienatationsWithMostPoints = new ui::Button(m_Optimization,"OnlyKeepOrienatationsWithMostPoints");
   m_OnlyKeepOrienatationsWithMostPoints->setText("Only keep orienatations with most points");
   m_OnlyKeepOrienatationsWithMostPoints->setState(GA::s_OnlyKeepOrientationWithMostPoints);
   m_OnlyKeepOrienatationsWithMostPoints->setCallback([this](bool state)
   {
      GA::s_OnlyKeepOrientationWithMostPoints = state;
   });


   m_MaxCoverage1Menu = new ui::Menu(m_Optimization, "MaxCoverage1");
   m_MaxCoverage1Menu->setText("Max Coverage 1");

   m_WeightingPrio1 = new ui::Slider(m_MaxCoverage1Menu,"Weighting Prio1");
   m_WeightingPrio1->setIntegral(true);
   m_WeightingPrio1->setText("Weighting Prio1: ");
   m_WeightingPrio1->setBounds(1, 10);
   m_WeightingPrio1->setValue(GA::s_PropsMaxCoverage1.weightingFactorPRIO1);
   m_WeightingPrio1->setCallback([](double value, bool released)
   {
      if(released)
         GA::s_PropsMaxCoverage1.weightingFactorPRIO1 = value;
   });

   m_Penalty = new ui::Slider(m_MaxCoverage1Menu,"Penalty");
   m_Penalty->setIntegral(true);
   m_Penalty->setText("Penalty too few cameras: ");
   m_Penalty->setBounds(1, 8000);
   m_Penalty->setValue(GA::s_PropsMaxCoverage1.Penalty);
   m_Penalty->setCallback([](double value, bool released)
   {
      if(released)
         GA::s_PropsMaxCoverage1.Penalty = value;
   });


   m_MaxCoverage2Menu = new ui::Menu(m_Optimization, "MaxCoverage2");
   m_MaxCoverage2Menu->setText("Max Coverage 2");

   m_RequiredCoverage = new ui::Slider(m_MaxCoverage2Menu,"Required Coverage");
   m_RequiredCoverage->setIntegral(true);
   m_RequiredCoverage->setText("Required Coverage: ");
   m_RequiredCoverage->setBounds(0.1, 1.0);
   m_RequiredCoverage->setValue(GA::s_PropsMaxCoverage2.m_RequiredCoverage);
   m_RequiredCoverage->setCallback([](double value, bool released)
   {
      if(released)
         GA::s_PropsMaxCoverage2.m_RequiredCoverage = value;
   });

   m_PenaltyConstant = new ui::Slider(m_MaxCoverage2Menu,"PenaltyConstant");
   m_PenaltyConstant->setIntegral(true);
   m_PenaltyConstant->setText("Penalty: ");
   m_PenaltyConstant->setBounds(500, 8000);
   m_PenaltyConstant->setValue(GA::s_PropsMaxCoverage2.m_PenaltyConstant);
   m_PenaltyConstant->setCallback([](double value, bool released)
   {
      if(released)
         GA::s_PropsMaxCoverage2.m_PenaltyConstant = value;
   });



   m_Results = new ui::Menu(m_Optimization, "Results");
   m_Results->setText("Results");

   m_TotalCoverage = new ui::Label(m_Results, "TotalCoverage");
   m_TotalCoverage->setText("Total Coverage:");
   m_Prio1Coverage = new ui::Label(m_Results, "Prio1Coverage");
   m_Prio1Coverage->setText("Prio1 Coverage:");
   m_Prio2Coverage = new ui::Label(m_Results, "Prio2Coverage");
   m_Prio2Coverage->setText("Prio2 Coverage:");
   m_Fitness = new ui::Label(m_Results, "Fitness");
   m_Fitness->setText("Fitness:");
   m_OptimizationTime = new ui::Label(m_Results, "OptimizationTime");
   m_OptimizationTime->setText("Optimization time:");
   m_NbrCameras = new ui::Label(m_Results, "NbrCameras");
   m_NbrCameras->setText("Cameras: ");
   m_NbrControlPoints = new ui::Label(m_Results, "NbrControlPoints");
   m_NbrControlPoints->setText("Control points:");



   //Demonstrator Menu ---------------------------------------------------------------------

   m_Demonstrator = new ui::Menu(m_MainMenu, "Demonstrator");
   m_Demonstrator->setText("Demonstrator");

   m_cameraPositions = new ui::Button(m_Demonstrator,"Camera_positions");
   m_cameraPositions->setText("show possible camera positions");
   m_cameraPositions->setState(false);
   m_cameraPositions->setCallback([this](bool state)
   {
      if(state)
      {
         osg::Matrix m = osg::Matrix::translate(osg::Vec3(1.28,-0.04,0.28));
         DataManager::AddSensorZone(Factory::createSensorZone(SensorType::Camera, m, 0.7));
         
         m = osg::Matrix::translate(osg::Vec3(0.22,-0.04,0.28));
         DataManager::AddSensorZone(Factory::createSensorZone(SensorType::Camera, m, 0.7));

         m = osg::Matrix::translate(osg::Vec3(1.28,1.54,0.28));
         DataManager::AddSensorZone(Factory::createSensorZone(SensorType::Camera, m, 0.7));
         
         const char *covisedir = getenv("COVISEDIR");

         osg::ref_ptr<osg::Node> cameraSphere= osgDB::readNodeFile(std::string(covisedir)+ "/CameraSphere.obj");
         if (!cameraSphere.valid())
         {
               osg::notify( osg::FATAL ) << "Unable to load camera sphere" << std::endl;
         }
         else
         {
            // const unsigned int nodeMask = UINT32_MAX & ~opencover::Isect::Intersection & ~opencover::Isect::Pick;
            // cameraSphere->setNodeMask(nodeMask);
            // osg::Matrix spherePos1 = osg::Matrix::translate(osg::Vec3(1.28,-0.04,0.28));
            // osg::Matrix spherePos2 = osg::Matrix::translate(osg::Vec3(0.22,-0.04,0.28));
            // osg::Matrix spherePos3 = osg::Matrix::translate(osg::Vec3(1.28,1.54,0.28));
// 
            // osg::ref_ptr<osg::MatrixTransform> mt1 = new osg::MatrixTransform(spherePos1);
            // osg::ref_ptr<osg::MatrixTransform> mt2 = new osg::MatrixTransform(spherePos2);
            // osg::ref_ptr<osg::MatrixTransform> mt3 = new osg::MatrixTransform(spherePos3);
// 
            // mt1->addChild(cameraSphere);
            // mt2->addChild(cameraSphere);
            // mt3->addChild(cameraSphere);
// 
            // DataManager::GetRootNode()->addChild(mt1);
            // DataManager::GetRootNode()->addChild(mt2);
            // DataManager::GetRootNode()->addChild(mt3);
         }
      }

   });


    //UDP Menu-------------------------------------------------------------------------------
   m_UDP = new ui::Menu(m_Demonstrator, "UDP");
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

