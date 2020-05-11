#include "UI.h"
#include "Helper.h"

#include "SensorPlacement.h"

using namespace opencover;

UI::UI() : ui::Owner("SensorPlacementUI", cover->ui)
{
   // Main menu
    m_MainMenu = new ui::Menu("SensorPlacement",this);

    m_AddCamera = new ui::Action(m_MainMenu,"AddCamera");
    m_AddCamera-> setText("Add Camera");
    m_AddCamera-> setCallback([]()
    {
       osg::Matrix m;
       DataManager::AddSensor(myHelpers::make_unique<Camera>(m));
    }
    );

    m_AddSafetyZone = new ui::Action(m_MainMenu,"AddSafetyZoe");
    m_AddSafetyZone-> setText("Add Safety Zone");
    m_AddSafetyZone-> setCallback([]()
    {
       osg::Matrix m;
       DataManager::AddSafetyZone(myHelpers::make_unique<SafetyZone>(m));
    }
    );

    m_Delete = new ui::Button(m_MainMenu,"Delete");
    m_Delete->setText("Remove");
    m_Delete->setState(m_DeleteStatus);
    m_Delete->setCallback([this](bool state){
      m_DeleteStatus = state;
      if(m_DeleteStatus)
         std::cout<<"Delete Status"<<m_DeleteStatus<<std::endl;
    });

   // Camera properties
    m_CameraProps = new ui::Menu(m_MainMenu,"CameraProps");
    m_CameraProps->setText("Camera Properties");

    m_Visibility = new ui::Slider(m_CameraProps,"Visibility");
    m_Visibility->setText("Visibility [ % ]");
    m_Visibility->setBounds(10., 100.);
    m_Visibility->setCallback([](double value, bool released)
    {

    });

};