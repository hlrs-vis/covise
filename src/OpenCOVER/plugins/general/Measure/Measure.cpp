/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Switch>
#include <osg/Matrix>
#include <OpenVRUI/osg/mathUtils.h>

#include <cover/coVRLabel.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>


#include "Measure.h"

/*
	Measure is a COVISE plugin used to create a virtual measuring tape.
	You are able to set the initial coneSize and textSize by adding those values
	into the config XML file.
*/


Measure::Measure()
: coVRPlugin(COVER_PLUGIN_NAME)
, opencover::ui::Owner("MeasurePlugin", cover->ui)
, m_menu(new ui::Menu("Measure", this))
, m_scaling(m_menu)
, m_addLinearDimension(new ui::Action(m_menu, "tape measure"))
, m_removeCurrentDimension(new ui::Action(m_menu, "remove last"))
, m_clearDimensions(new ui::Action(m_menu, "clear all"))
{
    m_addLinearDimension->setCallback([this]()
    {
        for (size_t i = 0;; i++)
        {
            auto &d = m_dimensions[i];
            if(!d)
            {
                d = std::make_unique<LinearDimension>(i, m_menu, m_scaling);
                break;
            }
        }
    });
    m_removeCurrentDimension->setCallback([this](){
        m_dimensions.erase(m_currentDimension->getID());
    });
    m_clearDimensions->setCallback([this](){
        m_dimensions.clear();
    });
    m_scaling.setCallback([this](){
        for(auto &dim : m_dimensions)
        {
            dim.second->setScaling(m_scaling);
        }
    });
}


/** Opencover calls this function before each frame is rendered */
void Measure::preFrame()
{
    std::chrono::system_clock::time_point time;
    for (auto &dim : m_dimensions)
    {
        dim.second->update();
        if(dim.second->getLastChange() > time)
        {
            time = dim.second->getLastChange();
            m_currentDimension = dim.second.get();
        }
    }
}

void Measure::setCurrentMeasure(const Pin &pin)
{
    m_currentDimension = m_dimensions.find(pin.getDimensionID())->second.get();
}


COVERPLUGIN(Measure)
