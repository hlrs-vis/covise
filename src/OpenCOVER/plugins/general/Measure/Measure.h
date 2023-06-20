/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MEASURE_H
#define _MEASURE_H

#include "Pin.h"
#include "LinearDimension.h"


#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Material>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osgText/Text>
#include <cover/coVRPluginSupport.h>

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/osg/OSGVruiNode.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>

#include <array>
#include <vector>


using namespace vrui;
using namespace opencover;


/** Line between markers */


/** Main plugin class */
class Measure : public coVRPlugin, public ui::Owner
{
public:

    Measure();
    void preFrame() override;
    void setCurrentMeasure(const Pin &pin);


private:


    ui::Menu *m_menu;
    ui::Action *m_addLinearDimension;
    ui::Action *m_removeCurrentDimension;
    ui::Action *m_clearDimensions;
    
    Scaling m_scaling;
    Dimension *m_currentDimension = nullptr;    
    std::map<int, std::unique_ptr<Dimension>> m_dimensions;


};

#endif
