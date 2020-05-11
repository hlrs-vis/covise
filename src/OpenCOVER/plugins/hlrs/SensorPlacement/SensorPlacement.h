/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
/****************************************************************************\
 **                                                            (C)2020 HLRS  **
 **                                                                          **
 ** Description: Camera position and orientation optimization                **
 **                                                                          **
 **                                                                          **
 ** Author: Matthias Epple	                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** April 2020  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <vector>
#include <memory>


#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>

#include "Zone.h"
#include "Helper.h"
#include "Sensor.h"
#include "UI.h"
#include "DataManager.h"


class SensorPlacementPlugin :public opencover::coVRPlugin 
{
    public:
    SensorPlacementPlugin();
    ~SensorPlacementPlugin();
    bool init() override;
    bool destroy() override;
    void preFrame() override;

    private:

    std::unique_ptr<UI> m_UI;
};