/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVER_PLUGIN_TOOL_MASCHIE_H
#define COVER_PLUGIN_TOOL_MASCHIE_H



#include <cover/coVRPluginSupport.h>
#include <osg/Vec3>
#include <cover/ui/Owner.h>
#include <open62541/client.h>

class ToolMaschinePlugin : public opencover::coVRPlugin, opencover::ui::Owner
{
public:
    ToolMaschinePlugin();
private:
    void key(int type, int keySym, int mod) override;
    
    std::array<float, 5> m_axisPositions{0,0,0,0,0}; //A, C, X, Y, Z
};




#endif
