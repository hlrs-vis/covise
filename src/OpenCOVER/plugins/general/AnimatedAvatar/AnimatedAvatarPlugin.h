/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Animated_Avatar_Plugin
#define _Animated_Avatar_Plugin

#include "AnimatedAvatar.h"

#include <cover/coVRPluginSupport.h>
#include <osgCal/CoreModel>//abstraktes Modell
#include <osgCal/Model>//konkretes Modell in der Szene
#include <cover/ui/FileBrowser.h>
#include <cover/ui/Owner.h>
#include <osg/MatrixTransform>
#include <map>


using namespace covise;
using namespace opencover;
using namespace ui;

class PLUGINEXPORT AnimatedAvatarPlugin : public coVRPlugin, public ui::Owner
{
public:

    AnimatedAvatarPlugin();
    bool update() override; //wird in jedem Frame aufgerufen: check neue/gegangene partner/ (bei coVRCommunication) und update avatare
private:
    std::map<std::string, osg::ref_ptr<osgCal::CoreModel>>m_coreModels; //Übersetzung von File-Name auf CoreModel damit man kein core model 2x lädt
    std::map<int , std::shared_ptr<AnimatedAvatar>> m_avatars;

    void removeAvatar();
    void addPartner();
};

#endif