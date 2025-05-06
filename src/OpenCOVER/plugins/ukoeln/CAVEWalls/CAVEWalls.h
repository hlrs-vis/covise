/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CAVEWALLS_H
#define CAVEWALLS_H
/****************************************************************************************\ 
 **                                                                                    **
 **                                                                                    **
 ** Description: Virtual walls for the CAVE. Detects when user is near the walls.      **
 **                                                                                    **
 **                                                                                    **
 ** Author: Tze Sheng Ng <tng1@uni-koeln.de>                                           **
 **                                                                                    **
\****************************************************************************************/

#include <cover/coVRPlugin.h>

#include <util/coTypes.h>

#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coUIContainer.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coBackground.h>

#include <osg/Transform>
#include <osg/PositionAttitudeTransform>
#include <osg/Material>
#include <osg/Texture2D>
#include <osg/Image>

// #include <mutex>

#include <cover/ui/Owner.h>

namespace opencover
{
    namespace ui
    {
        class Menu;
        class Action;
        class Button;
        class ButtonGroup;
        class EditField;
    }
}

using namespace opencover;
using namespace vrui;

class CAVEWalls : public coVRPlugin, public ui::Owner
{
    public:
        enum Panel
        {
            FLOOR,
            FRONT_WALL,
            LEFT_WALL,
            RIGHT_WALL,
            BACK_WALL,
            CEILING
        };

    private:
        std::string m_name = "CAVEWalls";
        bool isPlatform = false;
        bool isTracking = false;

        void initPanel(CAVEWalls::Panel panel);
        void resizeRoom(double new_width, double new_length, double new_height, double old_width, double old_length, double old_height);
    
    public:
        CAVEWalls();
        CAVEWalls(const CAVEWalls&) = delete;
        ~CAVEWalls() override;

        static CAVEWalls& getInstance()
        {
            static CAVEWalls instance;
            return instance;
        };

        ui::Menu* CAVEWalls_menu = nullptr;
        ui::Button* platform_button = nullptr;
        ui::Button* room_button = nullptr;
        ui::EditField* floorWidth_edit = nullptr;
        ui::EditField* floorLength_edit = nullptr;
        ui::EditField* roomHeight_edit = nullptr;

        double floor_width = 3000.0;
        double floor_length = 3000.0;
        double room_height = 3000.0;
        
        std::string image_filepath = "";

        bool init() override;
        void preFrame() override;
    
    private:
        osg::Image* panel_image = nullptr;
        osg::Texture2D* panel_texture = nullptr;

        osg::PositionAttitudeTransform* room_transform = nullptr;

        osg::PositionAttitudeTransform* floor_transform = nullptr;
        osg::Geode* floor_geode = nullptr;
        osg::Material* floor_material = nullptr;

        osg::PositionAttitudeTransform* frontWall_transform = nullptr;
        osg::Geode* frontWall_geode = nullptr;
        osg::Material* frontWall_material = nullptr;

        osg::PositionAttitudeTransform* leftWall_transform = nullptr;
        osg::Geode* leftWall_geode = nullptr;
        osg::Material* leftWall_material = nullptr;

        osg::PositionAttitudeTransform* rightWall_transform = nullptr;
        osg::Geode* rightWall_geode = nullptr;
        osg::Material* rightWall_material = nullptr;

        osg::PositionAttitudeTransform* backWall_transform = nullptr;
        osg::Geode* backWall_geode = nullptr;
        osg::Material* backWall_material = nullptr;

        osg::PositionAttitudeTransform* ceiling_transform = nullptr;
        osg::Geode* ceiling_geode = nullptr;
        osg::Material* ceiling_material = nullptr;
};

#endif