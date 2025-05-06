/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CAVEWalls.h"

#include <osgDB/ReadFile>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRSelectionManager.h>
#include <cover/RenderObject.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/EditField.h>
// #include <cover/ui/Manager.h>
// #include <cover/ui/TextField.h>
// #include <cover/ui/FileBrowser.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coBackground.h>

#include <osg/Group>
#include <osg/Transform>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Vec2>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/PositionAttitudeTransform>
#include <osg/Math>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osg/Material>
#include <osg/Image>
#include <osg/Texture2D>

#include <osgViewer/Viewer>

#include <config/CoviseConfig.h>

#include <cassert>
#include <cstdlib>
#include <string>
#include <algorithm>

using namespace opencover;
using namespace vrui;

CAVEWalls::CAVEWalls()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("CAVEWalls", cover->ui)
{
    fprintf(stderr, "CAVEWalls::CAVEWalls ON\n");

    float x = covise::coCoviseConfig::getFloat("x", "COVER.Plugin.CAVEWalls.Dimensions", 3000.0);
    float y = covise::coCoviseConfig::getFloat("y", "COVER.Plugin.CAVEWalls.Dimensions", 3000.0);
    float z = covise::coCoviseConfig::getFloat("z", "COVER.Plugin.CAVEWalls.Dimensions", 3000.0);

    floor_width = static_cast<double>(x);
    floor_length = static_cast<double>(y);
    room_height = static_cast<double>(z);

    std::string covise_dir = std::string(std::getenv("COVISEDIR")) + "/";
    std::string default_rel_filepath = "src/OpenCOVER/plugins/ukoeln/CAVEWalls/checkerboard.png";

    image_filepath = covise::coCoviseConfig::getEntry(
                                                "filepath",
                                                "COVER.Plugin.CAVEWalls.Image",
                                                covise_dir + default_rel_filepath
                                                );
}

// this is called if the plugin is removed at runtime
CAVEWalls::~CAVEWalls()
{
    cover->getMenuGroup()->removeChild(room_transform);
    fprintf(stderr, "CAVEWalls::CAVEWalls OFF\n");
}

bool CAVEWalls::init()
{
    CAVEWalls_menu = new ui::Menu("CAVEWalls_menu", this);
    CAVEWalls_menu->setText("CAVEWalls");

    // platform_room_buttonGroup = new ui::ButtonGroup(CAVEWalls_menu, "platform_room_buttonGroup");
    // platform_room_buttonGroup->enableDeselect(true);

    platform_button = new ui::Button(CAVEWalls_menu, "platform_button");
    platform_button->setText("Platform");

    room_button = new ui::Button(CAVEWalls_menu, "room_button");
    room_button->setText("Room");

    floorWidth_edit = new ui::EditField(CAVEWalls_menu, "floorWidth_edit");
    floorWidth_edit->setText("Floor width:");
    floorWidth_edit->setValue(floor_width);

    floorLength_edit = new ui::EditField(CAVEWalls_menu, "floorLength_edit");
    floorLength_edit->setText("Floor length:");
    floorLength_edit->setValue(floor_length);

    roomHeight_edit = new ui::EditField(CAVEWalls_menu, "roomHeight_edit");
    roomHeight_edit->setText("Room height:");
    roomHeight_edit->setValue(room_height);

    floorWidth_edit->setCallback(
        [this](std::string edit_value)
        {
            if (floorWidth_edit->number() < 0.1)
            {
                floorWidth_edit->setValue(0.1);
            }

            this->resizeRoom(floorWidth_edit->number(), floor_length, room_height, floor_width, floor_length, room_height);

            floor_width = floorWidth_edit->number();

            std::cout << "Floor width set to " << std::to_string(floor_width) << "." << std::endl;
        }
    );

    floorLength_edit->setCallback(
        [this](std::string edit_value)
        {
            if (floorLength_edit->number() < 0.1)
            {
                floorLength_edit->setValue(0.1);
            }

            this->resizeRoom(floor_width, floorLength_edit->number(), room_height, floor_width, floor_length, room_height);

            floor_length = floorLength_edit->number();

            std::cout << "Floor length set to " << std::to_string(floor_length) << "." << std::endl;
        }
    );

    roomHeight_edit->setCallback
    (
        [this](std::string edit_value)
        {
            if (roomHeight_edit->number() < 0.1)
            {
                roomHeight_edit->setValue(0.1);
            }

            this->resizeRoom(floor_width, floor_length, roomHeight_edit->number(), floor_width, floor_length, room_height);

            room_height = roomHeight_edit->number();

            std::cout << "Room height set to " << std::to_string(room_height) << "." << std::endl;
        }
    );

    panel_image = osgDB::readImageFile(image_filepath);

    if (panel_image)
    {
        panel_texture = new osg::Texture2D();
        panel_texture->setImage(0, panel_image);
        // panel_texture->setTextureSize(1024, 1024);
    }

    room_transform = new osg::PositionAttitudeTransform();
    // room_transform->setPosition(cover->getViewerMat().getTrans());

    cover->getMenuGroup()->addChild(room_transform);
    
    initPanel(CAVEWalls::Panel::FLOOR);
    initPanel(CAVEWalls::Panel::FRONT_WALL);
    initPanel(CAVEWalls::Panel::LEFT_WALL);
    initPanel(CAVEWalls::Panel::RIGHT_WALL);
    initPanel(CAVEWalls::Panel::BACK_WALL);
    initPanel(CAVEWalls::Panel::CEILING);

    platform_button->setCallback(
        [this](const int state_int)
        {
            if (state_int == 1)
            {
                isPlatform = true;
                floor_transform->setNodeMask(0xffffffff);
                floor_material->setAlpha(osg::Material::FRONT, 1.0);
            }
            else
            {
                isPlatform = false;
                floor_transform->setNodeMask(0x00);
                floor_material->setAlpha(osg::Material::FRONT, 0.5);
            }
        }
    );

    room_button->setCallback(
        [this](const int state_int)
        {
            if (state_int == 1)
            {
                isTracking = true;

                /*
                frontWall_transform->setNodeMask(0xffffffff);
                leftWall_transform->setNodeMask(0xffffffff);
                rightWall_transform->setNodeMask(0xffffffff);
                backWall_transform->setNodeMask(0xffffffff);
                ceiling_transform->setNodeMask(0xffffffff);
                */
            }
            else
            {
                isTracking = false;

                frontWall_transform->setNodeMask(0x00);
                leftWall_transform->setNodeMask(0x00);
                rightWall_transform->setNodeMask(0x00);
                backWall_transform->setNodeMask(0x00);
                ceiling_transform->setNodeMask(0x00);
            }
        }
    );

    return true;
}

void CAVEWalls::initPanel(CAVEWalls::Panel panel)
{
    // osg::PositionAttitudeTransform* panel_transform = new osg::PositionAttitudeTransform();
    // osg::Geode* panel_geode = new osg::Geode();
    // osg::StateSet* panel_stateSet = panel_geode->getOrCreateStateSet();
    osg::Geometry* panel_geometry = new osg::Geometry();
    
    osg::Vec3Array* vertices = new osg::Vec3Array;
    osg::Vec4d color = osg::Vec4d(0.5, 0.5, 0.5, 1.0);
    osg::Vec3Array* normals = new osg::Vec3Array;
    osg::Vec2Array* texCoord = new osg::Vec2Array;

    switch (panel)
    {
        case CAVEWalls::Panel::FLOOR:
        {
            // The four vertices for the floor plane.
            vertices->push_back(osg::Vec3d(- 0.5 * floor_width, - 0.5 * floor_length, 0.0));
            vertices->push_back(osg::Vec3d(- 0.5 * floor_width, 0.5 * floor_length, 0.0));
            vertices->push_back(osg::Vec3d(0.5 * floor_width, 0.5 * floor_length, 0.0));
            vertices->push_back(osg::Vec3d(0.5 * floor_width, - 0.5 * floor_length, 0.0));

            // The normal for the floor.
            normals->push_back(osg::Vec3d(0.0, 0.0, 1.0));

            // The texture coordinates for the floor.
            texCoord->push_back(osg::Vec2f(0.0f, 0.0f));
            texCoord->push_back(osg::Vec2f(0.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 0.0f));
        }
        break;

        case CAVEWalls::Panel::FRONT_WALL:
        {
            // The four vertices for the front wall plane.
            vertices->push_back(osg::Vec3d(- 0.5 * floor_width, 0.0, - 0.5 * room_height));
            vertices->push_back(osg::Vec3d(- 0.5 * floor_width, 0.0, 0.5 * room_height));
            vertices->push_back(osg::Vec3d(0.5 * floor_width, 0.0, 0.5 * room_height));
            vertices->push_back(osg::Vec3d(0.5 * floor_width, 0.0, - 0.5 * room_height));

            // The normal for the front wall.
            normals->push_back(osg::Vec3d(0.0, -1.0, 0.0));

            // The texture coordinates for the front wall.
            texCoord->push_back(osg::Vec2f(0.0f, 0.0f));
            texCoord->push_back(osg::Vec2f(0.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 0.0f));
        }
        break;

        case CAVEWalls::Panel::LEFT_WALL:
        {
            // The four vertices for the left wall plane.
            vertices->push_back(osg::Vec3d(0.0, - 0.5 * floor_length, - 0.5 * room_height));
            vertices->push_back(osg::Vec3d(0.0, - 0.5 * floor_length, 0.5 * room_height));
            vertices->push_back(osg::Vec3d(0.0, 0.5 * floor_length, 0.5 * room_height));
            vertices->push_back(osg::Vec3d(0.0, 0.5 * floor_length, - 0.5 * room_height));

            // The normal for the left wall.
            normals->push_back(osg::Vec3d(1.0, 0.0, 0.0));

            // The texture coordinates for the left wall.
            texCoord->push_back(osg::Vec2f(0.0f, 0.0f));
            texCoord->push_back(osg::Vec2f(0.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 0.0f));
        }
        break;

        case CAVEWalls::Panel::RIGHT_WALL:
        {
            // The four vertices for the right wall plane.
            vertices->push_back(osg::Vec3d(0.0, 0.5 * floor_length, - 0.5 * room_height));
            vertices->push_back(osg::Vec3d(0.0, 0.5 * floor_length, 0.5 * room_height));
            vertices->push_back(osg::Vec3d(0.0, - 0.5 * floor_length, 0.5 * room_height));
            vertices->push_back(osg::Vec3d(0.0, - 0.5 * floor_length, - 0.5 * room_height));

            // The normal for the right wall.
            normals->push_back(osg::Vec3d(-1.0, 0.0, 0.0));

            // The texture coordinates for the right wall.
            texCoord->push_back(osg::Vec2f(0.0f, 0.0f));
            texCoord->push_back(osg::Vec2f(0.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 0.0f));
        }
        break;

        case CAVEWalls::Panel::BACK_WALL:
        {
            // The four vertices for the back wall plane.
            vertices->push_back(osg::Vec3d(0.5 * floor_width, 0.0, - 0.5 * room_height));
            vertices->push_back(osg::Vec3d(0.5 * floor_width, 0.0, 0.5 * room_height));
            vertices->push_back(osg::Vec3d(- 0.5 * floor_width, 0.0, 0.5 * room_height));
            vertices->push_back(osg::Vec3d(- 0.5 * floor_width, 0.0, - 0.5 * room_height));

            // The normal for the back wall.
            normals->push_back(osg::Vec3d(0.0, 1.0, 0.0));

            // The texture coordinates for the back wall.
            texCoord->push_back(osg::Vec2f(0.0f, 0.0f));
            texCoord->push_back(osg::Vec2f(0.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 0.0f));
        }
        break;

        case CAVEWalls::Panel::CEILING:
        {    
            // The four vertices for the ceiling plane.
            vertices->push_back(osg::Vec3d(- 0.5 * floor_width, - 0.5 * floor_length, 0.0));
            vertices->push_back(osg::Vec3d(- 0.5 * floor_width, 0.5 * floor_length, 0.0));
            vertices->push_back(osg::Vec3d(0.5 * floor_width, 0.5 * floor_length, 0.0));
            vertices->push_back(osg::Vec3d(0.5 * floor_width, - 0.5 * floor_length, 0.0));

            // The normal for the ceiling.
            normals->push_back(osg::Vec3d(0.0, 0.0, -1.0));

            // The texture coordinates for the ceiling.
            texCoord->push_back(osg::Vec2f(0.0f, 0.0f));
            texCoord->push_back(osg::Vec2f(0.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 1.0f));
            texCoord->push_back(osg::Vec2f(1.0f, 0.0f));
        }
        break;

        default:
            return;
            break;
    }

    panel_geometry->setVertexArray(vertices);
    panel_geometry->setNormalArray(normals);
    panel_geometry->setTexCoordArray(0, texCoord);
    panel_geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, vertices->size()));

    switch (panel)
    {
        case CAVEWalls::Panel::FLOOR:
        {
            floor_geode = new osg::Geode();
            floor_transform = new osg::PositionAttitudeTransform();

            floor_geode->addDrawable(panel_geometry);
            floor_transform->addChild(floor_geode);
            room_transform->addChild(floor_transform);

            osg::StateSet* floor_stateSet = floor_geode->getOrCreateStateSet();
            
            floor_transform->setPosition(osg::Vec3d(0.0, 0.0, - 0.5 * room_height));
            floor_transform->setNodeMask(0x00);

            floor_material = dynamic_cast<osg::Material*>(floor_stateSet->getAttribute(osg::StateAttribute::MATERIAL));

            if (!floor_material)
            {
                floor_material = new osg::Material;
            }

            floor_material->setColorMode(osg::Material::EMISSION);
            floor_material->setEmission(osg::Material::FRONT, color);
            floor_material->setAlpha(osg::Material::FRONT, 0.5);

            if (panel_texture && panel_texture->getImage())
            {
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER, osg::Texture2D::FilterMode::LINEAR);
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER, osg::Texture2D::FilterMode::LINEAR);

                floor_stateSet->setTextureAttribute(0, panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                floor_stateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                floor_stateSet->setAttributeAndModes(panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }

            floor_stateSet->setAttributeAndModes(floor_material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            floor_stateSet->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            floor_stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            floor_stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        }
        break;

        case CAVEWalls::Panel::FRONT_WALL:
        {
            frontWall_geode = new osg::Geode();
            frontWall_transform = new osg::PositionAttitudeTransform();

            frontWall_geode->addDrawable(panel_geometry);
            frontWall_transform->addChild(frontWall_geode);
            room_transform->addChild(frontWall_transform);

            osg::StateSet* frontWall_stateSet = frontWall_geode->getOrCreateStateSet();

            frontWall_transform->setPosition(osg::Vec3d(0.0, 0.5 * floor_length, 0.0));
            frontWall_transform->setNodeMask(0x00);

            frontWall_material = dynamic_cast<osg::Material*>(frontWall_stateSet->getAttribute(osg::StateAttribute::MATERIAL));

            if (!frontWall_material)
            {
                frontWall_material = new osg::Material;
            }

            frontWall_material->setColorMode(osg::Material::EMISSION);
            frontWall_material->setEmission(osg::Material::FRONT, color);    
            frontWall_material->setAlpha(osg::Material::FRONT, 0.5);

            if (panel_texture && panel_texture->getImage())
            {
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER, osg::Texture2D::FilterMode::LINEAR);
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER, osg::Texture2D::FilterMode::LINEAR);

                frontWall_stateSet->setTextureAttribute(0, panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                frontWall_stateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

                frontWall_stateSet->setAttributeAndModes(panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }

            frontWall_stateSet->setAttribute(frontWall_material, osg::StateAttribute::OVERRIDE);
            frontWall_stateSet->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            frontWall_stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            frontWall_stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        }
        break;

        case CAVEWalls::Panel::LEFT_WALL:
        {
            leftWall_geode = new osg::Geode();
            leftWall_transform = new osg::PositionAttitudeTransform();

            leftWall_geode->addDrawable(panel_geometry);
            leftWall_transform->addChild(leftWall_geode);
            room_transform->addChild(leftWall_transform);

            osg::StateSet* leftWall_stateSet = leftWall_geode->getOrCreateStateSet();

            leftWall_transform->setPosition(osg::Vec3d(- 0.5 * floor_width, 0.0, 0.0));
            leftWall_transform->setNodeMask(0x00);

            leftWall_material = dynamic_cast<osg::Material*>(leftWall_stateSet->getAttribute(osg::StateAttribute::MATERIAL));

            if (!leftWall_material)
            {
                leftWall_material = new osg::Material;
            }

            leftWall_material->setColorMode(osg::Material::EMISSION);
            leftWall_material->setEmission(osg::Material::FRONT, color);    
            leftWall_material->setAlpha(osg::Material::FRONT, 0.5);

            if (panel_texture && panel_texture->getImage())
            {
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER, osg::Texture2D::FilterMode::LINEAR);
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER, osg::Texture2D::FilterMode::LINEAR);

                leftWall_stateSet->setTextureAttribute(0, panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                leftWall_stateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

                leftWall_stateSet->setAttributeAndModes(panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }

            leftWall_stateSet->setAttribute(leftWall_material, osg::StateAttribute::OVERRIDE);
            leftWall_stateSet->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            leftWall_stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            leftWall_stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        }
        break;

        case CAVEWalls::Panel::RIGHT_WALL:
        {
            rightWall_geode = new osg::Geode();
            rightWall_transform = new osg::PositionAttitudeTransform();

            rightWall_geode->addDrawable(panel_geometry);
            rightWall_transform->addChild(rightWall_geode);
            room_transform->addChild(rightWall_transform);

            osg::StateSet* rightWall_stateSet = rightWall_geode->getOrCreateStateSet();

            rightWall_transform->setPosition(osg::Vec3d(0.5 * floor_width, 0.0, 0.0));
            rightWall_transform->setNodeMask(0x00);

            rightWall_material = dynamic_cast<osg::Material*>(rightWall_stateSet->getAttribute(osg::StateAttribute::MATERIAL));

            if (!rightWall_material)
            {
                rightWall_material = new osg::Material;
            }

            rightWall_material->setColorMode(osg::Material::EMISSION);
            rightWall_material->setEmission(osg::Material::FRONT, color);    
            rightWall_material->setAlpha(osg::Material::FRONT, 0.5);

            if (panel_texture && panel_texture->getImage())
            {
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER, osg::Texture2D::FilterMode::LINEAR);
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER, osg::Texture2D::FilterMode::LINEAR);

                rightWall_stateSet->setTextureAttribute(0, panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                rightWall_stateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

                rightWall_stateSet->setAttributeAndModes(panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }

            rightWall_stateSet->setAttribute(rightWall_material, osg::StateAttribute::OVERRIDE);
            rightWall_stateSet->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            rightWall_stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            rightWall_stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        }
        break;

        case CAVEWalls::Panel::BACK_WALL:
        {
            backWall_geode = new osg::Geode();
            backWall_transform = new osg::PositionAttitudeTransform();

            backWall_geode->addDrawable(panel_geometry);
            backWall_transform->addChild(backWall_geode);
            room_transform->addChild(backWall_transform);

            osg::StateSet* backWall_stateSet = backWall_geode->getOrCreateStateSet();

            backWall_transform->setPosition(osg::Vec3d(0.0, - 0.5 * floor_length, 0.0));
            backWall_transform->setNodeMask(0x00);

            backWall_material = dynamic_cast<osg::Material*>(backWall_stateSet->getAttribute(osg::StateAttribute::MATERIAL));

            if (!backWall_material)
            {
                backWall_material = new osg::Material;
            }

            backWall_material->setColorMode(osg::Material::EMISSION);
            backWall_material->setEmission(osg::Material::FRONT, color);    
            backWall_material->setAlpha(osg::Material::FRONT, 0.5);

            if (panel_texture && panel_texture->getImage())
            {
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER, osg::Texture2D::FilterMode::LINEAR);
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER, osg::Texture2D::FilterMode::LINEAR);

                backWall_stateSet->setTextureAttribute(0, panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                backWall_stateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

                backWall_stateSet->setAttributeAndModes(panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }

            backWall_stateSet->setAttribute(backWall_material, osg::StateAttribute::OVERRIDE);
            backWall_stateSet->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            backWall_stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            backWall_stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        }
        break;

        case CAVEWalls::Panel::CEILING:
        {
            ceiling_geode = new osg::Geode();
            ceiling_transform = new osg::PositionAttitudeTransform();

            ceiling_geode->addDrawable(panel_geometry);
            ceiling_transform->addChild(ceiling_geode);
            room_transform->addChild(ceiling_transform);

            osg::StateSet* ceiling_stateSet = ceiling_geode->getOrCreateStateSet();

            ceiling_transform->setPosition(osg::Vec3d(0.0, 0.0, 0.5 * room_height));
            ceiling_transform->setNodeMask(0x00);

            ceiling_material = dynamic_cast<osg::Material*>(ceiling_stateSet->getAttribute(osg::StateAttribute::MATERIAL));

            if (!ceiling_material)
            {
                ceiling_material = new osg::Material;
            }

            ceiling_material->setColorMode(osg::Material::EMISSION);
            ceiling_material->setEmission(osg::Material::FRONT, color);    
            ceiling_material->setAlpha(osg::Material::FRONT, 0.5);

            if (panel_texture && panel_texture->getImage())
            {
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER, osg::Texture2D::FilterMode::LINEAR);
                panel_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER, osg::Texture2D::FilterMode::LINEAR);

                ceiling_stateSet->setTextureAttribute(0, panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                ceiling_stateSet->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

                ceiling_stateSet->setAttributeAndModes(panel_texture, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            }

            ceiling_stateSet->setAttribute(ceiling_material, osg::StateAttribute::OVERRIDE);
            ceiling_stateSet->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            ceiling_stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            ceiling_stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        }
        break;

        default:
            return;
            break;
    }
}

void CAVEWalls::resizeRoom(double new_width, double new_length, double new_height, double old_width, double old_length, double old_height)
{
    osg::Vec3d currentScale = room_transform->getScale();
    osg::Vec3d relativeScale = osg::Vec3d(new_width / old_width, new_length / old_length, new_height / old_height);
    osg::Vec3d newScale = osg::Vec3d(relativeScale.x() * currentScale.x(), relativeScale.y() * currentScale.y(), relativeScale.z() * currentScale.z());

    room_transform->setScale(newScale);
}

void CAVEWalls::preFrame()
{
    if (!isTracking)
    {
        return;
    }

    osg::Vec3d flystick_pos = cover->getPointerMat().getTrans();
    osg::Vec3d viewer_pos = cover->getViewerMat().getTrans();

    if ((flystick_pos.x() < 0.8 * (- 0.5 * floor_width)) || (viewer_pos.x() < 0.8 * (- 0.5 * floor_width)))
    {
        leftWall_transform->setNodeMask(0xfffffff);

        if ((flystick_pos.x() < 0.9 * (- 0.5 * floor_width)) || (viewer_pos.x() < 0.9 * (- 0.5 * floor_width)))
        {
            leftWall_material->setAlpha(osg::Material::FRONT, 1.0);
        }
        else
        {
            leftWall_material->setAlpha(osg::Material::FRONT, 0.5);
        }
    }
    else
    {
        leftWall_transform->setNodeMask(0x00);
    }
    
    if ((flystick_pos.x() > 0.8 * (0.5 * floor_width)) || (viewer_pos.x() > 0.8 * (0.5 * floor_width)))
    {
        rightWall_transform->setNodeMask(0xffffffff);

        if ((flystick_pos.x() > 0.9 * (0.5 * floor_width)) || (viewer_pos.x() > 0.9 * (0.5 * floor_width)))
        {
            rightWall_material->setAlpha(osg::Material::FRONT, 1.0);
        }
        else
        {
            rightWall_material->setAlpha(osg::Material::FRONT, 0.5);
        }
    }
    else
    {
        rightWall_transform->setNodeMask(0x00);
    }
    
    if ((flystick_pos.y() > 0.8 * (0.5 * floor_length)) || (viewer_pos.y() > 0.8 * (0.5 * floor_length)))
    {
        frontWall_transform->setNodeMask(0xfffffff);

        if ((flystick_pos.y() > 0.9 * (0.5 * floor_length)) || (viewer_pos.y() > 0.9 * (0.5 * floor_length)))
        {
            frontWall_material->setAlpha(osg::Material::FRONT, 1.0);
        }
        else
        {
            frontWall_material->setAlpha(osg::Material::FRONT, 0.5);
        }
    }
    else
    {
        frontWall_transform->setNodeMask(0x00);
        // frontWall_transform->setNodeMask(0xffffffff);
        // frontWall_material->setAlpha(osg::Material::FRONT, 0.5);
    }

    /*
    if ((flystick_pos.y() < 0.8 * (- 0.5 * floor_length)) || (viewer_pos.y() < 0.8 * (- 0.5 * floor_length)))
    {
        backWall_transform->setNodeMask(0xfffffff);

        if ((flystick_pos.y() < 0.9 * (- 0.5 * floor_width)) || (viewer_pos.y() < 0.9 * (- 0.5 * floor_width)))
        {
            backWall_material->setAlpha(osg::Material::FRONT, 1.0);
        }
        else
        {
            backWall_material->setAlpha(osg::Material::FRONT, 0.5);
        }
    }
    else
    {
        backWall_transform->setNodeMask(0x00);
    }
    */

    if ((flystick_pos.z() > 0.8 * (0.5 * room_height)) || (viewer_pos.z() > 0.8 * (0.5 * room_height)))
    {
        ceiling_transform->setNodeMask(0xfffffff);

        if ((flystick_pos.z() > 0.9 * (0.5 * room_height)) || (viewer_pos.z() > 0.9 * (0.5 * room_height)))
        {
            ceiling_material->setAlpha(osg::Material::FRONT, 1.0);
        }
        else
        {
            ceiling_material->setAlpha(osg::Material::FRONT, 0.5);
        }
    }
    else
    {
        ceiling_transform->setNodeMask(0x00);
    }

    if (isPlatform)
    {
        return;
    }

    if ((flystick_pos.z() < 0.8 * (- 0.5 * room_height)) || (viewer_pos.z() < 0.8 * (- 0.5 * room_height)))
    {
        floor_transform->setNodeMask(0xfffffff);

        if ((flystick_pos.z() < 0.9 * (- 0.5 * room_height)) || (viewer_pos.z() < 0.9 * (- 0.5 * room_height)))
        {
            floor_material->setAlpha(osg::Material::FRONT, 1.0);
        }
        else
        {
            floor_material->setAlpha(osg::Material::FRONT, 0.5);
        }
    }
    else
    {
        floor_transform->setNodeMask(0x00);
    }
}

COVERPLUGIN(CAVEWalls)