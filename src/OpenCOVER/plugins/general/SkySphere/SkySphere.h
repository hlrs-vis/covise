/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SKYSPHERE_H
#define SKYSPHERE_H

#include <optional>
#include <osg/Texture2D>
#include <vector>
#include <osg/PositionAttitudeTransform>
#include <osgTerrain/Terrain>
#include <cover/coVRPlugin.h>
#include <cover/coVRShader.h>
#include <proj.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Button.h>
#include <cover/ui/SelectionList.h>
#include <filesystem>

#ifdef HAVE_EPHEMERIS
#include "EphemeralSky.h"
#endif

struct SkyEntry
{
    std::string name;
    std::string fileName;

    // All in degrees
    double longitude = 0.0;
    double latitude = 0.0;
    double trueNorth = 0.0;

    osg::ref_ptr<osg::Texture2D> texture;
};

class SkySphere : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    SkySphere();
    bool init();
    ~SkySphere();

    static SkySphere *instance();

    virtual bool update();
    virtual void message(int toWhom, int type, int length, const void *data);

    void setSky(std::string_view nameOrFile);

    void setTop(float t);
    void setBottom(float b);
    void setFloorColor(osg::Vec4 fc);

    void setTrueNorth(float trueNorth);

private:
    enum SkyMode
    {
        DISABLED,
        TEXTURE,
        EPHEMERIS,
        AUTO,
    } m_mode;

    void loadSkies();
    std::optional<std::reference_wrapper<SkyEntry>> addSkyFile(std::filesystem::path path);
    void updateSkyMenu();

    void setSkyDisabled();
    void setSkyTexture(std::string_view nameOrFile);
    void setSkyEphemeris();
    void setSkyAuto();
    void removeExistingSky();

    static SkySphere *s_instance;
    std::vector<SkyEntry> m_skies;

    osg::ref_ptr<osg::MatrixTransform> skyRootNode;
    osg::ref_ptr<osg::Geode> texturedSphere;

#ifdef HAVE_EPHEMERIS
    std::unique_ptr<EphemeralSky> m_ephemeralSky;
#endif

    opencover::coVRShader *shader = nullptr;
    opencover::coVRUniform *topUniform = nullptr;
    opencover::coVRUniform *bottomUniform = nullptr;
    opencover::coVRUniform *floorColorUniform = nullptr;

    opencover::ui::Menu *geoDataMenu;
    opencover::ui::Group *skyGroup;
    opencover::ui::SelectionList *skyList;
    opencover::ui::Slider *skyNorthSlider = nullptr;

    float northAngle;
    std::string skyPath;
};
#endif
