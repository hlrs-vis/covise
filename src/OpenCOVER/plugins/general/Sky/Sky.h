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
public:
    SkyEntry(const std::string &name, const std::string &displayName, const std::string &fileName, double longitude, double latitude, double altitude, double trueNorth)
        : name(name), displayName(displayName), fileName(fileName), longitude(longitude), latitude(latitude), altitude(altitude), trueNorth(trueNorth) {}
    std::string name;
    std::string displayName;
    std::string fileName;

    // All in degrees
    double longitude = 0.0;
    double latitude = 0.0;
    double trueNorth = 0.0;

    // in meters
    double altitude = 0.0;

    osg::ref_ptr<osg::Texture2D> texture;
};

class Sky : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    Sky();
    bool init();
    ~Sky();

    static Sky *instance();

    virtual bool update();
    virtual void message(int toWhom, int type, int length, const void *data);

    void setSky(std::string_view nameOrFile);

    void setTop(float t);
    void setBottom(float b);
    void setFloorColor(osg::Vec4 fc);

    void setTrueNorth(float trueNorth);
    void setSkyEphemeris(bool enable = true);
    void setHour(int hourOfDay);

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
    void setSkyEntry(SkyEntry &sky);
    void setSkyAuto();
    void updateAutoSky();
    SkyEntry *findClosestSky(const osg::Vec3 &globalPosition);
    void removeExistingSky();

    static Sky *s_instance;
    std::vector<SkyEntry> m_skies;

    SkyEntry *m_currentAutoSky = nullptr;

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
    int skyListNameStart = 0;
    opencover::ui::Button *autoSkyButton;
    opencover::ui::Button *jumpToSkyButton;
    opencover::ui::Slider *skyNorthSlider = nullptr;

    float northAngle;
    std::string skyPath;
};
#endif
