/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GENERAL_GEOMETRY_INTERACTION_H
#define _GENERAL_GEOMETRY_INTERACTION_H

#include <PluginUtil/ModuleFeedbackManager.h>

#include <osg/Geode>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>

namespace vrui
{
class coRowMenu;
class coPotiMenuItem;
}

class GeneralGeometryInteraction : public opencover::ModuleFeedbackManager
{
private:
    bool paramsOk();
    bool newObject_;

    virtual void menuEvent(vrui::coMenuItem *menuItem);
    vrui::coPotiMenuItem *transparencyPoti_;
    osg::ref_ptr<osg::Drawable> geoset_;
    bool firsttime_;
    float transparency_;
    void setTransparencyWithoutShader(float transparency);

public:
    // constructor
    GeneralGeometryInteraction(const opencover::RenderObject *container, opencover::coInteractor *inter, const char *pluginName);
    GeneralGeometryInteraction(const opencover::RenderObject *container, const opencover::RenderObject *geomObject, const char *pluginName);

    // destructor
    virtual ~GeneralGeometryInteraction();

    // update covise stuff and menus
    virtual void update(const opencover::RenderObject *container, const opencover::RenderObject *geomObject);

    // direct interaction
    virtual void preFrame();

    // mesagge from gui
    void setColor(int *color);
    //void setShader(const char* shaderName, const char* paraFloat, const char* paraVec2, const char* paraVec3, const char* paraVec4, const char* paraInt, const char* paraBool, const char* paraMat2, const char* paraMat3, const char* paraMat4);
    void setTransparency(float transparency);
    void setMaterial(int *ambient, int *diffuse, int *specular, float shininess, float transpareny);
};

#endif
