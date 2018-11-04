/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SURFACE_PLUGIN_H_
#define _SURFACE_PLUGIN_H_

#include <util/common.h>

#include "ModuleFeedbackPlugin.h"
#include "ModuleFeedbackManager.h"
#include "coVRPlugin.h"

class coVRPlugin;

namespace osg
{
class Group;
class Node;
class MatrixTransform;
class Switch;
};

#include <osg/Matrix>
#include <osg/Vec3>

#include <OpenVRUI/coMenu.h>
class RenderObject;
class PLUGIN_UTILEXPORT SurfacePlugin : public coVRPlugin, public ModuleFeedbackPlugin, public coMenuFocusListener
{
public:
    SurfacePlugin(const char *iconname, string sectionName);
    virtual ~SurfacePlugin();
    void VerwaltePointer(bool show);
    void GetPoint(osg::Vec3 &vect) const;
    void GetNormal(osg::Vec3 &vect) const;
    void SubstitutePointer(const char *iconname);

    void AddObject(const char *objName, RenderObject *colorOrText);
    void AddNode(const char *objName, osg::Node *node);
    void AddContainer(const char *contName, const char *objName);
    void RemoveObject(const char *objName);
    void RemoveNode(osg::Node *);
    RenderObject *GetColor(const char *objName);
    osg::Node *GetNode(const char *objName);
    void ToggleVisibility(string objName);
    virtual void SuppressOther3DTex(ModuleFeedbackManager *);
    virtual void DeleteInteractor(coInteractor *i);
    void GetPointerParams(float &angle, float &scaleFactor, float &displacement);
    void AddFixedIcon();
    void RemoveFixedIcon();
    virtual void preFrame()
    {
        ModuleFeedbackPlugin::preFrame();
    };
    // Change Pointer state: true=active, false=non-active
    void setActive(bool isActive);

protected:
    virtual void focusEvent(bool focus, coMenu *menu);
    coVRPlugin *_module;

private:
    map<string, osg::Node *> _findNode; // object, node
    map<osg::Node *, string> _findNodeSym; // object, node
    map<osg::Node *, osg::Group *> _parentNode; // object, parent object

    map<string, string> _findObject; // container, object
    map<string, string> _findObjectSym; // object, container

    map<string, RenderObject *> _findColor; // object, colorObject

    // pointer
    osg::Switch *readPointer(const char *basename);
    bool _show;
    osg::Group *_rootNode;
    osg::MatrixTransform *_scale;
    osg::MatrixTransform *_fixed_scale;
    osg::Matrix *_fixed_matrix;
    osg::Matrix *_matrix;
    osg::Switch *_pointer; // pointer icon
    float _cos_angle;
    float _sin_angle;
    float _angle; // degrees
    float _displacement;
    float _scaleFactor;
    bool _inFocus;
    string _iconName;
};
#endif
