/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef APPEARANCE_BEHAVIOR_H
#define APPEARANCE_BEHAVIOR_H

#include "Behavior.h"
#include "MyShader.h"
#include "../Events/PreFrameEvent.h"

#include <osg/Material>

#include <QRegExp>

struct Appearance
{
    MyShader *shader;
    osg::ref_ptr<osg::Material> material;
    osg::Vec4 color;
};

struct Scope
{
    std::string name;
    QString regexp;
    Appearance appearance;
    std::string geoNameSpace;
};

class AppearanceBehavior : public Behavior
{
public:
    AppearanceBehavior();
    virtual ~AppearanceBehavior();

    virtual int attach(SceneObject *);
    virtual int detach();

    virtual EventErrors::Type receiveEvent(Event *e);

    virtual bool buildFromXML(QDomElement *behaviorElement);

private:
    void setAppearance(QRegExp regexp, osg::Node *node, Appearance appearance, bool remove);
    void setAppearance(osg::Node *, Appearance appearance, bool remove);

    osg::ref_ptr<osg::Material> createMaterial(osg::Vec4 color);

    MyShader *buildShaderFromXML(QDomElement *shaderElement);

    std::vector<Scope> _scopeList;
};

#endif
