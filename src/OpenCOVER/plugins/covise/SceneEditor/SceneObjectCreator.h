/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCENE_OBJECT_CREATOR
#define SCENE_OBJECT_CREATOR

#include "SceneObject.h"

#include <QDomElement>

class SceneObjectCreator
{
public:
    SceneObjectCreator();
    virtual ~SceneObjectCreator();

    virtual SceneObject *createFromXML(QDomElement *root);

protected:
    virtual bool buildFromXML(SceneObject *so, QDomElement *root);

private:
    bool _createBehaviorsFromXML(SceneObject *so, QDomElement *root);
    bool _createBehaviorFromXML(SceneObject *so, QDomElement *behaviorElement);
};

#endif
