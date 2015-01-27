/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef LIGHT_CREATOR
#define LIGHT_CREATOR

#include "SceneObject.h"
#include "SceneObjectCreator.h"
#include "Light.h"

#include <QDomElement>

class LightCreator : public SceneObjectCreator
{
public:
    LightCreator();
    virtual ~LightCreator();

    virtual SceneObject *createFromXML(QDomElement *root);

protected:
    virtual bool buildFromXML(SceneObject *so, QDomElement *root);

private:
    bool buildGeometryFromXML(Light *light, QDomElement *root);
};

#endif
