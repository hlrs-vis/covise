/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GROUND_CREATOR
#define GROUND_CREATOR

#include "SceneObject.h"
#include "SceneObjectCreator.h"
#include "Ground.h"

#include <QDomElement>

class GroundCreator : public SceneObjectCreator
{
public:
    GroundCreator();
    virtual ~GroundCreator();

    virtual SceneObject *createFromXML(QDomElement *root);

protected:
    virtual bool buildFromXML(SceneObject *so, QDomElement *root);

private:
    bool buildGeometryFromXML(Ground *ground, QDomElement *root);
};

#endif
