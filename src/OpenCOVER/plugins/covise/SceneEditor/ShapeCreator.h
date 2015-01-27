/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SHAPE_CREATOR
#define SHAPE_CREATOR

#include "SceneObject.h"
#include "SceneObjectCreator.h"
#include "Shape.h"

#include <QDomElement>

class ShapeCreator : public SceneObjectCreator
{
public:
    ShapeCreator();
    virtual ~ShapeCreator();

    virtual SceneObject *createFromXML(QDomElement *root);

protected:
    virtual bool buildFromXML(SceneObject *so, QDomElement *root);

private:
    bool buildGeometryFromXML(Shape *shape, QDomElement *root);
};

#endif
