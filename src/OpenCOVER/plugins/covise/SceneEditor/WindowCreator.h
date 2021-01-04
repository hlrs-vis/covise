/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WINDOW_CREATOR
#define WINDOW_CREATOR

#include "SceneObject.h"
#include "SceneObjectCreator.h"
#include "Window.h"

#include <QDomElement>

class WindowCreator : public SceneObjectCreator
{
public:
    WindowCreator();
    virtual ~WindowCreator();

    virtual SceneObject *createFromXML(QDomElement *root);

protected:
    virtual bool buildFromXML(SceneObject *so, QDomElement *root);

private:
    bool buildGeometryFromXML(Window *win, QDomElement *root);
};

#endif
