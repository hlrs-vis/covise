/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_PANEL_GEOMETRY_H
#define CO_PANEL_GEOMETRY_H

#include <util/coTypes.h>

#include <string>

namespace vrui
{

class vruiTransformNode;
class vruiPanelGeometryProvider;

class OPENVRUIEXPORT coPanelGeometry
{
public:
    coPanelGeometry(const std::string &textureName);
    virtual ~coPanelGeometry();
    virtual void attachGeode(vruiTransformNode *node);
    virtual float getWidth();
    virtual float getHeight();
    virtual float getDepth();

    const char *getTextureName() const;

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    coPanelGeometry();

private:
    std::string textureName;
    mutable vruiPanelGeometryProvider *provider;
};
}
#endif
