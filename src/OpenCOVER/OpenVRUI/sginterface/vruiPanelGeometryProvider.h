/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_PANEL_GEOMETRY_H
#define VRUI_PANEL_GEOMETRY_H

#include <OpenVRUI/coPanelGeometry.h>

namespace vrui
{

class vruiTransformNode;
class coPanelGeometry;

class OPENVRUIEXPORT vruiPanelGeometryProvider
{
public:
    vruiPanelGeometryProvider(coPanelGeometry *geometry)
    {
        this->geometry = geometry;
    }
    virtual ~vruiPanelGeometryProvider();
    virtual void attachGeode(vruiTransformNode *node) = 0;
    virtual float getWidth() const = 0;
    virtual float getHeight() const = 0;
    virtual float getDepth() const = 0;

private:
    coPanelGeometry *geometry;
};
}
#endif
