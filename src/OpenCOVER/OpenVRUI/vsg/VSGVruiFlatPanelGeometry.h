/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_FLAT_PANEL_GEOMETRY_H
#define OSG_VRUI_FLAT_PANEL_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiPanelGeometryProvider.h>

#include <OpenVRUI/coUIElement.h>

#include <vsg/maths/vec3.h>
#include <vsg/core/ref_ptr.h>
#include <vsg/nodes/Node.h>

namespace vrui
{

class coFlatPanelGeometry;

class VSGVRUIEXPORT VSGVruiFlatPanelGeometry : public vruiPanelGeometryProvider
{
public:
    VSGVruiFlatPanelGeometry(coFlatPanelGeometry *geometry);
    virtual ~VSGVruiFlatPanelGeometry();

    vsg::ref_ptr<vsg::Node> createQuad(const vsg::vec3& origin, const vsg::vec3& horizontal, const vsg::vec3& vertical);

    virtual void attachGeode(vruiTransformNode *node);
    virtual float getWidth() const;
    virtual float getHeight() const;
    virtual float getDepth() const;

protected:
    coFlatPanelGeometry* panelGeometry;

private:
    //shared coord and color list
    static float A;
    static float B;


};
}
#endif
