/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <OpenVRUI/sginterface/vruiPanelGeometryProvider.h>

#include <vsg/state/Sampler.h>

namespace vrui
{

class VSGVRUIEXPORT VSGVruiPanelGeometry : public virtual vruiPanelGeometryProvider
{
public:
    VSGVruiPanelGeometry(coPanelGeometry *geometry);
    virtual ~VSGVruiPanelGeometry();
    virtual void attachGeode(vruiTransformNode *node);
    virtual float getWidth() const;
    virtual float getHeight() const;
    virtual float getDepth() const;

    virtual void createSharedLists();

private:
    coPanelGeometry *geometry;

    static float A;
    static float B;
    static float C;


    vsg::ref_ptr<vsg::Sampler> texture;
};
}

