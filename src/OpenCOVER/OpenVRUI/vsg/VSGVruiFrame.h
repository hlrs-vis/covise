/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_FRAME_H
#define OSG_VRUI_FRAME_H

#include <util/coTypes.h>

#include <OpenVRUI/vsg/VSGVruiUIContainer.h>

#include <vsg/state/Sampler.h>

#include <string>

#ifdef _WIN32
typedef unsigned short ushort;
#endif

namespace vrui
{

class coFrame;

/** This class provides a flat textured frame arround objects.
  A frame should contain only one child, use another container to layout
  multiple children inside the frame.
  A frame can be configured to fit tight around its child or
  to maximize its size to always fit into its parent container
*/

class VSGVRUIEXPORT VSGVruiFrame : public VSGVruiUIContainer
{

public:
    VSGVruiFrame(coFrame *frame, const std::string &textureName = "UI/Frame");
    virtual ~VSGVruiFrame();

    void createGeometry();

protected:
    virtual void resizeGeometry();
    void realign();

    coFrame *frame;

private:
    //shared coord and color list
    void createSharedLists();



    vsg::ref_ptr<vsg::Sampler> texture;

};
}
#endif
