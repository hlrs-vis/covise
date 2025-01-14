/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-

#ifndef OSG_VRUI_TEXTURED_BACKGROUND_H
#define OSG_VRUI_TEXTURED_BACKGROUND_H

#include <OpenVRUI/vsg/VSGVruiUIContainer.h>

#include <vsg/state/Sampler.h>
#include <vsg/nodes/StateGroup.h>
#include <vsg/nodes/MatrixTransform.h>

#include <OpenVRUI/coTexturedBackground.h>

namespace vrui
{

/** This class provides background for GUI elements.
  The color of this background changes according to the elements state
  (normal/highlighted/disabled)
  A background should contain only one child, use another container to layout
  multiple chlidren inside the frame.
*/
class VSGVRUIEXPORT VSGVruiTexturedBackground
    : public VSGVruiUIContainer
{
public:
    VSGVruiTexturedBackground(coTexturedBackground *background);
    virtual ~VSGVruiTexturedBackground();

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual void update();

    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);

protected:
    void createSharedLists();

    void rescaleTexture();
    void createTexturesFromFiles();
    void createTexturesFromArrays(const uint *normalImage, const uint *highlightImage, const uint *disabledImage,
                                  int comp, int ns, int nt, int nr);

    coTexturedBackground *background;
    coTexturedBackground::TextureSet *tex;

private:
    //shared coord and color list
    vsg::ref_ptr<vsg::vec3Array> coord; ///< Coordinates of background geometry
    vsg::ref_ptr<vsg::vec2Array> texcoord; ///< Texture coordinates of background geometry

    static vsg::ref_ptr<vsg::vec3Array> normal; ///< Normal of background geometry

    vsg::ref_ptr<vsg::StateGroup> state; ///< Normal geometry color
    vsg::ref_ptr<vsg::StateGroup> highlightState; ///< Highlighted geometry color
    vsg::ref_ptr<vsg::StateGroup> disabledState; ///< Disabled geometry color

    vsg::ref_ptr<vsg::Node> geometryNode; ///< Geometry node
    vsg::ref_ptr<vsg::Node> geometry; ///< Geometry object

    vsg::ref_ptr<vsg::Sampler> texNormal; ///< Texture object for normal texture
    vsg::ref_ptr<vsg::Sampler> texHighlighted; ///< Texture object for highlighted texture
    vsg::ref_ptr<vsg::Sampler> texDisabled; ///< Texture object for disabled texture

    void createGeode();

    bool repeat; ///< wether to repeat the texture or not
    float texXSize; ///< horizontal size of the texture
    float texYSize; ///< vertical size of the texture
};
}
#endif
