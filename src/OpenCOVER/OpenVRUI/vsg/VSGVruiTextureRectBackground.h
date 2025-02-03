/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-

#ifndef OSG_VRUI_TEXTURERECT_BACKGROUND_H
#define OSG_VRUI_TEXTURERECT_BACKGROUND_H

#include <OpenVRUI/vsg/VSGVruiUIContainer.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/Switch.h>
#include <vsg/nodes/StateGroup.h>

#include <OpenVRUI/coTextureRectBackground.h>

namespace vrui
{

/** This class provides background for GUI elements.
  This class is specifically made for rectangular (non-power of 2)
  textures, which may me procedurraly updated. It is hardwired to
  vsg::TextureRectangle, no abstraction!
 */
class VSGVRUIEXPORT VSGVruiTextureRectBackground
    : public VSGVruiUIContainer
{
public:
    VSGVruiTextureRectBackground(coTextureRectBackground *background);
    virtual ~VSGVruiTextureRectBackground();

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual void update();

    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);

protected:
    void createSharedLists();

    void rescaleTexture();
    void createTexturesFromFiles();
    void createTexturesFromArrays(uint *normalImage, int comp, int ns, int nt, int nr);

    coTextureRectBackground *background;
    coTextureRectBackground::TextureSet *tex;

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


    void createGeode();

    bool repeat; ///< wether to repeat the texture or not
    float texXSize; ///< horizontal size of the texture
    float texYSize; ///< vertical size of the texture
};
}
#endif
