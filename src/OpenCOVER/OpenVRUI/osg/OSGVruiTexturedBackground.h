/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-

#ifndef OSG_VRUI_TEXTURED_BACKGROUND_H
#define OSG_VRUI_TEXTURED_BACKGROUND_H

#include <OpenVRUI/osg/OSGVruiUIContainer.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Texture2D>

#include <OpenVRUI/coTexturedBackground.h>

namespace vrui
{

/** This class provides background for GUI elements.
  The color of this background changes according to the elements state
  (normal/highlighted/disabled)
  A background should contain only one child, use another container to layout
  multiple chlidren inside the frame.
*/
class OSGVRUIEXPORT OSGVruiTexturedBackground
    : public OSGVruiUIContainer
{
public:
    OSGVruiTexturedBackground(coTexturedBackground *background);
    virtual ~OSGVruiTexturedBackground();

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
    osg::ref_ptr<osg::Vec3Array> coord; ///< Coordinates of background geometry
    osg::ref_ptr<osg::Vec2Array> texcoord; ///< Texture coordinates of background geometry

    static osg::ref_ptr<osg::Vec3Array> normal; ///< Normal of background geometry

    osg::ref_ptr<osg::StateSet> state; ///< Normal geometry color
    osg::ref_ptr<osg::StateSet> highlightState; ///< Highlighted geometry color
    osg::ref_ptr<osg::StateSet> disabledState; ///< Disabled geometry color

    osg::ref_ptr<osg::Geode> geometryNode; ///< Geometry node
    osg::ref_ptr<osg::Geometry> geometry; ///< Geometry object

    osg::ref_ptr<osg::Texture2D> texNormal; ///< Texture object for normal texture
    osg::ref_ptr<osg::Texture2D> texHighlighted; ///< Texture object for highlighted texture
    osg::ref_ptr<osg::Texture2D> texDisabled; ///< Texture object for disabled texture

    void createGeode();

    bool repeat; ///< wether to repeat the texture or not
    float texXSize; ///< horizontal size of the texture
    float texYSize; ///< vertical size of the texture
};
}
#endif
