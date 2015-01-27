/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_LABEL_H
#define SG_VRUI_LABEL_H

#include <OpenVRUI/opensg/SGVruiUIElement.h>

#include <OpenSG/OSGChunkMaterial.h>
#include <OpenSG/OSGText.h>
#include <OpenSG/OSGTextureChunk.h>
#include <OpenSG/OSGColor.h>

class coLabel;

/**
 * Label element.
 * A label consists of a text string and a background texture.
 */
class SGVRUIEXPORT SGVruiLabel : public SGVruiUIElement
{
public:
    SGVruiLabel(coLabel *label);
    virtual ~SGVruiLabel();

    virtual void createGeometry();

    virtual void resizeGeometry();

    virtual void update();

    virtual float getWidth() const;
    virtual float getHeight() const;
    virtual float getDepth() const;

    virtual void setHighlighted(bool highlighted);

protected:
    coLabel *label;

    osg::Color4ub textColor; ///< components of text color (RGBA)
    osg::Color4ub textColorHL; ///< components of text color when highlighted (RGBA)

    osg::Color4ub textForeground;
    osg::Color4ub textBackground;
    ///< Geostate of background texture
    osg::ChunkMaterialPtr material;
    osg::TextureChunkPtr textureChunk;
    osg::ImagePtr image;
    osg::Text labelText; ///< label text string in OpenSG format
    osg::GeoPositions3fPtr coord;

    void makeText();
};
#endif
