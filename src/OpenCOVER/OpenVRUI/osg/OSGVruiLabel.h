/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_LABEL_H
#define OSG_VRUI_LABEL_H

#include <OpenVRUI/osg/OSGVruiUIElement.h>

#include <osg/StateSet>
#include <osgText/Text>

namespace vrui
{

class coLabel;

/**
 * Label element.
 * A label consists of a text string and a background texture.
 */
class OSGVRUIEXPORT OSGVruiLabel : public OSGVruiUIElement
{
public:
    OSGVruiLabel(coLabel *label);
    virtual ~OSGVruiLabel();

    virtual void createGeometry();

    virtual void resizeGeometry();

    virtual void update();

    virtual float getWidth() const;
    virtual float getHeight() const;
    virtual float getDepth() const;

    virtual void setHighlighted(bool highlighted);

protected:
    coLabel *label;

    osg::Vec4 textColor; ///< components of text color (RGBA)
    osg::Vec4 textColorHL; ///< components of text color when highlighted (RGBA)

    ///< Geostate of background texture
    osg::ref_ptr<osg::StateSet> backgroundTextureState;
    osg::ref_ptr<osgText::Text> labelText; ///< label text string in OSG format

    void makeText();
};
}
#endif
