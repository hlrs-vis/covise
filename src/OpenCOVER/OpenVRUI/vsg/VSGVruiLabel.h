/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_LABEL_H
#define OSG_VRUI_LABEL_H

#include <OpenVRUI/vsg/VSGVruiUIElement.h>

#include <vsg/text/Text.h>

namespace vrui
{

class coLabel;

/**
 * Label element.
 * A label consists of a text string and a background texture.
 */
class VSGVRUIEXPORT VSGVruiLabel : public VSGVruiUIElement
{
public:
    VSGVruiLabel(coLabel *label);
    virtual ~VSGVruiLabel();

    virtual void createGeometry();

    virtual void resizeGeometry();

    virtual void update();

    virtual float getWidth() const;
    virtual float getHeight() const;
    virtual float getDepth() const;

    virtual void setHighlighted(bool highlighted);

protected:
    coLabel *label;

    vsg::vec4 textColor; ///< components of text color (RGBA)
    vsg::vec4 textColorHL; ///< components of text color when highlighted (RGBA)

    ///< Geostate of background texture
    vsg::ref_ptr<vsg::Text> labelText; ///< label text string in OSG format

    void makeText();
};
}
#endif
