/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_LABEL_H
#define CO_LABEL_H

#include <OpenVRUI/coUIElement.h>

#include <string>

namespace vrui
{

class coLabelGeometry;
class vruiUIElementProvider;

/**
 * Label element.
 * A label consists of a text string and a background texture.
 */
class OPENVRUIEXPORT coLabel : public coUIElement
{
public:
    enum DirectionsType /// valid orientations for text string
    {
        HORIZONTAL = 0,
        VERTICAL
    };

    enum Justify
    {
        LEFT = 0,
        CENTER,
        RIGHT
    };

    coLabel(const std::string &labelText = "");
    virtual ~coLabel();
    void resize();
    virtual void setPos(float x, float y, float z = 0.0f);
    virtual void setFontSize(float size);
    virtual void setString(const std::string &text);
    virtual void setSize(float size);
    virtual void setSize(float xs, float ys, float zs);
    virtual float getWidth() const // see superclass for comment
    {
        return myWidth;
    }
    virtual float getHeight() const // see superclass for comment
    {
        return myHeight;
    }
    virtual float getDepth() const // see superclass for comment
    {
        return myDepth;
    }
    virtual float getXpos() const // see superclass for comment
    {
        return myX;
    }
    virtual float getYpos() const // see superclass for comment
    {
        return myY;
    }
    virtual float getZpos() const // see superclass for comment
    {
        return myZ;
    }
    virtual float getFontSize() const
    {
        return fontSize;
    }
    virtual Justify getJustify() const
    {
        return justify;
    }
    virtual DirectionsType getDirection() const
    {
        return direction;
    }
    virtual const char *getString() const
    {
        return labelString.c_str();
    }
    virtual void setHighlighted(bool highlighted);

    virtual void resizeGeometry();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    float myX; ///< x position of label in object space [mm]
    float myY; ///< y position of label in object space [mm]
    float myZ; ///< z position of label in object space [mm]
    float myHeight; ///< label size in z direction [mm]
    float myWidth; ///< label size in x direction [mm]
    float myDepth; ///< label size in y direction [mm]

    //   float textColor[4];       ///< components of text color (RGBA)
    //   float textColorHL[4];     ///< components of text color when highlighted (RGBA)
    //   pfGeoState *backgroundTextureState; ///< Performer geostate of background texture
    //   pfText *labelText;        ///< label text string in Performer format

    std::string labelString; ///< text string which is displayed on the label

    Justify justify; ///< string justification, using Performer format. Default is left alignment.
    float fontSize; ///< label text size in mm
    DirectionsType direction; ///< direction into which the text string is written on the label

    //void makeText();
};
}
#endif
