/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_TEXTURERECT_BACKGROUND_H
#define CO_TEXTURERECT_BACKGROUND_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coBackground.h>

#ifdef _WIN32
typedef unsigned char uchar; ///< abbreviation for unsigned char
typedef unsigned short ushort; ///< abbreviation for unsigned short
typedef unsigned int uint; ///< abbreviation for unsigned int
typedef unsigned long ulong; ///< abbreviation for unsigned long
typedef signed char schar; ///< abbreviation for signed char
typedef signed short sshort; ///< abbreviation for signed short
typedef signed int sint; ///< abbreviation for signed int
typedef signed long slong; ///< abbreviation for signed long
#endif
#undef ACTION_BUTTON
#undef DRIVE_BUTTON
#undef XFORM_BUTTON

#include <string>

namespace vrui
{

class coTextureRectBackground;
class vruiHit;

/** Overwrite the routines of this class to get callback functions for
  when the pointer intersected the texture and the button was pressed.
 */
class OPENVRUIEXPORT coTextureRectBackgroundActor
{
public:
    virtual ~coTextureRectBackgroundActor()
    {
    }
    virtual void texturePointerClicked(coTextureRectBackground *, float, float);
    virtual void texturePointerReleased(coTextureRectBackground *, float, float);
    virtual void texturePointerDragged(coTextureRectBackground *, float, float);
    virtual void texturePointerMoved(coTextureRectBackground *, float, float);
    virtual void texturePointerLeft(coTextureRectBackground *);
};

/** This class provides background for GUI elements.
    The texture of this background changes according to the elements state
    (normal/highlighted/disabled)
    A background should contain only one child, use another container to layout
    multiple chlidren inside the frame.
 */
class OPENVRUIEXPORT coTextureRectBackground
    : public coBackground,
      public coAction
{
public:
    coTextureRectBackground(const std::string &normalTexture, coTextureRectBackgroundActor *actor = 0);
    coTextureRectBackground(uint *normalImage, int comp, int ns, int nt, int nr,
                            coTextureRectBackgroundActor *actor = 0);
    virtual ~coTextureRectBackground();

    class TextureSet
    {

    public:
        TextureSet(uint *nt, int comp, int s, int t, int r)
            : start(0.0f, 0.0f)
            , end(1.0f, 1.0f)
        {
            normalTextureImage = nt;
            this->comp = comp;
            this->s = s;
            this->t = t;
            this->r = r;
        }

        uint *normalTextureImage;
        int comp;
        int s;
        int t;
        int r;

        struct TexCoord
        {
            TexCoord(float xv, float yv)
                : x(xv)
                , y(yv)
            {
            }
            float x;
            float y;
        };

        TexCoord start;
        TexCoord end;
    };

    virtual int hit(vruiHit *hit);
    virtual void miss();

    /// set this widget to enabled
    virtual void setEnabled(bool en);

    /// set this widget to highlighted
    virtual void setHighlighted(bool hl);

    /// set texture repeat (deprecated!)
    void setRepeat(bool repeat);

    /// get texture repeat state (deprecated!)
    bool getRepeat() const;

    /** after setting the image data, set this to true.
       * after updating the texture, set this to false
       */
    void setUpdated(bool update)
    {
        this->updated = update;
    }

    /// returns whether
    bool getUpdated()
    {
        return updated;
    }

    /// set the texture dimensions (need to be pixel-exact)
    void setTexSize(float, float);

    /// returns the width of the texture
    float getTexXSize() const
    {
        return texXSize;
    }

    /// returns the height of the texture
    float getTexYSize() const
    {
        return texYSize;
    }

    /**
       *  set the image data
       *    Params:
       *    normalImage = texture data
       *    comp = components/Bytes per pixel
       *    ns = texture width
       *    nt = texture height
       *    nr = texture depth (unused, since this is a 2D-rect-texture!)
       */
    void setImage(uint *normalImage, int comp, int ns, int nt, int nr);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    inline TextureSet *getCurrentTextures()
    {
        return currentTextures;
    }

    inline const std::string &getNormalTexName() const
    {
        return normalTexName;
    }

protected:
    coTextureRectBackgroundActor *myActor; ///< action listener, triggered on pointer intersections

private:
    std::string normalTexName; ///< Name of the normal texture
    TextureSet *currentTextures;
    int comp; ///< color component (3: RGB, 4: RGBA)
    int s, t, r; ///< texture coordinates

    float texXSize; ///< texture width
    float texYSize; ///< texture height

    bool repeat; ///< repeat texture (deprecated!)

    bool updated; ///< was the texture data modified?
};
}
#endif
