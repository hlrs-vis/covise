/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_FRAME_H
#define CO_FRAME_H

#include <OpenVRUI/coUIContainer.h>

#include <string>

#ifdef _WIN32
typedef unsigned short ushort;
#endif

namespace vrui
{

/** This class provides a flat textured frame arround objects.
  A frame should contain only one child, use another container to layout
  multiple chlidren inside the frame.
  A frame can be configured to fit tight around its child or
  to maximize its size to always fit into its parent container
*/
class OPENVRUIEXPORT coFrame : public virtual coUIContainer
{

public:
    coFrame(const std::string &textureName = "UI/Frame");
    virtual ~coFrame();

    virtual void addElement(coUIElement *);
    virtual void removeElement(coUIElement *);

    virtual float getWidth() const
    {
        return myWidth;
    }
    virtual float getHeight() const
    {
        return myHeight;
    }
    virtual float getDepth() const
    {
        return myDepth;
    }
    virtual float getXpos() const
    {
        return myX;
    }
    virtual float getYpos() const
    {
        return myY;
    }
    virtual float getZpos() const
    {
        return myZ;
    }
    virtual float getBorderWidth() const
    {
        return bw;
    }
    virtual float getBorderHeight() const
    {
        return bw;
    }

    virtual const std::string &getTextureName() const
    {
        return textureName;
    }

    virtual void setPos(float x, float y, float z = 0);
    virtual void setWidth(float);
    virtual void setHeight(float);
    virtual void setDepth(float);
    virtual void setBorderWidth(float);
    virtual void setBorderHeight(float);
    virtual void setBorderDepth(float);
    virtual void setSize(float s);
    virtual void setSize(float nw, float nh, float nd);
    virtual void setBorder(float nw, float nh, float nd);
    virtual void resizeToParent(float, float, float, bool shrink = true);
    virtual void shrinkToMin();
    virtual void fitToParent();
    virtual void fitToChild();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    std::string textureName; ///< name of the texture file

    //shared coord and color list
    void realign();
    float myX; ///< Frame position X
    float myY; ///< Frame position Y
    float myZ; ///< Frame position Z
    float myWidth; ///< Frame width
    float myHeight; ///< Frame height
    float myDepth; ///< Frame depth
    float bw; ///< Border width
    float bh; ///< Border height
    float bd; ///< Border depth
    bool fitParent; ///< wether to fit the frame to its parent or child
};
}
#endif
