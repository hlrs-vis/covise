/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_BUTTON_GEOMETRY_H
#define CO_BUTTON_GEOMETRY_H
#ifdef _WIN32
typedef unsigned short ushort;
#endif
#include <util/coTypes.h>
#include <string>

namespace vrui
{

class vruiButtonProvider;
class vruiTransformNode;

/**
    this class describes an abstract bas for Button geometries
    new button shapes can be created by deriving from this class
*/
class OPENVRUIEXPORT coButtonGeometry
{
public:
    coButtonGeometry(const std::string &texture);
    virtual ~coButtonGeometry();

    enum ActiveGeometry
    {
        NORMAL = 0,
        PRESSED,
        HIGHLIGHT,
        HIGHLIGHT_PRESSED,
        DISABLED
    };

    virtual float getWidth() const; ///< get width of this geometry
    virtual float getHeight() const; ///< get height of this geometry
    ///< Switch the shown geometry
    virtual void switchGeometry(ActiveGeometry active);

    virtual const std::string &getTextureName() const
    {
        return texture;
    }

    virtual void createGeometry();
    virtual void resizeGeometry();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    virtual vruiTransformNode *getDCS();

    virtual vruiButtonProvider *getButtonProvider() const
    {
        return buttonGeometryProvider;
    }

protected:
    std::string texture; ///< name of the texture file
    mutable vruiButtonProvider *buttonGeometryProvider;
};
}
#endif
