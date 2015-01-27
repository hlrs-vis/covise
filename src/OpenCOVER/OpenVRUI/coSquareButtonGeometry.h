/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_SQUARE_BUTTON_GEOMETRY_H
#define CO_SQUARE_BUTTON_GEOMETRY_H

#include <OpenVRUI/coButtonGeometry.h>

#include <string>

/**
    this class implements a beveled button geometry
*/
namespace vrui
{

class OPENVRUIEXPORT coSquareButtonGeometry : public coButtonGeometry
{
public:
    coSquareButtonGeometry(const std::string &name);
    virtual ~coSquareButtonGeometry(); ///< Destructor.

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;
};
}
#endif
