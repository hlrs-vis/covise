/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DEFAULT_BUTTON_GEOMETRY_H
#define CO_DEFAULT_BUTTON_GEOMETRY_H

#include <OpenVRUI/coButtonGeometry.h>

#include <string>

namespace vrui
{

class OPENVRUIEXPORT coDefaultButtonGeometry : public coButtonGeometry
{
public:
    coDefaultButtonGeometry(const std::string &name);
    virtual ~coDefaultButtonGeometry();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;
};
}
#endif
