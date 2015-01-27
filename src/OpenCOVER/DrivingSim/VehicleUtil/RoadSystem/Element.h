/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Element_h
#define Element_h

#include <iostream>
#include <string>
#include <util/coExport.h>

class VEHICLEUTILEXPORT Element
{
public:
    Element(std::string);

    std::string getId();

    virtual ~Element()
    {
    }

protected:
    std::string id;
};

#endif
