/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_CHEMICAL_ELEMENT_H
#define CO_CHEMICAL_ELEMENT_H

#include <util/coExport.h>
#include <string>

/// list of all chemical elements

namespace covise
{

struct ALGEXPORT coChemicalElement
{
    int number;
    std::string name;
    std::string symbol;

    enum
    {
        Number = 118
    };
    static coChemicalElement all[Number];
};
}
#endif
