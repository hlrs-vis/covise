/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include "checkForPoint.h"
#include <stdlib.h>

bool checkForPoint(const std::string &point)
{
    std::istringstream locationInMeshChopper;
    locationInMeshChopper.str(point);
    std::string number_buffer;
    std::vector<std::string> number;
    while (!locationInMeshChopper.eof())
    {
        locationInMeshChopper >> number_buffer;
        number.push_back(number_buffer);
    }
    size_t i = number.size();
    if (i != 3)
    {
        return false;
    }
    for (int i = 0; i < 3; i++)
    {
        char *pEnd = NULL;
        double coordinate = strtod((number.at(i).c_str()), &pEnd);
        if (pEnd == (number.at(i)).c_str())
        {
            return false;
        }
    }
    return true;
}
