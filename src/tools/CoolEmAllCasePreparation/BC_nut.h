/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BC_nut_H
#define BC_nut_H

#include <string>
#include <iostream>
#include <fstream>
#include "FileReference.h"
class CoolEmAll;

class BC_nut
{
public:
    BC_nut(CoolEmAll *cc);
    ~BC_nut();
    void writeHeader();
    void writeSTL(std::string DataBase_Path, FileReference *ProductRevisionViewReference, FileReference *ProductInstanceReference, std::string transformedSTLFileName);
    void writeFooter();

private:
    std::ofstream file1;
    CoolEmAll *cool;
};

#endif
