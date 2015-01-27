/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Blockmesh_H
#define Blockmesh_H

#include <string>
#include <iostream>
#include <fstream>
#include "FileReference.h"
class CoolEmAll;

class Blockmesh
{
public:
    Blockmesh(CoolEmAll *cc);
    ~Blockmesh();
    void writeHeader();
    void writeBound(FileReference *ProductRevisionViewReference);
    void writeFooter();

private:
    std::ofstream file1;
    CoolEmAll *cool;
};

#endif
