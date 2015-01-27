/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshFileTransParser.h
 * a mesh file parser for changing (transient) meshes.
 */

//#include "MeshFileTransParser.h"  // a mesh file parser for changing (transient) meshes.

#ifndef __meshfilesetparser_h__
#define __meshfilesetparser_h__

#include "MeshDataTrans.h" // a container for mesh file data.
#include <util/coRestraint.h>

/**
 * a mesh file parser for changing (transient) meshes.
 */
class MeshFileTransParser
{
public:
    MeshFileTransParser(){};
    virtual ~MeshFileTransParser(){};

    virtual void parseMeshFile(std::string filename,
                               covise::coRestraint &sel,
                               bool subdivide,
                               const int &noOfTimeStepToSkip,
                               const int &noOfTimeStepsToParse) = 0;

    virtual MeshDataTrans *getMeshDataTrans(void) = 0;
};

#endif
