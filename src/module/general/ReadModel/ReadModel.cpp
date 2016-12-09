/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadModel
//
// This module reads polygon meshes in all formats supported by assimp
//

#include "ReadModel.h"
#include <do/coDoData.h>
#include <alg/coFeatureLines.h>

#ifndef _WIN32
#include <inttypes.h>
#endif


// Module set-up in Constructor
ReadModel::ReadModel(int argc, char *argv[])
    : coModule(argc, argv, "Read STL")
{
    // file browser parameter
    p_filename = addFileBrowserParam("file_path", "Data file path");
    p_filename->setValue("data/nofile.stl", "*.stl;*.STL/*");


    // activate automatic FixUsg ?
    p_triangulate = addBooleanParam("triangulate", "triangulate polygons");
    p_triangulate->setValue(1);

    // show feature lines and smooth surfaces (when p_removeDoubleVert is true)
    p_joinVertices = addBooleanParam("joinVertices", "join identical vertices");
    p_joinVertices->setValue(1);

    // Output ports
    p_polyOut = addOutputPort("mesh", "Polygons", "Polygons");
    p_normOut = addOutputPort("Normals", "Vec3", "velocity data");
    p_linesOut = addOutputPort("Feature_lines", "Lines", "Feature lines");

    // to be added later for coloured binary files
    p_colorOut = addOutputPort("colors", "RGBA", "color data");

}

ReadModel::~ReadModel()
{
}

// param callback read header again after all changes
void
ReadModel::param(const char *paraName, bool inMapLoading)
{
    if (inMapLoading)
        return;

   // if (strcmp(paraName, p_filename->getName()) == 0)
    //    readHeader();
}

int ReadModel::compute(const char *)
{
    int result = SUCCESS;
    return result;
}


MODULE_MAIN(IO, ReadModel)
