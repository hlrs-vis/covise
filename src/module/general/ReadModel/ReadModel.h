/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_MODEL_H
#define _READ_MODEL_H

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadModel
//
// This module reads polygon meshes in all formats supported by assimp
//

#include <api/coModule.h>
#include <assimp/scene.h>

using namespace covise;

class ReadModel : public coModule
{
private:
	virtual int compute(const char *port);

	coDistributedObject *load(const char *filename);
	void setPoints(const aiMesh *mesh, float *x_coord, float *y_coord, float *z_coord);
	void setNormals(const aiMesh *mesh);

    coFileBrowserParam *p_filename;

    coBooleanParam *p_triangulate;
    coBooleanParam *p_joinVertices;
	coBooleanParam *p_ignoreErrors;

	coOutputPort *p_polyOut;
	coOutputPort *p_pointOut;
	coOutputPort *p_normalOut;

public:
    ReadModel(int argc, char *argv[]);
    virtual ~ReadModel();
};
#endif
