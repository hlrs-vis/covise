/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_MODEL_H
#define _READ_MODEL_H

 // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 // MODULE ReadModel
 //
 // Read polygon meshes in all formats supported by Assimp library
 // assimp.sourceforge.net
 // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <api/coModule.h>

struct aiMesh;

class ReadModel : public covise::coModule
{
private:
	struct allGeometry {
		std::vector<covise::coDistributedObject *> allMeshes;
		std::vector<covise::coDistributedObject *> allNormals;
	};

	virtual int compute(const char *port);

	allGeometry load(std::string &filename, std::string &polyName, std::string &normalName);
	void setPoints(const aiMesh *mesh, float *x_coord, float *y_coord, float *z_coord);

	covise::coFileBrowserParam *p_filename;

	covise::coBooleanParam *p_triangulate;
	covise::coBooleanParam *p_joinVertices;
	covise::coBooleanParam *p_ignoreErrors;

	covise::coOutputPort *p_polyOut;
	covise::coOutputPort *p_normalOut;

public:
    ReadModel(int argc, char *argv[]);
    virtual ~ReadModel();
};
#endif
