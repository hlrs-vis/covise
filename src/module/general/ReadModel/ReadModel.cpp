/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 // MODULE ReadModel
 //
 // Read polygon meshes in all formats supported by Assimp library
 // assimp.sourceforge.net
 // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ReadModel.h"

#include <do/coDoData.h>
#include <alg/coFeatureLines.h>

#include <string>

#include <Importer.hpp>
#include <postprocess.h>

#ifndef _WIN32
#include <inttypes.h>
#endif


ReadModel::ReadModel(int argc, char *argv[])
	: coModule(argc, argv, "Read polygon meshes in all formats supported by Assimp library")
{
	// Parameters
	p_filename = addFileBrowserParam("polygonFile", "Polygon file path");
	p_filename->setValue("~", "*.*/*.3ds/*.ac/*.ase/*.blend/*.bvh/*.cob/*.csm/*.dae/*.dxf/*.hmp/*.ifc/*.irr/*.irrmesh/*.lwo/*.lws/*.lxo/*.material/*.mdl/*.md2/*.md3/*.md5mesh/*.md5anim/*.md5camera/*.mdc/*.mesh/*.ms3d/*.nff/*.obj/*.off/*.ply/*.pk3/*.q3o/*.q3s/*.raw/*.scn/*.skeleton/*.smd/*.stl/*.ter/*.vta/*.x/*.xgl/*.xml/*.zgl");

	p_triangulate = addBooleanParam("triangulate", "Triangulate polygons");
	p_triangulate->setValue(0);

	p_joinVertices = addBooleanParam("joinVertices", "Join identical vertices");
	p_joinVertices->setValue(0);

	p_ignoreErrors = addBooleanParam("ignoreErrors", "Ignore files that are not found");
	p_ignoreErrors->setValue(0);

	// Output ports
	p_polyOut = addOutputPort("GridOut0", "Polygons", "geometry polygons");
	p_pointOut = addOutputPort("GridOut1", "Points", "geometry points");
	p_normalOut = addOutputPort("DataOut0", "Vec3", "polygon normals");
}

ReadModel::~ReadModel() {
}

coDistributedObject *ReadModel::load(const char *filename) {

	coDistributedObject *distObj;
	Assimp::Importer importer;
	unsigned int readFlags = aiProcess_PreTransformVertices | aiProcess_SortByPType | aiProcess_ImproveCacheLocality | aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes;

	if (p_triangulate) {
		readFlags |= aiProcess_Triangulate;
	}

	if (p_joinVertices) {
		readFlags |= aiProcess_JoinIdenticalVertices;
	}

	const aiScene* scene = importer.ReadFile(filename, readFlags);

	if (!scene) {
		if (!p_ignoreErrors) {
			std::stringstream str;
			str << "failed to read " << filename << ": " << importer.GetErrorString() << std::endl;
			std::string s = str.str();
			sendError("%s", s.c_str());
		}
		return nullptr;
	}

	for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
		const aiMesh *mesh = scene->mMeshes[m];

		if (mesh->HasPositions()) {
			float *x_coord, *y_coord, *z_coord;
			auto numPoints = mesh->mNumVertices;
			// mesh contains polygons
			if (mesh->HasFaces()) {
				auto numPolygons = mesh->mNumFaces;
				int numCorners = 0;
				for (unsigned int f = 0; f < numPolygons; ++f) {
					numCorners += mesh->mFaces[f].mNumIndices;
				}
				coDoPolygons *poly(new coDoPolygons(p_polyOut->getNewObjectInfo(), numPoints, numCorners, numPolygons));
				int *cornerList, *polyList;
				poly->getAddresses(&x_coord, &y_coord, &z_coord, &cornerList, &polyList);
				unsigned int idx = 0, vertCount = 0;
				for (unsigned int f = 0; f < numPolygons; ++f) {
					polyList[idx++] = vertCount;
					const auto &face = mesh->mFaces[f];
					for (unsigned int i = 0; i < face.mNumIndices; ++i) {
						cornerList[vertCount++] = face.mIndices[i];
					}
				}
				setPoints(mesh, x_coord, y_coord, z_coord);
				distObj = poly;
			}
			// mesh contains only points, no polygons
			else {
				coDoPoints *points(new coDoPoints(p_pointOut->getNewObjectInfo(), numPoints));
				points->getAddresses(&x_coord, &y_coord, &z_coord);
				setPoints(mesh, x_coord, y_coord, z_coord);
				distObj = points;
			}
			// if mesh contains normals, set them
			setNormals(mesh);
		}
	}
	if (scene->mNumMeshes > 1) {
		sendInfo("file %s contains %d meshes, all but the first have been ignored", filename, scene->mNumMeshes);
	}
	return distObj;
}

void ReadModel::setPoints(const aiMesh *mesh, float *x_coord, float *y_coord, float *z_coord) {
	for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
		const auto &vert = mesh->mVertices[i];
		x_coord[i] = vert.x;
		y_coord[i] = vert.y;
		z_coord[i] = vert.z;
	}
}

void ReadModel::setNormals(const aiMesh *mesh) {
	if (mesh->HasNormals()) {
		int numNormals = 0;
		float *x_normals, *y_normals, *z_normals;
		coDoVec3 *normals(new coDoVec3(p_normalOut->getNewObjectInfo(), mesh->mNumVertices));
		normals->getAddresses(&x_normals, &y_normals, &z_normals);
		for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
			const auto &norm = mesh->mNormals[i];
			x_normals[i] = norm.x;
			y_normals[i] = norm.y;
			z_normals[i] = norm.z;
		}
		p_normalOut->setCurrentObject(normals);
	}
}

int ReadModel::compute(const char *) {
	const char *filename = p_filename->getValue();
	auto model = load(filename);

	if (!model) {
		if (!p_ignoreErrors) {
			sendError("failed to load %s", filename);
		}
		return FAIL;
	}

	if (coDoPolygons *polyModel = dynamic_cast<coDoPolygons *>(model)) {
		p_polyOut->setCurrentObject(polyModel);
	}
	else {
		coDoPoints *pointModel = dynamic_cast<coDoPoints *>(model);
		p_pointOut->setCurrentObject(pointModel);
	}
	return SUCCESS;
}

MODULE_MAIN(IO, ReadModel)
