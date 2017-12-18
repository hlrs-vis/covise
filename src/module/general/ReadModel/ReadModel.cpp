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
#include <do/coDoSet.h>
#include <alg/coFeatureLines.h>

#include <string>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

using namespace covise;

ReadModel::ReadModel(int argc, char *argv[])
	: coModule(argc, argv, "Read polygon meshes in all formats supported by Assimp library")
{
	// Parameters
	p_filename = addFileBrowserParam("polygonFile", "Polygon file path");
	p_filename->setValue("~", "*/*.dae/*.gltf,*.glb/*.blend/*.3ds,*.fbx,*.dxf,*.ase/*.obj/*.ifc/*.xgl,*.zgl/*.ply/*.lwo,*.lws,*.lxo/*.stl/*.x/*.ac/*.ms3d/*.cob,*.scn/*.bvh,*.csm/*.xml,*.irrmesh,*.irr/*.mdl,*.md2,*.md3,*.pk3,*.mdc,*.md5,*.smd,*.vta,*.ogex,*.3d/*.b3d/*.q3d,*.q3s/*.nff/*.off/*.raw/*.ter/*.mdl,*.hmp/*.ndo");

	p_triangulate = addBooleanParam("triangulate", "Triangulate polygons");
	p_triangulate->setValue(1);

	p_joinVertices = addBooleanParam("joinVertices", "Join identical vertices");
	p_joinVertices->setValue(0);

	p_ignoreErrors = addBooleanParam("ignoreErrors", "Ignore files that are not found");
	p_ignoreErrors->setValue(0);

	// Output ports
	p_polyOut = addOutputPort("PolyOut0", "Polygons", "geometry polygons");
	p_normalOut = addOutputPort("DataOut0", "Vec3", "polygon normals");
}

ReadModel::~ReadModel() {
}

ReadModel::allGeometry ReadModel::load(const std::string &filename, std::string polyName, std::string normalName) {

	Assimp::Importer importer;
	unsigned int readFlags = aiProcess_PreTransformVertices | aiProcess_SortByPType | aiProcess_ImproveCacheLocality | aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes;

	if (p_triangulate) {
		readFlags |= aiProcess_Triangulate;
	}

	if (p_joinVertices) {
		readFlags |= aiProcess_JoinIdenticalVertices;
	}

	const aiScene* scene = importer.ReadFile(filename, readFlags);
	ReadModel::allGeometry geoCollect = {};

	if (!scene) {
		if (!p_ignoreErrors) {
			sendError("failed to read %s : %s", filename.c_str(), importer.GetErrorString());
		}
		return geoCollect;
	}

	// create objects for all meshes in scene
	for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
		const aiMesh *mesh = scene->mMeshes[m];

		if (mesh->HasPositions()) {
			float *x_coord, *y_coord, *z_coord;
			unsigned int numPoints = mesh->mNumVertices;
			if (scene->mNumMeshes > 1) {
				std::stringstream polyNameS;
				polyNameS << polyName << m;
				polyName = polyNameS.str();
			}

			if (mesh->HasFaces()) {
				unsigned int numPolygons = mesh->mNumFaces, numCorners = 0;
				for (unsigned int f = 0; f < numPolygons; ++f) {
					numCorners += mesh->mFaces[f].mNumIndices;
				}
				coDoPolygons *poly(new coDoPolygons(polyName.c_str(), numPoints, numCorners, numPolygons));
				int *cornerList, *polyList;
				poly->getAddresses(&x_coord, &y_coord, &z_coord, &cornerList, &polyList);
				unsigned int idx = 0, vertCount = 0;
				for (unsigned int f = 0; f < numPolygons; ++f) {
					polyList[idx] = vertCount;
					idx++;
					const aiFace &face = mesh->mFaces[f];
					for (unsigned int i = 0; i < face.mNumIndices; ++i) {
						cornerList[vertCount] = face.mIndices[i];
						vertCount++;
					}
				}
				setPoints(mesh, x_coord, y_coord, z_coord);
				geoCollect.allMeshes.push_back(poly);
			}
			// mesh contains only points, no polygons
			else {
				coDoPoints *points(new coDoPoints(polyName.c_str(), numPoints));
				points->getAddresses(&x_coord, &y_coord, &z_coord);
				setPoints(mesh, x_coord, y_coord, z_coord);
				geoCollect.allMeshes.push_back(points);
			}
			// catch normals, if they exist
			if (mesh->HasNormals()) {
				unsigned int numNormals = 0;
				float *x_normals, *y_normals, *z_normals;
				if (scene->mNumMeshes > 1) {
					std::stringstream normalNameS;
					normalNameS << normalName << m;
					normalName = normalNameS.str();
				}
				coDoVec3 *normals(new coDoVec3(normalName.c_str(), mesh->mNumVertices));
				normals->getAddresses(&x_normals, &y_normals, &z_normals);
				for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
					const aiVector3D &norm = mesh->mNormals[i];
					x_normals[i] = norm.x;
					y_normals[i] = norm.y;
					z_normals[i] = norm.z;
				}
				geoCollect.allNormals.push_back(normals);
			}
		}
	}
	return geoCollect;
}

void ReadModel::setPoints(const aiMesh *mesh, float *x_coord, float *y_coord, float *z_coord) {
	for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
		const aiVector3D &vert = mesh->mVertices[i];
		x_coord[i] = vert.x;
		y_coord[i] = vert.y;
		z_coord[i] = vert.z;
	}
}

int ReadModel::compute(const char *) {
	const std::string filename = p_filename->getValue();
	std::string polyName = p_polyOut->getObjName();
	std::string normalName = p_normalOut->getObjName();

	allGeometry model = load(filename, polyName, normalName);

	if (model.allMeshes.empty()) {
		if (!p_ignoreErrors) {
			sendError("failed to load %s", filename.c_str());
		}
		return FAIL;
	}

	// create a coDoSet in case of multiple meshes
	if (model.allMeshes.size() > 1) {
		coDoSet *modelSet = new coDoSet(polyName.c_str(), (int)model.allMeshes.size(), &model.allMeshes.front());
		model.allMeshes.clear();
		p_polyOut->setCurrentObject(modelSet);
	}
	else {
		p_polyOut->setCurrentObject(model.allMeshes.front());
	}

	// create a coDoSet in case of multiple meshes
	if (model.allNormals.size() > 1) {
		coDoSet *normalSet = new coDoSet(normalName.c_str(), (int)model.allNormals.size(), &model.allNormals.front());
		model.allNormals.clear();
		p_normalOut->setCurrentObject(normalSet);
	}
	else {
		p_normalOut->setCurrentObject((coDoVec3 *)model.allNormals.front());
	}

	return SUCCESS;
}

MODULE_MAIN(IO, ReadModel)
