/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Readmodule for binary files in COVISE API              ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                              Anna Mack                              ++
// ++              High Performance Computing Center Stuttgart            ++
// ++                           Nobelstrasse 19                           ++
// ++                           70569 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  25.07.2019  V1.0 
// ++**********************************************************************/

// this includes our own class's headers
#include "ReadBinary.h"
#include <api/coModule.h>
#include <string.h>
#include <stdint.h>


// constructor
ReadBinary::ReadBinary(int argc, char *argv[])
    : coModule(argc, argv, "Annas first Reader")
{
	cout << "ReadBinary::ReadBinary()" << endl;


    grid = addOutputPort("mesh", "UnstructuredGrid", "unstructured grid");                                  //Output Gitter

    a_binaryData = addFileBrowserParam("filename", "Specify the filename of the binary data file(s).");     //Dialogfenster zum Dateneinlesen
    //a_binaryData->setValue("./mnt/raid/data/hidalgo/airflow/mesh_out.bin", "*.bin/");                       //Defaulf file, zulaessige Datentypen
	//a_binaryData->setValue("./.", "*.bin/");                       //Defaulf file, zulaessige Datentypen

    a_filename = new char[256];                                                                             //Dateiname der Binaerdaten speichern
    strcpy(a_filename, a_binaryData->getValue());
    uint8_t bswap =0;                                                                                       //byteswap ermoeglichen
}
// destructor
ReadBinary::~ReadBinary()
{
	cout << "ReadBinary::~ReadBinary()" << endl;

    if (a_filename != NULL)
    {
        delete[] a_filename;
        a_filename = NULL;
    }
}

// compute() is called once for every EXECUTE message
int ReadBinary::compute(const char *port)
{
	cout << "ReadBinary::compute()" << endl;

    (void)port;
    sendInfo("Annas zweites Modul!");
    read_mesh(a_filename, &mesh, bswap);            //binaerdatei wird gelesen und im filepointer mesh abgelegt
    getDataset();
    //grid->setCurrentObject(g);

    return SUCCESS;
}

// read function as used in dolfin-post
int ReadBinary::read_mesh(char *a_filename, Mesh *mesh, uint8_t bswap) {
  BinaryFileHeader mesh_hdr;
  FILE *mesh_fp;
  int *tmp_cells;
  double *tmp_vertices;

  if ( (mesh_fp = fopen(a_filename, "r")) == 0) {
    perror(a_filename);
    return -1;
  }

  fread(&mesh_hdr, sizeof(BinaryFileHeader), 1, mesh_fp);
//  if (bswap)
//    bswap_hdr(&mesh_hdr);
//  if ((mesh_hdr.magic != BINARY_MAGIC_V1 &&
//       mesh_hdr.magic != BINARY_MAGIC_V2) ||
//      mesh_hdr.type != BINARY_MESH_DATA)
//    return -1;


  fread(&(mesh->dim), sizeof(int), 1, mesh_fp);
  fread(&(mesh->type), sizeof(int), 1, mesh_fp);
  fread(&(mesh->nvertices), sizeof(int), 1, mesh_fp);

//  if(bswap) {
//    mesh->dim = bswap_int(mesh->dim);
//    mesh->type = bswap_int(mesh->type);
//    mesh->nvertices = bswap_int(mesh->nvertices);
//  }

  if (mesh_hdr.magic == BINARY_MAGIC_V1)
    mesh->type = (mesh->type + 3);
  else
    mesh->type = (mesh->type + 1);

  if (mesh->vertices) {
    tmp_vertices = mesh->vertices;
    mesh->vertices = (double*)realloc(tmp_vertices, mesh->dim * mesh->nvertices * sizeof(double));
  }
  else
    mesh->vertices = (double *)malloc(mesh->dim * mesh->nvertices * sizeof(double));
  fread(mesh->vertices, sizeof(double), mesh->dim * mesh->nvertices , mesh_fp);

  fread(&mesh->ncells, sizeof(int), 1, mesh_fp);

//  if (bswap) {
//    bswap_data(mesh->vertices, mesh->dim * mesh->nvertices);
//    mesh->ncells = bswap_int(mesh->ncells);
//  }

  if (mesh->cells) {
    tmp_cells = mesh->cells;
    mesh->cells = (int *)realloc(tmp_cells,
              mesh->ncells * mesh->type * sizeof(int));
  }
  else
    mesh->cells = (int *)malloc(mesh->ncells * mesh->type * sizeof(int));
  fread(mesh->cells, sizeof(int), mesh->ncells * mesh->type, mesh_fp);
  fclose(mesh_fp);

//  if (bswap)
//    bswap_iarr(mesh->cells, mesh->ncells * mesh->type);

  return 0;
}

// param function
void ReadBinary::param(const char *name, bool inMapLoading)
{
    if (strcmp(name, a_binaryData->getName()) == 0)
    {
        //filename aktualisieren nach Neueingabe
    strcpy(a_filename, a_binaryData->getValue());
    }
}

// Data functions
void ReadBinary::getDataset(){

    coDoUnstructuredGrid *g = new coDoUnstructuredGrid(grid->getObjName(), 2, 13, mesh.nvertices, 1);
    int *el, *cl, *tl;
    float *x, *y, *z;
    g->getAddresses(&el, &cl, &x, &y, &z);
    g->getTypeList(&tl);
    //getVertices(mesh.dim, mesh.vertices, x, y, z);
//    delete x;
//    delete y;
//    delete z;

}
void ReadBinary::getVertices(int dim, double* vertices, float* x, float* y, float* z){

    if(dim==2){
    sendInfo("dimension = 2");
    }
    else if (dim==3) {
    sendInfo("dimension = 3");
    //mesh.vertices(1)
    }
//    mesh.vertices
}
//int ReadBin::getNVertices(){}
//int ReadBin::getCells(){}
//int ReadBin::getNCells(){}
//int ReadBin::getDim(){}
//int ReadBin::getType(){}



MODULE_MAIN(IO, ReadBinary)
