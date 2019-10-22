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
// ++                  High Performance Center Stuttgart                  ++
// ++                           Nobelstrasse 19                           ++
// ++                           70569 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  25.07.2019  V1.0                                             ++
// ++**********************************************************************/

#ifndef _READBINARY_H
#define _READBINARY_H

// includes
#include <api/coModule.h>
#include <do/coDoUnstructuredGrid.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

// macros
#define BINARY_MAGIC_V1 0xBABE

using namespace covise;

class ReadBinary : public coModule
{

public:
    ReadBinary(int argc, char *argv[]);
    virtual ~ReadBinary();
    coOutputPort *grid;


private:
    // main
    virtual int compute(const char *port);

    // methods
    virtual void param(const char *name, bool inMapLoading);

    coFileBrowserParam *a_binaryData;                                           //eine Instanz der Klasse coFileBrowserParam, wird im Dialogfenster eingelesen
    char *a_filename;                                                           //der Name der eingelesenen Datei wird gespeichert

    typedef struct {
      double *vertices;
      int *cells;
      int ncells;
      int nvertices;
      int type;
      int dim;
    } Mesh;
    Mesh mesh;
    int read_mesh(char *a_filename, Mesh *mesh, uint8_t bswap);

	uint8_t bswap = 0;

    typedef struct {
      uint32_t magic;
      uint32_t bendian;
      uint32_t pe_size;
      uint32_t type;
    } BinaryFileHeader;

    void getDataset();
    void getVertices(int dim, double* vertices, float* x, float* y, float* z);
    int getNVertices();
    int getCells();
    int getNCells();
    int getDim();
    int getType();

	void myfread(void* ptr, size_t size, size_t count, FILE* stream);
};
#endif
