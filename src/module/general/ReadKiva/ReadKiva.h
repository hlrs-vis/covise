/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READIHS_H
#define _READIHS_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Ihs data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#ifdef BYTESWAP
#define bswap(x) byteSwap(x)
#else
#define bswap(x)
#endif

#define FORMAT_ASCII 10
#define FORMAT_BINARY 0

int readDouble;

class fileHeader
{
public:
    char dataDate[25];
    char dataName[81];
    float time;
    int ncyc;
    float crank;
    char jnm[9];
    int ifirst;
    int ncells;
    int nverts;
    float cylrad;
    float zpistn;
    float zhead;
    int np;
    int nrk;
    int nsp;
    int irez;
    int numBoundaryVertices; // (ibv) number of boundary vertices
    int *boundaryVertices;
    int iper;
    float rhop;
    float cmueps;
    int naxisj;
    int nregions;
    int numCoords;
    int numElem;

    void print();
};

class readBuf
{
private:
    char *buf;
    int bufSize;
    int blockSize;
    int fd;
    char *readPointer;

public:
    readBuf(int bs = 1000)
    {
        buf = new char[bs];
        bufSize = bs;
    };
    ~readBuf()
    {
        delete[] buf;
    };
    /*void init(int nint,int nfloat, int f)
      {
        fd=f;
        if(readDouble)
          blockSize = nint*4 + nfloat*8;
        else
          blockSize = nint*4 + nfloat*4;
      };*/
    void init(int Blocklen, int nitems, int f)
    {
        fd = f;
        blockSize = Blocklen / nitems;
    };
    int read()
    {
        if (::read(fd, buf, blockSize) < (blockSize))
        {
            Covise::sendError("unexpected end of file");
            return (-1);
        }
        readPointer = buf;
        return (blockSize);
    };
    void skip(int nint, int nfloat)
    {
        if (readDouble)
            readPointer += nint * 4 + nfloat * 8;
        else
            readPointer += nint * 4 + nfloat * 4;
    };
    void skip(int n)
    {
        lseek(fd, n * blockSize, SEEK_CUR);
    };
    void readFloat(float &f)
    {
        if (readDouble)
        {
            bswap(*((double *)readPointer));
            f = (float)(*((double *)readPointer));
            readPointer += 8;
        }
        else
        {
            f = *((float *)readPointer);
            bswap(f);
            readPointer += 4;
        }
    };
    void readInt(int &i)
    {
        i = *((int *)readPointer);
        bswap(i);
        readPointer += 4;
    };
};

class ReadKiva : public coSimpleModule
{

private:
    readBuf rb;
    //  member functions

    coOutputPort *p_mesh;
    coOutputPort *p_velocity;
    coOutputPort *p_pressure;
    coOutputPort *p_rho;
    coOutputPort *p_vol;
    coOutputPort *p_temperature;
    coOutputPort *p_amu;
    coOutputPort *p_tke;
    coOutputPort *p_eps;
    coOutputPort *p_particles;
    coOutputPort *p_pvelocity;
    coOutputPort *p_ptemperature;

    coFileBrowserParam *p_path;
    coChoiceParam *p_format;
    coIntScalarParam *p_numt;
    coIntScalarParam *p_skip;
    coIntScalarParam *p_pfactor;

    virtual int compute(const char *port);

    int readHeader(int fp);
    int readData(int fp);
    int readParticles(int fp);
    int readConn(int fp);
    int readFloat(int fp, float &);
    int readFloat(int fp, float *, int);
    int readInt(int fp, int &);

    int skipBlocks(int fp, int num); // skip n fortran blocks
    int beginRead(int fp); // skip fortran block header returns -1 on error
    int endRead(int fp); // skip fortran block footer returns -1 on error
    //  Parameter names
    const char *dataPath;
    const char *Mesh;
    const char *Veloc;
    const char *Press;
    const char *rho_name;
    const char *vol_name;
    const char *temp_name;
    const char *amu_name;
    const char *tke_name;
    const char *eps_name;
    const char *p_name;
    const char *pv_name;
    const char *pt_name;

    //  Local data
    int blockLen;
    int n_coord, n_elem;
    int *el, *vl, *tl;
    long pfactor;
    float *x_coord;
    float *y_coord;
    float *z_coord;
    float *eps;
    float *u, *v, *w;
    float *pu, *pv, *pw;
    float *px, *py, *pz, *ptemp;
    float *p;
    float *rho;
    float *vol;
    float *temp;
    float *amu;
    float *tke;
    int isAscii;
    fileHeader header;
    int *boundaryVertices;

    //  Shared memory data
    coDoUnstructuredGrid *mesh;
    coDoVec3 *DOveloc;
    coDoFloat *DOpress;
    coDoFloat *DOrho;
    coDoFloat *DOvol;
    coDoFloat *DOtemperature;
    coDoFloat *DOamu;
    coDoFloat *DOtke;
    coDoFloat *DOeps;
    coDoFloat *DOptemperature;
    coDoVec3 *DOpveloc;
    coDoPoints *DOparticles;

public:
    ReadKiva(int argc, char **argv);
    virtual ~ReadKiva()
    {
    }
};
#endif // _READIHS_H
