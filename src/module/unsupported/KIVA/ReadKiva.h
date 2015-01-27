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

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
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
            f = (float)(*((double *)readPointer));
            readPointer += 8;
        }
        else
        {
            f = *((float *)readPointer);
            readPointer += 4;
        }
    };
    void readInt(int &i)
    {
        i = *((int *)readPointer);
        readPointer += 4;
    };
};
class Application
{

private:
    readBuf rb;
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

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
    char *dataPath;
    char *Mesh;
    char *Veloc;
    char *Press;
    char *rho_name;
    char *vol_name;
    char *temp_name;
    char *amu_name;
    char *tke_name;
    char *eps_name;
    char *p_name;
    char *pv_name;
    char *pt_name;

    //  Local data
    int blockLen;
    int n_coord, n_elem;
    int *el, *vl, *tl, pfactor;
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
    Application(int argc, char *argv[])

    {

        Covise::set_module_description("Read KIVA");
        Covise::add_port(OUTPUT_PORT, "mesh", "coDoUnstructuredGrid", "Grid");
        Covise::add_port(OUTPUT_PORT, "velocity", "Vec3", "velocity");
        Covise::add_port(OUTPUT_PORT, "pressure", "coDoFloat", "pressure");
        Covise::add_port(OUTPUT_PORT, "rho", "coDoFloat", "rho");
        Covise::add_port(OUTPUT_PORT, "vol", "coDoFloat", "vol");
        Covise::add_port(OUTPUT_PORT, "temperature", "coDoFloat", "temperature");
        Covise::add_port(OUTPUT_PORT, "amu", "coDoFloat", "amu");
        Covise::add_port(OUTPUT_PORT, "tke", "coDoFloat", "tke");
        Covise::add_port(OUTPUT_PORT, "eps", "coDoFloat", "eps");
        Covise::add_port(OUTPUT_PORT, "particles", "coDoPoints", "Particles");
        Covise::add_port(OUTPUT_PORT, "pvelocity", "Vec3", "velocity of Particles");
        Covise::add_port(OUTPUT_PORT, "ptemperature", "coDoFloat", "temperature of Particles");
        Covise::add_port(PARIN, "path", "Browser", "Data file path");
        Covise::add_port(PARIN, "numt", "Scalar", "Nuber of Timesteps to read");
        Covise::add_port(PARIN, "skip", "Scalar", "Nuber of Timesteps to skip");
        Covise::add_port(PARIN, "format", "Choice", "Double or Float");
        Covise::add_port(PARIN, "pfactor", "Scalar", "Output each n'th Particle ");
        Covise::set_port_default("path", "data/kiva/otape9 *9*");
        Covise::set_port_default("numt", "1");
        Covise::set_port_default("skip", "0");
        Covise::set_port_default("format", "1 Double Float");
        Covise::set_port_default("pfactor", "1");
        Covise::init(argc, argv);
        Covise::set_quit_callback(Application::quitCallback, this);
        Covise::set_start_callback(Application::computeCallback, this);
    }

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _READIHS_H
