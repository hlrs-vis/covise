/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__READWMATRIX_H)
#define __READWMATRIX_H

#include <appl/ApplInterface.h>
using namespace covise;

class ReadWMatrix
{
private:
    int sizeX, sizeY, sizeZ; // number of points in each dimension
    float *matrixPoints; // scalar values for matrix points
    FILE *fp; // matrix data file

    // Member functions:
    void compute(void *);
    void quit(void *);
    int openFile();
    int readWMFile();
    void generateMatrix(float);
    int getWord(char *);
    int getNumber(float *);
    void freeWMMemory();

    //  Static callback stubs:
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

public:
    ReadWMatrix(int argc, char *argv[]);
    ~ReadWMatrix(){};

    void run()
    {
        Covise::main_loop();
    }
};
#endif
