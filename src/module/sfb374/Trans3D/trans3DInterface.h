/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TRANS_INTERFACE
#define TRANS_INTERFACE
typedef double prec;

class trans3DInterface
{
public:
    trans3DInterface();
    ~trans3DInterface();
    void init(const char *filename);
    int Calculate(int numStaps);
    int executeScript();
    int initCalculation();
    void setIntensity(float factor);
    void setRadius(float factor);
    void setLaserPos(float x, float y, float z);
    void getLaserPos(float &x, float &y, float &z);
    void getGridSize(int &xDim, int &yDim, int &zDim);
    void getValues(int i, int j, int k, float *xc, float *yc, float *zc, float *t, float *q);

private:
    int nstep;
    prec ress, resd, dtmax, dimtime;
};

extern trans3DInterface trans3D;
#endif
