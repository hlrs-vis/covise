/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__LINEREDUCE_H)
#define __LINEREDUCE_H

class LineReduce;

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

class LineReduce : public coSimpleModule
{
private:
    // ports
    coInputPort *linesIn, *dataIn;
    coOutputPort *linesOut, *dataOut;

    // parameters
    coFloatParam *maxAngle, *maxDifference;
    //      coBooleanParam *coordReduce;
    //      coFloatParam *redDist;

    // main-callback
    virtual int compute(const char *port);

    // line reduction
    void lineReduce(int inNumLines, int *inLineList, int inNumCorners, int *inCornerList,
                    int inNumCoord, float *inXCoord, float *inYCoord, float *inZCoord,
                    int &outNumLines, int *outLineList, int &outNumCorners, int *outCornerList,
                    int &outNumCoord, float *outXCoord, float *outYCoord, float *outZCoord,
                    float maxAng, float *dIn[3], int numComp, float *dOut[3]);

    // compute the angle between two lines (A-B  and  C-D)
    float getAngle(int A, int B, int C, int D, float *XCoord, float *YCoord, float *ZCoord);
    float getAngle(int A, int B, int C, int D, int cs, float *XCoord, float *YCoord, float *ZCoord);
    float getAngle(float ax, float ay, float az, float bx, float by, float bz);
    // get the difference of value between four points
    // 1 if difference is smaller than value, which is choosen from the user, else 0
    int getDifference(float A, float B, float C, float D);
    float m_fDifference;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///// LOOKUP-Table,   [case][m_cur][m_test]    (0=abcd, 1=abdc, 2=bacd, 3=badc)
    /////                                          (m_cur is mFlag of current line, m_test that of the line we might merge)
    /////                                              )----(    0     () open,   * connected
    /////                                              *----(    1
    /////                                              )----*    2
    /////                                              *----*    3
    /////                  -> { doMerge, newMCur, newMTest }
    struct
    {
        int doMerge;
        char newMCur, newMTest;
    } caseTable[4][4][4];
    void loadTable();
    void loadTable2(int c, int mC, int mT, int dM, char nC, char nT);

public:
    LineReduce(int argc, char **argv);
    virtual ~LineReduce();
};

//   int *contLine[2];  // 0=line#, 1=0 if none|1 ABCD|2 ABDC|3 BACD|4 BADC
#endif // __LINEREDUCE_H
