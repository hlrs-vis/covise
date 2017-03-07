/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LineReduce.h"
#include <util/coviseCompat.h>
#include <do/coDoData.h>

LineReduce::LineReduce(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Merge/Reduce lines")
{
    // parameters
    maxAngle = addFloatParam("maxAngle", "used for line-reduction (in radians)");
    maxAngle->setValue(0.01f);

    //    coordReduce = addBooleanParam( "coordReduce", "should we reduce coordinates ?!" );
    //    coordReduce->setValue( 0 );
    //    redDist = addFloatParam( "redDist", "max. distance between coordinates for reduction" );
    //    redDist->setValue( 0.001 );
    maxDifference = addFloatParam("maxDifference", "maximum difference between two points");
    maxDifference->setValue(0.1f);
    // ports
    linesIn = addInputPort("linesIn", "Lines", "Lines input");
    dataIn = addInputPort("dataIn", "Vec3|Float", "Data input");
    dataIn->setRequired(0);
    linesOut = addOutputPort("linesOut", "Lines", "Lines output");
    dataOut = addOutputPort("dataOut", "Vec3|Float", "Data output");
    dataOut->setDependencyPort(dataIn);

// we don't handle any multiblock/timestep stuff, because this is a REALLY SIMPLE module
#ifndef YAC
    setComputeTimesteps(0);
    setComputeMultiblock(0);
    // and the API should take care of attributes
    setCopyAttributes(1);
#endif
    // done
    return;
}

LineReduce::~LineReduce()
{
    // dummy
}

int LineReduce::compute(const char *)
{
    int i;

    float maxAng; // , maxDist
    //int cRed;

    const coDistributedObject *lObjIn, *dObjIn;
    const coDoLines *lIn;
    coDoLines *lOut;
    coDoFloat *s3d;
    coDoVec3 *v3d;
    float *dIn[3], *dOut[3];
    int numComp;

    int *inLineList, *inCornerList;
    float *inXCoord, *inYCoord, *inZCoord;
    int inNumLines, inNumCoord, inNumCorners;

    int *outLineList, *outCornerList;
    float *outXCoord, *outYCoord, *outZCoord;
    int outNumLines, outNumCoord, outNumCorners;
    int iNumOfData;

    // KM+12.09.01 initialize the variables
    iNumOfData = 0;
    s3d = NULL;
    v3d = NULL;
    lObjIn = NULL;
    dObjIn = NULL;

    for (i = 0; i < 3; i++)
    {
        dIn[i] = NULL;
        dOut[i] = NULL;
    }

    lObjIn = linesIn->getCurrentObject();
    dObjIn = dataIn->getCurrentObject();

    // check/get input
    if (!lObjIn)
    {
        sendError("no linesIn-object available");
        return (0);
    }
    if (!lObjIn->isType("LINES"))
    {
        sendError("linesIn-object is not of type LINES");
        return (0);
    }
    numComp = 0; // no data at all
    if (dObjIn)
    {
        // we have data, so get it
        if (dObjIn->isType("USTSDT"))
        {
            s3d = (coDoFloat *)dObjIn;
            s3d->getAddress(&(dIn[0]));
            numComp = 1;
        }
        else if (dObjIn->isType("USTVDT"))
        {
            v3d = (coDoVec3 *)dObjIn;
            v3d->getAddresses(&(dIn[0]), &(dIn[1]), &(dIn[2]));
            numComp = 3;
        }
    }
    else
    {
        // KM 30.01.02 this input port is optional
        /*Covise::sendError("no data input!");
      return STOP_PIPELINE;*/
    }
    // no scalar or vector data
    //	if (!s3d && !v3d)
    //		sendInfo( "Warning: There is no scalar or vector data input!" );

    lIn = (coDoLines *)lObjIn;
    lIn->getAddresses(&inXCoord, &inYCoord, &inZCoord, &inCornerList, &inLineList);
    inNumCoord = lIn->getNumPoints();
    inNumCorners = lIn->getNumVertices();
    inNumLines = lIn->getNumLines();

    // get parameters
    maxAng = maxAngle->getValue();
    //    maxDist = redDist->getValue();
    //    cRed = coordReduce->getValue();

    // alloc space for output....
    //   note that we will definitely never produce more than we got as input
    outXCoord = new float[inNumCoord];
    outYCoord = new float[inNumCoord];
    outZCoord = new float[inNumCoord];
    outCornerList = new int[inNumCorners];
    outLineList = new int[inNumLines];

    if (s3d)
    {
        // get Number of Data Values
        iNumOfData = s3d->getNumPoints();
        if (iNumOfData != inNumCoord)
        {
            sendError("Number of Points != Number of Data");
            return STOP_PIPELINE;
        }
    }

    // KM 12.09.01 Error No 176
    // we need more memory store... (before: new float[inNumLines] )
    // but only when there are scalar or vector data
    if (s3d || v3d)
        for (i = 0; i < numComp; i++)
            dOut[i] = new float[inNumCoord];

    // init
    outNumCoord = 0;
    outNumLines = 0;
    outNumCorners = 0;

    // compute
    //    if( cRed )
    //       sendInfo( "LineReduce: coordinate reduction/merging not yet available. coordReduce ignored." );

    lineReduce(inNumLines, inLineList, inNumCorners, inCornerList, inNumCoord, inXCoord, inYCoord, inZCoord,
               outNumLines, outLineList, outNumCorners, outCornerList, outNumCoord, outXCoord, outYCoord, outZCoord,
               maxAng, dIn, numComp, dOut);

    /*
   // debug
   outNumCoord = inNumCoord;
   for( i=0; i<inNumCoord; i++ )
   {
      outXCoord[i] = inXCoord[i];
      outYCoord[i] = inYCoord[i];
      outZCoord[i] = inZCoord[i];
   }
   outNumLines = inNumLines;
   for( i=0; i<inNumLines; i++ )
   outLineList[i] = inLineList[i];
   outNumCorners = inNumCorners;
   for( i=0; i<inNumCorners; i++ )
   outCornerList[i] = inCornerList[i];
   */

    // create output
    lOut = new coDoLines(linesOut->getObjName(), outNumCoord, outXCoord, outYCoord, outZCoord,
                         outNumCorners, outCornerList, outNumLines, outLineList);

    if (numComp == 1)
        s3d = new coDoFloat(dataOut->getObjName(), outNumCoord, dOut[0]);
    else if (numComp == 3)
        v3d = new coDoVec3(dataOut->getObjName(), outNumCoord, dOut[0], dOut[1], dOut[2]);

    // clean up
    delete[] outXCoord;
    delete[] outYCoord;
    delete[] outZCoord;
    delete[] outCornerList;
    delete[] outLineList;

    if (s3d || v3d)
        for (i = 0; i < numComp; i++)
            delete[] dOut[i];

    // finish work
    linesOut->setCurrentObject(lOut);
    if (numComp == 1)
        dataOut->setCurrentObject(s3d);
    else if (numComp == 3)
        dataOut->setCurrentObject(v3d);

    // done
    return CONTINUE_PIPELINE;
}

void LineReduce::lineReduce(int inNumLines, int *inLineList, int inNumCorners, int *inCornerList,
                            int inNumCoord, float *inXCoord, float *inYCoord, float *inZCoord,
                            int &outNumLines, int *outLineList, int &outNumCorners, int *outCornerList,
                            int &outNumCoord, float *outXCoord, float *outYCoord, float *outZCoord,
                            float maxAng, float *dIn[3], int numComp, float *dOut[3])
{
    (void)numComp;

    int i, j, n, k;
    int A, B, C, D;

    int *lines[3]; // 0=start, 1=stop, 2=length
    char *mFlag; // a line-segment can only be part of ONE SINGLE line
    // contCase: 0 if none|1 ABCD|2 ABDC|3 BACD|4 BADC
    struct contInfoStru
    {
        int Acont, Bcont, AcontCase, BcontCase;
    } *contInfo;
    char *cPtr, *mPtr;

    float minAngle, curAngle;
    int minLine[5]; // contLine[0][i], contLine[1][i], mFlag[i], mFlag[contLine[0][i]], AorB
    int maxLine;
    int hasMinLine;

    int curCase, useFlag, curCont;

    int AorB[4] = { 1, 1, 0, 0 };
    int CorD[4] = { 0, 1, 0, 1 };

    // this is for speed-up
    lines[0] = new int[inNumLines];
    lines[1] = new int[inNumLines];
    lines[2] = new int[inNumLines];
    mFlag = new char[inNumLines];
    for (i = 0; i < inNumLines; i++)
    {
        mFlag[i] = 0;
        lines[0][i] = inCornerList[inLineList[i]];
        if (i == inNumLines - 1)
        {
            lines[1][i] = inCornerList[inNumCorners - 1];
            lines[2][i] = inNumCorners - inLineList[i];
        }
        else
        {
            lines[1][i] = inCornerList[inLineList[i + 1] - 1];
            lines[2][i] = inLineList[i + 1] - inLineList[i];
        }
    }

    // we need this
    contInfo = new contInfoStru[inNumLines];
    for (i = 0; i < inNumLines; i++)
    {
        contInfo[i].Acont = -1;
        contInfo[i].Bcont = -1;
        contInfo[i].AcontCase = 0;
        contInfo[i].AcontCase = 0;
    }

    // first we mark all doublicated lines as doublicated...
    for (i = 0; i < inNumLines - 1; i++)
    {
        A = lines[0][i];
        B = lines[1][i];
        // KM 15.01.02:it's a point because start and stop of the line are the same
        if (A == B)
            mFlag[i] = 5;
        else if (!mFlag[i])
        {
            for (n = i + 1; n < inNumLines; n++)
            {
                C = lines[0][n];
                D = lines[1][n];
                // KM 15.01.02: 2 lines can be put together
                if ((A == C && B == D) || (A == D && B == C))
                    mFlag[n] = 4;
            }
        }
    }

    // now try to 'assemble' the line-segments to longer lines
    loadTable();
    for (i = 0; i < inNumLines - 1; i++)
    {
        // init
        minAngle = 4.0;
        minLine[0] = 0; // which line
        minLine[1] = 0; // which case
        minLine[2] = 0; // mFlag for this line
        minLine[3] = 0; // mFlag for other line (minLine[0])
        minLine[4] = 0; // AorB
        //AorB = -1;      // 0: *A*    1: *B*
        hasMinLine = 0;

        A = lines[0][i];
        B = lines[1][i];
        cPtr = mFlag + i;

        // this is fun
        k = 0;
        if (*cPtr < 3)
        {
            // now check if we have some other line that starts/ends with the same "corner(s)"
            for (n = i + 1; n < inNumLines; n++)
            {
                C = lines[0][n];
                D = lines[1][n];
                mPtr = mFlag + n;
                if (*mPtr < 3)
                {
                    useFlag = 0;
                    if (B == C) // ABCD
                    {
                        curCase = 0;
                        useFlag = 1;
                    }
                    else if (B == D) // ABDC
                    {
                        curCase = 1;
                        useFlag = 1;
                    }
                    else if (A == C) // BACD
                    {
                        curCase = 2;
                        useFlag = 1;
                    }
                    else if (A == D) // BADC
                    {
                        curCase = 3;
                        useFlag = 1;
                    }

                    // did we find something ?
                    if (useFlag)
                    {
                        // compute angle and maybe store results
                        if (caseTable[curCase][int(*cPtr)][int(*mPtr)].doMerge)
                        {
                            curAngle = getAngle(A, B, C, D, curCase, inXCoord, inYCoord, inZCoord);
                            k++;
                            // the lookup-table makes things easy
                            m_fDifference = maxDifference->getValue();
                            if (dIn[0])
                            {
                                if (curAngle < minAngle && getDifference(dIn[0][A], dIn[0][B], dIn[0][C], dIn[0][D]))
                                {
                                    minAngle = curAngle;
                                    minLine[0] = n;
                                    minLine[1] = curCase + 1;
                                    minLine[2] = caseTable[curCase][int(*cPtr)][int(*mPtr)].newMCur;
                                    minLine[3] = caseTable[curCase][int(*cPtr)][int(*mPtr)].newMTest;
                                    minLine[4] = AorB[curCase];
                                    hasMinLine = 1;
                                }
                            }
                            else
                            {
                                if (curAngle < minAngle)
                                {
                                    minAngle = curAngle;
                                    minLine[0] = n;
                                    minLine[1] = curCase + 1;
                                    minLine[2] = caseTable[curCase][int(*cPtr)][int(*mPtr)].newMCur;
                                    minLine[3] = caseTable[curCase][int(*cPtr)][int(*mPtr)].newMTest;
                                    minLine[4] = AorB[curCase];
                                    hasMinLine = 1;
                                }
                            }
                        }
                    }
                } // C!=D
            } // for(n)
        } // A!=B / mFlag

        // store results
        if (hasMinLine)
        {
            n = minLine[0];
            if (minLine[4]) // B connected
            {
                contInfo[i].Bcont = n;
                contInfo[i].BcontCase = minLine[1];
                if (minLine[1] == 1) // ABCD
                {
                    contInfo[n].Acont = i;
                    contInfo[n].AcontCase = 4;
                }
                else // ABDC
                {
                    contInfo[n].Bcont = i;
                    contInfo[n].BcontCase = 2;
                }
            }
            else // A connected
            {
                contInfo[i].Acont = n;
                contInfo[i].AcontCase = minLine[1];
                if (minLine[1] == 3) // BACD
                {
                    contInfo[n].Acont = i;
                    contInfo[n].AcontCase = 3;
                }
                else // BADC
                {
                    contInfo[n].Bcont = i;
                    contInfo[n].BcontCase = 1;
                }
            }
            mFlag[i] = (char)minLine[2];
            mFlag[n] = (char)minLine[3];

            // maybe retry ?
            if (k > 1 && mFlag[i] < 3)
                i--;
        }
    }

    // keep coordinates (for now)
    outNumCoord = inNumCoord;
    for (i = 0; i < inNumCoord; i++)
    {
        outXCoord[i] = inXCoord[i];
        outYCoord[i] = inYCoord[i];
        outZCoord[i] = inZCoord[i];
        // KM+12.09.01 dIn = Null if there is no scalar or vector data
        if (dIn[0])
            (dOut[0])[i] = (dIn[0])[i];
    }

    // now merge the lines
    outNumLines = 0;
    outNumCorners = 0;
    for (i = 0; i < inNumLines; i++)
    {
        switch (mFlag[i])
        {
        case 0: //    )----(
            // simply copy this line
            outLineList[outNumLines] = outNumCorners;
            outNumLines++;
            for (j = 0; j < lines[2][i]; j++)
            {
                outCornerList[outNumCorners] = inCornerList[inLineList[i] + j];
                outNumCorners++;
            }
            break;
        case 1: //    *----(
            // we add one line
            outLineList[outNumLines] = outNumCorners;
            outNumLines++;
            // then copy the current line (reverse)
            for (j = lines[2][i] - 1; j >= 0; j--)
            {
                outCornerList[outNumCorners] = inCornerList[inLineList[i] + j];
                outNumCorners++;
            }

            // and add all following lines
            curCont = contInfo[i].Acont;
            curCase = contInfo[i].AcontCase;
            n = curCont;
            k = i;
            while (mFlag[n] == 3)
            {
                // add this line (n)
                if (curCase == 1 || curCase == 3) // CD
                {
                    // as is
                    for (j = 0; j < lines[2][n]; j++)
                    {
                        outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                        outNumCorners++;
                    }
                    curCont = contInfo[n].Bcont;
                    curCase = contInfo[n].BcontCase;
                }
                else // DC
                {
                    // reverse
                    for (j = lines[2][n] - 1; j >= 0; j--)
                    {
                        outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                        outNumCorners++;
                    }
                    curCont = contInfo[n].Acont;
                    curCase = contInfo[n].AcontCase;
                }
                // mark this one as used
                mFlag[n] = 6;
                // on to the next one
                n = curCont;
            }

            // this is the tail of the line, so we will have to add it
            if (curCase == 1 || curCase == 3) // CD
            {
                // as is
                for (j = 0; j < lines[2][n]; j++)
                {
                    outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                    outNumCorners++;
                }
                curCont = contInfo[n].Bcont;
                curCase = contInfo[n].BcontCase;
            }
            else // DC
            {
                // reverse
                for (j = lines[2][n] - 1; j >= 0; j--)
                {
                    outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                    outNumCorners++;
                }
                curCont = contInfo[n].Acont;
                curCase = contInfo[n].AcontCase;
            }

            // make sure not to add this line twice (e.g. in reverse order)
            mFlag[n] = 7;
            break;
        case 2: //    )----*
            // we add one line
            outLineList[outNumLines] = outNumCorners;
            outNumLines++;
            // then copy the current line
            for (j = 0; j < lines[2][i]; j++)
            {
                outCornerList[outNumCorners] = inCornerList[inLineList[i] + j];
                outNumCorners++;
            }

            // and add all following lines
            curCont = contInfo[i].Bcont;
            curCase = contInfo[i].BcontCase;
            n = curCont;
            k = i;
            while (mFlag[n] == 3)
            {
                // add this line (n)
                if (curCase == 1 || curCase == 3) // CD
                {
                    // as is
                    for (j = 0; j < lines[2][n]; j++)
                    {
                        outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                        outNumCorners++;
                    }
                    curCont = contInfo[n].Bcont;
                    curCase = contInfo[n].BcontCase;
                }
                else // DC
                {
                    // reverse
                    for (j = lines[2][n] - 1; j >= 0; j--)
                    {
                        outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                        outNumCorners++;
                    }
                    curCont = contInfo[n].Acont;
                    curCase = contInfo[n].AcontCase;
                }
                // mark this one as used
                mFlag[n] = 6;
                // on to the next one
                n = curCont;
            }

            // this is the tail of the line, so we will have to add it
            if (curCase == 1 || curCase == 3) // CD
            {
                // as is
                for (j = 0; j < lines[2][n]; j++)
                {
                    outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                    outNumCorners++;
                }
                curCont = contInfo[n].Bcont;
                curCase = contInfo[n].BcontCase;
            }
            else // DC
            {
                // reverse
                for (j = lines[2][n] - 1; j >= 0; j--)
                {
                    outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                    outNumCorners++;
                }
                curCont = contInfo[n].Acont;
                curCase = contInfo[n].AcontCase;
            }

            // make sure not to add this line twice (e.g. in reverse order)
            mFlag[n] = 7;
            break;
        case 3: //    *----*
            // do nothing
            break;
        }
    }

    // note that we might have closed-lines (where all segments have mFlag==3)
    //   check for them by searching for lines that still have mFlag==3
    for (i = 0; i < inNumLines; i++)
    {
        if (mFlag[i] == 3)
        {
            // here we go, find junction with maximum angle to split line there
            minAngle = 0.0; // max. angle yet found
            useFlag = 0; // contCase for the found line
            maxLine = i; // line on which the angle/cont was found
            curCont = contInfo[i].Acont;
            curCase = contInfo[i].AcontCase;
            A = lines[0][i];
            B = lines[1][i];
            while (curCont != i)
            {
                C = lines[0][curCont];
                D = lines[1][curCont];
                curAngle = getAngle(A, B, C, D, curCase, inXCoord, inYCoord, inZCoord);
                if (curAngle > minAngle)
                {
                    minAngle = curAngle;
                    useFlag = curCase;
                    maxLine = curCont;
                }

                // on to the next one
                A = C;
                B = D;
                n = curCont;
                if (curCase == 1 || curCase == 3) // we must continue via B
                {
                    curCont = contInfo[n].Bcont;
                    curCase = contInfo[n].BcontCase;
                }
                else
                {
                    curCont = contInfo[n].Acont;
                    curCase = contInfo[n].AcontCase;
                }
            }

            // found it
            if (AorB[useFlag - 1])
            {
                // split on B
                mFlag[maxLine] = 1;
                if (CorD[useFlag - 1]) // on D (B)
                    mFlag[contInfo[maxLine].Bcont] = 1;
                else
                    mFlag[contInfo[maxLine].Bcont] = 2;
            }
            else
            {
                // split on A
                mFlag[maxLine] = 2;
                if (CorD[useFlag - 1]) // on D (B)
                    mFlag[contInfo[maxLine].Acont] = 1;
                else
                    mFlag[contInfo[maxLine].Acont] = 2;
            }

            // temp
            useFlag = i;
            i = maxLine;

            // and add the line
            switch (mFlag[i])
            {
            case 1: //    *----(
                // we add one line
                outLineList[outNumLines] = outNumCorners;
                outNumLines++;
                // then copy the current line (reverse)
                for (j = lines[2][i] - 1; j >= 0; j--)
                {
                    outCornerList[outNumCorners] = inCornerList[inLineList[i] + j];
                    outNumCorners++;
                }

                // and add all following lines
                curCont = contInfo[i].Acont;
                curCase = contInfo[i].AcontCase;
                n = curCont;
                k = i;
                while (mFlag[n] == 3)
                {
                    // add this line (n)
                    if (curCase == 1 || curCase == 3) // CD
                    {
                        // as is
                        for (j = 0; j < lines[2][n]; j++)
                        {
                            outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                            outNumCorners++;
                        }
                        curCont = contInfo[n].Bcont;
                        curCase = contInfo[n].BcontCase;
                    }
                    else // DC
                    {
                        // reverse
                        for (j = lines[2][n] - 1; j >= 0; j--)
                        {
                            outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                            outNumCorners++;
                        }
                        curCont = contInfo[n].Acont;
                        curCase = contInfo[n].AcontCase;
                    }
                    // mark this one as used
                    mFlag[n] = 6;
                    // on to the next one
                    n = curCont;
                }

                // this is the tail of the line, so we will have to add it
                if (curCase == 1 || curCase == 3) // CD
                {
                    // as is
                    for (j = 0; j < lines[2][n]; j++)
                    {
                        outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                        outNumCorners++;
                    }
                    curCont = contInfo[n].Bcont;
                    curCase = contInfo[n].BcontCase;
                }
                else // DC
                {
                    // reverse
                    for (j = lines[2][n] - 1; j >= 0; j--)
                    {
                        outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                        outNumCorners++;
                    }
                    curCont = contInfo[n].Acont;
                    curCase = contInfo[n].AcontCase;
                }

                // make sure not to add this line twice (e.g. in reverse order)
                mFlag[n] = 7;
                break;
            case 2: //    )----*
                // we add one line
                outLineList[outNumLines] = outNumCorners;
                outNumLines++;
                // then copy the current line
                for (j = 0; j < lines[2][i]; j++)
                {
                    outCornerList[outNumCorners] = inCornerList[inLineList[i] + j];
                    outNumCorners++;
                }

                // and add all following lines
                curCont = contInfo[i].Bcont;
                curCase = contInfo[i].BcontCase;
                n = curCont;
                k = i;
                while (mFlag[n] == 3)
                {
                    // add this line (n)
                    if (curCase == 1 || curCase == 3) // CD
                    {
                        // as is
                        for (j = 0; j < lines[2][n]; j++)
                        {
                            outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                            outNumCorners++;
                        }
                        curCont = contInfo[n].Bcont;
                        curCase = contInfo[n].BcontCase;
                    }
                    else // DC
                    {
                        // reverse
                        for (j = lines[2][n] - 1; j >= 0; j--)
                        {
                            outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                            outNumCorners++;
                        }
                        curCont = contInfo[n].Acont;
                        curCase = contInfo[n].AcontCase;
                    }
                    // mark this one as used
                    mFlag[n] = 6;
                    // on to the next one
                    n = curCont;
                }

                // this is the tail of the line, so we will have to add it
                if (curCase == 1 || curCase == 3) // CD
                {
                    // as is
                    for (j = 0; j < lines[2][n]; j++)
                    {
                        outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                        outNumCorners++;
                    }
                    curCont = contInfo[n].Bcont;
                    curCase = contInfo[n].BcontCase;
                }
                else // DC
                {
                    // reverse
                    for (j = lines[2][n] - 1; j >= 0; j--)
                    {
                        outCornerList[outNumCorners] = inCornerList[inLineList[n] + j];
                        outNumCorners++;
                    }
                    curCont = contInfo[n].Acont;
                    curCase = contInfo[n].AcontCase;
                }

                // make sure not to add this line twice (e.g. in reverse order)
                mFlag[n] = 7;
                break;
            }

            // temp (undo)
            i = useFlag;
        }
    }

    // finally we can perform the reduction now
    // so we need a new lineList and cornerList
    int *redLineList, *redCornerList;
    int redNumLines, redNumCorners;
    redLineList = new int[outNumLines];
    redCornerList = new int[outNumCorners];
    redNumLines = 0;
    redNumCorners = 0;

    // again: speed-up (or at least: simplify loop-conditions ;-)))
    for (i = 0; i < outNumLines; i++)
    {
        lines[0][i] = outLineList[i];
        if (i == outNumLines - 1)
            lines[1][i] = outNumCorners;
        else
            lines[1][i] = outLineList[i + 1];
    }

    // GO !!!
    for (i = 0; i < outNumLines; i++)
    {
        A = outCornerList[lines[0][i]];
        B = A;
        C = A;
        D = A;
        redLineList[redNumLines] = redNumCorners;
        redNumLines++;
        redCornerList[redNumCorners] = A;
        redNumCorners++;
        minAngle = 0.0;
        for (n = lines[0][i] + 1; n < lines[1][i]; n++)
        {
            // continue the line
            B = outCornerList[n];

            // get angle between AB and CB
            curAngle = getAngle(A, B, C, B, inXCoord, inYCoord, inZCoord);
            if (curAngle >= maxAng)
            {
                // the angle is too big, so we have to keep point C
                if (C != A)
                {
                    redCornerList[redNumCorners] = C;
                    redNumCorners++;

                    // now we have to make sure that A is set to C so we get proper results
                    A = C;
                    D = C;
                    C = B;
                }
                else
                {
                    // seems as if we should keep all lines
                    redCornerList[redNumCorners] = B;
                    redNumCorners++;

                    // set A to B
                    A = B;
                    D = B;
                    C = B;
                }
            }
            else
            {
                // our second condition has to be fullfilled
                curAngle = getAngle(D, B, C, B, inXCoord, inYCoord, inZCoord);
                if (curAngle >= maxAng)
                {
                    // we have to keep this
                    redCornerList[redNumCorners] = C;
                    redNumCorners++;

                    // and restart from here
                    A = C;
                    D = C;
                    C = B;
                }
                else
                {
                    // continue
                    D = C;
                    C = B;
                }
            }
        }

        // make sure to allways keep the endpoint
        if (redCornerList[redNumCorners - 1] != B)
        {
            redCornerList[redNumCorners] = B;
            redNumCorners++;
        }
    }

    // we want the reduced stuff
    outNumLines = redNumLines;
    outNumCorners = redNumCorners;
    memcpy(outLineList, redLineList, redNumLines * sizeof(int));
    memcpy(outCornerList, redCornerList, redNumCorners * sizeof(int));

    // clean up the red. stuff
    delete[] redLineList;
    delete[] redCornerList;

    // clean up
    delete[] lines[0];
    delete[] lines[1];
    delete[] lines[2];
    delete[] mFlag;
    delete[] contInfo;

    // show some stats
    // char bfr[1024];
    // sprintf(bfr, "in:  lines: %d  corners: %d  coord: %d", inNumLines, inNumCorners, inNumCoord);
    // sendInfo(bfr);
    // sprintf(bfr, "out: lines: %d  corners: %d  coord: %d", outNumLines, outNumCorners, outNumCoord);
    // sendInfo(bfr);

    // done
    return;
}

float LineReduce::getAngle(int A, int B, int C, int D, int cs, float *XCoord, float *YCoord, float *ZCoord)
{
    float r = 4.0;
    switch (cs)
    {
    case 0:
        r = getAngle(A, B, C, D, XCoord, YCoord, ZCoord);
        break;
    case 1:
        r = getAngle(A, B, D, C, XCoord, YCoord, ZCoord);
        break;
    case 2:
        r = getAngle(B, A, C, D, XCoord, YCoord, ZCoord);
        break;
    case 3:
        r = getAngle(B, A, D, C, XCoord, YCoord, ZCoord);
        break;
    }
    return (r);
}

float LineReduce::getAngle(int A, int B, int C, int D, float *XCoord, float *YCoord, float *ZCoord)
{
    float ax, ay, az;
    float bx, by, bz;
    ax = XCoord[B] - XCoord[A];
    ay = YCoord[B] - YCoord[A];
    az = ZCoord[B] - ZCoord[A];
    bx = XCoord[D] - XCoord[C];
    by = YCoord[D] - YCoord[C];
    bz = ZCoord[D] - ZCoord[C];
    return (getAngle(ax, ay, az, bx, by, bz));
}

float LineReduce::getAngle(float ax, float ay, float az, float bx, float by, float bz)
{
    float al, bl, r;
    al = (float)(1.0 / sqrt(ax * ax + ay * ay + az * az));
    bl = (float)(1.0 / sqrt(bx * bx + by * by + bz * bz));
    r = al * bl * (ax * bx + ay * by + az * bz);

    if (r > 1.0)
        r = 1.0;
    else if (r < -1.0)
        r = -1.0;

    return (acos(r));
}

void LineReduce::loadTable()
{
    ///////////////////////////////////////////////
    // ABCD
    // (A) (B)
    loadTable2(0, 0, 0, 1, 2, 1); // (C) (D)   -->  (A) *B* *C* (D)
    loadTable2(0, 0, 1, 0, 0, 0); // *C* (D)
    loadTable2(0, 0, 2, 1, 2, 3); // (C) *D*   -->  (A) *B* *C* *D*
    loadTable2(0, 0, 3, 0, 0, 0); // *C* *D*
    // *A* (B)
    loadTable2(0, 1, 0, 1, 3, 1); // (C) (D)   -->  *A* *B* *C* (D)
    loadTable2(0, 1, 1, 0, 0, 0); // *C* (D)
    loadTable2(0, 1, 2, 1, 3, 3); // (C) *D*   -->  *A* *B* *C* *D*
    loadTable2(0, 1, 3, 0, 0, 0); // *C* *D*
    // (A) *B*
    loadTable2(0, 2, 0, 0, 0, 0); // (C) (D)
    loadTable2(0, 2, 1, 0, 0, 0); // *C* (D)
    loadTable2(0, 2, 2, 0, 0, 0); // (C) *D*
    loadTable2(0, 2, 3, 0, 0, 0); // *C* *D*
    // *A* *B*
    loadTable2(0, 3, 0, 0, 0, 0); // (C) (D)
    loadTable2(0, 3, 1, 0, 0, 0); // *C* (D)
    loadTable2(0, 3, 2, 0, 0, 0); // (C) *D*
    loadTable2(0, 3, 3, 0, 0, 0); // *C* *D*

    ///////////////////////////////////////////////
    // ABDC
    // (A) (B)
    loadTable2(1, 0, 0, 1, 2, 2); // (C) (D)   -->  (A) *B* (C) *D*
    loadTable2(1, 0, 1, 1, 2, 3); // *C* (D)   -->  (A) *B* *C* *D*
    loadTable2(1, 0, 2, 0, 0, 0); // (C) *D*
    loadTable2(1, 0, 3, 0, 0, 0); // *C* *D*
    // *A* (B)
    loadTable2(1, 1, 0, 1, 3, 2); // (C) (D)   -->  *A* *B* (C) *D*
    loadTable2(1, 1, 1, 1, 3, 3); // *C* (D)   -->  *A* *B* *C* *D*
    loadTable2(1, 1, 2, 0, 0, 0); // (C) *D*
    loadTable2(1, 1, 3, 0, 0, 0); // *C* *D*
    // (A) *B*
    loadTable2(1, 2, 0, 0, 0, 0); // (C) (D)
    loadTable2(1, 2, 1, 0, 0, 0); // *C* (D)
    loadTable2(1, 2, 2, 0, 0, 0); // (C) *D*
    loadTable2(1, 2, 3, 0, 0, 0); // *C* *D*
    // *A* *B*
    loadTable2(1, 3, 0, 0, 0, 0); // (C) (D)
    loadTable2(1, 3, 1, 0, 0, 0); // *C* (D)
    loadTable2(1, 3, 2, 0, 0, 0); // (C) *D*
    loadTable2(1, 3, 3, 0, 0, 0); // *C* *D*

    ///////////////////////////////////////////////
    // BACD
    // (A) (B)
    loadTable2(2, 0, 0, 1, 1, 1); // (C) (D)   -->  *A* (B) *C* (D)
    loadTable2(2, 0, 1, 0, 0, 0); // *C* (D)
    loadTable2(2, 0, 2, 1, 1, 3); // (C) *D*   -->  *A* (B) *C* *D*
    loadTable2(2, 0, 3, 0, 0, 0); // *C* *D*
    // *A* (B)
    loadTable2(2, 1, 0, 0, 0, 0); // (C) (D)
    loadTable2(2, 1, 1, 0, 0, 0); // *C* (D)
    loadTable2(2, 1, 2, 0, 0, 0); // (C) *D*
    loadTable2(2, 1, 3, 0, 0, 0); // *C* *D*
    // (A) *B*
    loadTable2(2, 2, 0, 1, 3, 1); // (C) (D)   -->  *A* *B* *C* (D)
    loadTable2(2, 2, 1, 0, 0, 0); // *C* (D)
    loadTable2(2, 2, 2, 1, 3, 3); // (C) *D*   -->  *A* *B* *C* *D*
    loadTable2(2, 2, 3, 0, 0, 0); // *C* *D*
    // *A* *B*
    loadTable2(2, 3, 0, 0, 0, 0); // (C) (D)
    loadTable2(2, 3, 1, 0, 0, 0); // *C* (D)
    loadTable2(2, 3, 2, 0, 0, 0); // (C) *D*
    loadTable2(2, 3, 3, 0, 0, 0); // *C* *D*

    ///////////////////////////////////////////////
    // BADC
    // (A) (B)
    loadTable2(3, 0, 0, 1, 1, 2); // (C) (D)   -->  *A* (B) (C) *D*
    loadTable2(3, 0, 1, 1, 1, 3); // *C* (D)   -->  *A* (B) *C* *D*
    loadTable2(3, 0, 2, 0, 0, 0); // (C) *D*
    loadTable2(3, 0, 3, 0, 0, 0); // *C* *D*
    // *A* (B)
    loadTable2(3, 1, 0, 0, 0, 0); // (C) (D)
    loadTable2(3, 1, 1, 0, 0, 0); // *C* (D)
    loadTable2(3, 1, 2, 0, 0, 0); // (C) *D*
    loadTable2(3, 1, 3, 0, 0, 0); // *C* *D*
    // (A) *B*
    loadTable2(3, 2, 0, 1, 3, 2); // (C) (D)   -->  *A* *B* (C) *D*
    loadTable2(3, 2, 1, 1, 3, 3); // *C* (D)   -->  *A* *B* *C* *D*
    loadTable2(3, 2, 2, 0, 0, 0); // (C) *D*
    loadTable2(3, 2, 3, 0, 0, 0); // *C* *D*
    // *A* *B*
    loadTable2(3, 3, 0, 0, 0, 0); // (C) (D)
    loadTable2(3, 3, 1, 0, 0, 0); // *C* (D)
    loadTable2(3, 3, 2, 0, 0, 0); // (C) *D*
    loadTable2(3, 3, 3, 0, 0, 0); // *C* *D*

    // done
    return;
}

void LineReduce::loadTable2(int c, int mC, int mT, int dM, char nC, char nT)
{
    caseTable[c][mC][mT].doMerge = dM;
    caseTable[c][mC][mT].newMCur = nC;
    caseTable[c][mC][mT].newMTest = nT;
    return;
}

// KM 15.01.02 don't remove point, if the difference in colour is too big
int LineReduce::getDifference(float A, float B, float C, float D)
{
    int iRet = 1; // difference is smaller than value maxDifference
    float fDiff = 0.0;

    fDiff = fabs(A - B);
    if (fDiff >= m_fDifference)
        iRet = 0;
    fDiff = fabs(A - D);
    if (fDiff >= m_fDifference)
        iRet = 0;
    fDiff = fabs(C - B);
    if (fDiff >= m_fDifference)
        iRet = 0;
    fDiff = fabs(C - D);
    if (fDiff >= m_fDifference)
        iRet = 0;

    return iRet;
}

MODULE_MAIN(Filter, LineReduce)
