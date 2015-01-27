/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                           (C)2002 / 2003 VirCinity  ++
// ++ Description:                            ++
// ++             Implementation of class Refine                          ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "Refine.h"
#include <vector>
#include <values.h>

using namespace std;

inline float t(float x)
{
    if (x >= MAXFLOAT)
        return 0.;
    else
        return x;
}

//
// Constructor
//
Refine::Refine()
    : coSimpleModule("calculate refinement of a triangel mesh")
{
    surfPort_ = addInputPort("polyInPort", "coDoPolygons|Set_Polygons", "poly_in");
    inDataPort_ = addInputPort("dataInPort", "coDoFloat|coDoSet", "data_in");

    outSurfPort_ = addOutputPort("polyOut", "coDoPolygons|Set_Polygons", "out_polygons");
    outDataPort_ = addOutputPort("dataOut", "coDoFloat|coDoSet", "data_out");

    thresholdParam_ = addFloatSliderParam("refinement_threshold", "threshold for the refinement given in percent of the sqared gradient strenth");
    thresholdParam_->setValue(0.0, 100.0, 10.0);
}

//
// Method
//
int
Refine::compute()
{
    // receive data at ports
    coDistributedObject *surfObject = surfPort_->getCurrentObject();
    if (!surfObject)
    {
        sendError("DID NOT RECEIVE POLYGONS-OBJECT at Port %s", surfPort_->getName());
        return STOP_PIPELINE;
    }

    coDistributedObject *dataObject = inDataPort_->getCurrentObject();
    vector<float> dataIn;
    if (dataObject)
    {
        float *dat = NULL;
        coDoFloat *inData = (coDoFloat *)dataObject;
        int nDat = 0;
        inData->getAddress(&dat);
        nDat = inData->getNumPoints();
        copy(dat, dat + nDat, back_inserter(dataIn));
    }

    if (surfObject->isType("POLYGN"))
    {
        coDoPolygons *polygons = (coDoPolygons *)surfObject;
        // get dimensions
        int nPoints = polygons->getNumPoints();
        //int nCorners    = polygons->getNumVertices();
        int nPolygons = polygons->getNumPolygons();

        // create new arrays
        int *cl, *pl;
        float *coords[3];
        pl = NULL;
        cl = NULL;
        int i;
        for (i = 0; i < 3; ++i)
            coords[i] = NULL;

        polygons->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

        // loop over all triangs and partition each one
        float xc[3], yc[3], zc[3];

        vector<float> X(nPoints), Y(nPoints), Z(nPoints);
        copy(coords[0], coords[0] + nPoints, X.begin());
        copy(coords[1], coords[1] + nPoints, Y.begin());
        copy(coords[2], coords[2] + nPoints, Z.begin());

        vector<int> newConn;
        newConn.clear();

        vector<int> newPl;
        //int currPlIdx = 0;
        // if ( nPoints > 0 ) newPl.push_back(currPlIdx);

        int j, oIdx[3];

        // create a list of polygons to be refined
        vector<float> sqrGrd;
        vector<int> maxFloatCntList(nPolygons, 0);
        float gradientThreshold = 0.0;

        // 	// our polygon net may contain quads
        // T H E    F O L L O W I N G code may help
        // 	int multPoly=0;
        // 	int numCorn;
        // 	if (i<nPolygons-1) {
        // 	    numCorn = pl[i]-pl[i+1];
        // 	    triPass = numCorn - 2;
        // 	}
        // 	int pass=0;
        // 	for (pass=0; pass < triPass; pass++) {

        // 	}

        if (!dataIn.empty())
        {
            // calc maximal squared gradient for each triangle
            for (i = 0; i < nPolygons; ++i)
            {
                for (j = 0; j < 3; ++j)
                {
                    oIdx[j] = cl[pl[i] + j];
                    xc[j] = coords[0][oIdx[j]];
                    yc[j] = coords[1][oIdx[j]];
                    zc[j] = coords[2][oIdx[j]];
                }

                float dxsq = (xc[1] - xc[0]) * (xc[1] - xc[0]);
                float dysq = (yc[1] - yc[0]) * (yc[1] - yc[0]);
                float dzsq = (zc[1] - zc[0]) * (zc[1] - zc[0]);
                float dr1sq = dxsq + dysq + dzsq;

                dxsq = (xc[2] - xc[1]) * (xc[2] - xc[1]);
                dysq = (yc[2] - yc[1]) * (yc[2] - yc[1]);
                dzsq = (zc[2] - zc[1]) * (zc[2] - zc[1]);
                float dr2sq = dxsq + dysq + dzsq;

                dxsq = (xc[2] - xc[0]) * (xc[2] - xc[0]);
                dysq = (yc[2] - yc[0]) * (yc[2] - yc[0]);
                dzsq = (zc[2] - zc[0]) * (zc[2] - zc[0]);
                float dr3sq = dxsq + dysq + dzsq;

                float grd1 = dataIn[oIdx[1]] - dataIn[oIdx[0]];
                float grd2 = dataIn[oIdx[2]] - dataIn[oIdx[1]];
                float grd3 = dataIn[oIdx[0]] - dataIn[oIdx[2]];
                grd1 *= grd1;
                grd2 *= grd2;
                grd3 *= grd3;

                // TODO: check if dr==0 and set gradient to maxfloat in this case
                // find a solution for moulding simulations where unfilled cells
                // are marked by maxfloat
                grd1 /= dr1sq;
                grd2 /= dr2sq;
                grd3 /= dr3sq;

                int maxFloatCnt = 0;
                if (dataIn[oIdx[0]] >= MAXFLOAT)
                {
                    grd1 = 0.0;
                    maxFloatCnt++;
                }
                if (dataIn[oIdx[1]] >= MAXFLOAT)
                {
                    grd2 = 0.0;
                    maxFloatCnt++;
                }
                if (dataIn[oIdx[2]] >= MAXFLOAT)
                {
                    grd3 = 0.0;
                    maxFloatCnt++;
                }
                maxFloatCntList[i] = maxFloatCnt;

                float maxGrdSq = max(grd1, grd2);
                maxGrdSq = max(maxGrdSq, grd3);
                sqrGrd.push_back(maxGrdSq);
            }
            // find extremal squared gradient strength
            float maxGrdSq = *max_element(sqrGrd.begin(), sqrGrd.end());
            float minGrdSq = *min_element(sqrGrd.begin(), sqrGrd.end());

            float percentage = thresholdParam_->getValue() / 100.00;
            // we have to square percentage here
            gradientThreshold = minGrdSq + percentage * percentage * (maxGrdSq - minGrdSq);

            cerr << " max. gradient strength (per triangle) found : " << sqrt(maxGrdSq) << endl;
            cerr << " min. gradient strength (per triangle) found : " << sqrt(minGrdSq) << endl;
            cerr << " gradient threshold (per triangle)           : " << sqrt(gradientThreshold) << endl;
        }

        bool doRefine(true);
        for (i = 0; i < nPolygons; ++i)
        {
            if (!sqrGrd.empty())
            {
                if ((sqrGrd[i] > gradientThreshold)
                    || (maxFloatCntList[i] == 1)
                    || (maxFloatCntList[i] == 2))
                    doRefine = true;
                else
                    doRefine = false;
            }

            for (j = 0; j < 3; ++j)
            {
                oIdx[j] = cl[pl[i] + j];
                xc[j] = coords[0][oIdx[j]];
                yc[j] = coords[1][oIdx[j]];
                zc[j] = coords[2][oIdx[j]];
            }

            if (doRefine)
            {
                // edge midpoints
                X.push_back(xc[0] + 0.5 * (xc[1] - xc[0]));
                Y.push_back(yc[0] + 0.5 * (yc[1] - yc[0]));
                Z.push_back(zc[0] + 0.5 * (zc[1] - zc[0]));

                X.push_back(xc[1] + 0.5 * (xc[2] - xc[1]));
                Y.push_back(yc[1] + 0.5 * (yc[2] - yc[1]));
                Z.push_back(zc[1] + 0.5 * (zc[2] - zc[1]));

                X.push_back(xc[2] + 0.5 * (xc[0] - xc[2]));
                Y.push_back(yc[2] + 0.5 * (yc[0] - yc[2]));
                Z.push_back(zc[2] + 0.5 * (zc[0] - zc[2]));

                // data Interpolation is trivial here
                if (!dataIn.empty())
                {
                    dataIn.push_back(0.5 * (t(dataIn[oIdx[1]]) + t(dataIn[oIdx[0]])));
                    dataIn.push_back(0.5 * (t(dataIn[oIdx[2]]) + t(dataIn[oIdx[1]])));
                    dataIn.push_back(0.5 * (t(dataIn[oIdx[0]]) + t(dataIn[oIdx[2]])));
                }

                int startIdx = X.size() - 3;
                // ausgehend von den Koordinaten vollständig neue
                // Konnektivitätsliste bauen!!
                // tri 0 - n0 - n2
                newPl.push_back(newConn.size());
                newConn.push_back(oIdx[0]);
                newConn.push_back(startIdx);
                newConn.push_back(startIdx + 2);

                // tri n0 - 1 - n1
                newPl.push_back(newConn.size());
                newConn.push_back(startIdx);
                newConn.push_back(oIdx[1]);
                newConn.push_back(startIdx + 1);

                // tri n0 - n1 - n2
                newPl.push_back(newConn.size());
                newConn.push_back(startIdx);
                newConn.push_back(startIdx + 1);
                newConn.push_back(startIdx + 2);

                // tri n1 - 2 - n2
                newPl.push_back(newConn.size());
                newConn.push_back(startIdx + 1);
                newConn.push_back(oIdx[2]);
                newConn.push_back(startIdx + 2);
            }
            //no refinement needed
            else
            {
                newPl.push_back(newConn.size());
                newConn.push_back(oIdx[0]);
                newConn.push_back(oIdx[1]);
                newConn.push_back(oIdx[2]);
            }
        }

        // create out polygons
        int ll = strlen(outSurfPort_->getObjName());
        char *eleName = new char[ll + 20];
        sprintf(eleName, "%s", outSurfPort_->getObjName());

        /*coDoPolygons *outPoly = new coDoPolygons(eleName, X.size(), 
				  &X[0], &Y[0], &Z[0],
				  newConn.size(), &newConn[0], newPl.size(), &newPl[0]); */
        coDoPolygons *outPoly = new coDoPolygons(eleName, X.size(), newConn.size(), newPl.size());
        float *x, *y, *z;
        int *mpl, *mcl;
        outPoly->getAddresses(&x, &y, &z, &mcl, &mpl);
        for (i = 0; i < X.size(); i++)
        {
            x[i] = X[i];
            y[i] = Y[i];
            z[i] = Z[i];
        }
        for (i = 0; i < newConn.size(); i++)
            mcl[i] = newConn[i];
        for (i = 0; i < newPl.size(); i++)
            mpl[i] = newPl[i];

        outSurfPort_->setCurrentObject(outPoly);

        coDoFloat *outData;
        if (!dataIn.empty())
        {
            outData = new coDoFloat(outDataPort_->getObjName(),
                                    dataIn.size(),
                                    &dataIn[0]);
        }
        else
        {
            outData = new coDoFloat(outDataPort_->getObjName(), 0, NULL);
        }
        outDataPort_->setCurrentObject(outData);
    }
    else
    {
        sendError("NO POLYGONS-OBJECT at Port %s", surfPort_->getName());
    }
    return SUCCESS;
}

Refine::~Refine()
{
}

int main(int argc, char *argv[])
{
    Refine *application = new Refine;
    application->start(argc, argv);
    return 0;
}
