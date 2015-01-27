/****************************************************************************
// Plugin: FeatureTracking
// Description: Estimation of camera pose from corresponding 2D image points
// Date: 2010-07-01
// Author: RTW
//***************************************************************************/

#include "EpipolarGeometry.h"

EpipolarGeometry::EpipolarGeometry()
    : correctMatchesRate(0)
{
    camIntr.focLen = 3.7;
    camIntr.imgCtrX = 0.0;
    camIntr.imgCtrY = 0.0;
    camIntr.skew = 0.0;
    points3DVec = new std::vector<osg::Vec3>;
    correctMatchesRate;
    isInitMode = true;
    debugMode = false;
}

EpipolarGeometry::~EpipolarGeometry()
{
}

void EpipolarGeometry::resetEpipolarGeo()
{
    camIntr.focLen = 3.7;
    camIntr.imgCtrX = 0.0;
    camIntr.imgCtrY = 0.0;
    camIntr.skew = 0.0;
    points3DVec->clear();
    (const int)correctMatchesRate = 0;
    isInitMode = true;
    debugMode = false;
}

bool EpipolarGeometry::findCameraTransformation(std::vector<CorrPoint> *inMatchesVec, TrackingObject *inTrackObj_R, TrackingObject *inTrackObj_C)
{
    // help structures
    CvMat *tmpMat33 = cvCreateMat(3, 3, CV_32FC1);
    CvMat *tmpMat34a = cvCreateMat(3, 4, CV_32FC1);
    CvMat *tmpMat34b = cvCreateMat(3, 4, CV_32FC1);
    CvMat *BP0 = cvCreateMat(3, 1, CV_32FC1);
    CvMat *BP1 = cvCreateMat(3, 1, CV_32FC1);

    // rotation matrices
    CvMat *rotMatA = cvCreateMat(3, 3, CV_32FC1);
    CvMat *rotMatB = cvCreateMat(3, 3, CV_32FC1);

    // translation vector (as matrix)
    CvMat *trlMatA = cvCreateMat(3, 1, CV_32FC1);
    CvMat *trlMatB = cvCreateMat(3, 1, CV_32FC1);

    // projection matrices
    CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
    CvMat *P1 = cvCreateMat(3, 4, CV_32FC1);

    // intrinsic camera matrix
    CvMat *intrMat = cvCreateMat(3, 3, CV_32FC1);
    cvmSet(intrMat, 0, 0, camIntr.focLen);
    cvmSet(intrMat, 0, 1, 0.0);
    cvmSet(intrMat, 0, 2, camIntr.imgCtrX);
    cvmSet(intrMat, 1, 0, 0.0);
    cvmSet(intrMat, 1, 1, camIntr.focLen);
    cvmSet(intrMat, 1, 2, camIntr.imgCtrY);
    cvmSet(intrMat, 2, 0, 0.0);
    cvmSet(intrMat, 2, 1, 0.0);
    cvmSet(intrMat, 2, 2, 1.0);

    // 3D point
    CvMat *pt3D = cvCreateMat(4, 1, CV_32FC1);

    // if enough matches available for pose estimation
    if (inMatchesVec->size() >= 8)
    {
        const int numberOfSamples = inMatchesVec->size();
        CvMat *firstPts = cvCreateMat(2, numberOfSamples, CV_32FC1);
        CvMat *secondPts = cvCreateMat(2, numberOfSamples, CV_32FC1);

        for (int e = 0; e < numberOfSamples; e++)
        {
            cvmSet(firstPts, 0, e, (*inMatchesVec)[e].getFirstX());
            cvmSet(firstPts, 1, e, (*inMatchesVec)[e].getFirstY());
            cvmSet(secondPts, 0, e, (*inMatchesVec)[e].getSecondX());
            cvmSet(secondPts, 1, e, (*inMatchesVec)[e].getSecondY());
        }

        if (isInitMode)
        {
            // projection matrices of the reference and the captured image
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    cvmSet(tmpMat34a, i, j, inTrackObj_R->getCameraPose()(j, i));
                    cvmSet(tmpMat34b, i, j, inTrackObj_C->getCameraPose()(j, i));
                }
            }
            cvMatMul(intrMat, tmpMat34a, P0);
            cvMatMul(intrMat, tmpMat34a, P1);

            // choose an image point for 3D reconstruction
            int ptId1;
            if (inMatchesVec->size() >= 1)
            {
                ptId1 = 1;
            }
            else
            {
                ptId1 = inMatchesVec->size();
            }

            CvMat *A = cvCreateMat(4, 4, CV_32FC1);
            CvMat *U = cvCreateMat(4, 4, CV_32FC1);
            CvMat *S = cvCreateMat(4, 4, CV_32FC1);
            CvMat *V = cvCreateMat(4, 4, CV_32FC1);

            // reconstruct a 3D point from the projection matrices of the left and right camera image
            for (int j = 0; j < 4; j++)
            {
                cvmSet(A, 0, j, (cvmGet(P0, 2, j) * cvmGet(firstPts, 0, ptId1)) - cvmGet(P0, 0, j));
                cvmSet(A, 1, j, (cvmGet(P0, 2, j) * cvmGet(firstPts, 1, ptId1)) - cvmGet(P0, 1, j));
                cvmSet(A, 2, j, (cvmGet(P1, 2, j) * cvmGet(secondPts, 0, ptId1)) - cvmGet(P1, 0, j));
                cvmSet(A, 3, j, (cvmGet(P1, 2, j) * cvmGet(secondPts, 1, ptId1)) - cvmGet(P1, 1, j));
            }
            cvSVD(A, S, U, V, CV_SVD_MODIFY_A + CV_SVD_V_T);
            cvmSet(pt3D, 0, 0, cvmGet(V, 0, 3));
            cvmSet(pt3D, 1, 0, cvmGet(V, 1, 3));
            cvmSet(pt3D, 2, 0, cvmGet(V, 2, 3));
            cvmSet(pt3D, 3, 0, cvmGet(V, 3, 3));

            osg::Vec3 point3D;
            point3D.set(cvmGet(V, 0, 3), cvmGet(V, 1, 3), cvmGet(V, 2, 3));
            points3DVec->push_back(point3D);
        }

        // calculate fundamental matrix
        CvMat *fundMat = cvCreateMat(3, 3, CV_32FC1);
        CvMat *status = cvCreateMat(1, numberOfSamples, CV_8UC1);

        if (cvFindFundamentalMat(firstPts, secondPts, fundMat, CV_FM_RANSAC, 1.0, 0.99, status) == 1)
        //if (cvFindFundamentalMat(firstPts,secondPts,fundMat,CV_FM_8POINT,1.0,0.99,status) == 1)
        {
            CvMat *tpIntrMat = cvCreateMat(3, 3, CV_32FC1);
            cvTranspose(intrMat, tpIntrMat);

            // calculate essential matrix
            CvMat *esseMat = cvCreateMat(3, 3, CV_32FC1);
            cvMatMul(fundMat, intrMat, tmpMat33);
            cvMatMul(tpIntrMat, tmpMat33, esseMat);

            // do singular value decomposition
            CvMat *U = cvCreateMat(3, 3, CV_32FC1);
            CvMat *S = cvCreateMat(3, 3, CV_32FC1);
            CvMat *V = cvCreateMat(3, 3, CV_32FC1);
            cvSVD(esseMat, S, U, V, CV_SVD_MODIFY_A + CV_SVD_V_T);
            //cvSVD(fundMat,S3,U3,V3,CV_SVD_MODIFY_A + CV_SVD_V_T);

            // gives number of correct matches using a tolerance factor
            //(const int) correctMatchesRate = evaluateEpipolarMat33(esseMat, normalizeCoords(inMatchesVec));

            // matrix for calculating rotations
            CvMat *W = cvCreateMat(3, 3, CV_32FC1);
            cvmSet(W, 0, 0, 0.0);
            cvmSet(W, 0, 1, -1.0);
            cvmSet(W, 0, 2, 0.0);
            cvmSet(W, 1, 0, 1.0);
            cvmSet(W, 1, 1, 0.0);
            cvmSet(W, 1, 2, 0.0);
            cvmSet(W, 2, 0, 0.0);
            cvmSet(W, 2, 1, 0.0);
            cvmSet(W, 2, 2, 1.0);

            // first possible rotation
            cvMatMul(W, V, tmpMat33);
            cvMatMul(U, tmpMat33, rotMatA);

            // second possible rotation
            CvMat *tpW = cvCreateMat(3, 3, CV_32FC1);
            cvTranspose(W, tpW);
            cvMatMul(tpW, V, tmpMat33);
            cvMatMul(U, tmpMat33, rotMatB);

            // first possible translation
            cvmSet(trlMatA, 0, 0, cvmGet(U, 0, 2));
            cvmSet(trlMatA, 1, 0, cvmGet(U, 1, 2));
            cvmSet(trlMatA, 2, 0, cvmGet(U, 2, 2));

            // second possible translation
            cvmSet(trlMatB, 0, 0, -(cvmGet(U, 0, 2)));
            cvmSet(trlMatB, 1, 0, -(cvmGet(U, 1, 2)));
            cvmSet(trlMatB, 2, 0, -(cvmGet(U, 2, 2)));

            if (isInitMode)
            {
                // offset projection matrix
                cvmSet(tmpMat34a, 0, 0, 1.0);
                cvmSet(tmpMat34a, 0, 1, 0.0);
                cvmSet(tmpMat34a, 0, 2, 0.0);
                cvmSet(tmpMat34a, 0, 3, 0.0);
                cvmSet(tmpMat34a, 1, 0, 0.0);
                cvmSet(tmpMat34a, 1, 1, 1.0);
                cvmSet(tmpMat34a, 1, 2, 0.0);
                cvmSet(tmpMat34a, 1, 3, 0.0);
                cvmSet(tmpMat34a, 2, 0, 0.0);
                cvmSet(tmpMat34a, 2, 1, 0.0);
                cvmSet(tmpMat34a, 2, 2, 1.0);
                cvmSet(tmpMat34a, 2, 3, 0.0);
                cvMatMul(intrMat, tmpMat34a, P0);

                // calculate all possible projection matrices for the capture image
                for (int i = 0; i < rotMatA->rows; i++)
                {
                    for (int j = 0; j < rotMatA->cols; j++)
                    {
                        cvmSet(tmpMat34a, i, j, cvmGet(rotMatA, i, j));
                        cvmSet(tmpMat34b, i, j, cvmGet(rotMatB, i, j));
                    }
                }
                cvmSet(tmpMat34a, 0, 3, cvmGet(trlMatA, 0, 0));
                cvmSet(tmpMat34a, 1, 3, cvmGet(trlMatA, 1, 0));
                cvmSet(tmpMat34a, 2, 3, cvmGet(trlMatA, 2, 0));
                CvMat *PA = cvCreateMat(3, 4, CV_32FC1);
                cvMatMul(intrMat, tmpMat34a, PA);

                // calculate projection matrix B
                cvmSet(tmpMat34a, 0, 3, -(cvmGet(trlMatB, 0, 0)));
                cvmSet(tmpMat34a, 1, 3, -(cvmGet(trlMatB, 1, 0)));
                cvmSet(tmpMat34a, 2, 3, -(cvmGet(trlMatB, 2, 0)));
                CvMat *PB = cvCreateMat(3, 4, CV_32FC1);
                cvMatMul(intrMat, tmpMat34a, PB);

                // calculate projection matrix C
                cvmSet(tmpMat34b, 0, 3, cvmGet(trlMatA, 0, 0));
                cvmSet(tmpMat34b, 1, 3, cvmGet(trlMatA, 1, 0));
                cvmSet(tmpMat34b, 2, 3, cvmGet(trlMatA, 2, 0));
                CvMat *PC = cvCreateMat(3, 4, CV_32FC1);
                cvMatMul(intrMat, tmpMat34b, PC);

                // calculate projection matrix D
                cvmSet(tmpMat34b, 0, 3, -(cvmGet(trlMatB, 0, 0)));
                cvmSet(tmpMat34b, 1, 3, -(cvmGet(trlMatB, 1, 0)));
                cvmSet(tmpMat34b, 2, 3, -(cvmGet(trlMatB, 2, 0)));
                CvMat *PD = cvCreateMat(3, 4, CV_32FC1);
                cvMatMul(intrMat, tmpMat34b, PD);

                // find out which of the projection matrices is the correct one
                float zSign = 0.0;
                // do projection of the 3D point calculated with all projection matrices
                camTransMat.makeIdentity();
                cvMatMul(P0, pt3D, BP0);

                // correct projection matrix is PA
                cvMatMul(PA, pt3D, BP1);
                zSign = cvmGet(BP0, 2, 0) * cvmGet(BP1, 2, 0);
                if (zSign > 0.0)
                {
                    projMat = 1;
                }
                else
                {
                    // correct projection matrix is PB
                    cvMatMul(PB, pt3D, BP1);
                    zSign = cvmGet(BP0, 2, 0) * cvmGet(BP1, 2, 0);
                    if (zSign > 0.0)
                    {
                        projMat = 2;
                    }
                    else
                    {
                        // correct projection matrix is PC
                        cvMatMul(PC, pt3D, BP1);
                        zSign = cvmGet(BP0, 2, 0) * cvmGet(BP1, 2, 0);
                        if (zSign > 0.0)
                        {
                            projMat = 3;
                        }
                        else
                        {
                            // correct projection matrix is PD
                            cvMatMul(PD, pt3D, BP1);
                            zSign = cvmGet(BP0, 2, 0) * cvmGet(BP1, 2, 0);
                            if (zSign > 0.0)
                            {
                                projMat = 4;
                            }
                            else
                            {
                                return false;
                            }
                        }
                    }
                }
            }
        } // endif (isInitMode)
        // set projection matrix for capture image
        if (projMat == 1)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    camTransMat(j, i) = cvmGet(rotMatA, i, j);
                }
            }
            camTransMat(3, 0) = cvmGet(trlMatA, 0, 0);
            camTransMat(3, 1) = cvmGet(trlMatA, 1, 0);
            camTransMat(3, 2) = cvmGet(trlMatA, 2, 0);
        }
        if (projMat == 2)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    camTransMat(j, i) = cvmGet(rotMatA, i, j);
                }
            }
            camTransMat(3, 0) = cvmGet(trlMatB, 0, 0);
            camTransMat(3, 1) = cvmGet(trlMatB, 1, 0);
            camTransMat(3, 2) = cvmGet(trlMatB, 2, 0);
        }
        if (projMat == 3)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    camTransMat(j, i) = cvmGet(rotMatB, i, j);
                }
            }
            camTransMat(3, 0) = cvmGet(trlMatA, 0, 0);
            camTransMat(3, 1) = cvmGet(trlMatA, 1, 0);
            camTransMat(3, 2) = cvmGet(trlMatA, 2, 0);
        }
        if (projMat == 4)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    camTransMat(j, i) = cvmGet(rotMatB, i, j);
                }
            }
            camTransMat(3, 0) = cvmGet(trlMatB, 0, 0);
            camTransMat(3, 1) = cvmGet(trlMatB, 1, 0);
            camTransMat(3, 2) = cvmGet(trlMatB, 2, 0);
        }
        else
        {
            return false;
        }
        return true;
    }
    // inMatchesVec contains less than 8 elements
    else
    {
        return false;
    }
}

float EpipolarGeometry::calcDistanceScale(std::vector<osg::Vec3> *in3DPoints)
{
    float scaleMedian = 0.0;
    osg::Vec3 distVec;
    int numOf3DPoints = in3DPoints->size();
    std::vector<float> scaleValue;

    for (int n = 1; n < numOf3DPoints; n++)
    {
        distVec = ((*in3DPoints)[n]) - ((*in3DPoints)[n - 1]);
        scaleValue[n - 1] = distVec.length();
    }

    for (int n = 1; n < scaleValue.size(); n++)
    {
        //############################
        // TODO: compute median here
        // else TODO
        //############################
    }
    return scaleMedian;
}

// Can be used for verification of the fundamental matrix calculation
std::vector<CorrPoint> *EpipolarGeometry::do2DTrafo(int inOp)
{
    // given 2D points (homogenous coordinates)
    CvMat firstPts_0 = cvMat(3, 1, CV_32FC1);
    firstPts_0.data.fl[0] = 0.0;
    firstPts_0.data.fl[1] = 0.0;
    firstPts_0.data.fl[2] = 1.0;
    CvMat firstPts_1 = cvMat(3, 1, CV_32FC1);
    firstPts_1.data.fl[0] = 1.0;
    firstPts_1.data.fl[1] = 0.0;
    firstPts_1.data.fl[2] = 1.0;
    CvMat firstPts_2 = cvMat(3, 1, CV_32FC1);
    firstPts_2.data.fl[0] = 2.0;
    firstPts_2.data.fl[1] = 0.0;
    firstPts_2.data.fl[2] = 1.0;
    CvMat firstPts_3 = cvMat(3, 1, CV_32FC1);
    firstPts_3.data.fl[0] = 2.0;
    firstPts_3.data.fl[1] = 1.0;
    firstPts_3.data.fl[2] = 1.0;
    CvMat firstPts_4 = cvMat(3, 1, CV_32FC1);
    firstPts_4.data.fl[0] = 1.0;
    firstPts_4.data.fl[1] = 1.0;
    firstPts_4.data.fl[2] = 1.0;
    CvMat firstPts_5 = cvMat(3, 1, CV_32FC1);
    firstPts_5.data.fl[0] = 1.0;
    firstPts_5.data.fl[1] = 3.0;
    firstPts_5.data.fl[2] = 1.0;
    CvMat firstPts_6 = cvMat(3, 1, CV_32FC1);
    firstPts_6.data.fl[0] = 0.0;
    firstPts_6.data.fl[1] = 3.0;
    firstPts_6.data.fl[2] = 1.0;
    CvMat firstPts_7 = cvMat(3, 1, CV_32FC1);
    firstPts_7.data.fl[0] = 0.0;
    firstPts_7.data.fl[1] = 1.0;
    firstPts_7.data.fl[2] = 1.0;

    // identity (homogeneous coordinates)
    CvMat idMat = cvMat(3, 3, CV_32FC1);
    idMat.data.fl[0] = 1.0;
    idMat.data.fl[1] = 0.0;
    idMat.data.fl[2] = 0.0;
    idMat.data.fl[3] = 0.0;
    idMat.data.fl[4] = 1.0;
    idMat.data.fl[5] = 0.0;
    idMat.data.fl[6] = 0.0;
    idMat.data.fl[7] = 0.0;
    idMat.data.fl[8] = 1.0;

    // rotation = 90 degrees (homogeneous coordinates)
    CvMat rotMat = cvMat(3, 3, CV_32FC1);
    rotMat.data.fl[0] = 0.0;
    rotMat.data.fl[1] = -1.0;
    rotMat.data.fl[2] = 0.0;
    rotMat.data.fl[3] = 1.0;
    rotMat.data.fl[4] = 0.0;
    rotMat.data.fl[5] = 0.0;
    rotMat.data.fl[6] = 0.0;
    rotMat.data.fl[7] = 0.0;
    rotMat.data.fl[8] = 1.0;

    // translation x-axis = -2 (homogeneous coordinates)
    CvMat trlMat = cvMat(3, 3, CV_32FC1);
    trlMat.data.fl[0] = 1.0;
    trlMat.data.fl[1] = 0.0;
    trlMat.data.fl[2] = -2.0;
    trlMat.data.fl[3] = 0.0;
    trlMat.data.fl[4] = 1.0;
    trlMat.data.fl[5] = 0.0;
    trlMat.data.fl[6] = 0.0;
    trlMat.data.fl[7] = 0.0;
    trlMat.data.fl[8] = 1.0;

    // transformation matrix in 2D space (homogeneous coordinates)
    CvMat D = cvMat(3, 3, CV_32FC1);

    // identity
    if (inOp == 0)
    {
        fprintf(stderr, "\n*** doing identity ...\n");
        D = idMat;
    }

    // rotation only
    if (inOp == 1)
    {
        fprintf(stderr, "\n*** doing rotation = 90 ...\n");
        D = rotMat;
    }

    // translation only
    if (inOp == 2)
    {
        fprintf(stderr, "\n*** doing translation x-axis = -2 ...\n");
        D = trlMat;
    }

    // rotation and then translation
    if (inOp == 3)
    {
        fprintf(stderr, "\n*** doing rotation = 90, translation x-axis = -2 ...\n");
        cvMatMul(&trlMat, &rotMat, &D);
    }

    if (debugMode)
    {
        fprintf(stderr, "\n*** D: \n");
        displayFloatMat(&D);
    }

    // computed points (homogeneous coordinates)
    CvMat p_0 = cvMat(3, 1, CV_32FC1);
    cvMatMul(&D, &firstPts_0, &p_0);
    CvMat p_1 = cvMat(3, 1, CV_32FC1);
    cvMatMul(&D, &firstPts_1, &p_1);
    CvMat p_2 = cvMat(3, 1, CV_32FC1);
    cvMatMul(&D, &firstPts_2, &p_2);
    CvMat p_3 = cvMat(3, 1, CV_32FC1);
    cvMatMul(&D, &firstPts_3, &p_3);
    CvMat p_4 = cvMat(3, 1, CV_32FC1);
    cvMatMul(&D, &firstPts_4, &p_4);
    CvMat p_5 = cvMat(3, 1, CV_32FC1);
    cvMatMul(&D, &firstPts_5, &p_5);
    CvMat p_6 = cvMat(3, 1, CV_32FC1);
    cvMatMul(&D, &firstPts_6, &p_6);
    CvMat p_7 = cvMat(3, 1, CV_32FC1);
    cvMatMul(&D, &firstPts_7, &p_7);

    /*if (debugMode) 
   {
      fprintf(stderr,"\n*** p_0: ");   fprintf(stderr,"(%f,%f)", p_0.data.fl[0], p_0.data.fl[1]);
      fprintf(stderr,"\n*** p_1: ");   fprintf(stderr,"(%f,%f)", p_1.data.fl[0], p_1.data.fl[1]);
      fprintf(stderr,"\n*** p_2: ");   fprintf(stderr,"(%f,%f)", p_2.data.fl[0], p_2.data.fl[1]);
      fprintf(stderr,"\n*** p_3: ");   fprintf(stderr,"(%f,%f)", p_3.data.fl[0], p_3.data.fl[1]);
      fprintf(stderr,"\n*** p_4: ");   fprintf(stderr,"(%f,%f)", p_4.data.fl[0], p_4.data.fl[1]);
      fprintf(stderr,"\n*** p_5: ");   fprintf(stderr,"(%f,%f)", p_5.data.fl[0], p_5.data.fl[1]);
      fprintf(stderr,"\n*** p_6: ");   fprintf(stderr,"(%f,%f)", p_6.data.fl[0], p_6.data.fl[1]);
      fprintf(stderr,"\n*** p_7: ");   fprintf(stderr,"(%f,%f)", p_7.data.fl[0], p_7.data.fl[1]);
      fprintf(stderr,"\n\n");
   }*/

    std::vector<CorrPoint> *point2DVector = new std::vector<CorrPoint>;
    CorrPoint match;

    match.setFirstX(firstPts_0.data.fl[0]);
    match.setFirstY(firstPts_0.data.fl[1]);
    match.setSecondX(p_0.data.fl[0]);
    match.setSecondY(p_0.data.fl[1]);
    point2DVector->push_back(match);
    match.setFirstX(firstPts_1.data.fl[0]);
    match.setFirstY(firstPts_1.data.fl[1]);
    match.setSecondX(p_1.data.fl[0]);
    match.setSecondY(p_1.data.fl[1]);
    point2DVector->push_back(match);
    match.setFirstX(firstPts_2.data.fl[0]);
    match.setFirstY(firstPts_2.data.fl[1]);
    match.setSecondX(p_2.data.fl[0]);
    match.setSecondY(p_2.data.fl[1]);
    point2DVector->push_back(match);
    match.setFirstX(firstPts_3.data.fl[0]);
    match.setFirstY(firstPts_3.data.fl[1]);
    match.setSecondX(p_3.data.fl[0]);
    match.setSecondY(p_3.data.fl[1]);
    point2DVector->push_back(match);
    match.setFirstX(firstPts_4.data.fl[0]);
    match.setFirstY(firstPts_4.data.fl[1]);
    match.setSecondX(p_4.data.fl[0]);
    match.setSecondY(p_4.data.fl[1]);
    point2DVector->push_back(match);
    match.setFirstX(firstPts_5.data.fl[0]);
    match.setFirstY(firstPts_5.data.fl[1]);
    match.setSecondX(p_5.data.fl[0]);
    match.setSecondY(p_5.data.fl[1]);
    point2DVector->push_back(match);
    match.setFirstX(firstPts_6.data.fl[0]);
    match.setFirstY(firstPts_6.data.fl[1]);
    match.setSecondX(p_6.data.fl[0]);
    match.setSecondY(p_6.data.fl[1]);
    point2DVector->push_back(match);
    match.setFirstX(firstPts_7.data.fl[0]);
    match.setFirstY(firstPts_7.data.fl[1]);
    match.setSecondX(p_7.data.fl[0]);
    match.setSecondY(p_7.data.fl[1]);
    point2DVector->push_back(match);

    return point2DVector;
}

int EpipolarGeometry::evaluateEpipolarMat33(CvMat *inMat, std::vector<CorrPoint> *inMatchesVec)
{
    CvMat *tVecFirst = cvCreateMat(3, 1, CV_32FC1);
    CvMat *tVecSecond = cvCreateMat(1, 3, CV_32FC1);
    CvMat *vecMat = cvCreateMat(3, 1, CV_32FC1);
    CvMat *resultMat = cvCreateMat(1, 1, CV_32FC1);
    int correctMatches = 0;

    for (int i = 0; i < inMatchesVec->size(); i++)
    {
        tVecFirst->data.fl[0] = inMatchesVec->at(i).getFirstX();
        tVecFirst->data.fl[1] = inMatchesVec->at(i).getFirstY();
        tVecFirst->data.fl[2] = 1.0;
        tVecSecond->data.fl[0] = inMatchesVec->at(i).getSecondX();
        tVecSecond->data.fl[1] = inMatchesVec->at(i).getSecondY();
        tVecSecond->data.fl[2] = 1.0;

        cvMatMul(inMat, tVecFirst, vecMat);
        cvMatMul(tVecSecond, vecMat, resultMat);

        /* get correct matches */
        if ((float)(abs(resultMat->data.fl[0])) < 0.1)
        {
            correctMatches++;
        }
        //if (debugMode) { displayFloatMat(resultMat); }
    }

    if (inMatchesVec->size() > 0)
    {
        return (correctMatches / inMatchesVec->size() * 100);
    }
    else
    {
        return 0;
    }
}

std::vector<CorrPoint> *EpipolarGeometry::normalizeCoords(std::vector<CorrPoint> *inMatchesVec)
{
    // center of gravity
    float cogPt1[2] = { 0.0, 0.0 };
    float cogPt2[2] = { 0.0, 0.0 };
    // summed distance */
    float distSum1[2] = { 0.0, 0.0 };
    float distSum2[2] = { 0.0, 0.0 };
    // scale factors
    float scaleFac1 = 0.0;
    float scaleFac2 = 0.0;
    const float numPts = inMatchesVec->size();
    // reciprocal of number of points
    const float reciNumPts = 1.0 / inMatchesVec->size();

    for (int i = 0; i < numPts; i++)
    {
        cogPt1[0] += inMatchesVec->at(i).getFirstX();
        cogPt1[1] += inMatchesVec->at(i).getFirstY();
        cogPt2[0] += inMatchesVec->at(i).getSecondX();
        cogPt2[1] += inMatchesVec->at(i).getSecondY();
        distSum1[0] += inMatchesVec->at(i).getFirstX() * inMatchesVec->at(i).getFirstX();
        distSum1[1] += inMatchesVec->at(i).getFirstY() * inMatchesVec->at(i).getFirstY();
        distSum2[0] += inMatchesVec->at(i).getSecondX() * inMatchesVec->at(i).getSecondX();
        distSum2[1] += inMatchesVec->at(i).getSecondY() * inMatchesVec->at(i).getSecondY();
    }
    cogPt1[0] *= reciNumPts;
    cogPt1[1] *= reciNumPts;
    cogPt2[0] *= reciNumPts;
    cogPt2[1] *= reciNumPts;
    scaleFac1 = reciNumPts * sqrt(distSum1[0] + distSum1[1] - numPts * (cogPt1[0] * cogPt1[0] + cogPt1[1] * cogPt1[1]));
    scaleFac2 = reciNumPts * sqrt(distSum2[0] + distSum2[1] - numPts * (cogPt2[0] * cogPt2[0] + cogPt2[1] * cogPt2[1]));

    scaleFac1 = sqrt(2.0) / scaleFac1;
    scaleFac2 = sqrt(2.0) / scaleFac2;

    std::vector<CorrPoint> *normVector = new std::vector<CorrPoint>;

    for (int i = 0; i < numPts; i++)
    {
        CorrPoint normPoint;
        normPoint.setFirstX((inMatchesVec->at(i).getFirstX() - cogPt1[0]) * scaleFac1);
        normPoint.setFirstY((inMatchesVec->at(i).getFirstY() - cogPt1[1]) * scaleFac1);
        normPoint.setSecondX((inMatchesVec->at(i).getSecondX() - cogPt2[0]) * scaleFac2);
        normPoint.setSecondY((inMatchesVec->at(i).getSecondY() - cogPt2[1]) * scaleFac2);
        normVector->push_back(normPoint);
    }
    return normVector;
}

void EpipolarGeometry::displayFloatMat(CvMat *inMat)
{
    fprintf(stderr, " \n");
    for (int i = 0; i < inMat->rows; i++)
    {
        for (int j = 0; j < inMat->cols; j++)
        {
            fprintf(stderr, " %f ", cvmGet(inMat, i, j));
        }
        fprintf(stderr, " \n");
    }
}

void EpipolarGeometry::setFocalLength(float inFocalLen)
{
    camIntr.focLen = inFocalLen;
}

void EpipolarGeometry::setImageCenter(float inImgCtrX, float inImgCtrY)
{
    camIntr.imgCtrX = inImgCtrX;
    camIntr.imgCtrY = inImgCtrY;
}

void EpipolarGeometry::setSkewParameter(float inSkew)
{
    camIntr.skew = inSkew;
}

void EpipolarGeometry::setInitMode(bool inMode)
{
    isInitMode = inMode;
}

void EpipolarGeometry::setDebugMode(bool inMode)
{
    debugMode = inMode;
}

osg::Matrix EpipolarGeometry::getCameraTransform()
{
    return camTransMat;
}

float EpipolarGeometry::getCameraFocalLength()
{
    return camIntr.focLen;
}

float EpipolarGeometry::getCameraImageCenterX()
{
    return camIntr.imgCtrX;
}

float EpipolarGeometry::getCameraImageCenterY()
{
    return camIntr.imgCtrY;
}

float EpipolarGeometry::getSkewParameter()
{
    return camIntr.skew;
}

const int EpipolarGeometry::getRateOfCorrectMatches()
{
    return (const int)correctMatchesRate;
}

// EOF
