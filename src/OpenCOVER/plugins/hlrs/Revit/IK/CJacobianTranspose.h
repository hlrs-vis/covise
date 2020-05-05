#ifndef __JACOBIANTRANSPOSE__
#define __JACOBIANTRANSPOSE__

#include "CAlgoAbstract.h"
#include "CMatrixFactory.h"

class CJacobianTranspose : public CAlgoAbstract
{
    CMatrixFactory *  mtxinstance;
    float             lamda_coefficent;
    VectorXf        & _desired_position;
    VectorXf          current_position;
    CRobot          & _robot;

public:
    CJacobianTranspose(
                            IN VectorXf        & desired_position , 
                            IN CRobot          & robot
                       );

    OUT VectorXf CalculateData();
    void         SetAdditionalParametr(IN float & add_in);
    void UpdateJoints(VectorXf & delta_theta);
};

#endif