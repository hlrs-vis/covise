#ifndef __CJACOBIANPSEUDOINVERSE__
#define __CJACOBIANPSEUDOINVERSE__

#include "CAlgoAbstract.h"
#include "CMatrixFactory.h"
#include "CRobot.h"

class CJacobianPseudoInverse : public CAlgoAbstract
{
    CMatrixFactory * mtxinstance;
    VectorXf        & _desired_position;
    CRobot          & _robot;
    VectorXf          current_position;

public:
    CJacobianPseudoInverse(
                                IN VectorXf        & desired_position , 
                                IN CRobot          & robot
                           );

    OUT VectorXf CalculateData();
    void         SetAdditionalParametr(IN float & add_in){}
    void UpdateJoints(VectorXf & delta_theta);
};


#endif