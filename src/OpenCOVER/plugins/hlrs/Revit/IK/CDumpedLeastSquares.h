#ifndef __CDUMPLEASTSQUARES__
#define __CDUMPLEASTSQUARES__

#include "CAlgoAbstract.h"
#include "CMatrixFactory.h"
#include "CRobot.h"

class CDumpedLeastSquares : public CAlgoAbstract
{
    CMatrixFactory * mtxinstance;
    VectorXf        & _desired_position;
    CRobot          & _robot;
    VectorXf          current_position;
    float             nu;
public:
    CDumpedLeastSquares(
                            IN VectorXf        & desired_position , 
                            IN CRobot          & robot
                        );
    OUT VectorXf CalculateData();
    void         SetAdditionalParametr(IN float & add_in){ nu = add_in;}
    void UpdateJoints(VectorXf & delta_theta);
};

#endif