#ifndef __CALGOABSTRACT__
#define __CALGOABSTRACT__

#include "SharedTypes.h"
#include "RobotTypes.h"
#include "CRobot.h"

enum AlgType
{
    JACOBIANTRANSPOSE,
    JACOBIANPSEVDOINVERSE,
    DUMPEDLEASTSQUARES,
    SELECTIVEDUMPEDLEASTSQUARES,
    CCD
};

//Abstarct class for all algorithm
class CAlgoAbstract
{
public:
             CAlgoAbstract(){}
    virtual ~CAlgoAbstract(){}
    virtual OUT VectorXf CalculateData() = 0;
    virtual void         SetAdditionalParametr(IN float & add_in) = 0;
};

#endif