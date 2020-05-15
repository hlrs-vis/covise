#ifndef __CALGOFACTORY__
#define __CALGOFACTORY__

#include "CAlgoAbstract.h"
#include "CRobot.h"

class CAlgoFactory
{
public:
    CAlgoFactory(){}
    CAlgoAbstract * GiveMeSolver(
                                    IN AlgType atype,
                                    IN VectorXf        & desired_position , 
                                    IN CRobot          & ptrRobot
                                 );
};

#endif