#include "CAlgoFactory.h"
#include "CJacobianTranspose.h"
#include "CJacobianPseudoInverse.h"
#include "CDumpedLeastSquares.h"

CAlgoAbstract * CAlgoFactory::GiveMeSolver(
                                                IN AlgType atype,
                                                IN VectorXf        & desired_position , 
                                                IN CRobot          & ptrRobot
                                           )
{
    CAlgoAbstract * ptr = NULL;

    switch(atype)
    {
    case JACOBIANTRANSPOSE:
        ptr = new CJacobianTranspose(desired_position , ptrRobot);
        break;
    case JACOBIANPSEVDOINVERSE:
        ptr = new CJacobianPseudoInverse(desired_position , ptrRobot);
        break;
    case DUMPEDLEASTSQUARES:
        ptr = new CDumpedLeastSquares(desired_position , ptrRobot);
        break;
    case SELECTIVEDUMPEDLEASTSQUARES:
        //TODO
        break;
    }

    return ptr;
}