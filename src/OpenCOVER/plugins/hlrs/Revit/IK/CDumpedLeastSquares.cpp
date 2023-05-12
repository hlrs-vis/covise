#include "CDumpedLeastSquares.h"
#include "CJacobian.h"
#include <iostream>

CDumpedLeastSquares::CDumpedLeastSquares( IN VectorXf & desired_position , IN CRobot & robot ):
                                                                                                mtxinstance(CMatrixFactory::GetInstance()),
                                                                                                _desired_position(desired_position),
                                                                                                _robot(robot)
{
    current_position.resize(6);
}
template<typename _Matrix_Type_>
bool pseudoInverse(const _Matrix_Type_& a, _Matrix_Type_& result, double
	epsilon = std::numeric_limits<typename _Matrix_Type_::Scalar>::epsilon())
{
	if (a.rows() < a.cols())
		return false;
	Eigen::JacobiSVD< _Matrix_Type_ > svd = a.jacobiSvd(ComputeThinU | ComputeThinV);

	typename _Matrix_Type_::Scalar tolerance = epsilon * std::max(a.cols(),
		a.rows()) * svd.singularValues().array().abs().maxCoeff();

	result = svd.matrixV() * _Matrix_Type_((svd.singularValues().array().abs() >
		tolerance).select(svd.singularValues().
			array().inverse(), 0)).asDiagonal() * svd.matrixU().adjoint();
    return true;
 }

OUT VectorXf CDumpedLeastSquares::CalculateData()
{
    //Result vector
    VectorXf delta_theta(_robot.GiveMeMatrixHolder().size());
    //Vector for delta moving in space
    VectorXf delta_translation(6);

    VectorXf _temp = _desired_position;

    CJacobian * jac = CJacobian::GetInstance();
    //TODO
    //should be a parameter of function
    //Desired accuracy
    float epsilon = 0.1f;

    for (int it=0;it< 1000;it++)
    {
        jac->CalculateJacobian(_robot.GiveMeMatrixHolder(),_robot.GiveMeJoints(),_robot.GiveMeFullHM());

        float thetaX;
        float thetaY;
        float thetaZ;
        if (_robot.GiveMeFullHM()(2, 0) < +1)
        {
            if (_robot.GiveMeFullHM()(2, 0) > -1)
            {
                thetaY = asin(-_robot.GiveMeFullHM()(2, 0));
                thetaX = atan2(_robot.GiveMeFullHM()(1, 0), _robot.GiveMeFullHM()(0, 0));
                thetaZ = atan2(_robot.GiveMeFullHM()(2, 1), _robot.GiveMeFullHM()(2, 2));
            }
            else
            {
                thetaY = (float)(M_PI / 2.0);
                thetaX = -atan2(-_robot.GiveMeFullHM()(1, 2), _robot.GiveMeFullHM()(1, 1));
                thetaZ = 0;
            }
        }
        else//r02=+1
        {
            thetaY = (float)( -M_PI / 2);
            thetaX = atan2(-_robot.GiveMeFullHM()(1, 2), _robot.GiveMeFullHM()(1, 1));
            thetaZ = 0;
        }

        //calculation delta
        current_position << _robot.GiveMeFullHM()(0,3) ,    //X
            _robot.GiveMeFullHM()(1,3) ,    //Y
            _robot.GiveMeFullHM()(2,3) ,    //Z
            0,                           //Orientation
            0,                           //Orientation
            0;                           //Orientation
		



#ifdef JACOBIANDEBUGOUTPUT
        std::cout<<"Current position"<<std::endl<<current_position<<std::endl;
        std::cout<<"Desired position"<<std::endl<<_desired_position<<std::endl;
#endif

        delta_translation = _desired_position - current_position;

#ifdef JACOBIANDEBUGOUTPUT
        std::cout<<"Delta position"<<std::endl<<delta_translation<<std::endl;
#endif
        //compare delta with desired accuracy
        float n = delta_translation.norm();
        if (n < epsilon)
        {
            fprintf(stderr, "foundSolution\n");
            //Done
            break;
        }

        //Algorithm
        //TODO optimization needed

        MatrixXf one = jac->GetJacobian();
        MatrixXf two = one * jac->GetJacobian().transpose();
        MatrixXf id = MatrixXf::Identity(two.rows() , two.cols());
        id = (nu*nu) * id;

        MatrixXf result = two + id ;
        MatrixXf result_out;

        //JacobiSVD<MatrixXf> svd;
        //svd.compute(result , Eigen::ComputeThinU | Eigen::ComputeThinV);
        //was:         svd.pinv(result_out);
        //now:
        pseudoInverse(result, result_out);

#ifdef JACOBIANDEBUGOUTPUT
        std::cout<<"Result"<<std::endl<<result_out<<std::endl;
#endif

        result_out = jac->GetJacobian().transpose() * result_out;
        
        delta_theta =  result_out * delta_translation;

#ifdef JACOBIANDEBUGOUTPUT
        std::cout<<"Delta theta"<<std::endl<<delta_theta<<std::endl;
#endif
        UpdateJoints(delta_theta);
    }

    return _temp;
}

void CDumpedLeastSquares::UpdateJoints( VectorXf & delta_theta )
{
    JointHandler & _j = _robot.GiveMeJoints();

    for (int i = 0 ; i < delta_theta.size() ; i++)
    {
        //TODO ADD CONSTRAINTS
        //new var = old var + delta var
        float res = delta_theta[i] + _j[i].GiveMeVariableParametr();
        _robot.SetJointVariable(i,res);
    }
}