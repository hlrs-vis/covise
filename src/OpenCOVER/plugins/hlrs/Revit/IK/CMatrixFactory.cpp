#include "CMatrixFactory.h"
#include <iostream>

Matrix4f CMatrixFactory::CalculateHTranslationMatrix(
                                                        float alpha,
                                                        float a,
                                                        float d,
                                                        float theta
                                                     )
{
    Matrix4f mat_tr;

    //First row
    float r11 = 
        static_cast<float>(cos(CONVERT_TO_RAD(theta)));
    float r12 = 
        static_cast<float>(
                           -sin(CONVERT_TO_RAD(theta))*cos(CONVERT_TO_RAD(alpha))
                          );
    float r13 = 
        static_cast<float>(
                           sin(CONVERT_TO_RAD(theta))*sin(CONVERT_TO_RAD(alpha))
                          );
    float r14 = 
        static_cast<float>(
                           a*cos(CONVERT_TO_RAD(theta))
                          );
    //Second row
    float r21 = 
        static_cast<float>(
                           sin(CONVERT_TO_RAD(theta))
                          );
    float r22 = 
        static_cast<float>(
                           cos(CONVERT_TO_RAD(theta))*cos(CONVERT_TO_RAD(alpha))
                          );
    float r23 = 
        static_cast<float>(
                           -cos(CONVERT_TO_RAD(theta))*sin(CONVERT_TO_RAD(alpha))
                          );
    float r24 = 
        static_cast<float>(
                           a*sin(CONVERT_TO_RAD(theta))
                          );
    //Third row
    float r31 = 0.0f;
    float r32 = 
        static_cast<float>(
                           sin(CONVERT_TO_RAD(alpha))
                          );
    float r33 = 
        static_cast<float>(
                           cos(CONVERT_TO_RAD(alpha))
                          );
    float r34 = d;
    //Forth row
    float r41 = 0.0f;
    float r42 = 0.0f;;
    float r43 = 0.0f;;
    float r44 = 1.0f;;

    mat_tr <<   r11 , r12 , r13 , r14 ,
                r21 , r22 , r23 , r24 , 
                r31 , r32 , r33 , r34 , 
                r41 , r42 , r43 , r44 ;

    return mat_tr;
}

CMatrixFactory * CMatrixFactory::_instance = NULL;

CMatrixFactory * CMatrixFactory::GetInstance()
{
    if(!_instance)
        _instance = new CMatrixFactory();

    return _instance;
}

Matrix4f CMatrixFactory::CreateRotationMatrixAroundZ( float alpha )
{
    Matrix4f _temp;

    //First row
    float r11 = static_cast<float>(
                                    cos(CONVERT_TO_RAD(alpha))
                                   );
    float r12 = static_cast<float>(
                                    -sin(CONVERT_TO_RAD(alpha))
                                   );
    float r13 = 0.0f;
    float r14 = 0.0f;
    //Second row
    float r21 = static_cast<float>(
                                    sin(CONVERT_TO_RAD(alpha))
                                   );
    float r22 = static_cast<float>(
                                    cos(CONVERT_TO_RAD(alpha))
                                   );
    float r23 = 0.0f;
    float r24 = 0.0f;
    //Third row
    float r31 = 0.0f;
    float r32 = 0.0f;
    float r33 = 1.0f;
    float r34 = 0.0f;
    //Forth row
    float r41 = 0.0f;
    float r42 = 0.0f;
    float r43 = 0.0f;
    float r44 = 0.0f;

    _temp << r11 , r12 , r13 , r14 ,
             r21 , r22 , r23 , r24 ,
             r31 , r32 , r33 , r34 , 
             r41 , r42 , r43 , r44 ;

    return _temp; 
}

Matrix4f CMatrixFactory::CreateRotationMatrixAroundY( float beta )
{
    Matrix4f _temp;

    //First row
    float r11 = static_cast<float>(
                                    cos(CONVERT_TO_RAD(beta))
                                   );
    float r12 = 0.0f;
    float r13 = static_cast<float>(
                                    sin(CONVERT_TO_RAD(beta))
                                   );
    float r14 = 0.0f;
    //Second row
    float r21 = 0.0f;
    float r22 = 1.0f;
    float r23 = 0.0f;
    float r24 = 0.0f;
    //Third row
    float r31 = static_cast<float>(
                                    -sin(CONVERT_TO_RAD(beta))
                                   );
    float r32 = 0.0f;
    float r33 = static_cast<float>(
                                    cos(CONVERT_TO_RAD(beta))
                                   );
    float r34 = 0.0f;
    //Forth row
    float r41 = 0.0f;
    float r42 = 0.0f;
    float r43 = 0.0f;
    float r44 = 0.0f;

    _temp << r11 , r12 , r13 , r14 ,
             r21 , r22 , r23 , r24 ,
             r31 , r32 , r33 , r34 , 
             r41 , r42 , r43 , r44 ;

    return _temp;
}

Matrix4f CMatrixFactory::CreateRotationMatrixAroundX( float gamma )
{
    Matrix4f _temp;

    //First row
    float r11 = 1.0f;
    float r12 = 0.0f;
    float r13 = 0.0f;
    float r14 = 0.0f;
    //Second row
    float r21 = 0.0f;
    float r22 = static_cast<float>(
                                    cos(CONVERT_TO_RAD(gamma))
                                   );
    float r23 = static_cast<float>(
                                    -sin(CONVERT_TO_RAD(gamma))
                                   );
    float r24 = 0.0f;
    //Third row
    float r31 = 0.0f;
    float r32 = static_cast<float>(
                                    sin(CONVERT_TO_RAD(gamma))
                                   );
    float r33 = static_cast<float>(
                                    cos(CONVERT_TO_RAD(gamma))
                                   );
    float r34 = 0.0f;
    //Forth row
    float r41 = 0.0f;
    float r42 = 0.0f;
    float r43 = 0.0f;
    float r44 = 0.0f;

    _temp << r11 , r12 , r13 , r14 ,
             r21 , r22 , r23 , r24 ,
             r31 , r32 , r33 , r34 , 
             r41 , r42 , r43 , r44 ;

    return _temp;
}

Matrix4f CMatrixFactory::CreateTranslateMatrixX( float dx )
{
    Matrix4f _temp = Matrix4f::Identity();  

    _temp(0,3) = dx;

    return _temp;
}

Matrix4f CMatrixFactory::CreateTranslateMatrixY( float dy )
{
    Matrix4f _temp = Matrix4f::Identity();  

    _temp(1,3) = dy;

    return _temp;
}

Matrix4f CMatrixFactory::CreateTranslateMatrixZ( float dz )
{
    Matrix4f _temp = Matrix4f::Identity();  

    _temp(2,3) = dz;

    return _temp;
}

Matrix3f CMatrixFactory::ExtractRotationMatrix( Matrix4f & hm )
{
    return hm.block(0,0,3,3);
}

Vector3f CMatrixFactory::ExtractTranslationVector( Matrix4f & hm )
{
    Vector3f vec;

    vec(0) = hm(0,3);
    vec(1) = hm(1,3);
    vec(2) = hm(2,3);

    return vec;
}

Vector3f CMatrixFactory::MultiplyVectors( Vector3f & vec_one , Vector3f & vec_two )
{
    Vector3f vec_res;
    //Equation [A,B] = (Ay*Bz - Az*By , AzBx - Ax*Bz , AxBy - AyBx)
    vec_res(0)  = vec_one(1)*vec_two(2) - vec_one(2)*vec_two(1);
    vec_res(1)  = vec_one(2)*vec_two(0) - vec_one(0)*vec_two(2);
    vec_res(2)  = vec_one(0)*vec_two(1) - vec_one(1)*vec_two(0);

#ifdef JACOBIANDEBUGOUTPUT
    std::cout<<"k*d"<<std::endl<<vec_res<<std::endl;
#endif

    return vec_res;
}

OUT float CMatrixFactory::GetLengthOfVector( IN VectorXf & invec )
{
    float res = static_cast<float>(
                                        sqrt(invec(0)*invec(0) + invec(1)*invec(1) + invec(2)*invec(2))
                                   );

    return res;
}


