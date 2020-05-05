#ifndef __CMATRIXFACTORY__
#define __CMATRIXFACTORY__

/************************************************************************/
/* CMatrixFactory                                                       */
/************************************************************************/

#include "SharedTypes.h"

#define PI 3.14159265
#define CONVERT_TO_RAD(x) x*(PI/180)

class CMatrixFactory
{
    CMatrixFactory() {}

    static CMatrixFactory * _instance;   

public:

/************************************************************************/
/* Rotation matrix                                                      */
/************************************************************************/

    OUT Matrix4f CreateRotationMatrixAroundZ( IN float alpha );
    OUT Matrix4f CreateRotationMatrixAroundY( IN float beta );
    OUT Matrix4f CreateRotationMatrixAroundX( IN float gamma );


/************************************************************************/
/* Translation matrix                                                   */
/************************************************************************/
    
    OUT Matrix4f CreateTranslateMatrixX( IN float dx );
    OUT Matrix4f CreateTranslateMatrixY( IN float dy );    
    OUT Matrix4f CreateTranslateMatrixZ( IN float dz );    

/************************************************************************/
/* Transformation matrix from frame i to i-1                            */
/************************************************************************/

    OUT Matrix4f CalculateHTranslationMatrix(
                                                IN float alpha,
                                                IN float a,
                                                IN float d,
                                                IN float theta
                                            );
/************************************************************************/
/* Help functions                                                       */
/************************************************************************/
    OUT Matrix3f ExtractRotationMatrix( IN Matrix4f & hm );
    OUT Vector3f ExtractTranslationVector ( IN Matrix4f & hm );
    OUT Vector3f MultiplyVectors(IN Vector3f & vec_one , IN Vector3f & vec_two);
    OUT float    GetLengthOfVector(IN VectorXf & invec);
/************************************************************************/
/* Full calculation algo                                                */
/************************************************************************/
    

    static CMatrixFactory * GetInstance();
    
};

#endif