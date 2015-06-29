/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                     (C)2001 Vircinity  **
 **                                                                        **
 ** Description:  COVISE TensScal  application module                      **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  S. Leseduarte                                                 **
 **                                                                        **
 **                                                                        **
 ** Date:  01.12.97  V1.0                                                  **
 ** Date   08.11.00                                                        **
\**************************************************************************/

#include "TensorScal.h"
#include "jacobi_eigenvalue.hpp"
#include <do/coDoData.h>
#include <do/coDoData.h>

# define N 3

TensScal::TensScal(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Calculate invariants from Tensors")
{
    const char *ChoiseVal[]  = { "Spur", "Stress effective value" ,
				 "1st-Principal", "2nd-Principal", "3rd-Principal", "Signed Max." };
    const char *VChoiseVal[] = { "1st-Principal", "2nd-Principal", "3rd-Principal", "Signed Max." };

    //parameters

    p_option = addChoiceParam("option", "Options");
    p_option->setValue(6, ChoiseVal, 1);

    p_voption = addChoiceParam("vector", "Vector Options");
    p_voption->setValue(4, VChoiseVal, 0);

    //ports
    p_inPort   = addInputPort("vdataIn", "Tensor", "input tensor data");
    p_outPort  = addOutputPort("sdataOut", "Float", "output scalar data");
    p_voutPort = addOutputPort("vdataOut", "Vector", "output vector data");
}

int TensScal::compute(const char *)
{

    coDoFloat *u_scalar_data = NULL;

    //int option = p_option->getValue();

    const coDistributedObject *obj = p_inPort->getCurrentObject();
    coDoVec3 *vdata=NULL;
    if (!obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort->getName());
        return FAIL;
    }
    else if (obj->isType("USTTDT"))
    {
        // unstructured tensor data
        const coDoTensor *tensor = (const coDoTensor *)(obj);
        coDoTensor::TensorType type = tensor->getTensorType();
        int option = p_option->getValue();
        int voption = p_voption->getValue();
        float *scalar = NULL;
        float *xvector = NULL;
        float *yvector = NULL;
        float *zvector = NULL;
        int nopoints = tensor->getNumPoints();
        float *t_addr = NULL;
        tensor->getAddress(&t_addr);

	//Allocate Output Port Vector Data Object *****************************
	vdata = new coDoVec3(p_voutPort->getObjName(),
				       nopoints);

	// if object was not properly allocated *******************************
	if (!vdata->objectOk())
	  {
	    sendError("Failed to create object '%s' for port '%s'",
		      p_voutPort->getObjName(), p_voutPort->getName());
	    return FAIL;
	  }

	vdata->getAddresses(&xvector,&yvector,&zvector);
	
        // it would be better to introduce new methods in the
        // tensor class, but now we may not change the libraries...
        switch (type)
        {
        case coDoTensor::S2D:
            switch (option)
            {
            case 0:
                scalar = S2D_Spur(nopoints, t_addr);
                break;
            case 1:
                scalar = S2D_Stress(nopoints, t_addr);
                break;
            case 2:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            case 3:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            case 4:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            case 5:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            default:
                break;
            }

	    sendWarning("Vector results will only be produced for tensor type S3D");

            break;
        case coDoTensor::F2D:
            switch (option)
            {
            case 0:
                scalar = F2D_Spur(nopoints, t_addr);
                break;
            case 1:
                sendWarning("Stress effective value only defined for symmetric tensors");
            case 2:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            case 3:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            case 4:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            case 5:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            default:
                break;

	    sendWarning("Vector results will only be produced for tensor type S3D");

            }
            break;
        case coDoTensor::S3D:
            switch (option)
            {
            case 0:
                scalar = S3D_Spur(nopoints, t_addr);
                break;
            case 1:
                scalar = S3D_Stress(nopoints, t_addr);
                break;
            case 2:
 	        scalar = S3D_Principal_Scalar(nopoints, t_addr, 2);
                break;
            case 3:
 	        scalar = S3D_Principal_Scalar(nopoints, t_addr, 3);
                break;
            case 4:
 	        scalar = S3D_Principal_Scalar(nopoints, t_addr, 4);
                break;
            case 5:
 	        scalar = S3D_Principal_Scalar(nopoints, t_addr, 5);
                break;

            default:
                break;
            }

	    S3D_Principal(nopoints, t_addr, voption, 
			  xvector, yvector, zvector);

            break;
        case coDoTensor::F3D:
            switch (option)
            {
            case 0:
                scalar = S3D_Spur(nopoints, t_addr);
                break;
            case 1:
                sendWarning("Stress effective value only defined for symmetric tensors");
                break;
            case 2:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            case 3:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            case 4:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            case 5:
                sendWarning("Principal results will only be produced for tensor type S3D");
                break;
            default:
                break;
            }

	    sendWarning("Vector results will only be produced for tensor type S3D");

            break;
        case coDoTensor::UNKNOWN:
        default:
            break;
        }

        if (scalar == NULL)
            return FAIL;

        // unstructured scalar data output
        u_scalar_data = new coDoFloat(p_outPort->getObjName(), nopoints, scalar);
        delete[] scalar;
        if (!u_scalar_data->objectOk())
        {
            sendError("Failed to create the object '%s' for the port '%s'", 
		      p_outPort->getObjName(), p_outPort->getName());
            return FAIL;
        }
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort->getName());
        return FAIL;
    }
    p_outPort->setCurrentObject(u_scalar_data);
    if(vdata)
    {
        p_voutPort->setCurrentObject(vdata);
    }
    
    return SUCCESS;
}

float *
TensScal::S2D_Spur(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = t_addr[point * 3];
        ret[point] += t_addr[point * 3 + 1];
        // ret[point] *= 0.5;
    }
    return ret;
}

float *
TensScal::S2D_Stress(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = sqrt((t_addr[point * 3] - t_addr[point * 3 + 1]) * 
			  (t_addr[point * 3] - t_addr[point * 3 + 1]) + 
			  4.0f * t_addr[point * 3 + 2] * t_addr[point * 3 + 2]);
    }
    return ret;
}

float *
TensScal::S3D_Spur(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = t_addr[point * 6];
        ret[point] += t_addr[point * 6 + 1];
        ret[point] += t_addr[point * 6 + 2];
        // ret[point] *= 0.33333333333333333;
    }
    return ret;
}

float *
TensScal::S3D_Stress(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        float spur = t_addr[point * 6] + t_addr[point * 6 + 1] + t_addr[point * 6 + 2];
        spur *= 0.3333333333333f;
        float contrib = t_addr[point * 6] - spur;
        ret[point] = contrib * contrib;
        contrib = t_addr[point * 6 + 1] - spur;
        ret[point] += contrib * contrib;
        contrib = t_addr[point * 6 + 2] - spur;
        ret[point] += contrib * contrib;
        contrib = t_addr[point * 6 + 3];
        ret[point] += 2.0f * contrib * contrib;
        contrib = t_addr[point * 6 + 4];
        ret[point] += 2.0f * contrib * contrib;
        contrib = t_addr[point * 6 + 5];
        ret[point] += 2.0f * contrib * contrib;

        ret[point] *= 1.5;

        ret[point] = sqrt(ret[point]);
    }
    return ret;
}

float *
TensScal::F2D_Spur(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = t_addr[point * 4];
        ret[point] += t_addr[point * 4 + 3];
        // ret[point] *= 0.5;
    }
    return ret;
}

float *
TensScal::F3D_Spur(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = t_addr[point * 9];
        ret[point] += t_addr[point * 9 + 4];
        ret[point] += t_addr[point * 9 + 8];
        // ret[point] *= 0.33333333333333333;
    }
    return ret;
}

void
TensScal::S3D_Principal(int nopoints, const float *t_addr, int voption,
			float *xv, float *yv, float *zv)
{

  int point;

  double a[N*N];
  double d[N];
  double error_frobenius;
  int it_max;
  int it_num;
  int n = N;
  int rot_num;
  double v[N*N];
  
  it_max = 100;

  for (point = 0; point < nopoints; ++point) {

    // ----------------- | +0  +1  +2  +3  +4  +5
    // S3D Tensor order  : XX, YY, ZZ, XY, YZ, ZX
    // -----------------------------------------
    //                   |  1   2   3
    //                   +-----------
    //                  1| +0  +3  +5
    // A sym. matrix :  1| +3  +1  +4
    //                  1| +5  +4  +2

    a[0] = t_addr[point * 6  ];
    a[1] = t_addr[point * 6+3];
    a[2] = t_addr[point * 6+5];
    a[3] = t_addr[point * 6+3];
    a[4] = t_addr[point * 6+1];
    a[5] = t_addr[point * 6+4];
    a[6] = t_addr[point * 6+5];
    a[7] = t_addr[point * 6+4];
    a[8] = t_addr[point * 6+2];
    
    jacobi_eigenvalue ( n, a, it_max, v, d, it_num, rot_num );

    // First Principal **************************
    if (voption == 0) {

      xv[point] = d[0] * v[0];
      yv[point] = d[0] * v[1];
      zv[point] = d[0] * v[2]; 

    // Second Principal *************************
    } else if (voption == 1) {

      xv[point] = d[1] * v[3];
      yv[point] = d[1] * v[4];
      zv[point] = d[1] * v[5]; 
    
    // Third Principal **************************
    } else if  (voption == 2) {

      xv[point] = d[2] * v[6];
      yv[point] = d[2] * v[7];
      zv[point] = d[2] * v[8]; 
    
    // Signed Max. Principal ********************
    } else if  (voption == 3) {

      // Third bigger than abs(first) ***********
      if ( fabs(d[2]) >= fabs(d[0]) ) {
	xv[point] = d[2] * v[6];
	yv[point] = d[2] * v[7];
	zv[point] = d[2] * v[8]; 

      // abs(First) bigger than third ***********
      } else {
	xv[point] = d[0] * v[0];
	yv[point] = d[0] * v[1];
	zv[point] = d[0] * v[2]; 
      }
    
    }

    
  }

  return;
}

float *
TensScal::S3D_Principal_Scalar(int nopoints, const float *t_addr, int option)
{

  int point;
  float *ret = new float[nopoints];
  double a[N*N];
  double d[N];
  double error_frobenius;
  int it_max;
  int it_num;
  int n = N;
  int rot_num;
  double v[N*N];
  
  it_max = 100;

  for (point = 0; point < nopoints; ++point) {

    // ----------------- | +0  +1  +2  +3  +4  +5
    // S3D Tensor order  : XX, YY, ZZ, XY, YZ, ZX
    // -----------------------------------------
    //                   |  1   2   3
    //                   +-----------
    //                  1| +0  +3  +5
    // A sym. matrix :  1| +3  +1  +4
    //                  1| +5  +4  +2

    a[0] = t_addr[point * 6  ];
    a[1] = t_addr[point * 6+3];
    a[2] = t_addr[point * 6+5];
    a[3] = t_addr[point * 6+3];
    a[4] = t_addr[point * 6+1];
    a[5] = t_addr[point * 6+4];
    a[6] = t_addr[point * 6+5];
    a[7] = t_addr[point * 6+4];
    a[8] = t_addr[point * 6+2];
    
    jacobi_eigenvalue ( n, a, it_max, v, d, it_num, rot_num );

    // First Principal **************************
    if (option == 2) {
      ret[point] = d[0];

    // Second Principal *************************
    } else if (option == 3) {
      ret[point] = d[1];
    
    // Third Principal **************************
    } else if  (option == 4) {
      ret[point] = d[2];
    
    // Signed Max. Principal ********************
    } else if  (option == 5) {

      // Third bigger than abs(first) ***********
      if ( fabs(d[2]) >= fabs(d[0]) ) {
	ret[point] = d[2];

      // abs(First) bigger than third ***********
      } else {
	ret[point] = d[0];

      }
    
    }

    
  }

  return ret;
}

MODULE_MAIN(Tools, TensScal)
