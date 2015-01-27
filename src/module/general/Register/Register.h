/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __REGISTER_H
#define __REGISTER_H

/*=========================================================================
 *   Program:   Covise
 *   Module:    Register
 *   Language:  C++
 *   Date:      $Date: 2007/09/07 14:17:42 $
 *   Version:   $Revision:  $
 *=========================================================================*/

#include <api/coModule.h>
using namespace covise;
#include <do/coDoUniformGrid.h>
#include <itkCommand.h>
//ITK Image
#include <itkImage.h>
#include <itkImageIOBase.h>
#include <itkRGBPixel.h>
//Registration
#include <itkMultiResolutionImageRegistrationMethod.h>
#include <itkRegularStepGradientDescentOptimizer.h>
//Transforms
#include <itkTranslationTransform.h>
#include <itkCenteredRigid2DTransform.h>
//Iterators
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIteratorWithIndex.h>

using namespace std;

class Register : public coModule
{
private:
    // ports
    coInputPort *piGrid;

    coInputPort *piRGBR;
    coInputPort *piRGBG;
    coInputPort *piRGBB;

    coOutputPort *poGrid;

    coOutputPort *poRGBR;
    coOutputPort *poRGBG;
    coOutputPort *poRGBB;

    // Parameter & IO:
    coFileBrowserParam *pbrImageFiles;
    coIntScalarParam *piSequenceBegin;
    coIntScalarParam *piSequenceEnd;
    coIntScalarParam *piSequenceInc;
    coBooleanParam *pboCustomOutput;
    coFileBrowserParam *outImgfile;
    coChoiceParam *pFormatChoice;
    //StepLength
    coFloatParam *pfsMaxStepLength;
    coFloatParam *pfsMinStepLength;
    coFloatParam *pfsMinStepLengthDivisor;

    //Registration parameters
    coIntScalarParam *piFillColor;
    coIntScalarParam *piIterations;
    coIntScalarParam *piPyramid;

    //Method parameters
    coChoiceParam *pMetricChoice;
    coChoiceParam *pOptimizerChoice;

public:
    Register(int argc, char *argv[]);
    virtual ~Register();

    // main-callback
    virtual int compute(const char *port);

    typedef unsigned char GrayScalePixelType;
    typedef itk::RGBPixel<unsigned char> RGBPixelType;
    typedef float InternalPixelType;

    typedef itk::Image<RGBPixelType, 2> ColorImage2DType;
    typedef itk::Image<RGBPixelType, 3> ColorImage3DType;
    typedef itk::Image<GrayScalePixelType, 2> GrayScaleImage2DType;
    typedef itk::Image<InternalPixelType, 2> InternalImageType;

    typedef itk::ImageRegionConstIteratorWithIndex<ColorImage2DType> ConstIteratorType;
    typedef itk::ImageRegionIteratorWithIndex<ColorImage3DType> IteratorType;

    coDoUniformGrid *convertITKtoCOVISE(const ColorImage3DType::Pointer itkVolume);
    int convertCOVISEtoITK(Register::ColorImage3DType::Pointer *ITKGrid);
};

#endif // __Register_H
