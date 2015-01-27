/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __WRITEITKSLICES_H
#define __WRITEITKSLICES_H

/*=========================================================================
 *   Program:   Covise
 *   Module:    WriteITKSlices
 *   Language:  C++
 *   Date:      $Date: 2008/07/14 11:00:00 $
 *   Version:   $Revision:  1$
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

class WriteITKSlices : public coModule
{
private:
    // ports
    coInputPort *piGrid;

    coInputPort *piRGBR;
    coInputPort *piRGBG;
    coInputPort *piRGBB;

    // Parameter & IO:
    coFileBrowserParam *outImgfile;
    coChoiceParam *pFormatChoice;

public:
    WriteITKSlices(int argc, char *argv[]);
    virtual ~WriteITKSlices();

    // main-callback
    virtual int compute(const char *port);

    typedef itk::RGBPixel<unsigned char> RGBPixelType;
    typedef itk::Image<RGBPixelType, 2> ColorImage2DType;
    typedef itk::Image<RGBPixelType, 3> ColorImage3DType;

    typedef itk::ImageRegionIteratorWithIndex<ColorImage3DType> IteratorType;

    int convertCOVISEtoITK(WriteITKSlices::ColorImage3DType::Pointer *ITKGrid);
    itk::ImageIOBase::Pointer imageIO(int filename);
};

#endif // __WRITEITKSLICES_H
