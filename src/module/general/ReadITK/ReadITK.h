/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __READITK_H
#define __READITK_H

/*=========================================================================
 *   Program:   Covise
 *   Module:    ReadITK
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
//Iterators
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIteratorWithIndex.h>

using namespace std;

class ReadITK : public coModule
{
private:
    // ports
    coOutputPort *poGrid;

    coOutputPort *poRGBR;
    coOutputPort *poRGBG;
    coOutputPort *poRGBB;

    // Parameter & IO:
    coFileBrowserParam *pbrImageFiles;
    coIntScalarParam *piSequenceBegin;
    coIntScalarParam *piSequenceEnd;
    coIntScalarParam *piSequenceInc;

    //Voxels:
    coBooleanParam *pboCustomSize;
    coFloatParam *pfsVoxWidth;
    coFloatParam *pfsVoxHeight;
    coFloatParam *pfsVoxDepth;

    //ImageOrigin
    coBooleanParam *pboCustomOrigin;
    coFloatParam *pfsOriginX;
    coFloatParam *pfsOriginY;
    coFloatParam *pfsOriginZ;

public:
    ReadITK(int argc, char *argv[]);
    virtual ~ReadITK();

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

    coDoUniformGrid *convertITKtoCovise(const ColorImage3DType::Pointer itkVolume);
    itk::ImageIOBase::Pointer imageIO(string filename);
};

#endif // __ReadITK_H
