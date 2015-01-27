/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//Header
#include <iostream>
#include <string>
#include <sstream>
#include <do/coDoData.h>
//ITK Image IO
#include <itkPNGImageIO.h>
#include <itkTIFFImageIO.h>
#include <itkJPEGImageIO.h>
//Writer
#include <itkImageSeriesWriter.h>
//ReadWrite
#include <itkNumericSeriesFileNames.h>
#include "WriteITKSlices.h"

//CONSTRUCTOR
WriteITKSlices::WriteITKSlices(int argc, char *argv[])
    : coModule(argc, argv, "Write image stack using ITK")
{
    //PORTS:
    piGrid = addInputPort("VolumeGridIn", "UniformGrid", "Grid");

    piRGBR = addInputPort("redin", "Float", "RGB - Red");
    piRGBG = addInputPort("greenin", "Float", "RGB - Green");
    piRGBB = addInputPort("bluein", "Float", "RGB - Blue");

    //PARAMETERS:
    pFormatChoice = addChoiceParam("SaveAs", "Image format");
    const char *imageformat[] = { "jpg", "png", "tif" };
    pFormatChoice->setValue(3, imageformat, 0);

    outImgfile = addFileBrowserParam("Output", "Output filename pattern (printf format string)");
    outImgfile->setValue("output-%02d.jpg", "*.tif;*.tiff;*.TIF/*.png;*.PNG/*.jpg;*.jpeg;*.JPG");
    outImgfile->show();

    return;
}

//DESTRUCTOR
WriteITKSlices::~WriteITKSlices()
{
}

//COMPUTE-ROUTINE
int WriteITKSlices::compute(const char *)
{
    const coDoUniformGrid *covisegrid = dynamic_cast<const coDoUniformGrid *>(piGrid->getCurrentObject());
    if (covisegrid == 0)
    {
        sendError("No data found. The first port only accepts uniform grids.");
        return STOP_PIPELINE;
    }

    ///#######################################################################################
    // function convertCOVISEtoITK => uniform grid to ITK object
    ///#######################################################################################
    ColorImage3DType::Pointer inputVolume;
    if (convertCOVISEtoITK(&inputVolume) == STOP_PIPELINE)
    {
        return STOP_PIPELINE;
    }
    ///#######################################################################################

    ColorImage3DType::RegionType inputRegion = inputVolume->GetLargestPossibleRegion();
    ColorImage3DType::SizeType size = inputRegion.GetSize();

    // save amount of slices in variable (end)
    int end = size[2];

    inputVolume->SetRequestedRegion(inputVolume->GetLargestPossibleRegion());

    ///#### Output #############################################################
    typedef itk::ImageSeriesWriter<ColorImage3DType, ColorImage2DType> SeriesWriterType;
    SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
    typedef itk::NumericSeriesFileNames NameGeneratorType;
    NameGeneratorType::Pointer nameGenerator = NameGeneratorType::New();
    nameGenerator->SetStartIndex(1);
    nameGenerator->SetEndIndex(end);
    nameGenerator->SetIncrementIndex(1);
    nameGenerator->SetSeriesFormat(outImgfile->getValue());

    seriesWriter->SetInput(inputVolume);

    itk::ImageIOBase::Pointer imageIOType = imageIO(pFormatChoice->getValue());
    if (imageIOType.IsNotNull())
    {
        seriesWriter->SetImageIO(imageIOType);
    }
    else
        return EXIT_FAILURE;

    seriesWriter->SetFileNames(nameGenerator->GetFileNames());

    try
    {
        seriesWriter->Update();
    }
    catch (itk::ExceptionObject &excp)
    {
        std::cerr << "Error reading the series " << std::endl;
        std::cerr << excp << std::endl;
    }
    return EXIT_SUCCESS;
}

int WriteITKSlices::convertCOVISEtoITK(WriteITKSlices::ColorImage3DType::Pointer *ITKGrid)
{
    ///get grid

    const coDoUniformGrid *covisegrid = dynamic_cast<const coDoUniformGrid *>(piGrid->getCurrentObject());
    if (covisegrid == 0)
    {
        sendError("No data found. The first port only accepts uniform grids.");
        return STOP_PIPELINE;
    }

    ///COVISE-geometry to ITK
    ///size
    int gridsize[3];
    covisegrid->getGridSize(&gridsize[0], &gridsize[1], &gridsize[2]);
    ///space
    float gridspace[3];
    covisegrid->getDelta(&gridspace[0], &gridspace[1], &gridspace[2]);
    ///origin
    float gridminmax[6];
    covisegrid->getMinMax(&gridminmax[0], &gridminmax[1], &gridminmax[2], &gridminmax[3], &gridminmax[4], &gridminmax[5]);
    float gridorigin[3];
    gridorigin[0] = (gridminmax[0] + gridminmax[1]) / 2;
    gridorigin[1] = (gridminmax[2] + gridminmax[3]) / 2;
    gridorigin[2] = (gridminmax[4] + gridminmax[5]) / 2;

    ///Creation of a ITK Volume with the size of the grid
    *ITKGrid = ColorImage3DType::New();

    int numVoxels = int(gridsize[0]) * int(gridsize[1]) * int(gridsize[2]);

    ColorImage3DType::SizeType size;
    size[0] = gridsize[0];
    size[1] = gridsize[1];
    size[2] = gridsize[2];
    (*ITKGrid)->SetRegions(size);
    (*ITKGrid)->Allocate();

    ColorImage3DType::SpacingType spacing;
    spacing[0] = gridspace[0];
    spacing[1] = gridspace[1];
    spacing[2] = gridspace[2];
    (*ITKGrid)->SetSpacing(spacing);

    ColorImage3DType::PointType origin;
    origin[0] = gridorigin[0];
    origin[1] = gridorigin[1];
    origin[2] = gridorigin[2];
    (*ITKGrid)->SetOrigin(origin);

    ColorImage3DType::RegionType itkGridRegion = (*ITKGrid)->GetLargestPossibleRegion();

    ///Iterator
    IteratorType itITK(*ITKGrid, itkGridRegion);
    itITK.GoToBegin();

    ///RGB data grid
    const coDoFloat *coDoRed = dynamic_cast<const coDoFloat *>(piRGBR->getCurrentObject());
    const coDoFloat *coDoGreen = dynamic_cast<const coDoFloat *>(piRGBG->getCurrentObject());
    const coDoFloat *coDoBlue = dynamic_cast<const coDoFloat *>(piRGBB->getCurrentObject());

    if (coDoRed == 0 || coDoGreen == 0 || coDoBlue == 0)
    {
        sendError("No rgb data found.");
        return STOP_PIPELINE;
    }

    float *red = NULL;
    float *green = NULL;
    float *blue = NULL;

    coDoRed->getAddress(&red);
    coDoGreen->getAddress(&green);
    coDoBlue->getAddress(&blue);

    int dim[3] = { int(size[0]), int(size[1]), int(size[2]) };

    for (int i = 0; i < numVoxels; i++)
    {
        int index[3] = { (int)itITK.GetIndex()[0], (int)itITK.GetIndex()[1], (int)itITK.GetIndex()[2] };
        int rgbIndex = coIndex(index, dim);

        RGBPixelType newPixel;
        newPixel.SetRed(255 * red[rgbIndex]);
        newPixel.SetGreen(255 * green[rgbIndex]);
        newPixel.SetBlue(255 * blue[rgbIndex]);
        itITK.Set(newPixel);

        ++itITK;
    }

    return CONTINUE_PIPELINE;
}

itk::ImageIOBase::Pointer WriteITKSlices::imageIO(int fileformat)
{
    int localfileformat(fileformat);
    itk::ImageIOBase::Pointer localIO;

    if (localfileformat == 2)
    {
        localIO = itk::TIFFImageIO::New();
    }
    else if (localfileformat == 1)
    {
        localIO = itk::PNGImageIO::New();
    }
    else if (localfileformat == 0)
    {
        localIO = itk::JPEGImageIO::New();
    }
    else
    {
        sendError("Image format could not be written.");
    }
    return localIO;
}

MODULE_MAIN(IO, WriteITKSlices)
