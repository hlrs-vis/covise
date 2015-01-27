/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//Header
#include "ReadITK.h"
#include <iostream>
#include <string>
#include <sstream>
#include <do/coDoData.h>
//ITK Image IO
#include <itkPNGImageIO.h>
#include <itkTIFFImageIO.h>
#include <itkJPEGImageIO.h>
//Reader
#include <itkImageFileReader.h>
#include <itkImageSeriesReader.h>
//Writer
#include <itkImageFileWriter.h>
#include <itkImageSeriesWriter.h>
//ReadWrite
#include <itkNumericSeriesFileNames.h>

//CONSTRUCTOR
ReadITK::ReadITK(int argc, char *argv[])
    : coModule(argc, argv, "Read image stack using ITK")
{
    //PORTS:
    poGrid = addOutputPort("VolumeGrid", "UniformGrid", "Grid");

    poRGBR = addOutputPort("red", "Float", "RGB - Red");
    poRGBG = addOutputPort("green", "Float", "RGB - Green");
    poRGBB = addOutputPort("blue", "Float", "RGB - Blue");

    //PARAMETERS:
    pbrImageFiles = addFileBrowserParam("FilePath", "Path of series image files including printf format string");
    pbrImageFiles->setValue(" ", "*.tif;*.tiff;*.TIF/*.png;*.PNG/*.jpg;*.jpeg;*.JPG/*");
    pbrImageFiles->show();

    piSequenceBegin = addInt32Param("SequenceBegin", "First file number in sequence");
    piSequenceBegin->setValue(0);
    piSequenceBegin->show();

    piSequenceEnd = addInt32Param("SequenceEnd", "Last file number in sequence");
    piSequenceEnd->setValue(0);
    piSequenceEnd->show();

    piSequenceInc = addInt32Param("SequenceInc", "Increment in sequence");
    piSequenceInc->setValue(1);
    piSequenceInc->show();

    pboCustomSize = addBooleanParam("VoxelSize", "Off: use default values, on: use voxelsize values from below");
    pboCustomSize->setValue(false);
    pboCustomSize->show();

    pfsVoxWidth = addFloatParam("VoxelWidth", "Voxel width");
    pfsVoxWidth->setValue(1.0);
    pfsVoxWidth->hide();

    pfsVoxHeight = addFloatParam("VoxelHeight", "Voxel height");
    pfsVoxHeight->setValue(1.0);
    pfsVoxHeight->hide();

    pfsVoxDepth = addFloatParam("VoxelDepth", "Voxel depth");
    pfsVoxDepth->setValue(13.5);
    pfsVoxDepth->show();

    pboCustomOrigin = addBooleanParam("ImageOrigin", "Off: use default values, on: use values from below");
    pboCustomOrigin->setValue(false);
    pboCustomOrigin->hide();

    pfsOriginX = addFloatParam("xorigin", "x origin");
    pfsOriginX->setValue(1.0);
    pfsOriginX->hide();

    pfsOriginY = addFloatParam("yorigin", "y origin");
    pfsOriginY->setValue(1.0);
    pfsOriginY->hide();

    pfsOriginZ = addFloatParam("zorigin", "z origin");
    pfsOriginZ->setValue(3.0);
    pfsOriginZ->hide();

    return;
}

//DESTRUCTOR
ReadITK::~ReadITK()
{
}

//COMPUTE-ROUTINE
int ReadITK::compute(const char *)
{

    typedef itk::ImageSeriesReader<ColorImage3DType> SeriesReaderType;
    typedef itk::ImageFileWriter<ColorImage2DType> WriterType;
    typedef itk::ImageFileWriter<ColorImage3DType> Writer3DType;
    typedef itk::ImageFileWriter<GrayScaleImage2DType> GrayWriterType;

    typedef itk::NumericSeriesFileNames NameGeneratorType;

    SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
    WriterType::Pointer writer = WriterType::New();
    Writer3DType::Pointer writer3D = Writer3DType::New();
    GrayWriterType::Pointer graywriter = GrayWriterType::New();

    // NAMEGENERATOR ##########################################################
    NameGeneratorType::Pointer nameGenerator = NameGeneratorType::New();
    nameGenerator->SetStartIndex(piSequenceBegin->getValue());
    nameGenerator->SetEndIndex(piSequenceEnd->getValue());
    nameGenerator->SetIncrementIndex(piSequenceInc->getValue());
    nameGenerator->SetSeriesFormat(pbrImageFiles->getValue());

    // IMAGE IO TYPE ##########################################################
    itk::ImageIOBase::Pointer imageIOType = imageIO(pbrImageFiles->getValue());
    if (imageIOType.IsNotNull())
    {
        seriesReader->SetImageIO(imageIOType);
    }
    else
        return EXIT_FAILURE;

    seriesReader->SetFileNames(nameGenerator->GetFileNames());
    try
    {
        seriesReader->Update();
    }
    catch (itk::ExceptionObject &err)
    {
        sendError("%s", err.GetDescription());
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    //pointer for 3D object
    ColorImage3DType::Pointer inputVolume = seriesReader->GetOutput();
    inputVolume->SetRequestedRegion(inputVolume->GetLargestPossibleRegion());

    //set voxel dimensions
    ColorImage3DType::SpacingType spacing;
    // Note: measurement units (e.g., mm, inches, etc.) are defined by the application.
    if (pboCustomSize->getValue())
    {
        spacing[0] = pfsVoxWidth->getValue(); // spacing along X
        spacing[1] = pfsVoxHeight->getValue(); // spacing along Y
        spacing[2] = pfsVoxDepth->getValue(); // spacing along Z
    }
    else
    {
        spacing[0] = 1; // spacing along X
        spacing[1] = 1; // spacing along Y
        spacing[2] = 10; // spacing along Z
    }

    inputVolume->SetSpacing(spacing);

    //Set Origin
    ColorImage3DType::PointType origin;
    if (pboCustomOrigin->getValue())
    {
        origin[0] = pfsOriginX->getValue(); // spacing along X
        origin[1] = pfsOriginY->getValue(); // spacing along Y
        origin[2] = pfsOriginZ->getValue(); // spacing along Z
    }
    else
    {
        origin[0] = 0.0; // coordinates of the
        origin[1] = 0.0; // first pixel in N-D
        origin[2] = 0.0;
    }

    inputVolume->SetOrigin(origin);

    coDoUniformGrid *coviseVolume = convertITKtoCovise(inputVolume);
    poGrid->setCurrentObject(coviseVolume);

    return EXIT_SUCCESS;
}

coDoUniformGrid *ReadITK::convertITKtoCovise(const ColorImage3DType::Pointer itkVolume)
{
    const ColorImage3DType::SizeType &size = itkVolume->GetLargestPossibleRegion().GetSize();
    const ColorImage3DType::SpacingType &spacing = itkVolume->GetSpacing();
    const ColorImage3DType::PointType &origin = itkVolume->GetOrigin();

    float width = (spacing[0] * size[0]) / 2;
    float height = (spacing[1] * size[1]) / 2;
    float depth = (spacing[2] * size[2]) / 2;

    float minX = origin[0] - width;
    float minY = origin[1] - height;
    float minZ = origin[2] - depth;

    float maxX = origin[0] + width;
    float maxY = origin[1] + height;
    float maxZ = origin[2] + depth;

    //Creation of a grid with the size of the ITK Volume
    coDoUniformGrid *grid = new coDoUniformGrid(poGrid->getObjName(), int(size[0]), int(size[1]), int(size[2]), minX, maxX, minY, maxY, minZ, maxZ);

    //##############################################################################
    //##############################################################################

    int numVoxels = int(size[1]) * int(size[2]) * int(size[0]);

    ColorImage3DType::RegionType itkVolumeRegion = itkVolume->GetLargestPossibleRegion();
    IteratorType itITK(itkVolume, itkVolumeRegion);
    itITK.GoToBegin();

    coDoFloat *coDoRed = NULL;
    coDoFloat *coDoGreen = NULL;
    coDoFloat *coDoBlue = NULL;
    float *red = NULL;
    float *green = NULL;
    float *blue = NULL;

    coDoRed = new coDoFloat(poRGBR->getObjName(), numVoxels);
    coDoGreen = new coDoFloat(poRGBG->getObjName(), numVoxels);
    coDoBlue = new coDoFloat(poRGBB->getObjName(), numVoxels);

    coDoRed->getAddress(&red);
    coDoGreen->getAddress(&green);
    coDoBlue->getAddress(&blue);

    int dim[3] = { int(size[0]), int(size[1]), int(size[2]) };

    for (int i = 0; i < numVoxels; i++)
    {
        int index[3] = { int(itITK.GetIndex()[0]), int(itITK.GetIndex()[1]), int(itITK.GetIndex()[2]) };
        int rgbIndex = coIndex(index, dim);
        red[rgbIndex] = float(itITK.Get().GetRed() / 255.0f);
        green[rgbIndex] = float(itITK.Get().GetGreen() / 255.0f);
        blue[rgbIndex] = float(itITK.Get().GetBlue() / 255.0f);
        ++itITK;
    }

    poRGBR->setCurrentObject(coDoRed);
    poRGBG->setCurrentObject(coDoGreen);
    poRGBB->setCurrentObject(coDoBlue);

    //##################################################################
    //##################################################################

    return grid;
}

itk::ImageIOBase::Pointer ReadITK::imageIO(std::string filename)
{
    std::string localfilename(filename);
    itk::ImageIOBase::Pointer localIO;

    // search string
    string::size_type pos = localfilename.find(".tif");
    if (pos != string::npos)
    {
        localIO = itk::TIFFImageIO::New();
        cout << "Identified image format: TIFF" << endl;
    }
    else if (localfilename.find(".png") != string::npos)
    {
        localIO = itk::PNGImageIO::New();
        cout << "Identified image format: PNG" << endl;
    }
    else if (localfilename.find(".jpg") != string::npos || localfilename.find(".jpeg") != string::npos)
    {
        localIO = itk::JPEGImageIO::New();
        cout << "Identified image format: JPG" << endl;
    }
    // if string not found
    else
    {
        sendError("Image format could not be identified.");
    }
    return localIO;
}

MODULE_MAIN(IO, ReadITK)
