/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//Header
#include <iostream>
#include <string>
#include <sstream>
#include <do/coDoData.h>
#include <util/coWristWatch.h>
//ITK Image IO
#include <itkPNGImageIO.h>
#include <itkTIFFImageIO.h>
#include <itkJPEGImageIO.h>
//Writer
#include <itkImageSeriesWriter.h>
//ReadWrite
#include <itkNumericSeriesFileNames.h>
//Registration
#include <itkLinearInterpolateImageFunction.h>
#include <itkNormalVariateGenerator.h>
//Metrics
#include <itkMeanSquaresImageToImageMetric.h>
#include <itkMattesMutualInformationImageToImageMetric.h>
#include <itkMutualInformationImageToImageMetric.h>
//Optimizer
#include <itkOnePlusOneEvolutionaryOptimizer.h>
#include <itkGradientDescentOptimizer.h>
//Filter
#include <itkVectorResampleImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkMultiResolutionPyramidImageFilter.h>
#include <itkNormalizeImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkLightObject.h>
#include "Register.h"

//CONSTRUCTOR
Register::Register(int argc, char *argv[])
    : coModule(argc, argv, "Register slices in volumetric data using ITK")
{
    //PORTS:
    piGrid = addInputPort("VolumeGridIn", "UniformGrid", "Grid");

    piRGBR = addInputPort("redin", "Float", "RGB - Red");
    piRGBG = addInputPort("greenin", "Float", "RGB - Green");
    piRGBB = addInputPort("bluein", "Float", "RGB - Blue");

    poGrid = addOutputPort("VolumeGridOut", "UniformGrid", "Grid");

    poRGBR = addOutputPort("redout", "Float", "RGB - Red");
    poRGBG = addOutputPort("greenout", "Float", "RGB - Green");
    poRGBB = addOutputPort("blueout", "Float", "RGB - Blue");

    //PARAMETERS:
    pOptimizerChoice = addChoiceParam("Optimizer", "Optimizer type");
    const char *optimizers[] = { "Regular Step Gradient Descent", "Gradient Descent", "One Plus One Evolutionary" };
    pOptimizerChoice->setValue(3, optimizers, 0);

    pfsMaxStepLength = addFloatParam("MaxStepLength", "Maximum Step Length for Optimizer");
    pfsMaxStepLength->setValue(1.5);

    pfsMinStepLength = addFloatParam("MinStepLength", "Minimum Step Length for Optimizer");
    pfsMinStepLength->setValue(0.5);

    pfsMinStepLengthDivisor = addFloatParam("MinStepLengthDiv", "Minimum Step Length Divisor for higher Levels");
    pfsMinStepLengthDivisor->setValue(10);

    if (pOptimizerChoice->getValue() == 0)
    {
        pfsMaxStepLength->show();
        pfsMinStepLength->show();
        pfsMinStepLengthDivisor->show();
    }
    else
    {
        pfsMaxStepLength->hide();
        pfsMinStepLength->hide();
        pfsMinStepLengthDivisor->hide();
    }

    pMetricChoice = addChoiceParam("Metric", "Metric type");
    const char *metrics[] = { "Mean Squares", "Mutual Information by Mattes" };
    pMetricChoice->setValue(2, metrics, 1);

    piFillColor = addInt32Param("FillColor", "Gray scale fill color for background of transformated images");
    piFillColor->setValue(255);
    piFillColor->show();

    piIterations = addInt32Param("Iterations", "Number of Iterations");
    piIterations->setValue(200);
    piIterations->show();

    piPyramid = addInt32Param("Pyramid", "Number of Pyramidlevels");
    piPyramid->setValue(3);
    piPyramid->show();

    return;
}

//DESTRUCTOR
Register::~Register()
{
}

class CommandIterationUpdate : public itk::Command
{
public:
    typedef CommandIterationUpdate Self;
    typedef itk::Command Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    itkNewMacro(Self);

    typedef itk::SingleValuedNonLinearOptimizer OptimizerType;
    typedef const OptimizerType *OptimizerPointer;

    void Execute(itk::Object *caller, const itk::EventObject &event)
    {
        Execute((const itk::Object *)caller, event);
    }

    void Execute(const itk::Object *object, const itk::EventObject &event)
    {
        OptimizerPointer optimizer = dynamic_cast<OptimizerPointer>(object);

        if (!itk::IterationEvent().CheckEvent(&event))
        {
            return;
        }
        cout << optimizer->GetValue(optimizer->GetCurrentPosition()) << " " << optimizer->GetCurrentPosition() << endl;
    }

protected:
    CommandIterationUpdate(){};
    ~CommandIterationUpdate(){};
};

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command
{
public:
    typedef RegistrationInterfaceCommand Self;
    typedef itk::Command Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    itkNewMacro(Self);

protected:
    RegistrationInterfaceCommand(){};

public:
    typedef TRegistration RegistrationType;
    typedef RegistrationType *RegistrationPointer;
    typedef itk::SingleValuedNonLinearOptimizer OptimizerType;
    typedef OptimizerType *OptimizerPointer;

    float pfsMaxStepLength;
    float pfsMinStepLength;
    float pfsMinStepLengthDivisor;

    void Execute(itk::Object *object, const itk::EventObject &event)
    {
        if (!(itk::IterationEvent().CheckEvent(&event)))
        {
            return;
        }
        RegistrationPointer registration = dynamic_cast<RegistrationPointer>(object);
        itk::RegularStepGradientDescentOptimizer *optimizer = dynamic_cast<itk::RegularStepGradientDescentOptimizer *>(registration->GetOptimizer());

        if (optimizer)
        {
            if (registration->GetCurrentLevel() == 0)
            {
                optimizer->SetMaximumStepLength(pfsMaxStepLength);
                optimizer->SetMinimumStepLength(pfsMinStepLength);
            }
            else
            {
                optimizer->SetMaximumStepLength(optimizer->GetCurrentStepLength());
                optimizer->SetMinimumStepLength(optimizer->GetMinimumStepLength() / pfsMinStepLengthDivisor);
            }
        }
    }

    void Execute(const itk::Object *, const itk::EventObject &)
    {
        return;
    }
};

//COMPUTE-ROUTINE
int Register::compute(const char *)
{
    typedef itk::ImageFileWriter<ColorImage2DType> WriterType;
    typedef itk::ImageFileWriter<ColorImage3DType> Writer3DType;
    typedef itk::ImageFileWriter<GrayScaleImage2DType> GrayWriterType;

    WriterType::Pointer writer = WriterType::New();
    Writer3DType::Pointer writer3D = Writer3DType::New();
    GrayWriterType::Pointer graywriter = GrayWriterType::New();

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

    typedef itk::ExtractImageFilter<ColorImage3DType, ColorImage2DType> FilterType;
    FilterType::Pointer extractFixedImageFilter = FilterType::New();
    FilterType::Pointer extractMovingImageFilter = FilterType::New();

    // gray-scale images
    typedef itk::RGBToLuminanceImageFilter<ColorImage2DType, GrayScaleImage2DType> RGBToLuminanceFilterType;
    RGBToLuminanceFilterType::Pointer RGBToLuminanceFilterfix = RGBToLuminanceFilterType::New();
    RGBToLuminanceFilterType::Pointer RGBToLuminanceFiltermov = RGBToLuminanceFilterType::New();

    ColorImage3DType::RegionType inputRegion = inputVolume->GetLargestPossibleRegion();
    ColorImage3DType::SizeType size = inputRegion.GetSize();

    // save amount of slices in variable (end)
    int begin = 0;
    int end = size[2];

    coWristWatch _ww;

    /// RegistrationType ########################################################################################
    typedef itk::MultiResolutionImageRegistrationMethod<InternalImageType, InternalImageType> RegistrationType;
    RegistrationType::Pointer registration = RegistrationType::New();

    for (int i = begin; i < end - 1; i++)
    {
        size[2] = 0;
        ColorImage3DType::IndexType fix = inputRegion.GetIndex();
        ColorImage3DType::IndexType mov = inputRegion.GetIndex();
        const unsigned int fixedImageIndex = i;
        const unsigned int movingImageIndex = i + 1;
        fix[2] = fixedImageIndex;
        mov[2] = movingImageIndex;

        ColorImage3DType::RegionType fixedImage;
        ColorImage3DType::RegionType movingImage;

        fixedImage.SetSize(size);
        fixedImage.SetIndex(fix);
        movingImage.SetSize(size);
        movingImage.SetIndex(mov);

        //##########################################################################################################
        // read fixed image ############################################################################
        //##########################################################################################################
        extractFixedImageFilter->SetExtractionRegion(fixedImage);
        extractFixedImageFilter->SetInput(inputVolume);
        extractFixedImageFilter->Update();
        ColorImage2DType::Pointer fixedImagePointer = extractFixedImageFilter->GetOutput();
        // convert fixed image -> grey scale
        RGBToLuminanceFilterfix->SetInput(fixedImagePointer);
        RGBToLuminanceFilterfix->Update();
        GrayScaleImage2DType::Pointer grayfixedImagePointer = RGBToLuminanceFilterfix->GetOutput();

        //##########################################################################################################
        // read moving image ############################################################################
        //##########################################################################################################
        extractMovingImageFilter->SetExtractionRegion(movingImage);
        extractMovingImageFilter->SetInput(inputVolume);
        extractMovingImageFilter->Update();
        ColorImage2DType::Pointer movingImagePointer = extractMovingImageFilter->GetOutput();
        // convert moving image -> gray scale
        RGBToLuminanceFiltermov->SetInput(movingImagePointer);
        RGBToLuminanceFiltermov->Update();
        GrayScaleImage2DType::Pointer graymovingImagePointer = RGBToLuminanceFiltermov->GetOutput();

        // BEGIN OF REGISTRATION //////////////////////////////////////////////////////////////////

        ///##########################################################################################################
        /// COMPONENTS AND THEIR PARAMETERS #########################################################################
        ///##########################################################################################################

        /// TransformType ###########################################################################################

        typedef itk::CenteredRigid2DTransform<double> TransformType;
        TransformType::Pointer transform = TransformType::New();
        registration->SetTransform(transform);

        typedef GrayScaleImage2DType::SpacingType SpacingType;
        typedef GrayScaleImage2DType::PointType OriginType;
        typedef GrayScaleImage2DType::RegionType RegionType;
        typedef GrayScaleImage2DType::SizeType SizeType;

        //center of fixedImage
        const SpacingType fixedSpacing = grayfixedImagePointer->GetSpacing();
        const OriginType fixedOrigin = grayfixedImagePointer->GetOrigin();
        const RegionType fixedRegion = grayfixedImagePointer->GetLargestPossibleRegion();
        const SizeType fixedSize = fixedRegion.GetSize();

        TransformType::InputPointType centerFixed;

        centerFixed[0] = fixedOrigin[0] + fixedSpacing[0] * fixedSize[0] / 2.0;
        centerFixed[1] = fixedOrigin[1] + fixedSpacing[1] * fixedSize[1] / 2.0;

        //center of movingImage
        const SpacingType movingSpacing = graymovingImagePointer->GetSpacing();
        const OriginType movingOrigin = graymovingImagePointer->GetOrigin();
        const RegionType movingRegion = graymovingImagePointer->GetLargestPossibleRegion();
        const SizeType movingSize = movingRegion.GetSize();

        TransformType::InputPointType centerMoving;

        centerMoving[0] = movingOrigin[0] + movingSpacing[0] * movingSize[0] / 2.0;
        centerMoving[1] = movingOrigin[1] + movingSpacing[1] * movingSize[1] / 2.0;

        transform->SetCenter(centerFixed);
        transform->SetTranslation(centerMoving - centerFixed);

        transform->SetAngle(0.0);

        registration->SetInitialTransformParameters(transform->GetParameters());

        /// OptimizerType ###########################################################################################

        itk::SingleValuedNonLinearOptimizer::Pointer genericOptimizer = NULL;
        if (pOptimizerChoice->getValue() == 0)
        {
            typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
            OptimizerType::Pointer optimizer = OptimizerType::New();

            typedef OptimizerType::ScalesType OptimizerScalesType;
            OptimizerScalesType optimizerScales(registration->GetTransform()->GetNumberOfParameters());

            const double translationScale = 1.0 / 1000.0;
            optimizerScales[0] = 1.0;
            optimizerScales[1] = translationScale;
            optimizerScales[2] = translationScale;
            optimizerScales[3] = translationScale;
            optimizerScales[4] = translationScale;

            optimizer->SetScales(optimizerScales);
            optimizer->SetRelaxationFactor(0.6);
            optimizer->SetNumberOfIterations(piIterations->getValue());

            CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
            optimizer->AddObserver(itk::IterationEvent(), observer);

            registration->SetOptimizer(optimizer);
            genericOptimizer = optimizer;
        }
        else if (pOptimizerChoice->getValue() == 1)
        {
            typedef itk::GradientDescentOptimizer OptimizerType;
            OptimizerType::Pointer optimizer = OptimizerType::New();

            optimizer->SetLearningRate(15.0);
            optimizer->SetNumberOfIterations(piIterations->getValue());
            optimizer->MaximizeOn();

            CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
            optimizer->AddObserver(itk::IterationEvent(), observer);

            registration->SetOptimizer(optimizer);
            genericOptimizer = optimizer;
        }
        else if (pOptimizerChoice->getValue() == 2)
        {
            typedef itk::OnePlusOneEvolutionaryOptimizer OptimizerType;
            OptimizerType::Pointer optimizer = OptimizerType::New();

            typedef itk::Statistics::NormalVariateGenerator GeneratorType;
            GeneratorType::Pointer generator = GeneratorType::New();
            generator->Initialize(12345);

            optimizer->MaximizeOff();

            optimizer->SetNormalVariateGenerator(generator);
            optimizer->Initialize(10);
            optimizer->SetEpsilon(1.0);
            optimizer->SetMaximumIteration(piIterations->getValue());

            CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
            optimizer->AddObserver(itk::IterationEvent(), observer);

            registration->SetOptimizer(optimizer);
            genericOptimizer = optimizer;
        }
        else
        {
            sendError("unknown optimizer type");
            return STOP_PIPELINE;
        }

        /// MetricType ##############################################################################################

        typedef itk::CastImageFilter<GrayScaleImage2DType, InternalImageType> CastFilterType;
        CastFilterType::Pointer fixedCaster = CastFilterType::New();
        CastFilterType::Pointer movingCaster = CastFilterType::New();

        fixedCaster->SetInput(grayfixedImagePointer);
        movingCaster->SetInput(graymovingImagePointer);

        registration->SetFixedImage(fixedCaster->GetOutput());
        registration->SetMovingImage(movingCaster->GetOutput());

        fixedCaster->Update();

        registration->SetFixedImageRegion(fixedCaster->GetOutput()->GetBufferedRegion());

        if (pMetricChoice->getValue() == 0)
        {
            typedef itk::MeanSquaresImageToImageMetric<InternalImageType, InternalImageType> MetricType;
            MetricType::Pointer metric = MetricType::New();
            registration->SetMetric(metric);
        }
        else if (pMetricChoice->getValue() == 1)
        {
            typedef itk::MattesMutualInformationImageToImageMetric<InternalImageType, InternalImageType> MetricType;
            MetricType::Pointer metric = MetricType::New();
            registration->SetMetric(metric);

            // Parameter
            metric->SetNumberOfHistogramBins(20);
            metric->SetNumberOfSpatialSamples(10000);

            metric->ReinitializeSeed(76926294);
        }
        else
        {
            sendError("unknown metric type");
            return STOP_PIPELINE;
        }

        /// InterpolatorType ########################################################################################

        typedef itk::LinearInterpolateImageFunction<InternalImageType, double> InterpolatorType;

        InterpolatorType::Pointer interpolator = InterpolatorType::New();
        registration->SetInterpolator(interpolator);

        /// PyramidType #############################################################################################

        typedef itk::MultiResolutionPyramidImageFilter<InternalImageType, InternalImageType> FixedImagePyramidType;
        typedef itk::MultiResolutionPyramidImageFilter<InternalImageType, InternalImageType> MovingImagePyramidType;

        FixedImagePyramidType::Pointer fixedImagePyramid = FixedImagePyramidType::New();
        MovingImagePyramidType::Pointer movingImagePyramid = MovingImagePyramidType::New();

        registration->SetFixedImagePyramid(fixedImagePyramid);
        registration->SetMovingImagePyramid(movingImagePyramid);

        typedef RegistrationInterfaceCommand<RegistrationType> CommandType;
        CommandType::Pointer command = CommandType::New();
        command->pfsMaxStepLength = pfsMaxStepLength->getValue();
        command->pfsMinStepLength = pfsMinStepLength->getValue();
        command->pfsMinStepLengthDivisor = pfsMinStepLengthDivisor->getValue();

        registration->AddObserver(itk::IterationEvent(), command);

        registration->SetNumberOfLevels(piPyramid->getValue());

        ///##########################################################################################################
        /// REGISTRATION ########################################################################################
        ///##########################################################################################################

        try
        {
//atismer: backward compatibility of itk library
#if (ITK_VERSION_MAJOR <= 3)
            registration->StartRegistration();
#else
            registration->Update();
#endif
        }
        catch (itk::ExceptionObject &err)
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }

        /// PARAMETERS ##############################################################################################

        itk::Optimizer::ParametersType finalParameters = registration->GetLastTransformParameters();
        const double finalAngle = finalParameters[0];
        const double finalRotationCenterX = finalParameters[1];
        const double finalRotationCenterY = finalParameters[2];
        const double finalTranslationX = finalParameters[3];
        const double finalTranslationY = finalParameters[4];
        //const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
        const double bestValue = genericOptimizer->GetValue(genericOptimizer->GetCurrentPosition());
        const double finalAngleInDegrees = finalAngle * 45.0 / atan(1.0);

        std::cout << "Result = " << std::endl;
        std::cout << " Angle (radians)   = " << finalAngle << std::endl;
        std::cout << " Angle (degrees)   = " << finalAngleInDegrees << std::endl;
        std::cout << " Center X      = " << finalRotationCenterX << std::endl;
        std::cout << " Center Y      = " << finalRotationCenterY << std::endl;
        std::cout << " Translation X = " << finalTranslationX << std::endl;
        std::cout << " Translation Y = " << finalTranslationY << std::endl;
        //std::cout << " Iterations    = " << numberOfIterations << std::endl;
        std::cout << " Metric value  = " << bestValue << std::endl;

        std::cout << std::endl;

        /// RESAMPLER ###########################################################################################
        /// #####################################################################################################

        typedef itk::VectorResampleImageFilter<ColorImage2DType, ColorImage2DType> ResampleFilterType;

        ResampleFilterType::Pointer resample = ResampleFilterType::New();

        resample->SetTransform(registration->GetTransform());
        resample->SetInput(movingImagePointer);
        resample->SetSize(fixedImagePointer->GetLargestPossibleRegion().GetSize());
        resample->SetOutputOrigin(fixedImagePointer->GetOrigin());
        resample->SetOutputSpacing(fixedImagePointer->GetSpacing());
        resample->SetDefaultPixelValue(piFillColor->getValue());

        resample->Update();

        //##########################################################################################################
        // convert casted image to 3D objekt with 2 iterators #######################################
        //##########################################################################################################
        size[2] = 1;

        ColorImage3DType::RegionType inputVolumeRegion = inputVolume->GetLargestPossibleRegion();
        ColorImage2DType::Pointer newmov = resample->GetOutput();
        ColorImage2DType::RegionType newRegion = newmov->GetLargestPossibleRegion();

        ColorImage2DType::SizeType size2D;
        ColorImage2DType::IndexType index2D;
        size2D[0] = size[0];
        size2D[1] = size[1];
        index2D[0] = 0;
        index2D[1] = 0;

        newRegion.SetSize(size2D);
        newRegion.SetIndex(index2D);
        ConstIteratorType it2D(newmov, newRegion);

        inputVolumeRegion.SetSize(size);
        inputVolumeRegion.SetIndex(mov);
        IteratorType it3D(inputVolume, inputVolumeRegion);
        it3D.GoToBegin();
        it2D.GoToBegin();

        while (!it3D.IsAtEnd() && !it2D.IsAtEnd())
        {
            it2D.Get();
            it3D.Set(it2D.Get());
            ++it3D;
            ++it2D;
        }
        // END OF REGISTRATION /////////////////////////////////////////////////////////////////////////
    }

    float duration = _ww.elapsed();
    int min = int(duration) / 60;
    int sec = int(duration) % 60;

    const char *metricstring = registration->GetMetric()->GetNameOfClass();
    const char *optimizerstring = registration->GetOptimizer()->GetNameOfClass();

    sendInfo(" metric:  %s ", metricstring);
    sendInfo("optimizer:  %s", optimizerstring);
    sendInfo("runtime: %g seconds, that is %d minute(s) and %d second(s).", duration, min, sec);

    inputVolume->SetRequestedRegion(inputVolume->GetLargestPossibleRegion());

    ///#######################################################################################
    // function convertITKtoCOVISE => ITK object to uniform grid
    ///#######################################################################################
    coDoUniformGrid *coviseVolume = convertITKtoCOVISE(inputVolume);
    ///#######################################################################################

    // send information to outputport
    poGrid->setCurrentObject(coviseVolume);

    return EXIT_SUCCESS;
}

coDoUniformGrid *Register::convertITKtoCOVISE(const ColorImage3DType::Pointer itkVolume)
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

    return grid;
}

int Register::convertCOVISEtoITK(Register::ColorImage3DType::Pointer *ITKGrid)
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
        int index[3] = { int(itITK.GetIndex()[0]), int(itITK.GetIndex()[1]), int(itITK.GetIndex()[2]) };
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

MODULE_MAIN(Filter, Register)
