/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoText.h>
#include <do/coDoTexture.h>
#include <util/coviseCompat.h>
#include "ImageToTexture.h"
#include <float.h>
#include <string>

// the file name has changed and we want to reread the file
void
ImageToTexture::changeFileName()
{
    changeFileNameFailed_ = true;
    // assume physical-soze image dimension is available in file
    mode_ = TextureMapping::NONE;
    // open file
    bild_ = TIFFOpen(FileName_.c_str(), "r");

    if (bild_ == NULL)
    {
        sendWarning("Could not open file.");
        return;
    }

    // read width in pixels
    if (TIFFGetField(bild_, TIFFTAG_IMAGEWIDTH, &w_) != 1)
    {
        sendWarning("Could not allocate memory for raster_.");
        TIFFClose(bild_);
        bild_ = NULL;
        return;
    }
    // read height in pixels
    if (TIFFGetField(bild_, TIFFTAG_IMAGELENGTH, &h_) != 1)
    {
        sendWarning("Could not read image.");
        TIFFClose(bild_);
        bild_ = NULL;
        return;
    }

    // @@@ problems with segmented memory (Windows...)
    delete[] raster_;
    raster_ = new uint32[w_ * h_];

    if (raster_ == NULL)
    {
        sendWarning("Could not allocate memory for raster_.");
        TIFFClose(bild_);
        bild_ = NULL;
        return;
    }
    // read image content
    if (TIFFReadRGBAImage(bild_, w_, h_, raster_, 0) != 1)
    {
        sendWarning("Could not read TIFF file.");
        TIFFClose(bild_);
        delete[] raster_; // @@@
        bild_ = NULL;
        raster_ = NULL;
        return;
    }
    // Calculate image dimensions from the file (if possible)
    // first of all get resolution unit
    XResolution_ = 0.0, YResolution_ = 0.0;
    if (TIFFGetField(bild_, TIFFTAG_RESOLUTIONUNIT, &resUnit_) != 1
        || resUnit_ == RESUNIT_NONE)
    {
        sendInfo("The image has no resolution unit. Please, set physical image dimensions manually");
        mode_ = TextureMapping::MANUAL;
    }
    else // we have resolution unit
    {
        if (TIFFGetField(bild_, TIFFTAG_XRESOLUTION, &XResolution_) != 1)
        {
            sendInfo("Could not read X Resolution. Please, set physical image dimension manually");
            mode_ = TextureMapping::MANUAL;
        }
        else if (TIFFGetField(bild_, TIFFTAG_YRESOLUTION, &YResolution_) != 1)
        {
            sendInfo("Could not read Y Resolution. Assuming the same value as for the X dimension. You may always control the behaviour manually");
            YResolution_ = XResolution_;
            mode_ = TextureMapping::AUTOMATIC;
        }
        else
        {
            sendInfo("A valid image size has been read from file");
            mode_ = TextureMapping::AUTOMATIC;
        }
        if (mode_ == TextureMapping::AUTOMATIC) // give the user a hint
        {
            float Width, Height;
            Width = w_ / XResolution_;
            Height = h_ / YResolution_;
            if (resUnit_ == RESUNIT_INCH)
            {
                Width *= 2.54f;
                Height *= 2.54f;
            }
            sendInfo("Image size according to file information: width, %f; height, %f (cm)", Width, Height);
        }
    }
    TIFFClose(bild_);
    bild_ = NULL;
    changeFileNameFailed_ = false;
    // transform RGBA format from raster_ to image_
    return;
}

void
ImageToTexture::param(const char *paramName, bool /*inMapLoading*/)
{
    if (strcmp(paramName, p_file_->getName()) == 0)
    {
        if (strcmp(FileName_.c_str(), p_file_->getValue()) == 0 && !changeFileNameFailed_)
        {
            return;
        }
        FileName_ = p_file_->getValue();
        changeFileName(); // reread 'new' file
    }
    else if (strcmp(paramName, p_fit_->getName()) == 0)
    {
        if (p_fit_->getValue() == TextureMapping::FIT_MANUAL
            || (p_fit_->getValue() == TextureMapping::USE_IMAGE
                && mode_ == TextureMapping::MANUAL))
        {
            p_width_->enable();
            p_height_->enable();
        }
        else
        {
            p_width_->disable();
            p_height_->disable();
        }
    }
}

ImageToTexture::ImageToTexture(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Texturize polygons")
{
    p_file_ = addFileBrowserParam("TIFF_file", "RST file");
    p_file_->setValue("/var/tmp/foo.tif", "*.tif;*.tiff");

    p_geometry_ = addFileBrowserParam("Geometry_file", "Geometry file");
    p_geometry_->setValue("/var/tmp/foo.txt", "*.txt");

    p_fit_ = addChoiceParam("ImageSize", "Image size");
    const char *ImageSizeChoices[] = {
        "Set manually physical image size",
        "Fit to geometry",
        "Try using image size info, otherwise fit to geometry"
    };
    p_fit_->setValue(3, ImageSizeChoices, 0);
    p_orientation_ = addChoiceParam("ImageOrientation", "Image orientation");
    const char *ImageOrientationChoices[] = { "Free", "Portrait", "Landscape" };
    p_orientation_->setValue(3, ImageOrientationChoices, 0);

    p_width_ = addFloatParam("PhysImageW", "Physical image width");
    p_width_->setValue(0.0);
    p_height_ = addFloatParam("PhysImageH", "Physical image height");
    p_height_->setValue(0.0);

    const float minmax[2] = { 0.0, 1.0 };
    p_xlimits_ = addFloatVectorParam("MinMaxX", "MinMaxX");
    p_xlimits_->setValue(2, minmax);
    p_ylimits_ = addFloatVectorParam("MinMaxY", "MinMaxY");
    p_ylimits_->setValue(2, minmax);

    const char *autoOrManualDefaults[] = { "Automatic", "Manual" };
    p_AutoOrManual_ = addChoiceParam("AutoOrManual", "Automatic or manual pixel size");
    p_AutoOrManual_->setValue(2, autoOrManualDefaults, 1);

    p_XSize_ = addInt32Param("XPixelImageSize", "X PixelImage Size");
    p_XSize_->setValue(256);
    p_YSize_ = addInt32Param("YPixelImageSize", "Y PixelImage Size");
    p_YSize_->setValue(256);

    p_groupBB_ = addBooleanParam("GroupGeometry", "Group geometry per time step");
    p_groupBB_->setValue(0);

    p_mirror_ = addBooleanParam("MirrorImage", "Mirror image");
    p_mirror_->setValue(1);

    p_poly_ = addInputPort("GridIn0", "Polygons", "Input geometry");
    p_shift_ = addInputPort("DataIn0", "Vec3", "Material shift");
    p_shift_->setRequired(0);
    p_file_name_ = addInputPort("InFileName", "Text", "Image file");
    p_file_name_->setRequired(0);
    p_texture_ = addOutputPort("TextureOut0", "Texture", "Output texture");

    bild_ = NULL;
    raster_ = NULL;
    mode_ = TextureMapping::NONE;
    XResolution_ = 0.0, YResolution_ = 0.0;
    XSize_ = YSize_ = 0;
    pix_ = NULL;
    references_ = 0;

    preHandleDiagnose_ = PROBLEM;
    changeFileNameFailed_ = true;
    WasConnected_ = true;
}

ImageToTexture::~ImageToTexture()
{
    delete[] raster_;
}

// overriding param values through an "IMAGE_TO_TEXTURE" attribute
void
ImageToTexture::useImageToTextureAttribute(const coDistributedObject *inName)
{
    const char *wert = NULL;
    if (inName == NULL)
    {
        return;
    }
    wert = inName->getAttribute("IMAGE_TO_TEXTURE");
    if (wert == NULL) // perhaps the attribute is hidden in a set structure
    {
        if (inName->isType("SETELE"))
        {
            int no_elems;
            const coDistributedObject *const *setList = dynamic_cast<const coDoSet *>(inName)->getAllElements(&no_elems);
            int elem;
            for (elem = 0; elem < no_elems; ++elem)
            {
                useImageToTextureAttribute(setList[elem]);
            }
        }
        return;
    }
    istringstream pvalues(wert);
    char *value = new char[strlen(wert) + 1];
    while (pvalues.getline(value, strlen(wert) + 1))
    {
        int param;
        for (param = 0; param < hparams_.size(); ++param)
        {
            hparams_[param]->load(value);
        }
    }
    delete[] value;
}

void
ImageToTexture::postInst()
{
    hparams_.push_back(h_width_ = new coHideParam(p_width_));
    hparams_.push_back(h_height_ = new coHideParam(p_height_));
    hparams_.push_back(h_XSize_ = new coHideParam(p_XSize_));
    hparams_.push_back(h_YSize_ = new coHideParam(p_YSize_));
    hparams_.push_back(h_fit_ = new coHideParam(p_fit_));
    hparams_.push_back(h_orientation_ = new coHideParam(p_orientation_));
    hparams_.push_back(h_groupBB_ = new coHideParam(p_groupBB_));
}

void
ImageToTexture::preHandleObjects(coInputPort **inPorts)
{
    references_ = 0;
    preHandleDiagnose_ = PROBLEM;
    // check name file if 3rd port is connected
    int param;
    for (param = 0; param < hparams_.size(); ++param)
    {
        hparams_[param]->reset();
    }

    if (p_file_name_->isConnected())
    {
        WasConnected_ = true;
        const coDistributedObject *inObj = p_file_name_->getCurrentObject();
        if (!inObj->isType("DOTEXT"))
        {

            sendWarning("Object at port %s is not DOTEXT, ignoring", p_file_name_->getName());
            preHandleDiagnose_ = OUTPUT_DUMMY;
            return;
        }
        char *text;
        const coDoText *theText = dynamic_cast<const coDoText *>(inObj);
        theText->getAddress(&text);
        int size = theText->getTextLength();
        if (size == 0)
        {
            mode_ = TextureMapping::NONE;
            FileName_ = "";
            return;
        }
        if (changeFileNameFailed_ || FileName_ != (const char *)text)
        {
            FileName_ = text;
            changeFileName();
            if (changeFileNameFailed_)
            {
                return;
            }
        }
        useImageToTextureAttribute(inObj);
    }
    else
    {
        if (WasConnected_ || changeFileNameFailed_)
        {
            FileName_ = p_file_->getValue();
            changeFileName();
            if (changeFileNameFailed_)
            {
                return;
            }
        }
        WasConnected_ = false;
    }

    BBoxes_.clean();

    if (p_groupBB_->getValue())
    {
        coInputPort *polygons = inPorts[0];
        coInputPort *shift = inPorts[1];
        const coDistributedObject *inObj = polygons->getCurrentObject();
        const coDistributedObject *inShift = shift->getCurrentObject();
        if (inObj->isType("SETELE")) // if it is a single element, we will do
        {
            // the computation in the TextureMapping class
            const coDoSet *in_set = dynamic_cast<const coDoSet *>(inObj);
            const coDoSet *in_set_shift = NULL;
            if (inShift && !inShift->isType("SETELE"))
            {
                sendError("Geometry object is a set, but not so the vector field");
                return;
            }
            else if (inShift)
            {
                in_set_shift = dynamic_cast<const coDoSet *>(inShift);
            }
            int no_set_elems;
            if (in_set->getAttribute("TIMESTEP"))
            {
                const coDistributedObject *const *grid_objs = NULL;
                const coDistributedObject *const *shift_objs = NULL;
                grid_objs = in_set->getAllElements(&no_set_elems);
                if (in_set_shift)
                {
                    int no_set_elems_shift;
                    shift_objs = in_set_shift->getAllElements(&no_set_elems_shift);
                    if (no_set_elems_shift != no_set_elems)
                    {
                        sendError("Set structures do not match");
                        return;
                    }
                }
                BBoxes_.prepare(no_set_elems);
                int time;
                for (time = 0; time < no_set_elems; ++time)
                {
                    if (grid_objs[time]->isType("SETELE"))
                    {
                        if (shift_objs && !shift_objs[time]->isType("SETELE"))
                        {
                            sendError("Set structures do not match");
                            return;
                        }
                        if (shift_objs)
                        {
                            BBoxes_.FillBBox(dynamic_cast<const coDoSet *>(grid_objs[time]),
                                             dynamic_cast<const coDoSet *>(shift_objs[time]), time);
                        }
                        else
                        {
                            BBoxes_.FillBBox(dynamic_cast<const coDoSet *>(grid_objs[time]),
                                             NULL, time);
                        }
                    }
                    else if (grid_objs[time]->isType("POLYGN"))
                    {
                        if (shift_objs && !shift_objs[time]->isType("USTVDT"))
                        {
                            sendError("Set structures do not match, or an vector field has another type");
                            return;
                        }
                        if (shift_objs)
                        {
                            BBoxes_.FillBBox(dynamic_cast<const coDoPolygons *>(grid_objs[time]),
                                             dynamic_cast<const coDoVec3 *>(shift_objs[time]), time);
                        }
                        else
                        {
                            BBoxes_.FillBBox(dynamic_cast<const coDoPolygons *>(grid_objs[time]),
                                             NULL, time);
                        }
                    }
                }
            }
            else
            {
                BBoxes_.prepare(1);
                BBoxes_.FillBBox(in_set, in_set_shift, 0);
            }
        }
    }

    // make coDoPixelImage
    if (p_AutoOrManual_->getValue() == AUTO_SIZE)
    {
        XSize_ = w_;
        YSize_ = h_;
    }
    else
    {
        XSize_ = p_XSize_->getValue(); // dimensions of array for coDoPixelImage
        YSize_ = p_YSize_->getValue(); // dimensions of array for coDoPixelImage
    }

    // delete [] image_;
    char *image = new char[4 * XSize_ * YSize_];
    char *iPtr = image;
    //   int i_count,j_count;
    //   for(j_count=0;j_count < YSize_;++j_count){
    //      for(i_count=0;i_count < XSize_;++i_count){
    unsigned int pixel = 0;
    for (; pixel < XSize_ * YSize_; ++pixel)
    {
        int r_index;
        int x_index, y_index; // it extends up to w_-1,h_-1
        int i_count = pixel % XSize_;
        int j_count = pixel / XSize_;
        x_index = int((float(i_count) / float(XSize_)) * w_);
        y_index = int((float(j_count) / float(YSize_)) * h_);
        r_index = y_index * w_ + x_index;
        *iPtr = (char)TIFFGetR(raster_[r_index]);
        iPtr++;
        *iPtr = (char)TIFFGetG(raster_[r_index]);
        iPtr++;
        *iPtr = (char)TIFFGetB(raster_[r_index]);
        iPtr++;
        *iPtr = (char)TIFFGetA(raster_[r_index]);
        iPtr++;
    }

    std::string namebuf(p_texture_->getObjName());
    namebuf += "_PixImage";
    pix_ = new coDoPixelImage(namebuf, XSize_, YSize_, 4, 4, image);
    delete[] image;
    image = NULL;

    preHandleDiagnose_ = OK;
}

void
ImageToTexture::setIterator(coInputPort **inPorts, int t)
{
    const char *dataType;

    dataType = (inPorts[0]->getCurrentObject())->getType();
    if (strcmp(dataType, "SETELE") == 0 && inPorts[0]->getCurrentObject()->getAttribute("TIMESTEP"))
        lookUp_ = t;
}

void
ImageToTexture::postHandleObjects(coOutputPort **)
{
    BBoxes_.clean();
}

// if we do not know waht to do
void
ImageToTexture::outputDummies()
{
    std::string namebuf(p_texture_->getObjName());
    namebuf += "_PixImage";
    coDoPixelImage *pix = new coDoPixelImage(namebuf, 0, 0, 4, 4);
    coDoTexture *texture = new coDoTexture(p_texture_->getObjName(), pix, 0, 4, 0, 0, NULL, 0, NULL);
    (void)texture;
    // const_cast<float **>(txCoord));
}

void
MinMaxCalculation(float cx, float cy, float tifx, float tify, vector<float> &points,
                  float &minx, float &maxx, float &miny, float &maxy)
{
    cx += -1.0;
    cy += -1.0;
    tifx += -1.0;
    tify += -1.0;

    float a11 = 0.0, a22 = 0.0, a12 = 0.0, col1 = 0.0, col2 = 0.0; // matrix coef
    int point;
    for (point = 0; point < points.size() / 4; ++point)
    {
        a11 += (1.0f - points[4 * point] / cx) * (1.0f - points[4 * point] / cx);
        a12 += (1.0f - points[4 * point] / cx) * points[4 * point] / cx;
        a22 += points[4 * point] * points[4 * point] / (cx * cx);
        col1 += (1.0f - points[4 * point] / cx) * points[4 * point + 2] / tifx;
        col2 += (points[4 * point] / cx) * points[4 * point + 2] / tifx;
    }
    float det = a11 * a22 - a12 * a12;
    float b11 = a22 / det;
    float b22 = a11 / det;
    float b12 = -a12 / det;
    minx = b11 * col1 + b12 * col2;
    maxx = b12 * col1 + b22 * col2;
    // cerr << cx << ' ' << a11<< ' '<< a12 << ' '<< a22 <<' ' << col2<<endl;
    // cerr << minx << ' ' << maxx << endl;

    a11 = 0.0, a22 = 0.0, a12 = 0.0, col1 = 0.0, col2 = 0.0; // matrix coef
    for (point = 0; point < points.size() / 4; ++point)
    {
        a11 += (1.0f - points[4 * point + 1] / cy) * (1.0f - points[4 * point + 1] / cy);
        a12 += (1.0f - points[4 * point + 1] / cy) * points[4 * point + 1] / cy;
        a22 += points[4 * point + 1] * points[4 * point + 1] / (cy * cy);
        col1 += (1.0f - points[4 * point + 1] / cy) * points[4 * point + 3] / tify;
        col2 += (points[4 * point + 1] / cy) * points[4 * point + 3] / tify;
    }
    det = a11 * a22 - a12 * a12;
    b11 = a22 / det;
    b22 = a11 / det;
    b12 = -a12 / det;
    maxy = 1.0f - (b11 * col1 + b12 * col2);
    miny = 1.0f - (b12 * col1 + b22 * col2);
    // cerr << miny << ' ' << maxy << endl;
}

int
ImageToTexture::compute(const char *)
{
    if (preHandleDiagnose_ == PROBLEM)
    {
        return FAIL;
    }
    else if (preHandleDiagnose_ == OUTPUT_DUMMY)
    {
        outputDummies();
        return SUCCESS;
    }
    if (mode_ == NONE)
    {
        sendError("Error when opening or reading file.");
        return FAIL;
    }

    // get image size
    TextureMapping::SizeControl effMode = mode_;

    // the user wants manual control
    if (p_fit_->getValue() == TextureMapping::FIT_MANUAL)
    {
        effMode = TextureMapping::MANUAL;
    }

    switch (effMode)
    {
    case TextureMapping::AUTOMATIC: // use file info
        Width_ = w_ / XResolution_;
        Height_ = h_ / YResolution_;
        if (resUnit_ == RESUNIT_INCH)
        {
            Width_ *= 2.54f; // 2.54 cm per inch
            Height_ *= 2.54f;
        }
        break;
    case TextureMapping::MANUAL: // use parameter values
        Width_ = p_width_->getValue();
        Height_ = p_height_->getValue();
        break;
    case NONE:
        fprintf(stderr, "ImageToTexture: NONE not handled\n");
        break;
    }

    // check image size
    if (Width_ <= 0.0)
    {
        sendError("Please, set a width larger than zero");
        return FAIL;
    }
    if (Height_ <= 0.0)
    {
        sendInfo("A non-positive image height was calculated, correcting this value assuming the same resolution in the X and Y dimensions");
        Height_ = (Width_ * h_) / w_;
    }

    // open the polygons object and its content
    const coDistributedObject *in_obj = p_poly_->getCurrentObject();
    if (!in_obj)
    {
        sendError("No input object");
        return FAIL;
    }
    if (!in_obj->objectOk())
    {
        sendError("Object is not OK");
        return FAIL;
    }
    if (!in_obj->isType("POLYGN"))
    {
        sendError("Object is not coDoPolygons");
        return FAIL;
    }

    const coDoPolygons *in_poly = dynamic_cast<const coDoPolygons *>(in_obj);

    int no_points = in_poly->getNumPoints();

    int *v_l, *l_l;
    float *x_c, *y_c, *z_c;
    in_poly->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);

    // and prepare coDoTexture object
    TextureMapping::Orientation orientation_hint;
    switch (p_orientation_->getValue())
    {
    case TextureMapping::FREE:
        orientation_hint = TextureMapping::FREE;
        break;
    case TextureMapping::PORTRAIT:
        orientation_hint = TextureMapping::PORTRAIT;
        break;
    case TextureMapping::LANDSCAPE:
        orientation_hint = TextureMapping::LANDSCAPE;
        break;
    default:
        sendError("Wrong value for Orientation");
        return FAIL;
        break;
    }
    TextureMapping::Fit fit_hint;
    switch (p_fit_->getValue())
    {
    case TextureMapping::FIT_MANUAL:
        fit_hint = TextureMapping::FIT_MANUAL;
        break;
    case TextureMapping::FIT:
        fit_hint = TextureMapping::FIT;
        break;
    case TextureMapping::USE_IMAGE:
        fit_hint = TextureMapping::USE_IMAGE;
        break;
    default:
        sendError("Wrong value for Fit");
        return FAIL;
        break;
    }

    // use shift if available
    float *x_c_s = x_c;
    float *y_c_s = y_c;
    const coDistributedObject *shift = p_shift_->getCurrentObject();
    if (shift != NULL)
    {
        if (!shift->objectOk())
        {
            sendError("Input field is not OK.");
            return FAIL;
        }
        if (!shift->isType("USTVDT"))
        {
            sendError("Input field is no vector.");
            return FAIL;
        }
        const coDoVec3 *v_shift = dynamic_cast<const coDoVec3 *>(shift);
        if (v_shift->getNumPoints() != no_points)
        {
            sendError("Input field does not match number of points.");
            return FAIL;
        }
        float *ux = NULL;
        float *uy = NULL;
        float *uz = NULL;
        v_shift->getAddresses(&ux, &uy, &uz);
        int point;
        x_c_s = new float[no_points];
        y_c_s = new float[no_points];
        for (point = 0; point < no_points; point++)
        {
            x_c_s[point] = x_c[point] - ux[point];
            y_c_s[point] = y_c[point] - uy[point];
        }
    }
    TextureMapping tm(x_c_s, y_c_s, no_points, orientation_hint, fit_hint, mode_,
                      Width_, Height_, BBoxes_.getBBox(lookUp_), p_mirror_->getValue());
    if (shift != NULL)
    {
        delete[] x_c_s;
        delete[] y_c_s;
    }

    int point;
    int *txIndex = new int[no_points];

    for (point = 0; point < no_points; point++)
    {
        txIndex[point] = point;
    }

    const float *txCoord[2];

    // try to open file in p_geometry_
    ifstream geometry(p_geometry_->getValue());
    float minx = p_xlimits_->getValue(0);
    float maxx = p_xlimits_->getValue(1);
    float miny = p_ylimits_->getValue(0);
    float maxy = p_ylimits_->getValue(1);

    if (geometry.rdbuf()->is_open())
    {
        float cx, cy, tifx, tify;
        vector<float> points;
        if (geometry >> cx >> cy >> tifx >> tify)
        {
            float tmp[4];
            // cerr << "Input: " << cx << tifx << endl;
            while (geometry >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3])
            {
                points.push_back(tmp[0]);
                points.push_back(tmp[1]);
                points.push_back(tmp[2]);
                points.push_back(tmp[3]);
            }
        }
        if (points.size() > 8 && points.size() % 4 == 0)
        {
            MinMaxCalculation(cx, cy, tifx, tify, points, minx, maxx, miny, maxy);
            p_xlimits_->setValue(0, minx);
            p_xlimits_->setValue(1, maxx);
            p_ylimits_->setValue(0, miny);
            p_ylimits_->setValue(1, maxy);
        }
    }

    tm.getMapping(&txCoord[0], &txCoord[1], minx, maxx, miny, maxy);

    coDoTexture *texture = new coDoTexture(p_texture_->getObjName(), pix_,
                                           0, 4, 0, no_points, txIndex, no_points,
                                           const_cast<float **>(txCoord));
    if (references_ > 0)
    {
        pix_->incRefCount();
    }
    ++references_;

    delete[] txIndex;

    p_texture_->setCurrentObject(texture);

    return SUCCESS;
}

MODULE_MAIN(Tools, ImageToTexture)
