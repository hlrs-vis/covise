/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Logo.h"

int
main(int argc, char *argv[])
{
    Logo *application = new Logo;

    application->start(argc, argv);

    return 0;
}

Logo::Logo()
    : bild_(NULL)
    , raster_(NULL)
{
    // output ports
    p_do_geometry_ = addOutputPort("do_geometry", "coDoGeometry", "do_geometry object");
    // params
    p_file_ = addFileBrowserParam("logo_file", "Logo file");
    p_file_->setValue("/tmp/foo.tif", "*.tif;*.tiff");

    p_size_ = addFloatParam("SizeX", "x size");
    p_size_->setValue(1.0);

    p_position_ = addFloatVectorParam("Position", "position");
    p_position_->setValue(0.0, 0.0, 0.0);

    p_cardan_bryant_ = addFloatVectorParam("Euler_XYZ", "Euler XYZ");
    p_cardan_bryant_->setValue(0.0, 0.0, 0.0);

    p_logo_ = addBooleanParam("Fixed", "Fixed logo");
    p_logo_->setValue(1);
}

Logo::~Logo()
{
}

int
Logo::compute()
{
    // coDoPixelImage...
    float Height = changeFileName();
    if (Height < 0.0)
    {
        return FAIL;
    }

    // create the output geometry
    string poly_out_name = p_do_geometry_->getObjName();
    poly_out_name += "_Polygons";
    coDoPolygons *poly_out = new coDoPolygons(poly_out_name.c_str(), 4, 4, 1);
    float *x_c, *y_c, *z_c;
    int *v_l, *p_l;
    poly_out->getAddresses(&x_c, &y_c, &z_c, &v_l, &p_l);
    float corner_x = 0.5 * p_size_->getValue();
    float corner_y = 0.5 * Height;
    x_c[0] = corner_x;
    y_c[0] = corner_y;
    x_c[1] = -corner_x;
    y_c[1] = corner_y;
    x_c[2] = -corner_x;
    y_c[2] = -corner_y;
    x_c[3] = corner_x;
    y_c[3] = -corner_y;
    z_c[0] = z_c[1] = z_c[2] = z_c[3] = 0.0;
    v_l[0] = 0;
    v_l[1] = 1;
    v_l[2] = 2;
    v_l[3] = 3;
    p_l[0] = 0;
    // translate and rotate
    Transform(x_c, y_c, z_c);

    string namebuf = p_do_geometry_->getObjName();
    namebuf += "_Pix";
    char *image = new char[4 * w_ * h_];
    char *iPtr = image;
    int pixel = 0;
    for (; pixel < w_ * h_; ++pixel)
    {
        *iPtr = TIFFGetR(raster_[pixel]);
        iPtr++;
        *iPtr = TIFFGetG(raster_[pixel]);
        iPtr++;
        *iPtr = TIFFGetB(raster_[pixel]);
        iPtr++;
        *iPtr = TIFFGetA(raster_[pixel]);
        iPtr++;
    }
    coDoPixelImage *pix = new coDoPixelImage(namebuf.c_str(), w_, h_, 4, 4, image);
    delete[] image;
    // and now the texture...
    float *txCoord[2];
    float mapping[] = // X coord
        {
          1.0, 0.0, 0.0, 1.0,
          1.0, 1.0, 0.0, 0.0 // Y coord
        };
    txCoord[0] = mapping;
    txCoord[1] = mapping + 4;
    string texture_name = p_do_geometry_->getObjName();
    texture_name += "Texture";
    coDoTexture *texture = new coDoTexture(texture_name.c_str(), pix,
                                           0, 4, 0, 4 /* no_points */,
                                           v_l, 4 /* no_points */,
                                           txCoord);
    delete pix;

    // make a coDoGeometry...
    coDoGeometry *dogeom = new coDoGeometry(p_do_geometry_->getObjName(), poly_out);
    dogeom->setTexture(0, texture);

    if (p_logo_->getValue())
    {
        dogeom->addAttribute("LOGO", "");
    }

    p_do_geometry_->setCurrentObject(dogeom);

    delete poly_out;
    delete texture;

    return SUCCESS;
}

float
Logo::changeFileName()
{
    // open file
    bild_ = TIFFOpen(p_file_->getValue(), "r");
    if (bild_ == NULL)
    {
        sendError("Could not open file.");
        return -1.0;
    }
    // read width in pixels
    if (TIFFGetField(bild_, TIFFTAG_IMAGEWIDTH, &w_) != 1)
    {
        sendError("Could not allocate memory for raster_.");
        TIFFClose(bild_);
        bild_ = NULL;
        return -1.0;
    }
    // read height in pixels
    if (TIFFGetField(bild_, TIFFTAG_IMAGELENGTH, &h_) != 1)
    {
        sendError("Could not read image.");
        TIFFClose(bild_);
        bild_ = NULL;
        return -1.0;
    }
    // @@@ problems with segmented memory (Windows...)
    delete[] raster_;
    raster_ = new uint32[w_ * h_];

    if (raster_ == NULL)
    {
        sendError("Could not allocate memory for raster_.");
        TIFFClose(bild_);
        bild_ = NULL;
        return -1.0;
    }
    // read image content
    if (TIFFReadRGBAImage(bild_, w_, h_, raster_, 0) != 1)
    {
        sendError("Could not read TIFF image.");
        TIFFClose(bild_);
        delete[] raster_; // @@@ problems with segmented memory (Windows...)
        bild_ = NULL;
        raster_ = NULL;
        return -1.0;
    }

    float XResolution, YResolution;
    if (TIFFGetField(bild_, TIFFTAG_XRESOLUTION, &XResolution) != 1)
    {
        sendError("Could not read X Resolution.");
        return -1.0;
    }
    else if (TIFFGetField(bild_, TIFFTAG_YRESOLUTION, &YResolution) != 1)
    {
        sendInfo("Could not read Y Resolution. Assuming the same value as for the X dimension.");
        YResolution = XResolution;
    }
    TIFFClose(bild_);
    bild_ = NULL;

    float Width, Height;
    Width = w_ / XResolution;
    Height = h_ / YResolution;

    if (p_size_->getValue() <= 0.0)
    {
        sendError("A positive value is expected in SizeX");
        return -1.0;
    }

    return p_size_->getValue() * Height / Width;
}

#include <assert.h>
#include <math.h>

void
Rotation(float *x_c, float *y_c, float *z_c, float angle, int axis)
{
    assert(axis >= 0 && axis <= 2);
    float matrix[3][3] = {
        { 1.0, 0.0, 0.0 },
        { 0.0, 1.0, 0.0 },
        { 0.0, 0.0, 1.0 }
    };
    angle *= M_PI / 180.0;
    switch (axis)
    {
    case 0:
        matrix[1][1] = matrix[2][2] = cos(angle);
        matrix[2][1] = sin(angle);
        matrix[1][2] = -matrix[2][1];
        break;
    case 1:
        matrix[0][0] = matrix[2][2] = cos(angle);
        matrix[0][2] = sin(angle);
        matrix[2][0] = -matrix[0][2];
        break;
    case 2:
        matrix[0][0] = matrix[1][1] = cos(angle);
        matrix[1][0] = sin(angle);
        matrix[0][1] = -matrix[1][0];
        break;
    }

    float x_t[4], y_t[4], z_t[4];
    int point;
    for (point = 0; point < 4; ++point)
    {
        x_t[point] = matrix[0][0] * x_c[point] + matrix[0][1] * y_c[point] + matrix[0][2] * z_c[point];
        y_t[point] = matrix[1][0] * x_c[point] + matrix[1][1] * y_c[point] + matrix[1][2] * z_c[point];
        z_t[point] = matrix[2][0] * x_c[point] + matrix[2][1] * y_c[point] + matrix[2][2] * z_c[point];

        x_c[point] = x_t[point];
        y_c[point] = y_t[point];
        z_c[point] = z_t[point];
    }
}

void
Logo::Transform(float *x_c, float *y_c, float *z_c)
{
    Rotation(x_c, y_c, z_c, p_cardan_bryant_->getValue(2), 2);
    Rotation(x_c, y_c, z_c, p_cardan_bryant_->getValue(1), 1);
    Rotation(x_c, y_c, z_c, p_cardan_bryant_->getValue(0), 0);
    int i;
    for (i = 0; i < 4; ++i)
    {
        x_c[i] += p_position_->getValue(0);
        y_c[i] += p_position_->getValue(1);
        z_c[i] += p_position_->getValue(2);
    }
}
