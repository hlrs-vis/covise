/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE ImageToTexture
//
//  This module creates a texture for polygons given a tiff file (or other formats..)
//
//  Initial version: 2001-04-08 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _IMAGE_TO_TEXTURE_H
#define _IMAGE_TO_TEXTURE_H

#include <util/coviseCompat.h>
#include <do/coDoPixelImage.h>
#include <tiffio.h>

#define _HAS_UINT

#include <api/coSimpleModule.h>
using namespace covise;
#include <string>
#include "TextureMapping.h"
#include "BBoxes.h"
#include <api/coHideParam.h>

class ImageToTexture : public coSimpleModule
{
public:
    /// constructor
    ImageToTexture(int argc, char *argv[]);
    /// destructor
    virtual ~ImageToTexture();

protected:
private:
    virtual void param(const char *paramName, bool inMapLoading);
    /// the file name has changed and we want to reread the file
    /// return 0 if OK
    void changeFileName();
    /// overriding param values through an "IMAGE_TO_TEXTURE" attribute
    void useImageToTextureAttribute(const coDistributedObject *);
    std::string FileName_;

    virtual int compute(const char *port);
    virtual void postInst();
    virtual void preHandleObjects(coInputPort **);
    virtual void setIterator(coInputPort **, int);
    virtual void postHandleObjects(coOutputPort **);

    /// if we do not know waht to do
    void outputDummies();

    BBoxes BBoxes_; // bounding boxes...
    int lookUp_; // iteration index (for time steps?)

    coFileBrowserParam *p_file_;
    coFileBrowserParam *p_geometry_;
    // image "real" dimensions
    coFloatParam *p_width_;
    coFloatParam *p_height_;

    coFloatVectorParam *p_xlimits_;
    coFloatVectorParam *p_ylimits_;

    // pixel texture size
    enum
    {
        AUTO_SIZE = 1,
        MANUAL_SIZE = 2
    };
    coChoiceParam *p_AutoOrManual_;
    coIntScalarParam *p_XSize_;
    coIntScalarParam *p_YSize_;

    coChoiceParam *p_fit_;
    coChoiceParam *p_orientation_;
    // group objects in a time step
    coBooleanParam *p_groupBB_;
    // mirroring or translation
    coBooleanParam *p_mirror_;

    // overriding param values if attribute "IMAGE_TO_TEXTURE is available
    coHideParam *h_width_;
    coHideParam *h_height_;
    coHideParam *h_XSize_;
    coHideParam *h_YSize_;
    coHideParam *h_fit_;
    coHideParam *h_orientation_;
    coHideParam *h_groupBB_;
    std::vector<coHideParam *> hparams_;

    // the geometry
    coInputPort *p_poly_;
    // the projection (from original to actual geometry)
    coInputPort *p_shift_;
    // overriding p_file_
    coInputPort *p_file_name_;
    // the output
    coOutputPort *p_texture_;

    uint32 w_, h_; // image size in pixels

    uint16 resUnit_;
    float Width_; // width of file image
    float Height_; // height of file image
    // output limX, limY (texture coordinates) and modify orientation_
    /*
         void LimitMapping(float wo, // object (polygons) width
                           float ho, // object height
                           float *limX, // limit in X-texture coordinates
                           float *limY); // limit in Y-texture coordinates
      */
    float XResolution_; // pixels per unit length
    float YResolution_;
    TextureMapping::SizeControl mode_; // mapping (projection) mode

    uint32 *raster_; // tif image content as got through TIFFReadRGBAImage:
    // raw RGBA

    unsigned int XSize_; // pixel sizes of tif file
    unsigned int YSize_;
    // char *image_; // keep image content: "readable" RGBA

    TIFF *bild_; // file handler
    coDoPixelImage *pix_;
    int references_;

    enum
    {
        OUTPUT_DUMMY,
        PROBLEM,
        OK
    } preHandleDiagnose_;
    bool changeFileNameFailed_;
    bool WasConnected_;
};
#endif
