/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE Logo
//
//  This module creates a rectangle and a texture (logo)
//
//  Initial version: 2001-04-08 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _LOGO_H_
#define _LOGO_H_

#include <tiffio.h>

#define _HAS_UINT

#include <api/coModule.h>
using namespace covise;

class Logo : public coModule
{
public:
    /// constructor
    Logo();
    /// destructor
    ~Logo();

protected:
private:
    void Transform(float *x_c, float *y_c, float *z_c);
    virtual int compute();
    float changeFileName();
    // info from tif file
    uint32 w_, h_; // image size in pixels
    //uint16 resUnit_;
    uint32 *raster_;
    TIFF *bild_;
    // output ports
    coOutputPort *p_do_geometry_;
    // params
    coFileBrowserParam *p_file_;
    coFloatParam *p_size_; // x size
    coBooleanParam *p_logo_; // set LOGO attribute
    coFloatVectorParam *p_position_; // position of the middle point
    coFloatVectorParam *p_cardan_bryant_; // euler angles for orientation
};
#endif
