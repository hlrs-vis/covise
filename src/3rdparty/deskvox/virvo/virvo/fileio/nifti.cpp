// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#if VV_HAVE_NIFTI

#include <cassert>
#include <cfloat>
#include <climits>
#include <iostream>
#include <ostream>

#include <nifti1_io.h>

#include <virvo/vvvoldesc.h>

#include "exceptions.h"
#include "nifti.h"

namespace virvo { namespace nifti {

void load(vvVolDesc* vd)
{
    bool verbose = true;

    // read nifti header ----------------------------------

    nifti_image* header = nifti_image_read(vd->getFilename(), 0);

    if (!header)
    {
        throw fileio::exception();
    }


    // dimensions

    vd->vox[0] = header->nx;
    vd->vox[1] = header->ny;
    vd->vox[2] = header->nz;

    vd->setDist(header->dx, header->dy, header->dz);

    // no support for animation

    vd->frames = 1;

    // bytes per pixel and num channels

    vd->setChan(1); // default

    // Determine chan and bpc, set default mapping
    switch (header->datatype)
    {
    case NIFTI_TYPE_RGB24:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_RGB24\n";
        vd->setChan(3);
        vd->bpc = header->nbyper / 3;
        break;
    case NIFTI_TYPE_RGBA32:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_RGB32\n";
        vd->setChan(4);
        vd->bpc = header->nbyper / 4;
        break;
    case NIFTI_TYPE_INT8:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_INT8\n";
        assert(header->nbyper == 1);
        vd->bpc = 1;
        vd->mapping(0) = vec2(CHAR_MIN, CHAR_MAX);
        break;
    case NIFTI_TYPE_UINT8:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_UINT8\n";
        assert(header->nbyper == 1);
        vd->bpc = 1;
        vd->mapping(0) = vec2(0, UCHAR_MAX);
        break;
    case NIFTI_TYPE_INT16:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_INT16\n";
        assert(header->nbyper == 2);
        vd->bpc = 2;
        vd->mapping(0) = vec2(SHRT_MIN, SHRT_MAX);
        break;
    case NIFTI_TYPE_UINT16:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_UINT16\n";
        assert(header->nbyper == 2);
        vd->bpc = 2;
        vd->mapping(0) = vec2(0, USHRT_MAX);
        break;
    case NIFTI_TYPE_INT32:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_INT32\n";
        assert(header->nbyper == 4);
        vd->bpc = 4;
        vd->mapping(0) = vec2(INT_MIN, INT_MAX);
        break;
    case NIFTI_TYPE_UINT32:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_UINT32\n";
        assert(header->nbyper == 4);
        vd->bpc = 4;
        vd->mapping(0) = vec2(0, UINT_MAX);
        break;
    case NIFTI_TYPE_FLOAT32:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_FLOAT32\n";
        vd->mapping(0) = vec2(-FLT_MAX, FLT_MAX);
        vd->bpc = 4;
        break;
    default:
        if (verbose) std::cout << "Datatype: UNKNOWN\n";
        vd->bpc = header->nbyper;
        break;
    }


    // read image data ------------------------------------

    nifti_image* data_section = nifti_image_read(vd->getFilename(), 1);


    if (!data_section)
    {
        throw fileio::exception();
    }

    uint8_t* raw = new uint8_t[vd->getFrameBytes()];
    memcpy(raw, static_cast<uint8_t*>(data_section->data), vd->getFrameBytes());
    vd->addFrame(raw, vvVolDesc::ARRAY_DELETE);


    float slope = header->scl_slope;
    float inter = header->scl_inter;

    if (verbose)
    {
        std::cout << "Intercept: " << inter << ", slope: " << slope << '\n';
    }


    // adapt data formats

    if (header->datatype == NIFTI_TYPE_INT16)
    {
        vd->mapping(0) = vec2(SHRT_MIN * slope + inter, SHRT_MAX * slope + inter);

        // Remap data
        for (ssize_t z = 0; z < vd->vox[2]; ++z)
        {
            for (ssize_t y = 0; y < vd->vox[1]; ++y)
            {
                for (ssize_t x = 0; x < vd->vox[0]; ++x)
                {
                    uint8_t* bytes = (*vd)(x, y, z);
                    int32_t voxel = (int)*reinterpret_cast<int16_t*>(bytes);
                    voxel -= SHRT_MIN;
                    *reinterpret_cast<uint16_t*>(bytes) = voxel;
                }
            }
        }
    }
    else if (header->datatype == NIFTI_TYPE_INT32)
    {
        vd->mapping(0) = vec2(INT_MIN * slope + inter, INT_MAX * slope + inter);

        // Remap data to float
        for (ssize_t z = 0; z < vd->vox[2]; ++z)
        {
            for (ssize_t y = 0; y < vd->vox[1]; ++y)
            {
                for (ssize_t x = 0; x < vd->vox[0]; ++x)
                {
                    uint8_t* bytes = (*vd)(x, y, z);
                    int32_t i = *reinterpret_cast<int32_t*>(bytes);
                    float f = static_cast<float>(i);
                    *reinterpret_cast<float*>(bytes) = f;
                }
            }
        }
    }
    else if (header->datatype == NIFTI_TYPE_UINT32)
    {
        vd->mapping(0) = vec2(inter, UINT_MAX * slope + inter);

        // Remap data to float
        for (ssize_t z = 0; z < vd->vox[2]; ++z)
        {
            for (ssize_t y = 0; y < vd->vox[1]; ++y)
            {
                for (ssize_t x = 0; x < vd->vox[0]; ++x)
                {
                    uint8_t* bytes = (*vd)(x, y, z);
                    unsigned u = *reinterpret_cast<unsigned*>(bytes);
                    float f = static_cast<float>(u);
                    *reinterpret_cast<float*>(bytes) = f;
                }
            }
        }
    }
    else
    {
        vd->mapping(0) *= slope;
        vd->mapping(0) += inter;
    }


    for (int c = 0; c < vd->getChan(); ++c)
    {
        vd->findMinMax(c, vd->range(c)[0], vd->range(c)[1]);
        vd->tf[c].setDefaultColors(vd->getChan() == 1 ? 0 : 4 + c, vd->range(c)[0], vd->range(c)[1]);
        vd->tf[c].setDefaultAlpha(0, vd->range(c)[0], vd->range(c)[1]);
    }
}

void save(const vvVolDesc* vd)
{
    int dims[] = { 3, (int)vd->vox[0], (int)vd->vox[1], (int)vd->vox[2], 1, 0, 0, 0 };

    int datatype = 0;
    if (vd->bpc == 1)
        datatype = NIFTI_TYPE_UINT8;
    else if (vd->bpc == 2)
        datatype = NIFTI_TYPE_UINT16;
    else if (vd->bpc == 4)
        datatype = NIFTI_TYPE_FLOAT32;

    nifti_image* img = nifti_make_new_nim(dims, datatype, 1);

    nifti_set_filenames(img, vd->getFilename(), 0, 0);

    img->dx = vd->getDist()[0];
    img->dy = vd->getDist()[1];
    img->dz = vd->getDist()[2];

    img->data = vd->getRaw();

    nifti_image_write(img);
}

}} // namespace virvo::nifti

#endif // VV_HAVE_NIFTI
