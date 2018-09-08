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

#pragma once

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif


#if VV_HAVE_GDCM

#include <virvo/vvpixelformat.h>

class vvVolDesc;

namespace virvo { namespace gdcm {

struct dicom_meta
{
    PixelFormat format = PF_UNSPECIFIED;
    float slope = 1.0;
    float intercept = 0.0;
    int sequence = -1;
    int slice = -1;
    double spos = 0.0;
};

bool can_load(const vvVolDesc *vd);
dicom_meta load(vvVolDesc* vd);

}} // namespace virvo::gdcm

#endif // VV_HAVE_GDCM
