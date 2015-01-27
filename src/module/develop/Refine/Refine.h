/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// MODULE   Refine
//
// Description: Projects Magma Fill dato onto a given surface
//
// Initial version: dd.mm.2004
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2004 by VirCinity GmbH
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  Changes:
//

#ifndef MARKSURFACE_H
#define MARKSURFACE_H

#include <util/coviseCompat.h>
#include <api/coSimpleModule.h>

using namespace covise;

class Refine : public coSimpleModule
{
public:
    /// default CONSTRUCTOR
    Refine();

    /// DESTRUCTOR
    ~Refine();

private:
    virtual int compute();

    coInputPort *surfPort_;
    coInputPort *inDataPort_;

    coOutputPort *outSurfPort_;
    coOutputPort *outDataPort_;

    coFloatSliderParam *thresholdParam_;
};

#endif
