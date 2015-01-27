/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                           (C) 2005 VISENSO          ++
// ++ Description:                                                        ++
// ++             Implementation of class AutoColors                      ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@visenso.de)                               ++
// ++                                                                     ++
// ++               VISENSO GmbH                                          ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                               ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "AutoColors.h"
AutoColors *AutoColors::instance_ = NULL;

//
// Constructor
//
AutoColors::AutoColors()
    : idx_(0)
{
    // init
    colors_.clear();
    colors_.push_back("255 0 0");
    colors_.push_back("0 255 0");
    colors_.push_back("0 0 255");
    colors_.push_back("255 255 0");
    colors_.push_back("0 255 255");
    colors_.push_back("255 255 255");
    colors_.push_back("255 0 255");
    colors_.push_back("127 127 0");
    colors_.push_back("0 127 127");
    colors_.push_back("255 0 127");
}

//
// Acess Method
//
AutoColors *
AutoColors::instance()
{
    if (instance_ == NULL)
    {
        instance_ = new AutoColors();
    }
    return instance_;
}

string
AutoColors::next()
{
    size_t actSize = colors_.size();
    if (idx_ == actSize)
        idx_ = 0;
    string ret = colors_[idx_];
    idx_++;
    return ret;
}

void
AutoColors::reset()
{
    idx_ = 0;
}

//
// Destructor
//
AutoColors::~AutoColors()
{
}
