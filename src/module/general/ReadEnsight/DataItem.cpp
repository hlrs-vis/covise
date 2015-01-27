/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                            ++
// ++             Implementation of class DataItem                          ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "DataItem.h"

//
// Constructor
//
DataItem::DataItem()
{
}

DataItem::DataItem(const int &type, const string &file, const string &desc)
    : type_(type)
    , fileName_(file)
    , desc_(desc)

{
}

//
// Destructor
//
DataItem::~DataItem()
{
}
