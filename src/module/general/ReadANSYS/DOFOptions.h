/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DOF_OPTIONS_H
#define _DOF_OPTIONS_H

#include <vector>
#include <string>

struct DOFOptions
{
    std::string *options_;
    std::vector<int> *codes_;
    int num_options_;
    DOFOptions()
        : options_(NULL)
        , codes_(NULL)
        , num_options_(0)
    {
    }
};
#endif
