/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <vector>

struct HDRI
{
  void load(std::string fileName);
  unsigned width;
  unsigned height;
  unsigned numComponents;
  std::vector<float> pixel;
};




