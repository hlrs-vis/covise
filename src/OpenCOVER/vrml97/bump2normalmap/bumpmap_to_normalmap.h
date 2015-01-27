/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BUMPMAP_TO_NORMALMAP_H
#define BUMPMAP_TO_NORMALMAP_H

#include "glh_linear.h"
#include "glh_array.h"

void bumpmap_to_normalmap(const glh::array2<unsigned char> &src,
                          glh::array2<glh::vec3ub> &dst,
                          glh::vec3f scale = glh::vec3f());

void bumpmap_to_normalmap(const glh::array2<unsigned char> &src,
                          glh::array2<glh::vec3f> &dst,
                          glh::vec3f scale = glh::vec3f());

void bumpmap_to_normalmap(const glh::array2<unsigned char> &src,
                          glh::array2<glh::vec4f> &dst,
                          glh::vec3f scale = glh::vec3f());

void bumpmap_to_mipmap_normalmap(const glh::array2<unsigned char> &src,
                                 glh::array2<glh::vec3f> *&dst, int &nlevels,
                                 glh::vec3f scale = glh::vec3f());
#endif
