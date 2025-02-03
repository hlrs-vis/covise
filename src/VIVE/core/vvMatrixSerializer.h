/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

   * License: LGPL 2+ */
#ifndef MATRIX_SERIALIZER_H
#define MATRIX_SERIALIZER_H
#include <net/tokenbuffer_serializer.h>
#include <vsg/maths/mat4.h>
#include <util/coExport.h>

namespace covise {
template <>
void VVCORE_EXPORT serialize<vsg::dmat4>(covise::TokenBuffer &tb, const vsg::dmat4 &value);

template <>
void VVCORE_EXPORT deserialize<vsg::dmat4>(covise::TokenBuffer &tb, vsg::dmat4 &value);
}


#endif // !1
