/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

   * License: LGPL 2+ */
#ifndef MATRIX_SERIALIZER_H
#define MATRIX_SERIALIZER_H
#include <vrbclient/SharedStateSerializer.h>
#include <osg/Matrix>
#include <util/coExport.h>

namespace vrb {
template <>
void COVEREXPORT serialize<osg::Matrix>(covise::TokenBuffer &tb, const osg::Matrix &value);

template <>
void COVEREXPORT deserialize<osg::Matrix>(covise::TokenBuffer &tb, osg::Matrix &value);
}


#endif // !1
