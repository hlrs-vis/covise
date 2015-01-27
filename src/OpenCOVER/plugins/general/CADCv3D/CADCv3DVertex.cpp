/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                 (C)2007 Titus Miloi, ZAIK/RRZK, University of Cologne  **
 **                                                                        **
 ** Description: 3d container classes for geometry transfer                **
 **                                                                        **
 **                                                                        **
 ** Author: Titus Miloi                                                    **
 **                                                                        **
 ** History:                                                               **
 ** 2007-08-02 v0.1                                                        **
 **                                                                        **
 ** $LastChangedDate: 2009-03-25 17:16:38 +0100 (Mi, 25 Mrz 2009) $
 ** $Revision: 770 $
 ** $LastChangedBy: miloit $
 **                                                                        **
\****************************************************************************/
#include "CADCv3DVertex.h"

#include <QtGlobal>
#include <qendian.h>

/****************************************************************************/
bool CADCv3DVertex::read(const void *data, unsigned int size)
{
    uchar *d = (uchar *)data;
    unsigned int offset = 0;
    unsigned int element = sizeof(quint64);
    unsigned int i = calcSize();
    if (size == 0)
        size = i;
    i = i < size ? i : size;

    if (offset + element <= i)
    {
        xx.data = qFromBigEndian<quint64>(d + offset);
        offset += element;
    }
    if (offset + element <= i)
    {
        yy.data = qFromBigEndian<quint64>(d + offset);
        offset += element;
    }
    if (offset + element <= i)
    {
        zz.data = qFromBigEndian<quint64>(d + offset);
        offset += element;
    }

    return (offset == calcSize());
}

/****************************************************************************/
void CADCv3DVertex::write(void *data) const
{
    uchar *d = (uchar *)data;
    unsigned int offset = 0;
    unsigned int element = sizeof(quint64);

    *(reinterpret_cast<quint64 *>(d + offset)) = qToBigEndian<quint64>(xx.data);
    offset += element;
    *(reinterpret_cast<quint64 *>(d + offset)) = qToBigEndian<quint64>(yy.data);
    offset += element;
    *(reinterpret_cast<quint64 *>(d + offset)) = qToBigEndian<quint64>(zz.data);
}

/****************************************************************************/
const CADCv3DVertex &CADCv3DVertex::copy(const CADCv3DVertex &src)
{
    setX(src.x());
    setY(src.y());
    setZ(src.z());
    return *this;
}

/****************************************************************************/
unsigned int CADCv3DVertex::size()
{
    return 3 * sizeof(quint64);
}

// END OF FILE
