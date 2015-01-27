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
#include "CADCv3DColor.h"

#include <qendian.h>

/****************************************************************************/
bool CADCv3DColor::read(const void *data, unsigned int size)
{
    uchar *d = (uchar *)data;
    unsigned int offset = 0;
    unsigned int element = sizeof(quint32);
    unsigned int i = calcSize();
    if (size == 0)
        size = i;
    i = i < size ? i : size;

    if (offset + element <= i)
    {
        r.data = qFromBigEndian<quint32>(d + offset);
        offset += element;
    }
    if (offset + element <= i)
    {
        g.data = qFromBigEndian<quint32>(d + offset);
        offset += element;
    }
    if (offset + element <= i)
    {
        b.data = qFromBigEndian<quint32>(d + offset);
        offset += element;
    }
    if (offset + element <= i)
    {
        a.data = qFromBigEndian<quint32>(d + offset);
        offset += element;
    }

    return (offset == calcSize());
}

/****************************************************************************/
void CADCv3DColor::write(void *data) const
{
    uchar *d = (uchar *)data;
    unsigned int offset = 0;
    unsigned int element = sizeof(quint32);

    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>(r.data);
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>(g.data);
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>(b.data);
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>(a.data);
}

/****************************************************************************/
const CADCv3DColor &CADCv3DColor::copy(const CADCv3DColor &src)
{
    setRed(src.red());
    setGreen(src.green());
    setBlue(src.blue());
    setAlpha(src.alpha());
    return *this;
}

/****************************************************************************/
unsigned int CADCv3DColor::size()
{
    return 4 * sizeof(quint32);
}

// END OF FILE
