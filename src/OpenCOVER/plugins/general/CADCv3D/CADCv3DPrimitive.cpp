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
#include "CADCv3DPrimitive.h"

#include <qendian.h>

/****************************************************************************/
int CADCv3DPrimitive::getIndex(int pos) const
{
    if ((pos < 0) || (pos >= countIndices()))
        return -1;
    else
        return indices.at(pos);
}

/****************************************************************************/
void CADCv3DPrimitive::invert()
{
    std::vector<int> tmp;
    std::vector<int>::reverse_iterator i = indices.rbegin();
    while (i != indices.rend())
        tmp.push_back(*(i++));
    indices.swap(tmp);
}

/****************************************************************************/
void CADCv3DPrimitive::clear()
{
    indices.clear();
    type = TYPE_POINTS;
}

/****************************************************************************/
unsigned int CADCv3DPrimitive::calcSize() const
{
    // number of indices (quint32) + indices (each quint32)
    return (sizeof(quint32) * (2 + static_cast<unsigned int>(indices.size())));
}

/****************************************************************************/
bool CADCv3DPrimitive::read(const void *data, unsigned int size)
{
    clear();

    const quint32 *d = reinterpret_cast<const quint32 *>(data);
    unsigned int element = sizeof(quint32);

    if ((size < 2 * element) && (size != 0))
        return false;

    quint32 count = qFromBigEndian<quint32>(*d);
    quint32 t = qFromBigEndian<quint32>(*(d + 1));
    if (size == 0)
        size = (count + 2) * element;
    if ((count + 2) * element > size)
        return false;

    type = (int)t;
    for (unsigned int i = 0; i < count; i++)
        indices.push_back(qFromBigEndian<quint32>(*(d + 2 + i)));

    return true;
}

/****************************************************************************/
void CADCv3DPrimitive::write(void *data) const
{
    quint32 *d = static_cast<quint32 *>(data);
    quint32 is = (quint32)indices.size();
    quint32 it = (quint32)type;

    *d = qToBigEndian<quint32>(is);
    *(d + 1) = qToBigEndian<quint32>(it);

    for (unsigned int i = 0; i < indices.size(); i++)
        *(d + 2 + i) = qToBigEndian<quint32>(indices.at(i));
}

/****************************************************************************/
const CADCv3DPrimitive &CADCv3DPrimitive::copy(const CADCv3DPrimitive &src)
{
    clear();
    setType(src.getType());
    int c = countIndices();
    for (int i = 0; i < c; i++)
        pushIndex(src.getIndex(i));
    return *this;
}

// END OF FILE
