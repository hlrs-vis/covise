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
#include "CADCv3DGeometry.h"

#include <qendian.h>

/****************************************************************************/
bool CADCv3DGeometry::getColor(int pos, CADCv3DColor &col) const
{
    if ((pos < 0) || (pos >= countColors()))
        return false;
    else
    {
        col = *(colors.at(pos));
        return true;
    }
}

/****************************************************************************/
CADCv3DColor *CADCv3DGeometry::getColorP(int pos)
{
    if ((pos < 0) || (pos >= countColors()))
        return 0;
    else
        return colors.at(pos);
}

/****************************************************************************/
bool CADCv3DGeometry::getVertex(int pos, CADCv3DVertex &v) const
{
    if ((pos < 0) || (pos >= countVertices()))
        return false;
    else
    {
        v = *(vertices.at(pos));
        return true;
    }
}

/****************************************************************************/
CADCv3DVertex *CADCv3DGeometry::getVertexP(int pos)
{
    if ((pos < 0) || (pos >= countVertices()))
        return 0;
    else
        return vertices.at(pos);
}

/****************************************************************************/
bool CADCv3DGeometry::getNormal(int pos, CADCv3DVertex &n) const
{
    if ((pos < 0) || (pos >= countNormals()))
        return false;
    else
    {
        n = *(normals.at(pos));
        return true;
    }
}

/****************************************************************************/
CADCv3DVertex *CADCv3DGeometry::getNormalP(int pos)
{
    if ((pos < 0) || (pos >= countNormals()))
        return 0;
    else
        return normals.at(pos);
}

/****************************************************************************/
bool CADCv3DGeometry::getPrimitive(int pos, CADCv3DPrimitive &p) const
{
    if ((pos < 0) || (pos >= countPrimitives()))
        return false;
    else
    {
        p = *(primitives.at(pos));
        return true;
    }
}

/****************************************************************************/
CADCv3DPrimitive *CADCv3DGeometry::getPrimitiveP(int pos)
{
    if ((pos < 0) || (pos >= countPrimitives()))
        return 0;
    else
        return primitives.at(pos);
}

/****************************************************************************/
void CADCv3DGeometry::clear()
{
    name = "";
    colorBinding = 0;
    normalBinding = 0;
    // delete colors
    {
        std::vector<CADCv3DColor *>::iterator i = colors.begin();
        while (i != colors.end())
            delete *(i++);
        colors.clear();
    }
    // delete vertices
    {
        std::vector<CADCv3DVertex *>::iterator i = vertices.begin();
        while (i != vertices.end())
            delete *(i++);
        vertices.clear();
    }
    // delete normals
    {
        std::vector<CADCv3DVertex *>::iterator i = normals.begin();
        while (i != normals.end())
            delete *(i++);
        normals.clear();
    }
    // delete primitives
    {
        std::vector<CADCv3DPrimitive *>::iterator i = primitives.begin();
        while (i != primitives.end())
            delete *(i++);
        primitives.clear();
    }
}

/****************************************************************************/
unsigned int CADCv3DGeometry::calcSize() const
{
    // sizeof(quint32) * (name length + color binding + normal binding + number of colors +
    // number of vertices + number of normals + number of primitives) +
    // name length + colors + vertices + nomrals + primitives
    unsigned int primSize = 0;
    std::vector<CADCv3DPrimitive *>::const_iterator i = primitives.begin();
    while (i != primitives.end())
        primSize += (*(i++))->calcSize();

    return (
        (sizeof(quint32) * 7) + (static_cast<unsigned int>(colors.size()) * CADCv3DColor::size()) + (static_cast<unsigned int>(vertices.size()) * CADCv3DVertex::size()) + (static_cast<unsigned int>(normals.size()) * CADCv3DVertex::size()) + static_cast<unsigned int>(name.length()) + 1 + primSize);
}

/****************************************************************************/
bool CADCv3DGeometry::read(const void *data, unsigned int size)
{
    clear();

    const quint32 *d = static_cast<const quint32 *>(data);
    unsigned int element = sizeof(quint32);

    if ((size < 7 * element) && (size != 0))
        return false;

    quint32 nl = qFromBigEndian<quint32>(*(d++)); // name length
    quint32 cb = qFromBigEndian<quint32>(*(d++)); // color binding
    quint32 nb = qFromBigEndian<quint32>(*(d++)); // normal binding
    quint32 cn = qFromBigEndian<quint32>(*(d++)); // # colors
    quint32 vn = qFromBigEndian<quint32>(*(d++)); // # vertices
    quint32 nn = qFromBigEndian<quint32>(*(d++)); // # normals
    quint32 pn = qFromBigEndian<quint32>(*(d++)); // # primitives
    if (size > 0)
    {
        if (size >= (7 * element) + nl)
            size -= (7 * element) + nl;
        else
            return false;
    }
    const char *dc = reinterpret_cast<const char *>(d);
    name = "";
    if (nl > 0)
        for (quint32 i = 0; i < nl - 1; i++)
            name += *(dc++);
    d = reinterpret_cast<const quint32 *>(dc + 1);

    bool success = true;

    colorBinding = (int)cb;
    normalBinding = (int)nb;

    CADCv3DColor *c;
    for (quint32 i = 0; (i < cn) && success; i++)
    {
        c = new CADCv3DColor();
        if ((success = c->read(d, size)) == true)
        {
            colors.push_back(c);
            unsigned int s = c->calcSize();
            if (size > 0)
            {
                if (size >= s)
                    size -= s;
                else
                    return false;
            }
            d = reinterpret_cast<const quint32 *>(reinterpret_cast<const char *>(d) + s);
        }
        else
            delete c;
    }

    CADCv3DVertex *v;
    for (quint32 i = 0; (i < vn) && success; i++)
    {
        v = new CADCv3DVertex();
        if ((success = v->read(d, size)) == true)
        {
            vertices.push_back(v);
            unsigned int s = v->calcSize();
            if (size > 0)
            {
                if (size >= s)
                    size -= s;
                else
                    return false;
            }
            d = reinterpret_cast<const quint32 *>(reinterpret_cast<const char *>(d) + s);
        }
        else
            delete v;
    }

    for (quint32 i = 0; (i < nn) && success; i++)
    {
        v = new CADCv3DVertex();
        if ((success = v->read(d, size)) == true)
        {
            normals.push_back(v);
            unsigned int s = v->calcSize();
            if (size > 0)
            {
                if (size >= s)
                    size -= s;
                else
                    return false;
            }
            d = reinterpret_cast<const quint32 *>(reinterpret_cast<const char *>(d) + s);
        }
        else
            delete v;
    }

    CADCv3DPrimitive *p;
    for (quint32 i = 0; (i < pn) && success; i++)
    {
        p = new CADCv3DPrimitive();
        if ((success = p->read(d, size)) == true)
        {
            primitives.push_back(p);
            unsigned int s = p->calcSize();
            if (size > 0)
            {
                if (size >= s)
                    size -= s;
                else
                    return false;
            }
            d = reinterpret_cast<const quint32 *>(reinterpret_cast<const char *>(d) + s);
        }
        else
            delete p;
    }

    return success;
}

/****************************************************************************/
void CADCv3DGeometry::write(void *data) const
{
    quint8 *d = reinterpret_cast<quint8 *>(data);
    unsigned int element = sizeof(quint32);
    unsigned int offset = 0;
    unsigned int nl = static_cast<unsigned int>(name.length()) + 1;

    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>((quint32)nl);
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>((quint32)colorBinding);
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>((quint32)normalBinding);
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>((quint32)colors.size());
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>((quint32)vertices.size());
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>((quint32)normals.size());
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>((quint32)primitives.size());
    offset += element;

    // name
    for (unsigned int i = 0; i < name.length(); i++)
        *(d + (offset++)) = name[i];
    *(d + (offset++)) = '\0';

    // colors
    std::vector<CADCv3DColor *>::const_iterator i1 = colors.begin();
    while (i1 != colors.end())
    {
        (*i1)->write(reinterpret_cast<void *>(d + offset));
        offset += (*i1)->calcSize();
        i1++;
    }
    // vertices
    std::vector<CADCv3DVertex *>::const_iterator i2 = vertices.begin();
    while (i2 != vertices.end())
    {
        (*i2)->write(reinterpret_cast<void *>(d + offset));
        offset += (*i2)->calcSize();
        i2++;
    }
    // normals
    i2 = normals.begin();
    while (i2 != normals.end())
    {
        (*i2)->write(reinterpret_cast<void *>(d + offset));
        offset += (*i2)->calcSize();
        i2++;
    }
    // polygons
    std::vector<CADCv3DPrimitive *>::const_iterator i3 = primitives.begin();
    while (i3 != primitives.end())
    {
        (*i3)->write(reinterpret_cast<void *>(d + offset));
        offset += (*i3)->calcSize();
        i3++;
    }
}

/****************************************************************************/
const CADCv3DGeometry &CADCv3DGeometry::copy(const CADCv3DGeometry &src)
{
    int c, i;

    clear();
    setName(src.getName());
    setColorBinding(src.getColorBinding());
    setNormalBinding(src.getNormalBinding());

    CADCv3DGeometry &s = const_cast<CADCv3DGeometry &>(src);

    // copy the colors
    c = src.countColors();
    for (i = 0; i < c; i++)
        pushColorP(new CADCv3DColor(*(s.getColorP(i))));
    // copy the vertices
    c = src.countVertices();
    for (i = 0; i < c; i++)
        pushVertexP(new CADCv3DVertex(*(s.getVertexP(i))));
    // copy the normals
    c = src.countNormals();
    for (i = 0; i < c; i++)
        pushNormalP(new CADCv3DVertex(*(s.getNormalP(i))));
    // copy the primitives
    c = src.countPrimitives();
    for (i = 0; i < c; i++)
        pushPrimitiveP(new CADCv3DPrimitive(*(s.getPrimitiveP(i))));

    return *this;
}

// END OF FILE
