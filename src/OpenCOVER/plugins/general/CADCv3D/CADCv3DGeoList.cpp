/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                 (C)2007 Titus Miloi, ZAIK/RRZK, University of Cologne  **
 **                                                                        **
 ** Description: common files for the CAD Conversion client and server     **
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
#include "CADCv3DGeoList.h"

#include <QtCore>

/****************************************************************************/
bool CADCv3DGeoList::newGeo()
{
    CADCv3DGeometry *g = new CADCv3DGeometry();
    geometries.push_back(g);
    geoIter = geometries.size() - 1;
    prmIter = -1;
    vrtIter = -1;
    nrmIter = -1;
    colIter = -1;
    idxIter = -1;
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::firstGeo()
{
    if (geometries.empty())
        return false;

    geoIter = 0;
    prmIter = -1;
    vrtIter = -1;
    nrmIter = -1;
    colIter = -1;
    idxIter = -1;
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::nextGeo()
{
    if ((geoIter < 0) || (geoIter >= (int)geometries.size() - 1))
        return false;

    geoIter++;
    prmIter = -1;
    vrtIter = -1;
    nrmIter = -1;
    colIter = -1;
    idxIter = -1;
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::newGeoPrimitive(int type)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *g = geometries.at(geoIter);
    CADCv3DPrimitive *p = new CADCv3DPrimitive(type);
    g->pushPrimitiveP(p);
    prmIter = g->countPrimitives() - 1;
    idxIter = -1;
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::firstGeoPrimitive(int &type)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *g = geometries.at(geoIter);
    if (g->countPrimitives() == 0)
        return false;

    prmIter = 0;
    idxIter = -1;

    type = g->getPrimitiveP(prmIter)->getType();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::nextGeoPrimitive(int &type)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *g = geometries.at(geoIter);
    if ((prmIter < 0) || (prmIter >= g->countPrimitives() - 1))
        return false;

    prmIter++;
    idxIter = -1;

    type = g->getPrimitiveP(prmIter)->getType();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::newGeoVertex(double x, double y, double z)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *g = geometries.at(geoIter);
    CADCv3DVertex *v = new CADCv3DVertex(x, y, z);
    g->pushVertexP(v);
    vrtIter = g->countVertices() - 1;
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::firstGeoVertex(double &x, double &y, double &z)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *g = geometries.at(geoIter);
    if (g->countVertices() == 0)
        return false;

    vrtIter = 0;

    CADCv3DVertex *v = g->getVertexP(vrtIter);
    x = v->x();
    y = v->y();
    z = v->z();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::nextGeoVertex(double &x, double &y, double &z)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *g = geometries.at(geoIter);
    if ((vrtIter < 0) || (vrtIter >= g->countVertices() - 1))
        return false;

    vrtIter++;

    CADCv3DVertex *v = g->getVertexP(vrtIter);
    x = v->x();
    y = v->y();
    z = v->z();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::newGeoNormal(double x, double y, double z)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *g = geometries.at(geoIter);
    CADCv3DVertex *n = new CADCv3DVertex(x, y, z);
    g->pushNormalP(n);
    nrmIter = g->countNormals() - 1;
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::firstGeoNormal(double &x, double &y, double &z)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *g = geometries.at(geoIter);
    if (g->countNormals() == 0)
        return false;

    nrmIter = 0;

    CADCv3DVertex *n = g->getNormalP(nrmIter);
    x = n->x();
    y = n->y();
    z = n->z();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::nextGeoNormal(double &x, double &y, double &z)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *g = geometries.at(geoIter);
    if ((nrmIter < 0) || (nrmIter >= (int)g->countNormals() - 1))
        return false;

    nrmIter++;

    CADCv3DVertex *n = g->getNormalP(nrmIter);
    x = n->x();
    y = n->y();
    z = n->z();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::newGeoColor(float r, float g, float b, float a)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *geo = geometries.at(geoIter);
    CADCv3DColor *c = new CADCv3DColor(r, g, b, a);
    geo->pushColorP(c);
    colIter = geo->countColors() - 1;
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::firstGeoColor(float &r, float &g, float &b, float &a)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *geo = geometries.at(geoIter);
    if (geo->countColors() == 0)
        return false;

    colIter = 0;

    CADCv3DColor *c = geo->getColorP(colIter);
    r = c->red();
    g = c->green();
    b = c->blue();
    a = c->alpha();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::nextGeoColor(float &r, float &g, float &b, float &a)
{
    if (geoIter < 0)
        return false;

    CADCv3DGeometry *geo = geometries.at(geoIter);
    if ((colIter < 0) || (colIter >= geo->countColors() - 1))
        return false;

    colIter++;

    CADCv3DColor *c = geo->getColorP(colIter);
    r = c->red();
    g = c->green();
    b = c->blue();
    a = c->alpha();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::newPrimitiveIndex(int i)
{
    if ((geoIter < 0) || (prmIter < 0))
        return false;

    CADCv3DPrimitive *p = geometries.at(geoIter)->getPrimitiveP(prmIter);
    p->pushIndex(i);
    idxIter = p->countIndices() - 1;
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::firstPrimitiveIndex(int &i)
{
    if ((geoIter < 0) || (prmIter < 0))
        return false;

    CADCv3DPrimitive *p = geometries.at(geoIter)->getPrimitiveP(prmIter);
    if (p->countIndices() == 0)
        return false;

    idxIter = 0;

    i = p->getIndex(idxIter);
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::nextPrimitiveIndex(int &i)
{
    if ((geoIter < 0) || (prmIter < 0))
        return false;

    CADCv3DPrimitive *p = geometries.at(geoIter)->getPrimitiveP(prmIter);
    if ((idxIter < 0) || (idxIter >= p->countIndices() - 1))
        return false;

    idxIter++;

    i = p->getIndex(idxIter);
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::getPrimitiveIndexCount(int &i) const
{
    if ((geoIter < 0) || (prmIter < 0))
        return false;

    CADCv3DPrimitive *p = geometries.at(geoIter)->getPrimitiveP(prmIter);

    i = p->countIndices();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::invertPrimitiveIndices()
{
    if ((geoIter < 0) || (prmIter < 0))
        return false;

    CADCv3DPrimitive *p = geometries.at(geoIter)->getPrimitiveP(prmIter);
    p->invert();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::setGeoNormalBinding(int type)
{
    if (geoIter < 0)
        return false;

    geometries.at(geoIter)->setNormalBinding(type);
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::getGeoNormalBinding(int &type) const
{
    if (geoIter < 0)
        return false;

    type = geometries.at(geoIter)->getNormalBinding();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::setGeoColorBinding(int type)
{
    if (geoIter < 0)
        return false;

    geometries.at(geoIter)->setColorBinding(type);
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::getGeoColorBinding(int &type) const
{
    if (geoIter < 0)
        return false;

    type = geometries.at(geoIter)->getColorBinding();
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::setGeoName(const std::string &name)
{
    if (geoIter < 0)
        return false;

    geometries.at(geoIter)->setName(name);
    return true;
}

/****************************************************************************/
bool CADCv3DGeoList::getGeoName(std::string &name) const
{
    if (geoIter < 0)
        return false;

    name = geometries.at(geoIter)->getName();
    return true;
}

/****************************************************************************/
unsigned int CADCv3DGeoList::calcSize() const
{
    // sizeof(quint32) * (number of geometries + total size) + geometries
    unsigned int size = 2 * sizeof(quint32);

    std::vector<CADCv3DGeometry *>::const_iterator i = geometries.begin();
    while (i != geometries.end())
        size += (*(i++))->calcSize();

    return size;
}

/****************************************************************************/
bool CADCv3DGeoList::read(const void *data, unsigned int size)
{
    clear();

    const quint32 *d = static_cast<const quint32 *>(data);
    unsigned int element = sizeof(quint32);

    if ((size < 2 * element) && (size != 0))
        return false;

    quint32 gn = qFromBigEndian<quint32>(*(d++));
    quint32 ts = qFromBigEndian<quint32>(*(d++)) - (2 * element);
    if (size > 0)
        size -= (2 * element);

    bool success = true;

    CADCv3DGeometry *geo;
    for (quint32 i = 0; (i < gn) && success; i++)
    {
        geo = new CADCv3DGeometry();
        if ((success = geo->read(d, size)) == true)
        {
            geometries.push_back(geo);
            unsigned int s = geo->calcSize();
            if (size > 0)
                size -= s;
            ts -= s;
            d = reinterpret_cast<const quint32 *>(reinterpret_cast<const char *>(d) + s);
        }
        else
            delete geo;
    }
    success = success && (ts == 0);

    return success;
}

/****************************************************************************/
void CADCv3DGeoList::write(void *data) const
{
    quint8 *d = reinterpret_cast<quint8 *>(data);
    unsigned int element = sizeof(quint32);
    unsigned int offset = 0;

    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>((quint32)geometries.size());
    offset += element;
    *(reinterpret_cast<quint32 *>(d + offset)) = qToBigEndian<quint32>((quint32)calcSize());
    offset += element;

    std::vector<CADCv3DGeometry *>::const_iterator i = geometries.begin();
    while (i != geometries.end())
    {
        (*i)->write(reinterpret_cast<void *>(d + offset));
        offset += (*i)->calcSize();
        i++;
    }
}

/****************************************************************************/
void CADCv3DGeoList::clear()
{
    std::vector<CADCv3DGeometry *>::iterator i = geometries.begin();
    while (i != geometries.end())
        delete *(i++);
    geometries.clear();
    reset();
}

/****************************************************************************/
void CADCv3DGeoList::reset()
{
    geoIter = -1;
    prmIter = -1;
    vrtIter = -1;
    nrmIter = -1;
    colIter = -1;
    idxIter = -1;
}

/****************************************************************************/
const CADCv3DGeoList &CADCv3DGeoList::copy(const CADCv3DGeoList &src)
{
    clear();
    int c = src.geometries.size();
    for (int i = 0; i < c; i++)
        geometries.push_back(new CADCv3DGeometry(*(src.geometries.at(i))));

    return *this;
}

// END OF FILE
