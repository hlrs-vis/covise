/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
 \brief transformed COVISE data objects for cluster rendering

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2003
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   December 2003
 */

#ifndef RENDER_OBJ
#define RENDER_OBJ
#include <util/common.h>

namespace opencover
{

struct Bind
{
    enum
    {
        PerVertex = 0,
        PerFace = 1,
        None = 2,
        OverAll = 3,
    };
};

struct Pack
{
    enum
    {
        None = 2,
        RGBA = 4,
        Texture = 5,
        Float = 6,
    };
};

struct Field
{
    enum Id
    {
        Channel0 = 0,
        Channel1,
        Channel2,
        Channel3,
        Channel4,
        Channel5,
        Channel6,
        Channel7,
        NumChannels,   // invalid!
        X,
        Y,
        Z,
        Red,
        Green,
        Blue,
        RGBA,
        Byte,
        Texture,
        Elements,
        Connections,
        Types,
        ColorMap,
    };
};

// sonst muesste ObjectManager.h eingebunden wrden, was Fehlermeldungen beim
//compilieren gibt, gibt aber sicher irgendwann Probleme

inline void packRGBA(int *pc, int pos, float r, float g, float b, float a)
{
    if (pc)
    {
        unsigned char *chptr;

        chptr = (unsigned char *)&pc[pos];
#ifdef BYTESWAP
        *chptr = (unsigned char)(a * 255.0);
        chptr++;
        *chptr = (unsigned char)(b * 255.0);
        chptr++;
        *chptr = (unsigned char)(g * 255.0);
        chptr++;
        *chptr = (unsigned char)(r * 255.0);
#else
        *chptr = (unsigned char)(r * 255.0);
        chptr++;
        *chptr = (unsigned char)(g * 255.0);
        chptr++;
        *chptr = (unsigned char)(b * 255.0);
        chptr++;
        *chptr = (unsigned char)(a * 255.0);
#endif
    }
}

inline void unpackRGBA(const int *pc, int pos, float *r, float *g, float *b, float *a)
{
    if (pc)
    {
        const unsigned char *chptr;

        chptr = (const unsigned char *)&pc[pos];
#ifdef BYTESWAP
        *a = ((float)(*chptr)) / 255.0f;
        chptr++;
        *b = ((float)(*chptr)) / 255.0f;
        chptr++;
        *g = ((float)(*chptr)) / 255.0f;
        chptr++;
        *r = ((float)(*chptr)) / 255.0f;
#else
        *r = ((float)(*chptr)) / 255.0f;
        chptr++;
        *g = ((float)(*chptr)) / 255.0f;
        chptr++;
        *b = ((float)(*chptr)) / 255.0f;
        chptr++;
        *a = ((float)(*chptr)) / 255.0f;
#endif
    }
    else
    {
        *r = 1;
        *g = 1;
        *b = 1;
        *a = 1;
    }
}

//! base class for data received from visualization systems (e.g. COVISE or Vistle)
class COVEREXPORT RenderObject
{
public:
    RenderObject();
    virtual ~RenderObject();

    virtual const char *getName() const = 0;

    virtual bool isGeometry() const = 0;
    virtual RenderObject *getGeometry() const = 0;
    virtual RenderObject *getNormals() const = 0;
    virtual RenderObject *getColors() const = 0;
    virtual RenderObject *getTexture() const = 0;
    virtual RenderObject *getVertexAttribute() const = 0;
    virtual RenderObject *getColorMap(int idx) const = 0;

    virtual const char *getAttribute(const char *) const = 0;

    //XXX: hacks for Volume plugin and Tracer
    virtual bool isSet() const = 0;
    virtual size_t getNumElements() const = 0;
    virtual RenderObject *getElement(size_t idx) const = 0;

    virtual bool isUniformGrid() const = 0;
    virtual void getSize(int &nx, int &ny, int &nz) const = 0;
    virtual float getMin(int channel) const = 0;
    virtual float getMax(int channel) const = 0;
    virtual void getMinMax(float &xmin, float &xmax,
                           float &ymin, float &ymax,
                           float &zmin, float &zmax) const = 0;

    virtual bool isVectors() const = 0;
    virtual const unsigned char *getByte(Field::Id idx) const = 0;
    virtual const int *getInt(Field::Id idx) const = 0;
    virtual const float *getFloat(Field::Id idx) const = 0;

    virtual bool isUnstructuredGrid() const = 0;
};
}
#endif
