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

#ifndef COVISE_RENDER_OBJ
#define COVISE_RENDER_OBJ
#include <util/common.h>
#include <cover/RenderObject.h>

namespace covise
{
class coDistributedObject;
}
using std::string;

class CoviseRenderObject : public opencover::RenderObject
{
public:
    CoviseRenderObject(const covise::coDistributedObject *co, const std::vector<int> &assignedTo = std::vector<int>());
    ~CoviseRenderObject();
    const char *getType() const;
    bool isType(const char *t) const
    {
        return (strcmp(getType(), t) == 0);
    }
    bool isGeometry() const
    {
        return isType("GEOMET");
    }
    bool isSet() const
    {
        return isType("SETELE");
    }
    bool isUniformGrid() const
    {
        return isType("UNIGRD");
    }
    bool isUnstructuredGrid() const
    {
        return isType("UNSGRD");
    }
    bool isVectors() const
    {
        return isType("USTVDT");
    }
    const char *getName() const
    {
        return name;
    }
    const char *getAttribute(const char *attributeName) const;
    const char *getAttributeName(size_t idx) const;
    const char *getAttributeValue(size_t idx) const;
    size_t getNumAttributes() const;
    int getAllAttributes(char **&name, char **&value) const;
    CoviseRenderObject *getElement(size_t idx) const;
    RenderObject **getAllElements(int &numElements, std::vector<std::vector<int> > &assignments) const;
    RenderObject **getAllElements(int *numElements, std::vector<std::vector<int> > &assignments) const
    {
        return getAllElements(*numElements, assignments);
    }
    void getSize(int &u, int &v, int &w) const
    {
        u = sizeu;
        v = sizev;
        w = sizew;
    }
    void getSize(int &s) const
    {
        s = size;
    };
    int getRenderMethod() const
    {
        return geometryFlag;
    }
    void getAddresses(int *&i1)
    {
        i1 = pc;
    }
    void getAddresses(float *&f1, float *&f2, float *&f3, int *&i1, int *&i2)
    {
        f1 = farr1;
        f2 = farr2;
        f3 = farr3;
        i1 = iarr1;
        i2 = iarr2;
    }
    void getAddresses(float *&f1, float *&f2, float *&f3, int *&i1)
    {
        f1 = farr1;
        f2 = farr2;
        f3 = farr3;
        i1 = iarr1;
    }
    void getAddresses(float *&f1, float *&f2, float *&f3)
    {
        f1 = farr1;
        f2 = farr2;
        f3 = farr3;
    };
    void getAddresses(float *&f1, float *&f2, float *&f3, float *&f4, int *&i1)
    {
        f1 = farr1;
        f2 = farr2;
        f3 = farr3;
        f4 = farr4;
        i1 = iarr1;
    }
    void getAddresses(float *&f1, float *&f2, float *&f3, int *&i1, int *&i2, int *&i3)
    {
        f1 = farr1;
        f2 = farr2;
        f3 = farr3;
        i1 = iarr1;
        i2 = iarr2;
        i3 = iarr3;
    }
    void getMinMax(float &mix, float &max, float &miy, float &may, float &miz, float &maz) const
    {
        mix = minx;
        max = maxx;
        miy = miny;
        may = maxy;
        miz = minz;
        maz = maxz;
    }

    const unsigned char *getByte(opencover::Field::Id idx) const;
    const int *getInt(opencover::Field::Id idx) const;
    const float *getFloat(opencover::Field::Id idx) const;

    int getFloatRGBA(int pos, float *r, float *g, float *b, float *a) const;

    int getNumPoints() const
    {
        return size;
    }
    int getNumLines() const
    {
        return sizeu;
    }
    int getNumPolygons() const
    {
        return sizeu;
    }
    int getNumVertices() const
    {
        return sizev;
    }
    int getNumColors() const
    {
        return size;
    }
    int getNumNormals() const
    {
        return size;
    }
    int getNumStrips() const
    {
        return sizeu;
    }
    size_t getNumElements() const
    {
        return size;
    }

    bool isAssignedToMe() const;
    const std::vector<int> &getAssignment() const
    {
        return this->assignedTo;
    }

    CoviseRenderObject *getGeometry() const;
    CoviseRenderObject *getNormals() const;
    CoviseRenderObject *getColors() const;
    CoviseRenderObject *getTexture() const;
    CoviseRenderObject *getVertexAttribute() const;

    const covise::coDistributedObject *COVdobj;
    const covise::coDistributedObject *COVnormals;
    const covise::coDistributedObject *COVcolors;
    const covise::coDistributedObject *COVtexture;
    const covise::coDistributedObject *COVvertexAttribute;
    const covise::coDistributedObject *coviseObject;
    char type[7];
    char *name;
    unsigned char *texture;
    float **textureCoords;
    int *pc;
    unsigned char *byteData;
    int numTC;
    int numAttributes;
    char **attrNames;
    char **attributes;
    int size;
    int sizeu, sizev, sizew;
    float *farr1, *farr2, *farr3;
    float *farr4, *farr5;
    int *iarr1, *iarr2, *iarr3;
    int geometryFlag;
    float minx, maxx, miny, maxy, minz, maxz;
    bool cluster;
    mutable RenderObject **objs;
    mutable CoviseRenderObject *geometryObject;
    mutable CoviseRenderObject *normalObject;
    mutable CoviseRenderObject *colorObject;
    mutable CoviseRenderObject *textureObject;
    mutable CoviseRenderObject *vertexAttributeObject;
    bool IsTypeField(const std::vector<string> &types, bool strict_case);

    std::vector<int> assignedTo;
};
#endif
