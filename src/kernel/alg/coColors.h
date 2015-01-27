/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVCOLORS_H
#define COVCOLORS_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Universal Colormap class                               ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  03.08.2000  V3.0                                             ++
// ++**********************************************************************/

#include <do/coDoColormap.h>
#include <do/coDoTexture.h>
#include <do/coDoSet.h>
#include <do/coDoIntArr.h>

#include <utility>
#include <cfloat>

namespace covise
{

class ALGEXPORT AttributeList
{
public:
    void copyFrom(const AttributeList &attrs);
    void copyFrom(const coDistributedObject *source);
    void copyTo(coDistributedObject *dest);
    const char *getAttribute(const char *word) const;

private:
    std::vector<std::pair<std::string, std::string> > m_attrList;
};

class ALGEXPORT coColor
{
public:
    // RGBAX, [0..1]
    typedef float FlColor[5];
    coColor(int num_el, float *data_, const coDoColormap *colorMapIn);
    coDistributedObject *createColors(const coObjInfo &info);
    static FlColor *interpolateColormap(FlColor *map, int numColors, int numSteps);

private:
    FlColor *actMap_;
    int steps_;
    const char *annotation_;
    float min_, max_;
    int numElem;
    float *data;
    // color to be used for non-existing data values, nomallx 0x00000000
    unsigned long d_noDataColor;
    AttributeList _attributes;
};

class ALGEXPORT ScalarContainer
{
public:
    ScalarContainer();
    void Initialise(const coDistributedObject *);
    ScalarContainer(const ScalarContainer &);
    virtual ~ScalarContainer();
    void OpenList(int size);
    void CopyAllAttributes(const coDistributedObject *);
    void DumpAllAttributes(coDistributedObject *);
    const char *getAttribute(const char *) const;
    const char *getAttributeRecursive(const char *) const;
    void AddArray(int size, const float *scalar);

    const float *ScalarField() const;
    int SizeField() const;

    ScalarContainer &operator[](int i);
    int NoChildren() const;

    void MinMax(float &min, float &max) const;

private:
    float *_field;
    int _size_field;
    AttributeList _attributes;

    vector<ScalarContainer> _children;
};

class ALGEXPORT coColors
{
public:
    // RGBAX, [0..1]
    typedef float FlColor[5];

private:
    ////////// maximum size of colormap
    enum
    {
        MAX_CMAP_SIZE = 2048
    };

    const coDistributedObject *data_;
    const ScalarContainer *_scalar;
    AttributeList _attributes;

    FlColor *actMap_;
    int steps_;
    string annotation_;
    float min_, max_;
    int textureComponents_;

    // color to be used for non-existing data values, nomallx 0x00000000
    unsigned long d_noDataColor;

    // color output styles
    enum Outstyle
    {
        RGBA = 1,
        TEX = 4
    };

    // struct captures all info for one data input object
    struct recObj
    {
        const coDistributedObject *obj; // the object itself
        ScalarContainer *objSCont;
        const coDistributedObject *const *objList; // if object was a set: list of subobj.
        recObj *subObj; //                      recurse to subobj
        int numElem; //                      number
        float *data; // otherwise: pointer to data
        bool doDelete; // whether we have to delete this data
        // (from vect or int)
        recObj()
        {
            obj = 0;
            objSCont = 0;
            objList = 0;
            subObj = 0;
            numElem = 0;
            data = 0;
            doDelete = false;
        }
        ~recObj()
        {
            if (subObj)
                delete[] subObj;
            if (doDelete)
                delete[] data;
        }
    };

    // Data structure holding one color map
    struct ColorMap
    {
        FlColor *map;
        int numcoColors;
    } *d_cmap;

    int openObj(recObj &base, const coDistributedObject *obj, const char *&species);
    int openObj(recObj &base, const ScalarContainer *obj, const char *&species);
    coDistributedObject *createColors(recObj &base, const coObjInfo &info, int outStyle, int repeat);

public:
    void addColormapAttrib(const coObjInfo &info, coDistributedObject *outObj);

    coColors(const coDistributedObject *data, const coDoColormap *colorMapIn, bool transparentTextures);
    coColors(const ScalarContainer &data, const coDoColormap *colorMapIn, bool transparentTextures, const ScalarContainer *glData);
    coDistributedObject *getColors(const coObjInfo &outInfo,
                                   bool create_texture = false,
                                   bool createCMAP = true,
                                   int repeat = 1,
                                   float *min = NULL,
                                   float *max = NULL);
};
}
#endif
