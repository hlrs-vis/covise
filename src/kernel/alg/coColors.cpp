/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Universal Colormap module                              ++
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

#include "coColors.h"
#include <do/coDoData.h>
#include <do/coDoPixelImage.h>

#define FAIL 0
#define SUCCESS 1

using namespace covise;

void
AttributeList::copyFrom(const coDistributedObject *source)
{
    if (source)
    {
        const char **attr;
        const char **cont;
        int num = source->getAllAttributes(&attr, &cont);
        for (unsigned i = 0; i < (unsigned int)num; ++i)
        {
            string attrName = attr[i];
            string attrCont = cont[i];
            m_attrList.push_back(pair<string, string>(attrName, attrCont));
        }
    }
}

void
AttributeList::copyTo(coDistributedObject *dest)
{
    if (dest)
    {
        for (unsigned i = 0; i < m_attrList.size(); ++i)
        {
            string attrName = m_attrList.at(i).first;
            string attrCont = m_attrList.at(i).second;
            dest->addAttribute(attrName.c_str(), attrCont.c_str());
        }
    }
}

const char *
AttributeList::getAttribute(const char *word) const
{
    for (unsigned i = 0; i < m_attrList.size(); ++i)
    {
        if (m_attrList.at(i).first == word)
        {
            return m_attrList.at(i).second.c_str();
        }
    }
    return NULL;
}

void
AttributeList::copyFrom(const AttributeList &source)
{
    for (unsigned i = 0; i < source.m_attrList.size(); ++i)
    {
        string attrName = source.m_attrList.at(i).first;
        string attrCont = source.m_attrList.at(i).second;
        m_attrList.push_back(pair<string, string>(attrName, attrCont));
    }
}

// Data values of more than  NoDataColorPercent*FLT_MAX are non-data values
static const float NoDataColorPercent = 0.01f;

coColor::coColor(int num_el, float *data_, const coDoColormap *colorMapIn)
    : numElem(num_el)
    , data(data_)
{
    d_noDataColor = 0x00000000;

    if (colorMapIn)
    {
        min_ = colorMapIn->getMin();
        max_ = colorMapIn->getMax();
        actMap_ = (FlColor *)colorMapIn->getAddress();
        annotation_ = colorMapIn->getMapName();
        steps_ = colorMapIn->getNumSteps();
        _attributes.copyFrom(colorMapIn);
    }
    else
    {
        //use standard map
        FlColor s_normMap[3] = {
            { 0.000000, 0.000000, 1.000000, 1.000000, -1.0 },
            { 1.000000, 0.000000, 0.000000, 1.000000, -1.0 },
            { 1.000000, 1.000000, 0.000000, 1.000000, -1.0 }
        };
        steps_ = 256;
        actMap_ = interpolateColormap(s_normMap, 3, steps_);
        annotation_ = "Colors";
        min_ = FLT_MAX;
        max_ = -FLT_MAX;

        for (int i = 0; i < num_el; i++)
        {
            // do not care about FLT_MAX elements in min/max calc.
            if (data_[i] < NoDataColorPercent * FLT_MAX && data_[i] < min_)
                min_ = data_[i];
            if (data_[i] < NoDataColorPercent * FLT_MAX && data_[i] > max_)
                max_ = data_[i];
        }
    }
}

coDistributedObject *coColor::createColors(const coObjInfo &info)
{

    float delta = steps_ / (max_ - min_); // cmap steps
    int max_Idx = steps_ - 1;
    int i;

    coDoRGBA *res = new coDoRGBA(info, numElem);
    int *dPtr;
    res->getAddress(&dPtr);
    unsigned int *packed = (unsigned int *)dPtr;
    unsigned char r, g, b, a;
    for (i = 0; i < numElem; i++)
    {
        if (data[i] >= NoDataColorPercent * FLT_MAX)
        {
            packed[i] = d_noDataColor;
        }
        else
        {
            int idx = (int)((data[i] - min_) * delta);
            if (idx < 0)
                idx = 0;
            if (idx > max_Idx)
                idx = max_Idx;
            r = (unsigned char)(actMap_[idx][0] * 255);
            g = (unsigned char)(actMap_[idx][1] * 255);
            b = (unsigned char)(actMap_[idx][2] * 255);
            a = (unsigned char)(actMap_[idx][3] * 255);
            packed[i] = (r << 24) | (g << 16) | (b << 8) | a;
        }
    }

    res->copyAllAttributes(NULL);

    //char *buffer;
    //char colBuf[64];
    //buffer = new char [128 + 32*steps_];
    //sprintf(buffer,"%s\n%s\n%g\n%g\n%d\n%d",
    //   info.getName(),annotation_,min_,max_,steps_,0);
    //for (i=0; i<steps_; i++)
    //{
    //   sprintf(colBuf,"\n%4f\n%4f\n%4f\n%4f", actMap_[i][0],actMap_[i][1],actMap_[i][2],actMap_[i][3]);
    //   strcat(buffer,colBuf);
    //}
    //res->addAttribute("COLORMAP",buffer);
    //delete [] buffer;

    stringstream buffer;
    buffer << info.getName() << '\n' << annotation_ << '\n' << min_ << '\n' << max_ << '\n' << steps_ << '\n' << '0';
    buffer.precision(4);
    buffer << std::fixed;
    for (int i = 0; i < steps_; i++)
    {
        buffer << "\n" << (actMap_[i][0]) << "\n" << (actMap_[i][1]) << "\n" << (actMap_[i][2]) << "\n" << (actMap_[i][3]);
    }
    res->addAttribute("COLORMAP", buffer.str().c_str());

    return res;
}

coColor::FlColor *coColor::interpolateColormap(FlColor *map, int numColors, int numSteps)
{
    FlColor *actMap = new FlColor[numSteps];
    double delta = 1.0 / (numSteps - 1) * (numColors - 1);
    double x;
    int i;

    if (map[0][4] < 0)
    {
        for (i = 0; i < numSteps - 1; i++)
        {
            x = i * delta;
            int idx = (int)x;
            float d = (float)(x - idx);
            actMap[i][0] = (float)((1 - d) * map[idx][0] + d * map[idx + 1][0]);
            actMap[i][1] = (float)((1 - d) * map[idx][1] + d * map[idx + 1][1]);
            actMap[i][2] = (float)((1 - d) * map[idx][2] + d * map[idx + 1][2]);
            actMap[i][3] = (float)((1 - d) * map[idx][3] + d * map[idx + 1][3]);
            actMap[i][4] = -1;
        }
        actMap[numSteps - 1][0] = map[numColors - 1][0];
        actMap[numSteps - 1][1] = map[numColors - 1][1];
        actMap[numSteps - 1][2] = map[numColors - 1][2];
        actMap[numSteps - 1][3] = map[numColors - 1][3];
        actMap[numSteps - 1][4] = -1;
    }
    else
    {
        delta = 1.0 / (numSteps - 1);
        int idx = 0;
        for (i = 0; i < numSteps - 1; i++)
        {
            x = i * delta;
            while (map[idx + 1][4] <= x)
            {
                idx++;
                if (idx > numColors - 2)
                {
                    idx = numColors - 2;
                    break;
                }
            }
            double d = (x - map[idx][4]) / (map[idx + 1][4] - map[idx][4]);
            actMap[i][0] = (float)((1 - d) * map[idx][0] + d * map[idx + 1][0]);
            actMap[i][1] = (float)((1 - d) * map[idx][1] + d * map[idx + 1][1]);
            actMap[i][2] = (float)((1 - d) * map[idx][2] + d * map[idx + 1][2]);
            actMap[i][3] = (float)((1 - d) * map[idx][3] + d * map[idx + 1][3]);
            actMap[i][4] = -1;
        }
        actMap[numSteps - 1][0] = map[numColors - 1][0];
        actMap[numSteps - 1][1] = map[numColors - 1][1];
        actMap[numSteps - 1][2] = map[numColors - 1][2];
        actMap[numSteps - 1][3] = map[numColors - 1][3];
        actMap[numSteps - 1][4] = -1;
    }

    return actMap;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coColors::coColors(const ScalarContainer &scalar,
                   const coDoColormap *colorMapIn, bool transparentTextures,
                   const ScalarContainer *SCont)
    : data_(NULL)
    , _scalar(&scalar)
{
    d_noDataColor = 0x00000000;
    textureComponents_ = 3;
    if (transparentTextures)
    {
        textureComponents_ = 4;
    }

    min_ = FLT_MAX;
    max_ = -FLT_MAX;
    if (SCont)
    {
        SCont->MinMax(min_, max_);
    }
    else
    {
        _scalar->MinMax(min_, max_);
    }
    if (min_ > max_)
    {
        min_ = 0.0;
        max_ = 1.0;
    }
    //use standard map
    if (colorMapIn == NULL)
    {
        FlColor s_normMap[3] = {
            { 0.000000, 0.000000, 1.000000, 1.000000, -1.0 },
            { 1.000000, 0.000000, 0.000000, 1.000000, -1.0 },
            { 1.000000, 1.000000, 0.000000, 1.000000, -1.0 }
        };
        steps_ = 256;
        actMap_ = coColor::interpolateColormap(s_normMap, 3, steps_);
        char *annot = new char[7];
        strcpy(annot, "Colors");
        annotation_ = annot;
    }
    else
    {
        actMap_ = (FlColor *)colorMapIn->getAddress();
        annotation_ = colorMapIn->getMapName();
        steps_ = colorMapIn->getNumSteps();
        min_ = colorMapIn->getMin();
        max_ = colorMapIn->getMax();
        _attributes.copyFrom(colorMapIn);
    }
}

coColors::coColors(const coDistributedObject *data,
                   const coDoColormap *colorMapIn,
                   bool transparentTextures)
    : _scalar(NULL)
{
    data_ = data;

    d_noDataColor = 0x00000000;
    textureComponents_ = 3;
    if (transparentTextures)
    {
        textureComponents_ = 4;
    }
    if (colorMapIn == NULL)
    {
        ScalarContainer scalCont;
        scalCont.Initialise(data);
        min_ = FLT_MAX;
        max_ = -FLT_MAX;
        scalCont.MinMax(min_, max_);
        if (min_ > max_)
        {
            min_ = 0.0;
            max_ = 1.0;
        }
        FlColor s_normMap[3] = {
            { 0.000000, 0.000000, 1.000000, 1.000000, -1.0 },
            { 1.000000, 0.000000, 0.000000, 1.000000, -1.0 },
            { 1.000000, 1.000000, 0.000000, 1.000000, -1.0 }
        };
        steps_ = 256;
        actMap_ = coColor::interpolateColormap(s_normMap, 3, steps_);
        annotation_ = "Colors";
        if (data)
        {
            const char *species = scalCont.getAttributeRecursive("SPECIES");
            if (species)
                annotation_ = species;
        }
    }
    else
    {
        min_ = colorMapIn->getMin();
        max_ = colorMapIn->getMax();
        actMap_ = (FlColor *)colorMapIn->getAddress();
        annotation_ = colorMapIn->getMapName();
        steps_ = colorMapIn->getNumSteps();
        _attributes.copyFrom(colorMapIn);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  recursively open all objects and find own min/max
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int coColors::openObj(recObj &base, const coDistributedObject *obj, const char *&species)
{
    // save the object
    base.obj = obj;

    if (obj)
    {
        const char *specAttr = obj->getAttribute("SPECIES");
        if (specAttr)
            species = specAttr;

        const char *noDataAttr = obj->getAttribute("NO_DATA_COLOR");
        if (noDataAttr)
            d_noDataColor = strtoul(noDataAttr, NULL, 0);

        // if it's a set: recurse
        if (const coDoSet *set = dynamic_cast<const coDoSet *>(obj))
        {
            int i;
            base.objList = set->getAllElements(&base.numElem);
            base.subObj = new recObj[base.numElem];
            base.data = NULL;
            base.doDelete = false;
            for (i = 0; i < base.numElem; i++)
                if (openObj(base.subObj[i], base.objList[i], species) == FAIL)
                    return FAIL;
            return SUCCESS;
        }

        // otherwise: only valid data formats
        else if (const coDoFloat *uObj = dynamic_cast<const coDoFloat *>(obj))
        {
            base.objList = NULL;
            base.subObj = NULL;
            base.doDelete = false;
            uObj->getAddress(&base.data);
            base.numElem = uObj->getNumPoints();
            return SUCCESS;
        }

        else if (const coDoIntArr *iObj = dynamic_cast<const coDoIntArr *>(obj))
        {
            base.objList = NULL;
            base.subObj = NULL;
            base.doDelete = true;
            int numDim = iObj->getNumDimensions();
            if (numDim != 1)
            {
                return FAIL;
            }
            base.numElem = iObj->getDimension(0);
            base.data = new float[base.numElem];
            int *dataPtr = iObj->getAddress();
            int i;
            for (i = 0; i < base.numElem; i++)
                base.data[i] = (float)dataPtr[i];
            return SUCCESS;
        }

        else if (const coDoVec3 *vObj = dynamic_cast<const coDoVec3 *>(obj))
        {
            base.objList = NULL;
            base.subObj = NULL;
            base.doDelete = true;
            base.numElem = vObj->getNumPoints();
            float *u, *v, *w;
            vObj->getAddresses(&u, &v, &w);
            base.data = new float[base.numElem];
            for (int i = 0; i < base.numElem; i++)
                base.data[i] = sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
            return SUCCESS;
        }

        // invalid
        else
        {
            return FAIL;
        }
    }

    // ignore empty set elements, but NO ERROR
    base.objList = NULL;
    base.subObj = NULL;
    base.data = NULL;
    base.numElem = 0;

    return SUCCESS;
}

int coColors::openObj(recObj &base, const ScalarContainer *obj, const char *&species)
{
    // save the object
    // base.obj = obj; FIXME
    base.obj = NULL;
    base.objSCont = const_cast<ScalarContainer *>(obj);

    if (obj)
    {
        const char *specAttr = obj->getAttribute("SPECIES");
        if (specAttr)
            species = specAttr;

        const char *noDataAttr = obj->getAttribute("NO_DATA_COLOR");
        if (noDataAttr)
            d_noDataColor = strtoul(noDataAttr, NULL, 0);

        // if it's a set: recurse
        if (!obj->ScalarField())
        {
            int i;
            base.numElem = obj->NoChildren();
            base.objList = NULL; // ((coDoSet*)obj)->getAllElements(&base.numElem); // FIXME
            base.subObj = new recObj[base.numElem];
            base.data = NULL;
            base.doDelete = false;
            for (i = 0; i < base.numElem; i++)
            {
                ScalarContainer *obj_cast = const_cast<ScalarContainer *>(obj);
                const ScalarContainer *obj_i = &(obj_cast->operator[](i));
                // FIXME
                if (openObj(base.subObj[i], obj_i, species) == FAIL)
                    return FAIL;
            }
            return SUCCESS;
        }

        // otherwise
        else if (obj->SizeField() > 0)
        {
            base.objList = NULL;
            base.subObj = NULL;
            base.doDelete = false;
            //coDoFloat *uObj = (coDoFloat *) obj;
            // uObj->getAddress(&base.data);
            base.data = const_cast<float *>(obj->ScalarField());
            base.numElem = obj->SizeField(); // uObj->getNumPoints();
            return SUCCESS;
        }
    }

    // ignore empty set elements, but NO ERROR
    base.objList = NULL;
    base.subObj = NULL;
    base.data = NULL;
    base.numElem = 0;

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  recursively create float RGB coColors
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coDistributedObject *coColors::createColors(recObj &base, const coObjInfo &info, int outStyle, int repeat)
{

    // empty to empty
    if (base.obj == NULL && base.objSCont == NULL)
        return NULL;

    int i;

    // create set (recurse)
    if (base.subObj)
    {
        char namebuf[512];
        char namemask[512];
        sprintf(namemask, "%s_%%d", info.getName());

        // create array for Set
        coDistributedObject **setArr = new coDistributedObject *[base.numElem];

        // recursively create set objects
        for (i = 0; i < base.numElem; i++)
        {
            sprintf(namebuf, namemask, i);
            setArr[i] = createColors(base.subObj[i], coObjInfo(namebuf), outStyle, repeat);
        }

        // Create set
        coDoSet *set = new coDoSet(info, base.numElem, setArr);
        for (i = 0; i < base.numElem; i++)
            delete setArr[i];
        delete[] setArr;

        if (base.obj)
        {
            set->copyAllAttributes(base.obj);
        }
        else
        {
            base.objSCont->DumpAllAttributes(set);
        }
        return set;
    }

    // run over own object
    else
    {
        float *data = base.data; // where my data starts
        float delta = steps_ / (max_ - min_); // cmap steps
        int max_Idx = steps_ - 1;

        // packed RGBA data
        if (outStyle == RGBA)
        {
            coDoRGBA *res = new coDoRGBA(info, base.numElem * repeat);
            int *dPtr;
            res->getAddress(&dPtr);
            unsigned int *packed = (unsigned int *)dPtr;
            unsigned char r, g, b, a;
            for (i = 0; i < base.numElem; i++)
            {
                int base = repeat * i;
                if (data[i] >= NoDataColorPercent * FLT_MAX)
                {
                    // packed[i] = d_noDataColor;
                    int replicate;
                    for (replicate = 0; replicate < repeat; ++replicate)
                    {
                        packed[base + replicate] = d_noDataColor;
                    }
                }
                else
                {
                    int idx = (int)((data[i] - min_) * delta);
                    if (idx < 0)
                        idx = 0;
                    if (idx > max_Idx)
                        idx = max_Idx;
                    r = (unsigned char)(actMap_[idx][0] * 255);
                    g = (unsigned char)(actMap_[idx][1] * 255);
                    b = (unsigned char)(actMap_[idx][2] * 255);
                    a = (unsigned char)(actMap_[idx][3] * 255);
                    packed[base] = (r << 24) | (g << 16) | (b << 8) | a;
                    int replicate;
                    for (replicate = 1; replicate < repeat; ++replicate)
                    {
                        packed[base + replicate] = packed[base];
                    }
                }
            }
            if (base.obj)
            {
                res->copyAllAttributes(base.obj);
            }
            else
            {
                base.objSCont->DumpAllAttributes(res);
            }
            return res;
        }
        // packed RGBA data
        else if (outStyle == TEX)
        {
            // prepare the texture image
            char namebuf[512];
            int texSize = 256;
            while (texSize < steps_)
                texSize *= 2;
            sprintf(namebuf, "%s_Img", info.getName());
            unsigned char *image = new unsigned char[textureComponents_ * texSize];
            unsigned char *iPtr = image;
            float delta = 1.0f / (texSize - 1) * (steps_ - 0.00001f);
            for (i = 0; i < texSize; i++)
            {
                int k = (int)(delta * i);
                *iPtr = (unsigned char)(actMap_[k][0] * 255);
                iPtr++;
                *iPtr = (unsigned char)(actMap_[k][1] * 255);
                iPtr++;
                *iPtr = (unsigned char)(actMap_[k][2] * 255);
                iPtr++;
                if (textureComponents_ == 4)
                {
                    *iPtr = (unsigned char)(actMap_[k][3] * 255);
                    iPtr++;
                }
            }

            coDoPixelImage *pix = new coDoPixelImage(coObjInfo(namebuf), texSize, 1, textureComponents_, textureComponents_, (char *)image);

            // texture coordinate index
            int *txIndex = new int[base.numElem];
            float **txCoord = new float *[2];
            txCoord[0] = new float[base.numElem];
            txCoord[1] = new float[base.numElem];
            float fact = 1.0f / (max_ - min_);
            for (i = 0; i < base.numElem; i++)
            {
                float tx = (data[i] - min_) * fact;
                if (tx < 0.0)
                    tx = 0.0;
                if (tx > 1.0)
                    tx = 1.0;
                txCoord[0][i] = tx;
                txCoord[1][i] = 0.0;
                txIndex[i] = i;
            }

            coDoTexture *texture = new coDoTexture(info, pix, 0, textureComponents_, 0,
                                                   base.numElem, txIndex, base.numElem, txCoord);
            // coDoTexture *texture = new coDoTexture( info, pix, 0, 3, 0,
            //	                              base.numElem, txIndex, base.numElem, txCoord );

            delete[] txCoord[0];
            delete[] txCoord[1];
            delete[] txCoord;
            delete[] txIndex;

            if (base.obj)
            {
                texture->copyAllAttributes(base.obj);
            }
            else
            {
                base.objSCont->DumpAllAttributes(texture);
            }
            return texture;
        }
    }
    return NULL;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  add a COLORMAP attribute
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void coColors::addColormapAttrib(const coObjInfo &info, coDistributedObject *outObj)
{
    if (_attributes.getAttribute("COLORMAP"))
        _attributes.copyTo(outObj);

    //char *buffer;
    //char colBuf[64];
    //int i;
    //buffer = new char [128 + 32*steps_];
    //sprintf(buffer,"%s\n%s\n%g\n%g\n%d\n%d",
    //   info.getName(),annotation_.c_str(),min_,max_,steps_,0);
    //for (i=0; i<steps_; i++)
    //{
    //   sprintf(colBuf,"\n%4f\n%4f\n%4f\n%4f", actMap_[i][0],actMap_[i][1],actMap_[i][2],actMap_[i][3]);
    //   strcat(buffer,colBuf);
    //}
    //outObj->addAttribute("COLORMAP",buffer);
    //delete [] buffer;

    stringstream buffer;
    buffer << info.getName() << '\n' << annotation_ << '\n' << min_ << '\n' << max_ << '\n' << steps_ << '\n' << '0';
    buffer.precision(4);
    buffer << std::fixed;
    for (int i = 0; i < steps_; i++)
    {
        buffer << "\n" << (actMap_[i][0]) << "\n" << (actMap_[i][1]) << "\n" << (actMap_[i][2]) << "\n" << (actMap_[i][3]);
    }
    outObj->addAttribute("COLORMAP", buffer.str().c_str());
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coDistributedObject *coColors::getColors(const coObjInfo &outInfo,
                                         bool create_texture, bool createCMAP, int repeat,
                                         float *min, float *max)
{
    recObj base; // Data base pointer
    const char *species = NULL; // Description

    if (min && *min != FLT_MAX)
    {
        min_ = *min;
    }

    if (max && *max != -FLT_MAX)
    {
        max_ = *max;
    }

    if (data_)
    {
        if (openObj(base, data_, species) == FAIL)
            return FAIL;
    }
    else if (openObj(base, _scalar, species) == FAIL)
    {
        return FAIL;
    }

    coDistributedObject *outObj = NULL;

    // ---- Create the coColors

    if (/*data_ &&*/ !create_texture)
    {
        outObj = createColors(base, outInfo, RGBA, repeat);
    }

    else
    {
        outObj = createColors(base, outInfo, TEX, repeat);
    }

    if (createCMAP)
        addColormapAttrib(outInfo, outObj);

    if (min)
        *min = min_;
    if (max)
        *max = max_;

    return outObj;
}

///////////////////////////////////////////////////////////////////////////
///////  Read input lines for different color mappings from covise.config
///////////////////////////////////////////////////////////////////////////

#if 0
inline void clamp(float &x)
{
   if (x>1.0) x=1.0;
   if (x<0.0) x=0.0;
}


inline int readRGB(const char *entry, coColors::FlColor &color)
{
   if (sscanf(entry,"%f %f %f",&color[0],
      &color[1],
      &color[2]) == 3)
   {
      clamp(color[0]);
      clamp(color[1]);
      clamp(color[2]);
      color[3] =  1.0;
      color[4] = -1.0;
      return 1;
   }
   else
      return 0;
}


inline int readRGBA(const char *entry, coColors::FlColor &color)
{
   if ( sscanf(entry,"%f %f %f %f",&color[0],
      &color[1],
      &color[2],
      &color[3]) == 4
      )
   {
      clamp(color[0]);
      clamp(color[1]);
      clamp(color[2]);
      clamp(color[3]);
      color[4]=-1;
      return 1;
   }
   else
      return 0;
}


inline int readXRGB(const char *entry, coColors::FlColor &color)
{
   if ( sscanf(entry,"%f %f %f %f",&color[4],
      &color[0],
      &color[1],
      &color[2]) == 4 )
   {
      clamp(color[0]);
      clamp(color[1]);
      clamp(color[2]);
      color[3] = 1.0;
      clamp(color[4]);
      return 1;
   }
   else
      return 0;
}


inline int readXRGBA(const char *entry, coColors::FlColor &color)
{
   if ( sscanf(entry,"%f %f %f %f %f",&color[4],
      &color[0],
      &color[1],
      &color[2],
      &color[3]) == 5 )
   {
      clamp(color[0]);
      clamp(color[1]);
      clamp(color[2]);
      clamp(color[3]);
      clamp(color[4]);
      return 1;
   }
   else
      return 0;
}
#endif

ScalarContainer::ScalarContainer()
    : _field(NULL)
    , _size_field(0)
{
}

void
ScalarContainer::Initialise(const coDistributedObject *obj)
{
    if (obj == NULL)
        return;
    if (const coDoSet *set = dynamic_cast<const coDoSet *>(obj))
    {
        int no_e;
        const coDistributedObject *const *objs = set->getAllElements(&no_e);
        OpenList(no_e);
        int i;
        for (i = 0; i < no_e; ++i)
        {
            (this->operator[](i)).Initialise(objs[i]);
        }
        for (i = 0; i < no_e; ++i)
        {
            delete objs[i];
        }
        delete[] objs;
    }
    else if (const coDoFloat *sdata = dynamic_cast<const coDoFloat *>(obj))
    {
        float *field;
        sdata->getAddress(&field);
        AddArray(sdata->getNumPoints(), field);
    }
    else if (const coDoVec3 *vdata = dynamic_cast<const coDoVec3 *>(obj))
    {
        float *x, *y, *z;
        vdata->getAddresses(&x, &y, &z);
        int len = vdata->getNumPoints();
        float *field = new float[len];
        int i;
        for (i = 0; i < len; ++i)
        {
            field[i] = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        }
        AddArray(len, field);
        delete[] field;
    }
    CopyAllAttributes(obj);
}

ScalarContainer::~ScalarContainer()
{
    delete[] _field;
}

void
ScalarContainer::OpenList(int size)
{
    delete[] _field;
    _field = NULL;
    _size_field = 0;
    _children.resize(size);
}

void
ScalarContainer::CopyAllAttributes(const coDistributedObject *obj)
{
    _attributes.copyFrom(obj);
}

void
ScalarContainer::DumpAllAttributes(coDistributedObject *obj)
{
    _attributes.copyTo(obj);
}

/*
static bool
firstPart(pair<string, string> left,
          pair<string, string> right)
{
   return (left.first == right.first);
}
*/

const char *
ScalarContainer::getAttributeRecursive(const char *word) const
{
    const char *ret = getAttribute(word);
    if (ret)
    {
        return ret;
    }
    unsigned int i;
    for (i = 0; i < _children.size(); ++i)
    {
        ret = _children[i].getAttributeRecursive(word);
        if (ret)
            return ret;
    }
    return NULL;
}

const char *
ScalarContainer::getAttribute(const char *word) const
{
    return _attributes.getAttribute(word);
}

void
ScalarContainer::AddArray(int size, const float *scalar)
{
    _field = new float[size];
    _size_field = size;
    memcpy(_field, scalar, size * sizeof(float));

    _children.clear();
}

const float *
ScalarContainer::ScalarField() const
{
    return _field;
}

int
ScalarContainer::NoChildren() const
{
    return (int)_children.size();
}

ScalarContainer &
    ScalarContainer::
    operator[](int i)
{
    return _children[i];
}

void
ScalarContainer::MinMax(float &min, float &max) const
{
    int i;
    for (i = 0; i < _size_field; i++)
    {
        // do not care about FLT_MAX elements in min/max calc.
        if (_field[i] < NoDataColorPercent * FLT_MAX && _field[i] < min)
            min = _field[i];
        if (_field[i] < NoDataColorPercent * FLT_MAX && _field[i] > max)
            max = _field[i];
    }
    unsigned int ui;
    for (ui = 0; ui < _children.size(); ++ui)
    {
        _children[ui].MinMax(min, max);
    }
}

int
ScalarContainer::SizeField() const
{
    return _size_field;
}

ScalarContainer::ScalarContainer(const ScalarContainer &rhs)
{
    _attributes = rhs._attributes;
    _children = rhs._children;

    if (rhs.ScalarField())
    {
        _field = new float[rhs.SizeField()];
        _size_field = rhs.SizeField();
        memcpy(_field, rhs.ScalarField(), _size_field);
    }
    else
    {
        _field = NULL;
        _size_field = 0;
    }
}
