/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2000 Vircinity  ++
// ++ Description: Unify three scalar 3D-data-object to one vectorial     ++
// ++  3D-data-object                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                           Christof Schwenzer                        ++
// ++                        Vircinity GmbH Stuttgart                     ++
// ++                            Nobelstrasze 15                          ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  12.09.2000  V1.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "Scalar2Vector.h"
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
//the following methods are needed to unify the attributes of
//the input
void Scalar2Vector::summarizedNamesPushBack(char *name)
{
    summarizedNames[summarized] = name;
}

void Scalar2Vector::summarizedValuesPushBack(char *value)
{
    summarizedValues[summarized] = value;
    summarized++;
}

//Remove a trailing newline from a string
void Scalar2Vector::removeNewLine(char *string)
{
    int len = strlen(string);
    if (string[len - 1] == '\n')
    {
        string[len - 1] = '\0';
    }
}

//Summarize the values of attributes the name of which
//where already summarized.
void Scalar2Vector::summarizeValue(
    const char **xValue,
    const char **yValue,
    const char **zValue)
{
    char *sValue;
    int len;

    if ((strcasecmp(*xValue, *yValue) == 0) && (strcasecmp(*xValue, *zValue) == 0))
    {
        sValue = new char[strlen(*xValue) + 1 + 10];
        strcpy(sValue, *xValue);
        removeNewLine(sValue);
    }
    else
    {
        len = strlen(*xValue) + strlen(*yValue) + strlen(*zValue) + 4;
        sValue = new char[len + 10];
        sValue[0] = '\0';
        strcat(sValue, "(");
        strcat(sValue, *xValue);
        removeNewLine(sValue);
        strcat(sValue, ",");
        strcat(sValue, *yValue);
        removeNewLine(sValue);
        strcat(sValue, ",");
        strcat(sValue, *zValue);
        removeNewLine(sValue);
        strcat(sValue, ")");
    }
    summarizedValuesPushBack(sValue);
}

//summarize the attributes into one attribute
//The parameter manner specyfies the way how to do this.
void Scalar2Vector::summarizeEntry(
    const char **xName,
    const char **xValue,
    const char **yName,
    const char **yValue,
    const char **zName,
    const char **zValue,
    int manner,
    char *prefixOrSuffix)
{
    switch (manner)
    {
        char *sName;
        int len;
    case 0:
        //We do nothing because according to our heuristic
        //the names do not match
        break;
    case 1:
        sName = new char[strlen(*xName) + 1 + 10];
        *sName = '\0';
        strcpy(sName, *xName);
        removeNewLine(sName);
        summarizedNamesPushBack(sName);
        //mark the name as summarized
        *xName = NULL;
        *yName = NULL;
        *zName = NULL;
        summarizeValue(xValue, yValue, zValue);
        break;
    case 2:
        len = strlen(*xName);
        sName = new char[len + 5 + 10];
        sName[0] = '\0';
        sName[len + 4] = '\0';
        strcat(sName, prefixOrSuffix);
        strcat(sName, *xName + 1);
        removeNewLine(sName);
        summarizedNamesPushBack(sName);
        //mark the name as summarized
        *xName = NULL;
        *yName = NULL;
        *zName = NULL;
        summarizeValue(xValue, yValue, zValue);

        break;
    case 3:
        len = strlen(*xName);
        sName = new char[len + 4 + 10];
        sName[0] = '\0';
        strncat(sName, *xName, strlen(*xName) - 1);
        strcat(sName, prefixOrSuffix);
        removeNewLine(sName);

        //mark the names as summarized
        *xName = NULL;
        *yName = NULL;
        *zName = NULL;
        summarizedNamesPushBack(sName);
        summarizeValue(xValue, yValue, zValue);
        break;
    case 4:
        sName = new char[4 + 10];
        *sName = '\0';
        strcpy(sName, prefixOrSuffix);
        //mark the names as summarized
        *xName = NULL;
        *yName = NULL;
        *zName = NULL;
        removeNewLine(sName);

        summarizedNamesPushBack(sName);
        summarizeValue(xValue, yValue, zValue);
        break;
    }
}

//Find out in which way the three attributes match
int Scalar2Vector::matchAttributeNames(const char **xName, const char **yName, const char **zName, char *prefixOrSuffix)
{
    //At least one of the strings is marked as "spent" so the strings cannot match
    if ((*xName == NULL) || (*yName == NULL) || (*zName == NULL))
    {
        return 0;
    }

    //All Strings are equal i.e. they match fully that is in primary manner
    if ((strcasecmp(*xName, *yName) == 0) && (strcasecmp(*xName, *zName) == 0))
    {
        return 1;
    }

    //We check whether the strings match in quarternary mode
    //i.e. they sonsist of one letter but are different
    if ((strlen(*xName) == 1) && (strlen(*xName) == 1) && (strlen(*xName) == 1))
    {
        prefixOrSuffix[0] = **xName;
        prefixOrSuffix[1] = **yName;
        prefixOrSuffix[2] = **zName;
        prefixOrSuffix[3] = '\0';
        return 4;
    }

    //Now we check whether the strings match in secondary or tertiary manner i.e.
    //they have a suffix and a common prefix or a prefix and a common suffix
    //At the moment we assume that the prefix is one letter ( normally x, y, or z)
    if ((strlen(*xName) > 1) && (strlen(*xName) > 1) && (strlen(*xName) > 1))
    {
        if ((strcasecmp(*xName + 1, *yName + 1) == 0) && (strcasecmp(*xName + 1, *zName + 1) == 0))
        {
            prefixOrSuffix[0] = **xName;
            prefixOrSuffix[1] = **yName;
            prefixOrSuffix[2] = **zName;
            prefixOrSuffix[3] = '\0';
            return 2;
        }
        //Perhaps they have a common prefix and a different suffix
        if ((strlen(*xName) == strlen(*yName)) && (strlen(*xName) == strlen(*zName)))
        {
            int len = strlen(*xName);
            if ((strncasecmp(*xName, *yName, len - 1) == 0) && (strncasecmp(*xName, *zName, len - 1) == 0))
            {
                prefixOrSuffix[0] = (*xName)[len - 1];
                prefixOrSuffix[1] = (*yName)[len - 1];
                prefixOrSuffix[2] = (*zName)[len - 1];
                prefixOrSuffix[3] = '\0';
                return 3;
            }
        }
    }

    //Feel free to implement futher heuristics of matching strings
    //with ascending return codes
    //In this case you must implement a way to handle this code in the method
    // summarizeEntry
    return 0;
}

//Check all combinations of all attribute names whether
//they can be summarized.
void Scalar2Vector::summarizeAttributes(
    int xSize,
    const char **xNames,
    const char **xValues,
    int ySize,
    const char **yNames,
    const char **yValues,
    int zSize,
    const char **zNames,
    const char **zValues)
{
    int manner;
    int x, y, z;

    for (x = 0; x < xSize; x++)
        for (y = 0; y < ySize; y++)
            for (z = 0; z < zSize; z++)
            {
                char prefixOrSuffix[4];
                manner = matchAttributeNames(xNames + x, yNames + y, zNames + z, prefixOrSuffix);
                summarizeEntry(xNames + x, xValues + x,
                               yNames + y, yValues + y,
                               zNames + z, zValues + z,
                               manner, prefixOrSuffix);
            }
    //Now we append to the lists of summarized names
    //and attributes these ones  which could not have been simplified
    char *sValue, *sName;
    for (x = 0; x < xSize; x++)
    {
        if (xNames[x] != NULL)
        {
            sName = new char[strlen(xNames[x]) + 1 + 10];
            *sName = '\0';
            strcpy(sName, xNames[x]);
            removeNewLine(sName);
            summarizedNamesPushBack(sName);
            sValue = new char[strlen(xValues[x]) + 1 + 10];
            strcpy(sValue, xValues[x]);
            removeNewLine(sValue);
            summarizedValuesPushBack(sValue);
        }
    }
    for (y = 0; y < ySize; y++)
    {
        if (yNames[y] != NULL)
        {
            sName = new char[strlen(yNames[y]) + 1 + 10];
            *sName = '\0';
            strcpy(sName, yNames[y]);
            removeNewLine(sName);
            summarizedNamesPushBack(sName);
            sValue = new char[strlen(yValues[y]) + 1 + 10];
            strcpy(sValue, yValues[y]);
            removeNewLine(sValue);
            summarizedValuesPushBack(sValue);
        }
    }
    for (z = 0; z < zSize; z++)
    {
        if (zNames[z] != NULL)
        {
            sName = new char[strlen(zNames[z]) + 1 + 10];
            *sName = '\0';
            strcpy(sName, zNames[z]);
            removeNewLine(sName);
            summarizedNamesPushBack(sName);
            sValue = new char[strlen(zValues[z]) + 1 + 10];
            strcpy(sValue, zValues[z]);
            removeNewLine(sValue);
            summarizedValuesPushBack(sValue);
        }
    }
}

void Scalar2Vector::summarizeAttributes(
    const coDistributedObject *objU,
    const coDistributedObject *objV,
    const coDistributedObject *objW)
{
    /*   if(summarizedNames != NULL)
      {
         for(int i=0; i<summarized; i++)
         {
            delete [] summarizedNames[i];
            delete [] summarizedValues[i];
         }
         delete [] summarizedNames;
         delete [] summarizedValues;
         summarized = 0;
         }*/
    int xSize;
    const char **xNames;
    const char **xValues;
    int ySize;
    const char **yNames;
    const char **yValues;
    int zSize;
    const char **zNames;
    const char **zValues;

    xSize = objU->getAllAttributes(&xNames, &xValues);
    ySize = objV->getAllAttributes(&yNames, &yValues);
    zSize = objW->getAllAttributes(&zNames, &zValues);
    summarizedNames = new const char *[xSize + ySize + zSize];
    summarizedValues = new const char *[xSize + ySize + zSize];
    summarized = 0;
    summarizeAttributes(xSize, xNames, xValues,
                        ySize, yNames, yValues,
                        zSize, zNames, zValues);
}

////////////////////////////////////////////////////////////////////////////////

//check whether the input data are suitable
void Scalar2Vector::checkInPort(
    const coDistributedObject *obj,
    coInputPort *p_port,
    int &retVal,
    char *errMsg)
{
    if (obj == NULL)
    {
        strcat(errMsg, "Did not receive object at port");
        strcat(errMsg, p_port->getName());
        strcat(errMsg, "\n");
        retVal = FAIL;
    }
    else
    {
        if (!obj->isType("USTSDT"))
        {
            strcat(errMsg, "Received illegal type at port ");
            strcat(errMsg, p_port->getName());
            strcat(errMsg, "\n");
            retVal = FAIL;
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Scalar2Vector::Scalar2Vector(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Scalar2Vector")
{
    // Parameters

    // Ports
    p_inPortU = addInputPort("inPortU",
                             "Float", "Scalar input for U/red");
    p_inPortV = addInputPort("inPortV",
                             "Float", "Scalar input for V/green");
    p_inPortW = addInputPort("inPortW",
                             "Float", "Scalar input for W/blue");
    p_inPortA = addInputPort("inPortA", "Float", "Scalar input for alpha/opacity");
    p_inPortA->setRequired(0);
    p_outPort = addOutputPort("outPort", "Vec3", "Vector output");
    p_outPortPacked = addOutputPort("outPortRGBA", "RGBA", "Packed color output");

    // value to ignore for min/max
    const char *normalizeMode[] = { "Do not", "Ignore MAX_FLT", "Ignore user defined value", "User defined bounds" };
    p_paramNormalizeChoice = addChoiceParam("normalizeMode", "if and how the packed RGBA data is normalized");
    p_paramNormalizeChoice->setValue(4, normalizeMode, 0);
    p_paramNormalizeIgnore = addFloatParam("userIgnoreValue", "Value to ignore for min and max for normalization");
    p_paramNormalizeIgnore->setValue(0.0);
    p_paramNormalizeMin = addFloatVectorParam("userNormalizeMin", "Minima to use for normalization", 4);
    p_paramNormalizeMax = addFloatVectorParam("userNormalizeMax", "Maxima to use for normalization", 4);
    for (int i = 0; i < 4; i++)
    {
        p_paramNormalizeMin->setValue(i, 0.0);
        p_paramNormalizeMax->setValue(i, 1.0);
    }

    summarized = 0;
    summarizedNames = NULL;
    summarizedValues = NULL;

    setCopyNonSetAttributes(0);
}

//Check whether the input ports are OK
int Scalar2Vector::checkInPorts(
    const coDistributedObject *objU,
    const coDistributedObject *objV,
    const coDistributedObject *objW,
    char *errMsg)
{
    errMsg[0] = '\0';
    int retVal = SUCCESS;

    //Check whether we have objects
    checkInPort(objU, p_inPortU, retVal, errMsg);
    checkInPort(objV, p_inPortV, retVal, errMsg);
    checkInPort(objW, p_inPortW, retVal, errMsg);
    if (retVal == FAIL)
    {
        return FAIL;
    }
    //Check whether objects have the same type
    if (!(objU->isType("USTSDT") && objV->isType("USTSDT") && objW->isType("USTSDT")))
    {
        strcat(errMsg, "data types of Input ports ");
        strcat(errMsg, p_inPortU->getName());
        strcat(errMsg, p_inPortV->getName());
        strcat(errMsg, p_inPortW->getName());
        strcat(errMsg, "do not match");
        return FAIL;
    }

    //Check whether objects have the same sizes
    coDoFloat *pU = (coDoFloat *)objU;
    coDoFloat *pV = (coDoFloat *)objV;
    coDoFloat *pW = (coDoFloat *)objW;

    if (!(
            (pU->getNumPoints() == pV->getNumPoints()) && (pU->getNumPoints() == pW->getNumPoints())))
    {
        strcat(errMsg, "Dimensions of unstructured input data do not match");
        return FAIL;
    }
    return SUCCESS;
}

inline uint32_t tobyte(float f, float min, float scale)
{
    float v = (f - min) * scale;
    if (v < 0.f)
        v = 0.f;
    if (v > 255.f)
        v = 255.f;

    return uint32_t(v) & 0xff;
}

//combine structured scalar data to structured vector date
coDistributedObject *Scalar2Vector::computeStructuredPacked(
    const coDoFloat *uData,
    const coDoFloat *vData,
    const coDoFloat *wData,
    const coDoFloat *aData)
{
    if (!p_outPortPacked->isConnected())
    {
        return NULL;
    }

    bool computeNormBounds = true;
    bool normalize = true;

    float ignoreValue = p_paramNormalizeIgnore->getValue();
    switch (p_paramNormalizeChoice->getValue())
    {
    case 0:
        normalize = false;
        computeNormBounds = false;
        break;
    case 1:
        ignoreValue = FLT_MAX;
        break;
    case 2:
        ignoreValue = p_paramNormalizeIgnore->getValue();
        break;
    case 3:
        computeNormBounds = false;
        break;
    default:
        fprintf(stderr, "unhandled value for normalization parameter\n");
        break;
    }

    int nelem = uData->getNumPoints();
    float *r, *g, *b, *a = NULL;
    uData->getAddress(&r);
    vData->getAddress(&g);
    wData->getAddress(&b);
    if (aData)
    {
        aData->getAddress(&a);
    }

    float rMax = -FLT_MAX, gMax = -FLT_MAX, bMax = -FLT_MAX, aMax = -FLT_MAX;
    float rMin = FLT_MAX, gMin = FLT_MAX, bMin = FLT_MAX, aMin = FLT_MAX;

    if (computeNormBounds)
    {
        for (int i = 0; i < nelem; i++)
        {
            if (r[i] != ignoreValue)
            {
                if (rMax < r[i])
                    rMax = r[i];
                if (rMin > r[i])
                    rMin = r[i];
            }
            if (g[i] != ignoreValue)
            {
                if (gMax < g[i])
                    gMax = g[i];
                if (gMin > g[i])
                    gMin = g[i];
            }
            if (b[i] != ignoreValue)
            {
                if (bMax < b[i])
                    bMax = b[i];
                if (bMin > b[i])
                    bMin = b[i];
            }
            if (a && a[i] != ignoreValue)
            {
                if (aMax < a[i])
                    aMax = a[i];
                if (aMin > a[i])
                    aMin = a[i];
            }
        }
    }
    else
    {
        rMin = p_paramNormalizeMin->getValue(0);
        gMin = p_paramNormalizeMin->getValue(1);
        bMin = p_paramNormalizeMin->getValue(2);
        aMin = p_paramNormalizeMin->getValue(3);

        rMax = p_paramNormalizeMax->getValue(0);
        gMax = p_paramNormalizeMax->getValue(1);
        bMax = p_paramNormalizeMax->getValue(2);
        aMax = p_paramNormalizeMax->getValue(3);
    }

    const float scale = 255.99f;

    float rs = 0.f;
    if (rMax == rMin)
    {
        if (rMax >= 0.0 && rMax <= 1.0)
            rMin = 0.0;
        rs = scale;
    }
    else if (rMax != -FLT_MAX && rMin != FLT_MAX)
    {
        rs = 1.f / (rMax - rMin) * scale;
    }

    float gs = 0.f;
    if (gMax == gMin)
    {
        if (gMax >= 0.0 && gMax <= 1.0)
            gMin = 0.0;
        gs = scale;
    }
    else if (gMax != -FLT_MAX && gMin != FLT_MAX)
    {
        gs = 1.f / (gMax - gMin) * scale;
    }

    float bs = 0.f;
    if (bMax == bMin)
    {
        if (bMax >= 0.0 && bMax <= 1.0)
            bMin = 0.0;
        bs = scale;
    }
    else if (bMax != -FLT_MAX && bMin != FLT_MAX)
    {
        bs = 1.f / (bMax - bMin) * scale;
    }

    float as = 0.f;
    if (aMax == aMin)
    {
        if (aMax >= 0.0 && aMax <= 1.0)
            aMin = 0.0;
        as = scale;
    }
    else if (aMax != -FLT_MAX && aMin != FLT_MAX)
    {
        as = 1.f / (aMax - aMin) * scale;
    }

    uint32_t *pc = new uint32_t[nelem];
    if (a)
    {
        if (normalize)
        {
            for (int i = 0; i < nelem; i++)
            {
                pc[i] = (tobyte(r[i], rMin, rs) << 24) | (tobyte(g[i], gMin, gs) << 16) | (tobyte(b[i], bMin, bs) << 8) | tobyte(a[i], aMin, as);
            }
        }
        else
        {
            for (int i = 0; i < nelem; i++)
            {
                pc[i] = (tobyte(r[i], 0.f, scale) << 24) | (tobyte(g[i], 0.f, scale) << 16) | (tobyte(b[i], 0.f, scale) << 8) | tobyte(a[i], 0.f, scale);
            }
        }
    }
    else
    {
        if (normalize)
        {
            for (int i = 0; i < nelem; i++)
            {
                pc[i] = (tobyte(r[i], rMin, rs) << 24) | (tobyte(g[i], gMin, gs) << 16) | (tobyte(b[i], bMin, bs) << 8);
                float rr = (r[i] - rMin) * rs;
                float gg = (g[i] - gMin) * gs;
                float bb = (b[i] - bMin) * bs;
                pc[i] |= tobyte(rr + gg + bb, 0.f, 1.f / 3.f);
            }
        }
        else
        {
            for (int i = 0; i < nelem; i++)
            {
                pc[i] = (tobyte(r[i], 0.f, scale) << 24) | (tobyte(g[i], 0.f, scale) << 16) | (tobyte(b[i], 0.f, scale) << 8);
                pc[i] |= tobyte(r[i] + g[i] + b[i], 0.f, scale / 3.f);
            }
        }
    }

    coDoRGBA *outData
        = new coDoRGBA(
            p_outPortPacked->getObjName(),
            nelem,
            (int *)pc);
    return outData;
}

//combine unstructured scalar data to unstructured vector data
coDistributedObject *Scalar2Vector::computeUnstructured(
    const coDoFloat *uData,
    const coDoFloat *vData,
    const coDoFloat *wData)
{
    int numValues = uData->getNumPoints();
    float *uAddress, *vAddress, *wAddress;
    uData->getAddress(&uAddress);
    vData->getAddress(&vAddress);
    wData->getAddress(&wAddress);
    coDoVec3 *outData
        = new coDoVec3(
            p_outPort->getObjName(),
            numValues,
            uAddress,
            vAddress,
            wAddress);
    return outData;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int Scalar2Vector::compute(const char *)
{
    char errMsg[4096];
    const coDistributedObject *objU = p_inPortU->getCurrentObject();
    const coDistributedObject *objV = p_inPortV->getCurrentObject();
    const coDistributedObject *objW = p_inPortW->getCurrentObject();
    const coDistributedObject *objA = p_inPortA->getCurrentObject();

    //First consistency checks for all the in ports
    if (checkInPorts(objU, objV, objW, errMsg) == FAIL)
    {
        sendError("%s", errMsg);
        return FAIL;
    }

    //computation of the out port depending on the type of the in ports
    coDistributedObject *outData = NULL;
    coDistributedObject *outDataPacked = NULL;
    if (objU->isType("USTSDT"))
    {
        outDataPacked = computeStructuredPacked(
            static_cast<const coDoFloat *>(objU),
            static_cast<const coDoFloat *>(objV),
            static_cast<const coDoFloat *>(objW),
            static_cast<const coDoFloat *>(objA));
    }

    outData = computeUnstructured(
        static_cast<const coDoFloat *>(objU),
        static_cast<const coDoFloat *>(objV),
        static_cast<const coDoFloat *>(objW));
    if (outData == NULL)
    {
        sendError("Failed to create object '%s' for port '%s'",
                  p_outPort->getObjName(), p_outPort->getName());
        return FAIL;
    }

    //Now we unify the attributes of the three input objects

    summarizeAttributes(objU, objV, objW);

    outData->addAttributes(summarized, summarizedNames, summarizedValues);

    // sl: use setObj, if we want to see the output!!!
    p_outPort->setCurrentObject(outData);
    p_outPortPacked->setCurrentObject(outDataPacked);

    return SUCCESS;
}

MODULE_MAIN(Converter, Scalar2Vector)
