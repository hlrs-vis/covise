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

#include "Colors.h"
#include <config/CoviseConfig.h>

#include <float.h>
#include <limits.h>
#include <do/coDoColormap.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoPixelImage.h>
#include <do/coDoTexture.h>
#include <api/coFeedback.h>

// Data values of more than  NoDataColorPercent*FLT_MAX are non-data values
static const float NoDataColorPercent = 0.01f;

//////////////////////////////////////////////////////////////////////
//
// initialize parameters and ports for "Colors" appearance
//
//////////////////////////////////////////////////////////////////////

void Colors::initCOLORS()
{
    ////////////////// Parameters //////////////////

    static float defaultMinmaxValues[2] = { 0.0, 1.0 };
    p_minmax = addFloatVectorParam("MinMax", "Minimum and Maximum value");
    p_minmax->setValue(2, defaultMinmaxValues);

    p_colorMap = addColormapParam("EditableColormap", "Colormap Editor");
    p_colorNames = addColormapChoiceParam("Colormap", "Select a Colormap");
    readMaps();

    int confNumSteps = coCoviseConfig::getInt("Module.Colors.NumSteps", 256);

    if (confNumSteps < 2)
        confNumSteps = 2;
    if (confNumSteps > 256)
        confNumSteps = 256;
    p_numSteps = addInt32Param("numSteps", "Number of Steps in Map");
    p_numSteps->setValue(confNumSteps);

    p_autoScale = addBooleanParam("autoScales", "Automatically adjust Min and Max");
    p_autoScale->setValue(coCoviseConfig::isOn("Module.Colors.AutoScale", true));

    p_scaleNow = addBooleanParam("scaleNow", "Re-scale and execute immediately");
    p_scaleNow->setValue(0);

    p_alpha = addFloatParam("opacityFactor", "Global opacity multiplicator");
    p_alpha->setValue(1.0);

    p_annotation = addStringParam("annotation", "Colormap Annotation String");
    p_annotation->setValue("Colors");
    p_spikeAlgo = NULL;

    // new parameters for Spike removal: only if configured
    if (coCoviseConfig::isOn("Module.Colors.SpikeRemoval", true))
    {
        const char *spikeAlgoChoices[] = {
            "None", // = 0 = SPIKE_NONE
            "Adaptive", // = 1 = SPIKE_ADAPTIVE
            "Interval", // = 2 = SPIKE_INTERVAL
            "Elements", // = 3 = SPIKE_ELEMENTS
        };
        p_spikeAlgo = addChoiceParam("SpikeAlgo", "Select Spike removal algorithm");
        std::string algoConfig = coCoviseConfig::getEntry("Module.Colors.SpikeAlgo");
        if (!algoConfig.empty())
        {
            if (0 == strcasecmp("Adaptive", algoConfig.c_str()))
                p_spikeAlgo->setValue(4, spikeAlgoChoices, SPIKE_ADAPTIVE);
            else if (0 == strcasecmp("Interval", algoConfig.c_str()))
                p_spikeAlgo->setValue(4, spikeAlgoChoices, SPIKE_INTERVAL);
            else if (0 == strcasecmp("Elements", algoConfig.c_str()))
                p_spikeAlgo->setValue(4, spikeAlgoChoices, SPIKE_ELEMENTS);
            else
                p_spikeAlgo->setValue(4, spikeAlgoChoices, SPIKE_NONE);
        }
        else
            p_spikeAlgo->setValue(4, spikeAlgoChoices, SPIKE_NONE);

        float spikeLowFract = coCoviseConfig::getFloat("Module.Colors.SpikeLowFract", 0.05f);
        p_spikeBot = addFloatParam("SpikeLowFract", "Spike Ratio for low values");
        p_spikeBot->setValue(spikeLowFract);

        float spikeTopFract = coCoviseConfig::getFloat("Module.Colors.SpikeTopFract", 0.05f);
        p_spikeTop = addFloatParam("SpikeTopFract", "Spike Ratio for high values");
        p_spikeTop->setValue(spikeTopFract);

        // get some magical nubers for the spike removal algorithms
        numBinsAdaptive = coCoviseConfig::getInt("Module.Colors.NumBinsAdaptive", 50);

        numBinsElements = coCoviseConfig::getInt("Module.Colors.NumBinsElements", 50);
    }
    else
    {
        p_spikeAlgo = NULL;
        p_spikeBot = NULL;
        p_spikeTop = NULL;
    }

    // Input Ports
    p_data = addInputPort("DataIn0", "Vec3|IntArr|Int|Float|Byte", "scalar data");
    p_data->setRequired(0);

    p_alphaIn = addInputPort("DataIn1", "Vec3|IntArr|Int|Float|Byte", "scalar value");
    p_alphaIn->setRequired(0);

    p_histoIn = addInputPort("DataIn2", "Float", "histogram data");
    p_histoIn->setRequired(0);

    p_cmapIn = addInputPort("ColormapIn0", "ColorMap|MinMax_Data", "Colormap Input");
    p_cmapIn->setRequired(0);

    // Output ports
    p_texture = addOutputPort("TextureOut0", "Texture", "Data or colormap as texture");
    p_color = addOutputPort("DataOut0", "RGBA", "Data as colors");
    p_color->setDependencyPort(p_data);

    p_cmapOut = addOutputPort("ColormapOut0", "ColorMap", "Colormap Output");
}

// read all predefined colormaps
// global and local
void Colors::readMaps()
{
    // get names of colormaps ind config-colormap.xml
    coCoviseConfig::ScopeEntries keysEntries = coCoviseConfig::getScopeEntries("Colormaps");
    const char **keys = keysEntries.getValue();

    vector<string> mapNames;
    mapNames.push_back("Editable");
    if (keys)
    {
        int i = 0;
        while (keys[i] != NULL)
        {
            mapNames.push_back(keys[i]);
            i = i + 2;
        }
    }

    // allocate place for n colormaps
    numColormaps = mapNames.size();
    colormaps = new TColormapChoice[numColormaps];
    colormapAttributes.resize(numColormaps);

    // set first the module defined colormap
    colormaps[0].mapName = "Editable";
    colormap_type type;
    float min = 0.;
    float max = 1.;
    colormapAttributes[0].min = min;
    colormapAttributes[0].max = min;
    colormapAttributes[0].isAbsolute = false;

    const float *rgbax;
    int numSteps = p_colorMap->getValue(&min, &max, &type, &rgbax);
    for (int i = 0; i < numSteps; i++)
    {
        for (int c = 0; c < 5; c++)
            colormaps[0].mapValues.push_back(rgbax[i * 5 + c]);
    }

    // read all other colormaps from config-colormap.xml
    for (int i = 1; i < numColormaps; i++)
    {
        string name = "Colormaps." + mapNames[i];
        bool absolute = coCoviseConfig::isOn("absolute", name, false);
        coCoviseConfig::ScopeEntries entries = coCoviseConfig::getScopeEntries(name);
        const char **keys = entries.getValue();

        // read names
        if (keys)
        {
            int no;
            for (no = 0; keys[no] != NULL; no = no + 2)
            {
            }
            no = no / 2;

            // read all sampling points
            float diff = 1.0 / (no - 1);
            float pos = 0.0;
            for (int j = 0; j < no; j++)
            {
                ostringstream out;
                out << j;
                string tmp = name + ".Point:" + out.str();

                bool rgb = false;
                string rgba = coCoviseConfig::getEntry("rgba", tmp);
                if (rgba.empty())
                {
                    rgb = true;
                    rgba = coCoviseConfig::getEntry("rgb", tmp);
                }
                if (!rgba.empty())
                {
                    float a = 1.;
                    uint32_t c = strtol(rgba.c_str(), NULL, 16);
                    if (!rgb)
                    {
                        a = (c & 0xff) / 255.0;
                        c >>= 8;
                    }
                    float b = (c & 0xff) / 255.0;
                    c >>= 8;
                    float g = (c & 0xff) / 255.0;
                    c >>= 8;
                    float r = (c & 0xff) / 255.0;
                    colormaps[i].mapValues.push_back(r);
                    colormaps[i].mapValues.push_back(g);
                    colormaps[i].mapValues.push_back(b);
                    colormaps[i].mapValues.push_back(a);
                    colormaps[i].mapValues.push_back(coCoviseConfig::getFloat("x", tmp, pos));
                }
                else
                {
                    colormaps[i].mapValues.push_back(coCoviseConfig::getFloat("r", tmp, 1.0));
                    colormaps[i].mapValues.push_back(coCoviseConfig::getFloat("g", tmp, 1.0));
                    colormaps[i].mapValues.push_back(coCoviseConfig::getFloat("b", tmp, 1.0));
                    colormaps[i].mapValues.push_back(coCoviseConfig::getFloat("a", tmp, 1.0));
                    colormaps[i].mapValues.push_back(coCoviseConfig::getFloat("x", tmp, pos));
                }
                pos = pos + diff;
            }

            colormapAttributes[i].isAbsolute = absolute;
            colormapAttributes[i].min = 0.;
            colormapAttributes[i].max = 1.;

            if (absolute)
            {
                double min = colormaps[i].mapValues[0 + 4];
                double max = colormaps[i].mapValues[(no - 1) * 5 + 4];
                colormapAttributes[i].min = min;
                colormapAttributes[i].max = max;
                for (int j = 0; j < no; ++j)
                {
                    double x = colormaps[i].mapValues[j * 5 + 4];
                    colormaps[i].mapValues[j * 5 + 4] = (x - min) / (max - min);
                }
            }
            colormaps[i].mapName = mapNames[i];
        }
    }

    // set defined colormaps
    p_colorNames->setValue(numColormaps, 0, colormaps);

    /*
   // read values of local colormap files in .covise
   QString place = coConfigDefaultPaths::getDefaultLocalConfigFilePath()  + "colormaps";

   QDir directory(place);
   if(directory.exists())
   {
      QStringList filters;
      filters << "colormap_*.xml";
      directory.setNameFilters(filters);
      directory.setFilter(QDir::Files);
      QStringList files = directory.entryList();

      // loop over all found colormap xml files
      for(int j=0; j<files.size(); j++)
      {
         coConfigGroup *colorConfig = new coConfigGroup("Colormap");
         colorConfig->addConfig(place + "/" + files[j], "local", true);

         // read the name of the colormaps
         QStringList list;
         list = colorConfig->getVariableList("Colormaps");

         // loop over all colormaps in one file
         for(int i=0; i<list.size(); i++)
         {

            // remove global colormap with same name
            int index = mapNames.indexOf(list[i]);
            if(index != -1)
            {
               mapNames.removeAt(index);
               deleteMap(list[i]);
            }
            mapNames.append(list[i]);

            // get all definition points for the colormap
            QString cmapname = "Colormaps." + mapNames.last();
            QStringList variable = colorConfig->getVariableList(cmapname);

            mapSize.insert(list[i], variable.size());
            float *cval = new float[variable.size()*5];
            mapValues.insert(list[i], cval);

            // read the rgbax values
            int it = 0;
            for(int l=0; l<variable.size()*5; l=l+5)
            {
               QString tmp = cmapname + ".Point:" + QString::number(it);
               cval[l]   = (colorConfig->getValue("x", tmp," -1.0")).toFloat();
               cval[l+1] = (colorConfig->getValue("r", tmp, "1.0")).toFloat();
               cval[l+2] = (colorConfig->getValue("g", tmp, "1.0")).toFloat();
               cval[l+3] = (colorConfig->getValue("b", tmp, "1.0")).toFloat();
               cval[l+4] = (colorConfig->getValue("a", tmp, "1.0")).toFloat();
               it++;
            }
         }
         config->removeConfig(place + "/" + files[j]);
      }
   }*/
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Colors::Colors(int argc, char *argv[])
    : coModule(argc, argv, "Map scalars to colors")

{
    d_noDataColor = 0x00000000;
    colormaps = NULL;
    numColormaps = 0;
    textureComponents = 3;
    if (coCoviseConfig::isOn("Module.Colors.TransparentTextures", true))
    {
        textureComponents = 4;
    }

    initCOLORS();
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Param callback for scaleNow button
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Colors::param(const char *portName, bool inMapLoading)
{
    if (strcmp(portName, p_scaleNow->getName()) == 0)
    {
        // do NOT exec when loading from a map
        if (!inMapLoading && p_scaleNow->getValue())
        {
            static float values[2] = { 0.0, 0.0 };
            p_minmax->setValue(2, values);
            p_autoScale->setValue(1);
            selfExec();
        }
    }

    else if (strcmp(portName, p_minmax->getName()) == 0)
    {
        updateMinMax(p_minmax->getValue(0), p_minmax->getValue(1));
        if (!inMapLoading)
        {
            p_autoScale->setValue(0);
        }
    }

    // get new module defined colormap
    else if (strcmp(portName, p_colorMap->getName()) == 0)
    {
        float min = 0.0, max = 1.0;
        colormap_type type;
        const float *rgbax;

        int numSteps = p_colorMap->getValue(&min, &max, &type, &rgbax);
        colormaps[0].mapValues.clear();

        for (int i = 0; i < numSteps; i++)
        {
            for (int c = 0; c < 5; c++)
                colormaps[0].mapValues.push_back(rgbax[i * 5 + c]);
        }

        if (!inMapLoading)
            p_colorNames->setValue(numColormaps, 0, colormaps);
    }

    else if (strcmp(portName, p_colorNames->getName()) == 0)
    {
        int index = p_colorNames->getValue();
        if (index >= 0 && index < colormapAttributes.size()
            && colormapAttributes[index].isAbsolute)
        {
            updateMinMax(colormapAttributes[index].min,
                         colormapAttributes[index].max);
            p_autoScale->setValue(0);
        }
        else if (index < 0 || index >= colormapAttributes.size())
        {
            fprintf(stderr, "WARNING: colormap is out of range! index is set to 0 (%d)\n", index);
            index = 0;
            //TColormapChoice color = p_colorNames->getValue(index);
            //p_colorNames->setValue(numColormaps, 0, color);
            p_colorNames->setValue(index);
            if (colormapAttributes[index].isAbsolute)
            {
                updateMinMax(colormapAttributes[index].min, colormapAttributes[index].max);
                p_autoScale->setValue(0);
            }
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  recursively open all objects and find own min/max
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int Colors::openObj(recObj &base, const coDistributedObject *obj, const char *&species)
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

        else if (const coDoByte *bObj = dynamic_cast<const coDoByte *>(obj))
        {
            base.objList = NULL;
            base.subObj = NULL;
            base.doDelete = true;
            base.numElem = bObj->getNumPoints();
            base.data = new float[base.numElem];
            unsigned char *dataPtr = (unsigned char *)bObj->getAddress();
            for (int i = 0; i < base.numElem; i++)
                base.data[i] = dataPtr[i] / 255.f;
            return SUCCESS;
        }

        else if (const coDoInt *iObj = dynamic_cast<const coDoInt *>(obj))
        {
            base.objList = NULL;
            base.subObj = NULL;
            base.doDelete = true;
            base.numElem = iObj->getNumPoints();
            base.data = new float[base.numElem];
            int *dataPtr = iObj->getAddress();
            for (int i = 0; i < base.numElem; i++)
                base.data[i] = dataPtr[i];
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
                Covise::sendError("Cannot handle multi-dimensional integers");
                return FAIL;
            }
            base.numElem = ((coDoIntArr *)iObj)->getDimension(0);
            base.data = new float[base.numElem];
            int *dataPtr = iObj->getAddress();
            for (int i = 0; i < base.numElem; i++)
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
            int i;
            for (i = 0; i < base.numElem; i++)
                base.data[i] = sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
            return SUCCESS;
        }

        // invalid
        else
        {
            Covise::sendError("Cannot handle vector data");
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

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  recursively open all objects and find own min/max
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Colors::getMinMax(const recObj &base, float &min, float &max)
{
    if (base.obj == NULL)
        return; // we don't care for empty ones

    int i;

    // recurse
    if (base.subObj)
    {
        for (i = 0; i < base.numElem; i++)
            getMinMax(base.subObj[i], min, max);
    }

    // run over own object
    else
    {
        for (i = 0; i < base.numElem; i++)
        {
            // do not care about FLT_MAX elements in min/max calc.
            if ((base.data[i] < NoDataColorPercent * FLT_MAX) && (base.data[i] < min))
                min = base.data[i];
            if ((base.data[i] < NoDataColorPercent * FLT_MAX) && (base.data[i] > max))
                max = base.data[i];
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  recursively create float RGB colors
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coDistributedObject *Colors::createColors(recObj &base, recObj &alpha, FlColor *actMap,
                                          float min, float max, int numSteps,
                                          const char *name, int outStyle)
{
    //if (base.obj==NULL) return NULL;               // empty to empty

    int i;

    // create set (recurse)
    if (base.subObj)
    {
        // create array for Set
        coDistributedObject **setArr = new coDistributedObject *[base.numElem];

        // recursively create set objects
        for (i = 0; i < base.numElem; i++)
        {
            stringstream namebuf;
            namebuf << name << "_" << i;
            if (alpha.subObj)
            {
                setArr[i] = createColors(base.subObj[i], alpha.subObj[i], actMap, min, max,
                                         numSteps, namebuf.str().c_str(), outStyle);
            }
            else
            {
                setArr[i] = createColors(base.subObj[i], alpha, actMap, min, max,
                                         numSteps, namebuf.str().c_str(), outStyle);
            }
        }

        // Create set
        coDoSet *set = new coDoSet(name, base.numElem, setArr);
        for (i = 0; i < base.numElem; i++)
            delete setArr[i];
        delete[] setArr;

        set->copyAllAttributes(base.obj);
        return set;
    }

    // run over own object
    else
    {
        float *data = base.data; // where my data starts
        float *alphaData = alpha.data;
        float delta = numSteps / (max - min); // cmap steps
        int maxIdx = numSteps - 1;

        // packed RGBA data
        if (outStyle == RGBA)
        {
            coDoRGBA *res = new coDoRGBA(name, base.numElem);
            int *dPtr;
            res->getAddress(&dPtr);
            unsigned int *packed = (unsigned int *)dPtr;
            unsigned char r, g, b, a;
            for (i = 0; i < base.numElem; i++)
            {
                if (data[i] >= NoDataColorPercent * FLT_MAX)
                {
                    packed[i] = d_noDataColor;
                }
                else
                {
                    int idx = (int)((data[i] - min) * delta);
                    if (idx < 0)
                        idx = 0;
                    if (idx > maxIdx)
                        idx = maxIdx;
                    r = (int)(actMap[idx][0] * 255);
                    g = (int)(actMap[idx][1] * 255);
                    b = (int)(actMap[idx][2] * 255);
                    if (alphaData)
                        a = (int)(alphaData[i] * 255);
                    else
                        a = (int)(actMap[idx][3] * 255);
                    packed[i] = (r << 24) | (g << 16) | (b << 8) | a;
                }
            }
            res->copyAllAttributes(base.obj);
            return res;
        }
        // packed RGBA data
        else if (outStyle == TEX)
        {
            // prepare the texture image
            stringstream namebuf;
            int texSize = 256;
            while (texSize < numSteps)
                texSize *= 2;
            namebuf << name << "_Img";
            char *image = new char[textureComponents * texSize];
            char *iPtr = image;
            float delta = 1.0f / (texSize - 1) * (numSteps - 0.00001f);
            for (i = 0; i < texSize; i++)
            {
                int k = (int)(delta * i);
                *iPtr = (int)(actMap[k][0] * 255);
                iPtr++;
                *iPtr = (int)(actMap[k][1] * 255);
                iPtr++;
                *iPtr = (int)(actMap[k][2] * 255);
                iPtr++;
                if (textureComponents == 4)
                {
                    *iPtr = (int)(actMap[k][3] * 255);
                    iPtr++;
                }
            }

            coDoPixelImage *pix = new coDoPixelImage(namebuf.str().c_str(), texSize, 1, textureComponents, textureComponents, image);
            //coDoPixelImage *pix = new coDoPixelImage(namebuf,texSize,1,3,3,image);

            int *txIndex = NULL;
            float **txCoord = new float *[2];
            txCoord[0] = NULL;
            txCoord[1] = NULL;
            float fact = 1.0f / (max - min);
            if (data)
            {
                // texture coordinate index
                txIndex = new int[base.numElem];
                txCoord[0] = new float[base.numElem];
                txCoord[1] = new float[base.numElem];

                for (i = 0; i < base.numElem; i++)
                {
                    float tx = (data[i] - min) * fact;
                    if (tx < 0.0)
                        tx = 0.0;
                    if (tx > 1.0)
                        tx = 1.0;
                    txCoord[0][i] = tx;
                    txCoord[1][i] = 0.0;
                    txIndex[i] = i;
                }
            }
            else
            {
                // texture coordinate index
                txIndex = new int[1];
                txCoord[0] = new float[1];
                txCoord[1] = new float[1];

                for (i = 0; i < 1; i++)
                {
                    float tx = 0.0;
                    txCoord[0][i] = tx;
                    txCoord[1][i] = 0.0;
                    txIndex[i] = i;
                }
            }

            coDoTexture *texture = new coDoTexture(name, pix, 0, textureComponents, 0,
                                                   base.numElem, txIndex, base.numElem, txCoord);
            // coDoTexture *texture = new coDoTexture( name, pix, 0, 3, 0,
            //	                              base.numElem, txIndex, base.numElem, txCoord );

            delete[] txCoord[0];
            delete[] txCoord[1];
            delete[] txCoord;
            delete[] txIndex;

            if (base.obj)
            {
                texture->copyAllAttributes(base.obj);
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
void Colors::addColormapAttrib(const char *objName, float min, float max,
                               coDistributedObject *outObj, const char *annotation,
                               const FlColor *map, int numSteps)
{
    stringstream buffer;
    buffer << objName << '\n' << (annotation ? annotation : "dummy") << '\n'
           << min << '\n' << max << '\n' << numSteps << '\n' << '0';

    buffer.precision(4);
    buffer << std::fixed;
    for (int i = 0; i < numSteps; i++)
    {
        buffer << "\n" << (map[i][0]) << "\n" << (map[i][1]) << "\n" << (map[i][2]) << "\n" << (map[i][3]);
    }

    outObj->addAttribute("OBJECTNAME", getTitle());
    outObj->addAttribute("COLORMAP", buffer.str().c_str());
    outObj->addAttribute("MODULE", "ColorBars");

    coFeedback feedback("ColorBars");
    feedback.addPara(p_minmax);
    feedback.addPara(p_numSteps);
    feedback.addPara(p_annotation);
    feedback.addPara(p_autoScale);
    feedback.addPara(p_scaleNow);
    if (p_spikeAlgo)
    {
        feedback.addPara(p_spikeAlgo);
        feedback.addPara(p_spikeBot);
        feedback.addPara(p_spikeTop);
    }
    feedback.addString(buffer.str().c_str());
    feedback.apply(outObj);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int Colors::compute(const char *)
{
    recObj base, alpha = recObj(); // Data base pointer
    FlColor *actMap = NULL; // the map we use
    const char *species = NULL; // Description
    int numSteps = 0; // steps in Map
    float min = 0.0;
    float max = 0.0;
    const char *annotation = NULL; // What's written at the Map

    int index = p_colorNames->getValue();
    TColormapChoice color = p_colorNames->getValue(index);
    float alphaMult = p_alpha->getValue();
    numColors = color.mapValues.size() / 5;
    d_cmap.clear();
    d_cmap.assign(color.mapValues.begin(), color.mapValues.end());

    // ---- open input objects

    const coDistributedObject *data = p_data->getCurrentObject();
    if (data && openObj(base, data, species) == FAIL)
        return FAIL;

    const coDistributedObject *alphaObj = p_alphaIn->getCurrentObject();
    if (alphaObj && openObj(alpha, alphaObj, species) == FAIL)
        return FAIL;

    // ---- Handle input colormaps
    // ---- or minmax information

    // in_data_color is designed to point to an incoming
    // data map or incoming min-max information
    const coDistributedObject *in_data_color = NULL;

    // colorMapIn is the result of the conversion of
    // in_data_color in the case that we receive a coDoColormap
    // object
    const coDoColormap *colorMapIn = NULL;

    // get histogram data if available
    if (p_histoIn)
    {
        const coDistributedObject *in_histogram = p_histoIn->getCurrentObject();

        if (const coDoFloat *histoData = dynamic_cast<const coDoFloat *>(in_histogram))
        {
            float *mmdata;
            histoData->getAddress(&mmdata);
            int np = histoData->getNumPoints();
            p_colorMap->setData(np, mmdata);
        }
    }

    if (p_cmapIn)
    {
        in_data_color = p_cmapIn->getCurrentObject();

        if ((colorMapIn = dynamic_cast<const coDoColormap *>(in_data_color)))
        {
            min = colorMapIn->getMin();
            max = colorMapIn->getMax();
            actMap = (FlColor *)colorMapIn->getAddress();
            annotation = colorMapIn->getMapName();
            numSteps = colorMapIn->getNumSteps();
        }
    }

    if (p_scaleNow && p_scaleNow->getValue() == 1)
    {
        p_scaleNow->setValue(0);
        p_autoScale->setValue(0);
    }

    // ---- Handle minmax input

    // minmaxIn is the result of the conversion of
    // in_data_color in the case that we receive a min-max
    // object
    const coDoFloat *minmaxIn = dynamic_cast<const coDoFloat *>(in_data_color);
    if (minmaxIn)
    {
        int numVal = minmaxIn->getNumPoints();
        if (numVal != 2)
        {
            sendError("Illegal input at minmax port");
            return FAIL;
        }
        float *mmdata;
        minmaxIn->getAddress(&mmdata);
        min = mmdata[0];
        max = mmdata[1];
        updateMinMax(min, max);
    }

    // ---- If have got a colormap object, it overrides everything else !
    if (!colorMapIn)
    {
        if (!minmaxIn) // if min/max not set by objects, get from param
        {
            min = p_minmax->getValue(0);
            max = p_minmax->getValue(1);
        }

        // still no correct values and flag set: find out Min/Max
        if (min == max
            || ((p_autoScale && p_autoScale->getValue()) && (minmaxIn == NULL)))
        {
            min = FLT_MAX;
            max = -FLT_MAX;

            // search data for min and max
            if (data)
                getMinMax(base, min, max);

            // Oops, there wasn't even one element...
            if (min == FLT_MAX || max == -FLT_MAX)
            {
                min = 0.0;
                max = 1.0;
            }
            // Otherwise we might want to eliminate spikes - but only if we configured it
            else if (p_spikeAlgo)
            {
                switch (p_spikeAlgo->getValue())
                {
                case SPIKE_NONE:
                    break;

                case SPIKE_ADAPTIVE:
                    removeSpikesAdaptive(base, min, max);
                    updateMinMax(min, max);
                    break;

                case SPIKE_INTERVAL:
                    removeSpikesInterval(base, min, max);
                    updateMinMax(min, max);
                    break;

                case SPIKE_ELEMENTS:
                    removeSpikesElements(base, min, max);
                    updateMinMax(min, max);
                    break;

                default:
                    sendWarning("Parameter %s returned illegal value",
                                p_spikeAlgo->getName());
                    break;
                }
            }

            // min and max are same - would give random results: not pretty
            if (min == max)
                max = min + 1;

            // update values in Map (only done in new module styles)
            updateMinMax(min, max);
        }

        // number of steps in cmap
        numSteps = p_numSteps->getValue();
        if (numSteps < 2 || numSteps > 65536)
        {
            sendWarning("corrected illegal number of steps");
            numSteps = 256;
            p_numSteps->setValue(256);
        }

        // add Species if user gives no own attrib
        annotation = p_annotation->getValue();
        if (species && ((!*annotation) || strcmp(annotation, "Colors") == 0))
            annotation = species;

        // create the colormap: interpolate to selected number of steps
        actMap = interpolateColormap(numSteps, alphaMult);
    }

    const char *outName;
    coDistributedObject *outObj;

    // ---- Create the colors

    if (data && p_color)
    {
        outName = p_color->getObjName();
        outObj = createColors(base, alpha, actMap, min, max, numSteps, outName, RGBA);
        p_color->setCurrentObject(outObj);
        if (colorMapIn)
            outObj->copyAllAttributes(colorMapIn);
        else
            addColormapAttrib(outName, min, max, outObj, annotation, actMap, numSteps);
    }

    // ---- Create the color Texture object

    if (p_texture)
    {
        outName = p_texture->getObjName();
        outObj = createColors(base, alpha, actMap, min, max, numSteps, outName, TEX);
        p_texture->setCurrentObject(outObj);
        if (colorMapIn)
            outObj->copyAllAttributes(colorMapIn);
        else
            addColormapAttrib(outName, min, max, outObj, annotation, actMap, numSteps);
    }

    // ---- Create the Colormap output object
    if (p_cmapOut)
    {
        outName = p_cmapOut->getObjName();
        outObj = new coDoColormap(outName, numSteps, min, max, (float *)actMap, annotation);
        p_cmapOut->setCurrentObject(outObj);
        if (colorMapIn)
            outObj->copyAllAttributes(colorMapIn);
        else
            addColormapAttrib(outName, min, max, outObj, annotation, actMap, numSteps);
    }

    // delete the map ONLY if not read from a distributed object
    if (!colorMapIn)
        delete[] actMap;

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Interpolate a cmap to a given number of steps
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Colors::FlColor *Colors::interpolateColormap(int numSteps, float alphaMult)
{

    FlColor *actMap = new FlColor[numSteps];
    double delta = 1.0 / (numSteps - 1) * (numColors - 1);
    double x;
    int i;

    delta = 1.0 / (numSteps - 1);
    int idx = 0;
    for (i = 0; i < numSteps - 1; i++)
    {
        x = i * delta;
        while (d_cmap[(idx + 1) * 5 + 4] <= x)
        {
            idx++;
            if (idx > numColors - 2)
            {
                idx = numColors - 2;
                break;
            }
        }

        double d = (x - d_cmap[idx * 5 + 4]) / (d_cmap[(idx + 1) * 5 + 4] - d_cmap[idx * 5 + 4]);
        actMap[i][0] = (float)((1 - d) * d_cmap[idx * 5] + d * d_cmap[(idx + 1) * 5]);
        actMap[i][1] = (float)((1 - d) * d_cmap[idx * 5 + 1] + d * d_cmap[(idx + 1) * 5 + 1]);
        actMap[i][2] = (float)((1 - d) * d_cmap[idx * 5 + 2] + d * d_cmap[(idx + 1) * 5 + 2]);
        actMap[i][3] = (float)((1 - d) * d_cmap[idx * 5 + 3] + d * d_cmap[(idx + 1) * 5 + 3]) * alphaMult;
        actMap[i][4] = -1;
    }
    actMap[numSteps - 1][0] = d_cmap[(numColors - 1) * 5];
    actMap[numSteps - 1][1] = d_cmap[(numColors - 1) * 5 + 1];
    actMap[numSteps - 1][2] = d_cmap[(numColors - 1) * 5 + 2];
    actMap[numSteps - 1][3] = d_cmap[(numColors - 1) * 5 + 3] * alphaMult;
    actMap[numSteps - 1][4] = -1;

    return actMap;
}

///////////////////////////////////////////////////////////////////////////
///////  Read input lines for different color mappings from covise.config
///////////////////////////////////////////////////////////////////////////

inline void valueParse(char *line, char *token[])
{
    int count = 0;
    char *tp = strtok(line, " ");

    while (tp)
    {
        token[count] = tp;
        tp = strtok(NULL, " ");
        count++;
    }
    token[count] = NULL;
}

inline void clamp(float &x)
{
    if (x > 1.0)
        x = 1.0;
    if (x < 0.0)
        x = 0.0;
}

inline void readRGB(char *items[], int np, Colors::FlColor &color)
{
    int i = 0;
    for (int k = 0; k < np * 3; k = k + 3)
    {
        color[i] = atof(items[k]);
        color[i + 1] = atof(items[k + 1]);
        color[i + 2] = atof(items[k + 2]);
        color[i + 3] = 1.0;
        color[i + 4] = -1.0;
        clamp(color[0]);
        clamp(color[1]);
        clamp(color[2]);
        i = i + 5;
    }
}

inline void readRGBA(char *items[], int np, Colors::FlColor &color)
{
    int i = 0;
    for (int k = 0; k < np * 4; k = k + 4)
    {
        color[i] = atof(items[k]);
        color[i + 1] = atof(items[k + 1]);
        color[i + 2] = atof(items[k + 2]);
        color[i + 3] = atof(items[k + 3]);
        color[i + 4] = -1.0;
        clamp(color[0]);
        clamp(color[1]);
        clamp(color[2]);
        clamp(color[3]);
        i = i + 5;
    }
}

inline void readXRGB(char *items[], int np, Colors::FlColor &color)
{
    int i = 0;
    for (int k = 0; k < np * 4; k = k + 4)
    {
        color[i + 4] = atof(items[k]);
        color[i] = atof(items[k + 1]);
        color[i + 1] = atof(items[k + 2]);
        color[i + 2] = atof(items[k + 3]);
        color[i + 3] = 1.0;
        clamp(color[0]);
        clamp(color[1]);
        clamp(color[2]);
        i = i + 5;
    }
}

inline void readRGBX(char *items[], int np, Colors::FlColor &color)
{
    int i = 0;
    for (int k = 0; k < np * 4; k = k + 4)
    {
        color[i] = atof(items[k]);
        color[i + 1] = atof(items[k + 1]);
        color[i + 2] = atof(items[k + 2]);
        color[i + 3] = 1.0;
        color[i + 4] = atof(items[k + 2]);
        clamp(color[0]);
        clamp(color[1]);
        clamp(color[2]);
        i = i + 5;
    }
}

inline void readXRGBA(char *items[], int np, Colors::FlColor &color)
{
    int i = 0;
    for (int k = 0; k < np * 5; k = k + 5)
    {
        color[i + 4] = atof(items[k]);
        color[i] = atof(items[k + 1]);
        color[i + 1] = atof(items[k + 2]);
        color[i + 2] = atof(items[k + 3]);
        color[i + 3] = atof(items[k + 4]);
        clamp(color[0]);
        clamp(color[1]);
        clamp(color[2]);
        clamp(color[3]);
        clamp(color[4]);
        i = i + 5;
    }
}

inline void readRGBAX(char *items[], int np, Colors::FlColor &color)
{
    int i = 0;
    for (int k = 0; k < np * 5; k = k + 5)
    {
        color[i] = atof(items[k]);
        color[i + 1] = atof(items[k + 1]);
        color[i + 2] = atof(items[k + 2]);
        color[i + 3] = atof(items[k + 3]);
        color[i + 4] = atof(items[k + 4]);
        clamp(color[0]);
        clamp(color[1]);
        clamp(color[2]);
        clamp(color[3]);
        i = i + 5;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Colors::updateMinMax(float min, float max)
{
    float values[2] = { min, max };
    p_minmax->setValue(2, values);
    p_colorMap->setMinMax(min, max);
}

// #######################################################################
// #######################################################################
// #######################################################################
// #######################################################################
// ####                                                               ####
// ####                  Spike removal routines                       ####
// ####                                                               ####
// #######################################################################
// #######################################################################
// #######################################################################
// #######################################################################

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Adjust min/max to next existing data value between minV / maxV
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Colors::getMinMax(const recObj &base, float &min, float &max,
                       float minV, float maxV)
{
    if (base.obj == NULL)
        return; // we don't care for empty ones

    int i;

    // recurse
    if (base.subObj)
    {
        for (i = 0; i < base.numElem; i++)
            getMinMax(base.subObj[i], min, max, minV, maxV);
    }

    // run over own object
    else
    {
        for (i = 0; i < base.numElem; i++)
        {
            float actVal = base.data[i];
            if (actVal >= minV && actVal < min)
                min = actVal;
            if (actVal <= maxV && actVal > max)
                max = actVal;
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Count all data values between min and max into bins
// ++++
/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Colors::countBins(const recObj &base, float min, float max,
                       int numBins, int *bins)
{
    if (base.obj == NULL)
        return; // we don't care for empty ones

    int i;

    // recurse
    if (base.subObj)
    {
        for (i = 0; i < base.numElem; i++)
            countBins(base.subObj[i], min, max, numBins, bins);
    }

    // run over own object
    else
    {
        float delta = max - min;

        for (i = 0; i < base.numElem; i++)
        {
            float actData = base.data[i];

            // do not care about FLT_MAX elements in min/max calc.
            if (actData < NoDataColorPercent * FLT_MAX)
            {
                int binNo = (int)((actData - min) / delta * (numBins - 0.00000001));

                if (binNo >= 0 && binNo < numBins)
                    ++bins[binNo];
            }
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Adaptive algorithm
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Colors::removeSpikesAdaptive(const recObj &base, float &min, float &max)
{
    int i;
    int numBins = numBinsAdaptive;

    int *bin = new int[numBins];

    // recursively do algoritm until either concergent of cutoff-Limits (top/bot)
    // are reaches
    int numValuesOverall = -1;
    int cutoffLeftTop = 1;
    int cutoffLeftBot = 1;
    bool foundSpikes;

    do
    {
        foundSpikes = false;
        // count in bins
        for (i = 0; i < numBins; i++)
            bin[i] = 0;
        countBins(base, min, max, numBins, bin);
        int numValues = 0;
        for (i = 0; i < numBins; i++)
            numValues += bin[i];

        // 1st run : set global values
        if (numValuesOverall < 0)
        {
            numValuesOverall = numValues;
            cutoffLeftBot = (int)(numValuesOverall * p_spikeBot->getValue());
            cutoffLeftTop = (int)(numValuesOverall * p_spikeTop->getValue());

            if (cutoffLeftTop < 1)
                cutoffLeftTop = 1;
            if (cutoffLeftBot < 1)
                cutoffLeftBot = 1;
        }

        // -------------- TOP CLIP RUN --------------
        // Start at top, find 1st empty bin, count members in bin
        int topBin = numBins - 1;
        int numTopClip = 0;
        while (topBin > 0
               && bin[topBin] != 0
               && numTopClip + bin[topBin] <= cutoffLeftTop)
        {
            numTopClip += bin[topBin];
            --topBin;
        }

        // if we found an empty box before the # of spike values ran out, this was a spike
        if (bin[topBin] == 0)
        {
            // continue all empty bins
            while (topBin > 0 && bin[topBin] == 0)
            {
                --topBin;
            }
        }
        else
            topBin = numBins - 1;
        // ==> topBin now points to non-empty bin at top

        // -------------- BOT CLIP RUN --------------
        // Start at bot, find 1st empty bin, count members in bin
        int botBin = 0;
        int numBotClip = 0;
        while (botBin < numBins
               && bin[botBin] != 0
               && numBotClip + bin[botBin] <= cutoffLeftBot)
        {
            numBotClip += bin[botBin];
            ++botBin;
        }

        // if we found an empty box before the # of spike values ran out, this was a spike
        if (bin[botBin] == 0)
        {
            // continue all empty bins
            while (botBin < numBins && bin[botBin] == 0)
                ++botBin;
        }
        else
            botBin = 0;
        // ==> botBin now points to non-empty bin at top

        float newMin = min;
        float newMax = max;

        // -------------- BOT CLIP --------------
        if ((botBin != 0) // we found any top clip
            && (botBin < numBins) // we cannot exeed top
            && (topBin >= botBin) // at least one box is left
            && (cutoffLeftBot >= numBotClip)) // don't clip away more
        {
            // set new min
            newMin = min + (max - min) / numBins * botBin;
            cutoffLeftBot -= numBotClip; // book off spikes
            foundSpikes = true;
        }

        // -------------- TOP CLIP --------------
        if ((topBin < numBins - 1) // we found any top clip
            && (topBin >= 0) // we cannot hit ground
            && (topBin >= botBin) // at least one box is left
            && (cutoffLeftTop >= numTopClip)) // don't clip away more
        {
            // set new max
            newMax = min + (max - min) / numBins * (topBin + 1);
            cutoffLeftTop -= numTopClip; // book off spikes
            foundSpikes = true;
        }

        max = newMax;
        min = newMin;
    } while (foundSpikes);

    float newMin = FLT_MAX;
    float newMax = -FLT_MAX;

    // at last: use higest/lowest real data values instead of bin boundaries
    getMinMax(base, newMin, newMax, min, max);

    min = newMin;
    max = newMax;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  remove part of value range
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Colors::removeSpikesInterval(const recObj &base, float &min, float &max)
{
    (void)base;
    float oMin = min;
    float oMax = max;
    float delta = max - min;

    min = oMin + p_spikeBot->getValue() * delta;
    max = oMax - p_spikeTop->getValue() * delta;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Remove top/bot elements fraction
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Colors::removeSpikesElements(const recObj &base, float &min, float &max)
{
    int i;
    int numBins = numBinsElements;

    int *bin = new int[numBins];

    // recursively do algoritm until either concergent of cutoff-Limits (top/bot)
    // are reaches
    int numValuesOverall = -1;
    int cutoffLeftTop = 1;
    int cutoffLeftBot = 1;
    bool foundSpikes;

    do
    {
        foundSpikes = false;
        // count in bins
        for (i = 0; i < numBins; i++)
            bin[i] = 0;
        countBins(base, min, max, numBins, bin);
        int numValues = 0;
        for (i = 0; i < numBins; i++)
            numValues += bin[i];

        // 1st run : set global values
        if (numValuesOverall < 0)
        {
            numValuesOverall = numValues;
            cutoffLeftBot = (int)(numValuesOverall * p_spikeBot->getValue());
            cutoffLeftTop = (int)(numValuesOverall * p_spikeTop->getValue());

            if (cutoffLeftTop < 1)
                cutoffLeftTop = 1;
            if (cutoffLeftBot < 1)
                cutoffLeftBot = 1;
        }

        // -------------- TOP CLIP RUN --------------
        // Start at top, find 1st empty bin, count members in bin
        int topBin = numBins - 1;
        int numTopClip = 0;
        while (topBin > 0
               && numTopClip + bin[topBin] <= cutoffLeftTop)
        {
            numTopClip += bin[topBin];
            --topBin;
        }

        // ==> topBin now points to non-empty bin at top

        // -------------- BOT CLIP RUN --------------
        // Start at bot, find 1st empty bin, count members in bin
        int botBin = 0;
        int numBotClip = 0;
        while (botBin < numBins
               && numBotClip + bin[botBin] <= cutoffLeftBot)
        {
            numBotClip += bin[botBin];
            ++botBin;
        }

        float newMin = min;
        float newMax = max;

        // -------------- BOT CLIP --------------
        if ((botBin != 0) // we found any top clip
            && (botBin < numBins) // we cannot exeed top
            && (topBin >= botBin) // at least one box is left
            && (cutoffLeftBot >= numBotClip)) // don't clip away more
        {
            // set new min
            newMin = min + (max - min) / numBins * botBin;
            cutoffLeftBot -= numBotClip; // book off spikes
            foundSpikes = true;
        }

        // -------------- TOP CLIP --------------
        if ((topBin < numBins - 1) // we found any top clip
            && (topBin >= 0) // we cannot hit ground
            && (topBin >= botBin) // at least one box is left
            && (cutoffLeftTop >= numTopClip)) // don't clip away more
        {
            // set new max
            newMax = min + (max - min) / numBins * (topBin + 1);
            cutoffLeftTop -= numTopClip; // book off spikes
            foundSpikes = true;
        }

        max = newMax;
        min = newMin;
    } while (foundSpikes);

    float newMin = FLT_MAX;
    float newMax = -FLT_MAX;

    // at last: use higest/lowest real data values instead of bin boundaries
    getMinMax(base, newMin, newMax, min, max);

    min = newMin;
    max = newMax;
}

MODULE_MAIN(Mapper, Colors)
