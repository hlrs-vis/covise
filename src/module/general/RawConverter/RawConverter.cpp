/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                           (C)2008 HLRS **
**                                                                        **
** Description: Reduce pitcure values 16bit integer to float              **
**                                                                        **
**                                                                        **
** Name:        RawConverter                                              **
** Category:    Tools                                                     **
**                                                                        **
**                                                                        **
\****************************************************************************/

#include "RawConverter.h"
#include <do/coDoData.h>

/*! \brief constructor
*
* create In/Output Ports and module parameters
*/
RawConverter::RawConverter(int argc, char **argv)
    : coModule(argc, argv, "convert picture 16bit --> float")
{

    poIVolume = addInputPort("InputData", "Float|Int|Byte", "Scalar volume input data");
    poOVolume = addOutputPort("OutputData", "Float|Byte", "Scalar volume output data");

    //	// output port  polygons
    //	p_polyOut = addOutputPort("polygons","Polygons","polygons which form the cubes");

    // output port Lines
    //	p_polyOut = addOutputPort("Lines","Lines","Lines for ... ");

    p_Histo_In = addOutputPort("HistoInputValues", "Lines", "Lines for ... ");
    p_Histo_Out = addOutputPort("HistoOutputValues", "Lines", "Lines for ... ");

    // Create parameters:

    pboAutoSize = addBooleanParam("AutoSize", "On: Careless Programm ON");
    pboAutoSize->setValue(true);

    pboLOGScale = addBooleanParam("LOGScale", "On: Create log Scale");
    pboLOGScale->setValue(false);

    //	piSequenceBegin = addIntSliderParam("SequenceBegin", "First gray value in sequence");
    //	piSequenceBegin->setValue(0,65535,10);
    piSequenceBegin = addInt32Param("SequenceBegin", "First gray value in sequence");
    piSequenceBegin->setValue(0); // ,65535,10);

    piSequenceEnd = addInt32Param("SequenceEnd", "Last gray value in sequence");
    piSequenceEnd->setValue(0); //  63000, 63000);

    pfsResultsBegin = addFloatParam("ResultsBegin", "First color value in sequence");
    pfsResultsBegin->setValue(0.0); // ,1.0,0.0);

    pfsResultsEnd = addFloatParam("ResultsEnd", "Last color value in sequence");
    pfsResultsEnd->setValue(0.0); // ,1.0,1.0);

    pboByteOrFloats = addBooleanParam("ByteOrFloats", "true: send Byte;  false: send Floats");
    pboByteOrFloats->setValue(false);

    p_cutoff = addFloatSliderParam("cutoff", "Set the first Grayvalues to Black");
    p_cutoff->setValue(0.0, 1.0, 0.0);

    p_pre_cutoff = addFloatSliderParam("pre_cutoff", "Set the first Grayvalues to Black");
    p_pre_cutoff->setValue(0.0, 100.0, 0.0);

    p_logOffset = addFloatSliderParam("logOffset", "Offset for the log-Methode 1..100");
    p_logOffset->setValue(1.0, 100.0, 10.0);
}

RawConverter::~RawConverter()
{
}

int RawConverter::compute(const char * /* port */)
{
    bool Auto = bool(pboAutoSize->getValue());
    bool LOGScale = bool(pboLOGScale->getValue());
    int SB = int(piSequenceBegin->getValue());
    int SE = int(piSequenceEnd->getValue());
    float RB = float(pfsResultsBegin->getValue());
    float RE = float(pfsResultsEnd->getValue());
    bool BoF = bool(pboByteOrFloats->getValue());
    float cutoff = float(p_cutoff->getValue());
    float pre_cutoff = float(p_pre_cutoff->getValue());
    float logOffset = float(p_logOffset->getValue());

    float numColors, range_, rangef, result, param;
    bool getfloats;
    int intdummy[] = { 0 };
    //	int floatsdummy[] = {0.0};

    if (Auto)
        MakeAutoScale();

    range_ = 1.0f / (SE - SB);
    rangef = RE - RB;

    coDoByte *outObj = NULL;
    poOVolume->setCurrentObject(outObj);
    const coDistributedObject *inputObject = poIVolume->getCurrentObject();
    if (const coDoFloat *floatDataObject = dynamic_cast<const coDoFloat *>(inputObject))
    {
    }
    else if (const coDoInt *intDataObject = dynamic_cast<const coDoInt *>(inputObject))
    {
        int *dataValues;
        intDataObject->getAddress(&dataValues);
        int numValues = intDataObject->getNumPoints();

        if (BoF) // write Byte
        {
            unsigned char *outDataValues;
            coDoByte *outDataObject = new coDoByte(poOVolume->getNewObjectInfo(), numValues);
            outDataObject->getAddress(&outDataValues);

            //    -------------------------  COPY PASTE Beginn        This Part can be usesd exactly later, but the value  "outDataValues"  have a other type  !!
            numColors = 256 * 256;
            getfloats = false;
            //			MakeHistogram(  p_Histo_In, getfloats, numColors, numValues, dataValues, floatsdummy);

            if (LOGScale)
            {
                float lokscale;
                lokscale = log(logOffset);
                for (int i = 0; i < numValues; i++)
                {
                    param = float(dataValues[i] - int(pre_cutoff));
                    if (param < 0)
                        param = 0;
                    param = param + logOffset;
                    if (param <= 0)
                    {
                        sendError("log(val<0)not possibel");
                        param = -param;
                    }
                    result = ((log(param)) - lokscale);
                    outDataValues[i] = (unsigned char)(result * 0.2f);
                }
                if (Auto)
                {
                    float min = 256. * 256.;
                    float max = -min;
                    float range;
                    for (int i = 0; i < numValues; i++)
                    {
                        if (outDataValues[i] > max)
                            max = outDataValues[i];
                        if (outDataValues[i] < min)
                            min = outDataValues[i];
                    }
                    range = 1.0f / (max - min);
                    for (int i = 0; i < numValues; i++)
                    {
                        outDataValues[i] = (unsigned char)((outDataValues[i] - min) * range);
                        if (outDataValues[i] < cutoff)
                            outDataValues[i] = 0;
                    }
                }
            }
            else
            {
                for (int i = 0; i < numValues; i++)
                {
                    param = float(dataValues[i] - int(pre_cutoff));
                    if (param < 0)
                        param = 0;
                    outDataValues[i] = (unsigned char)(RB + ((param - SB) * range_ * rangef));
                    if (outDataValues[i] < cutoff)
                        outDataValues[i] = 0;

                    //if (outDataValues[i] < RB) 	   outDataValues[i] = RB ;
                    //if (outDataValues[i] > RE)       outDataValues[i] = RE ;
                }
            }
            numColors = 1.0;
            getfloats = true;
            //    -------------------------  COPY PASTE Ende
            // geht noch nicht 			MakeHistogram(  p_Histo_Out, getfloats, numColors, numValues,intdummy, outDataValues);
        }
        else // write Float
        {
            float *outDataValues;
            coDoFloat *outDataObject = new coDoFloat(poOVolume->getNewObjectInfo(), numValues);
            outDataObject->getAddress(&outDataValues);
            //    -------------------------  COPY PASTE Beginn   This Part can be usesd exactly later, but the value  "outDataValues"  have a other type  !!
            numColors = 256 * 256;
            getfloats = false;
            // ---- Noch nicht Möglich        MakeHistogram(  p_Histo_In, getfloats, numColors, numValues, dataValues, floatsdummy);

            if (LOGScale)
            {
                float lokscale;
                lokscale = log(logOffset);
                for (int i = 0; i < numValues; i++)
                {
                    param = float(dataValues[i] - int(pre_cutoff));
                    if (param < 0)
                        param = 0;
                    param = param + logOffset;
                    if (param <= 0)
                    {
                        sendError("log(val<0)not possibel");
                        param = -param;
                    }
                    result = ((log(param)) - lokscale);
                    outDataValues[i] = float(result * 0.2);
                }
                if (Auto)
                {
                    float min = 256. * 256.;
                    float max = -min;
                    float range;
                    for (int i = 0; i < numValues; i++)
                    {
                        if (outDataValues[i] > max)
                            max = outDataValues[i];
                        if (outDataValues[i] < min)
                            min = outDataValues[i];
                    }
                    range = 1.0f / (max - min);
                    for (int i = 0; i < numValues; i++)
                    {
                        outDataValues[i] = ((outDataValues[i] - min) * range);
                        if (outDataValues[i] < cutoff)
                            outDataValues[i] = 0.0;
                    }
                }
            }
            else
            {
                for (int i = 0; i < numValues; i++)
                {
                    param = float(dataValues[i] - int(pre_cutoff));
                    if (param < 0)
                        param = 0;
                    outDataValues[i] = float(RB + ((param - SB) * range_ * rangef));
                    if (outDataValues[i] < cutoff)
                        outDataValues[i] = 0.0;
                    //if (outDataValues[i] < RB) 	   outDataValues[i] = RB ;
                    //if (outDataValues[i] > RE)       outDataValues[i] = RE ;
                }
            }
            numColors = 1.0;
            getfloats = true;
            //    -------------------------  COPY PASTE Ende
            MakeHistogram(p_Histo_Out, getfloats, numColors, numValues, intdummy, outDataValues);
        }
    }
    else
    {
        sendError("unknown datatype at input port poIVolume");
    }

    return CONTINUE_PIPELINE;
}

void RawConverter::MakeHistogram(coOutputPort *myport, bool getfloats, float numColors, int numValues, int *intDataValues, float *floatDataValues)
{
    const char *lineObjName;
    coDoLines *lineObj;
    int polygonList[1] = { 0 };
    int const maxPKT = 10000; // besser wäre //MAX_GRAY_VALUES;
    float xCoords[maxPKT], yCoords[maxPKT], zCoords[maxPKT];
    int vertexList[maxPKT] = { 0 };

    float indexmanager = (float(maxPKT) / float(numColors));
    for (int j = 0; j < maxPKT; j++)
        yCoords[j] = 0;
    if (getfloats)
        for (int i = 0; i < numValues; i++)
            yCoords[int(float(floatDataValues[i]) * indexmanager + 1.0)]++;
    else
        for (int i = 0; i < numValues; i++)
            yCoords[int(float(intDataValues[i]) * indexmanager + 1.0)]++;

    // Ausgabe
    for (int i = 0; i < maxPKT; i++)
    {
        xCoords[i] = float(i) * 1.0f;
        yCoords[i] = log(yCoords[i] + 1.0f); //                 //  PUNKTKOORDINATEN
        zCoords[i] = 0.0;
    }
    for (int i = 0; i < maxPKT; i++)
        vertexList[i] = i;

    // get the data object name from the controller
    lineObjName = myport->getObjName();

    // create the polygons data object
    lineObj = new coDoLines(lineObjName, maxPKT, xCoords, yCoords, zCoords, maxPKT, vertexList, 1, polygonList); //erzeugt neues Polygon mit angegebenen Namen und Werten
    myport->setCurrentObject(lineObj); // legt  lineObj auf dem Outputport

    return;
}

void RawConverter::MakeAutoScale()
{
    const coDistributedObject *inputObject = poIVolume->getCurrentObject();
    if (const coDoFloat *floatDataObject = dynamic_cast<const coDoFloat *>(inputObject))
    {
    }
    else if (const coDoInt *intDataObject = dynamic_cast<const coDoInt *>(inputObject))
    {
        int *dataValues;
        intDataObject->getAddress(&dataValues);
        int numValues = intDataObject->getNumPoints();
        int min = 256 * 256;
        int max = -min;
        for (int i = 0; i < numValues; i++)
        {
            if (dataValues[i] > max)
                max = dataValues[i];
            if (dataValues[i] < min)
                min = dataValues[i];
        }
        piSequenceBegin->setValue(min); // ,65535,10);
        piSequenceEnd->setValue(max); //  63000, 63000);
        pfsResultsBegin->setValue(0.0); // ,1.0,0.0);
        pfsResultsEnd->setValue(1.0); // ,1.0,1.0);
    }
}

MODULE_MAIN(Filter, RawConverter) // Main !!
