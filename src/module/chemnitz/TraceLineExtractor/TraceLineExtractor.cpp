/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)1999 RUS   **
**                                                                          **
** Description: Simple Geometry Generation Module                           **
**              supports interaction from a COVER plugin                    **
**              feedback style is later than COVISE 4.5.2                   **
**                                                                          **
** Name:        TraceLineExtractor                                                        **
** Category:    examples                                                    **
**                                                                          **
** Author: D. Rainer                                                        **
**                                                                          **
** History:                                                                 **
** September-99                                                             **
**                                                                          **
**                                                                          **
\****************************************************************************/
//#define OLD_COVISE

#ifdef OLD_COVISE
#define addAttribute setAttribute
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "TraceLineExtractor.h"
#include "FortranData.h"
#include <api/coFeedback.h>
#include <do/coDoIntArr.h>
#include <do/coDoText.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

#ifndef _MSC_VER
#define sscanf_s sscanf
#define sprintf_s snprintf
#define fprintf_s fprintf
#define strtok_s(a, b, c) strtok(a, b)
#define strcpy_s(a, b, c) strncpy(a, c, b)
#define strncpy_s(a, b, c, d) strncpy(a, c, min(d, b))
#define stricmp strcasecmp
#define strnicmp strncasecmp
#define _Myptr base()
#endif

//#define WRITE_TO_FILE_ENABLE

TraceLineExtractor::TraceLineExtractor(int argc, char *argv[])
    : coModule(argc, argv, "Traceline extractor")
{
    nodes = addInputPort("Nodes", "Dataset", "Dataset15");
    traceLineIndices = addInputPort("TracelineIndices", "Dataset", "Dataset82");
    DOFs = addInputPort("FunctionAtNodalDOFs", "Dataset", "Dataset58");
    datasetFileheader = addInputPort("DatasetFileHeader", "Dataset", "Dataset151");
    datasetUnitDesc = addInputPort("DatasetUnitDescription", "Dataset", "Dataset164");

    DOFs->setRequired(0);
    nodes->setRequired(0);
    traceLineIndices->setRequired(0);
    datasetFileheader->setRequired(0);
    datasetUnitDesc->setRequired(0);

    traceLine = addOutputPort("Geometry", "UnstructuredGrid", "The resulting geometry");
    traceLineData = addOutputPort("Data", "Vec3", "Data on nodes");

    choice_traceLineID = addChoiceParam("TracelineID", "Choose a traceline");
    traceLineSelection = addStringParam("TracelineNumbers", "Specify the tracelines, you want to show, beginning at 0.");
    traceLineSelection->setValue("2");

    choice_traceLineID->hide();

    choice_functionType = addChoiceParam("Function", "Select the function, given by dataset58, which should be shown");
    choice_coordDir = addChoiceParam("CoordinateDirection", "Select a specific coordinate direction");
    const char *coordDirStrings[] = { "all directions", "X direction", "Y direction", "Z direction" };
    choice_coordDir->setValue(4, coordDirStrings, 0);

    dataPosSlider = addIntSliderParam("DataPosition", "Adjust the slider to the data postion, you want to show");
    dataPosSlider->setMin(0);
    dataPosSlider->setMax(0);

    animationStepsSlider = addIntSliderParam("AnimationSteps", "Choice the smoothness of the animation");
    animationStepsSlider->setMin(1);
    animationStepsSlider->setMax(200);
    animationStepsSlider->setValue(100);

    multiplierSlider = addFloatSliderParam("Multiplier", "Scale the output values");
    multiplierSlider->setMin(1000);
    multiplierSlider->setMax(100000);
    multiplierSlider->setValue(10000);

#ifdef WRITE_TO_FILE_ENABLE
    choice_writeToFile = addBooleanParam("WriteToFile", "Check this, if the current function selection should be written into a file");
    file_writeToFileName = addFileBrowserParam("Filename", "Specify the name of the output file");
    file_writeToFileName->setValue("D:\\Dokumente und Einstellungen\\afun\\Desktop\\", "*.unv");
#endif
    choice_useMapFile = addBooleanParam("UseMapFile", "Use a .map file, for mapping node numbers");
    choice_useMapFile->setValue(false);

    file_mapFile = addFileBrowserParam("MapFile", "Specify a map file, you want to use");
    file_mapFile->setValue("D:/Modalanalyse/HEC500/HEC500 Geom.map", "*.map");

    coordinates = NULL;
    lineList = NULL;
    vertList = NULL;

    cornerList = NULL;
    typeList = NULL;
    elementList = NULL;

    updateNodesInput = true;
    updateDOFsInput = true;
    updateTraceLineInput = true;

    use_mapFile = choice_useMapFile->getValue();
}

void TraceLineExtractor::postInst()
{
    choice_traceLineID->disable();
    traceLineSelection->disable();

    choice_functionType->disable();
    choice_coordDir->disable();

    dataPosSlider->disable();

    animationStepsSlider->disable();

    multiplierSlider->disable();

    choice_useMapFile->disable();

    file_mapFile->disable();

#ifdef WRITE_TO_FILE_ENABLE
    choice_writeToFile->disable();
    file_writeToFileName->disable();
#endif

    choice_traceIDText = NULL;
    numChoice_traceID = 0;

    choice_functionTypeText = NULL;
    numChoice_functionTypeText = 0;
}

void TraceLineExtractor::UnpackDataset(coDoSet *set)
{

    if (stricmp(set->getAttribute("Type"), "Dataset15") == 0)
    {
        coDoIntArr *nodes_labels = (coDoIntArr *)set->getElement(0);
        //coDoIntArr *nodes_defcosysnums	= (coDoIntArr*)set->getElement(1);
        //coDoIntArr *nodes_discosysnums	= (coDoIntArr*)set->getElement(2);
        //coDoIntArr *nodes_colors		= (coDoIntArr*)set->getElement(3);
        coDoPoints *nodes_points = (coDoPoints *)set->getElement(4);
        float *x;
        float *y;
        float *z;

        numCoordinates = nodes_points->getNumPoints();

        coordinates = new float *[3];
        coordinates[0] = new float[numCoordinates];
        coordinates[1] = new float[numCoordinates];
        coordinates[2] = new float[numCoordinates];

        int substPos = 0;

        nodes_points->getAddresses(&x, &y, &z);

        for (int i = 0; i < nodes_points->getNumPoints(); i++)
        {
            /*nodeMap[nodes_labels->getAddress()[i]] = new sPoint(nodes_defcosysnums->getAddress()[i],
				nodes_discosysnums->getAddress()[i],
				nodes_colors->getAddress()[i],
				x[i], y[i], z[i]);	
*/
            // create a node label substitution map, eg. point 10002 in the file has position 0 in the coordinate array
            if (substituteMap.find(nodes_labels->getAddress()[i]) == substituteMap.end())
            {
                coordinates[0][substPos] = x[i];
                coordinates[1][substPos] = y[i];
                coordinates[2][substPos] = z[i];
                substituteMap[nodes_labels->getAddress()[i]] = substPos;
                substPos++;
            }
        }
    }
    else if (stricmp(set->getAttribute("Type"), "Dataset58") == 0)
    {
        int numSetElements = set->getNumElements(); //number of datasets 58
        coDoSet *tmpDoSet = NULL;
        coDoIntArr *tmpIntArr = NULL;
        //map<short, char*>::iterator	 fTNit;										//functionTypeName iterator
        int funcType = 0;

        if (((coDoIntArr *)((coDoSet *)((coDoSet *)set->getElement(0))->getElement(6))->getElement(0))->getAddress()[2] == 0) //get the abscissa spacing value
        {
            coModule::sendInfo("Slider max = %d", (int)(((coDoIntArr *)((coDoSet *)((coDoSet *)set->getElement(0))->getElement(6))->getElement(0))->getAddress()[1] / 2));
            dataPosSlider->setMax(((coDoIntArr *)((coDoSet *)((coDoSet *)set->getElement(0))->getElement(6))->getElement(0))->getAddress()[1] / 2); //get the number of data pairs
        }
        else
        {
            dataPosSlider->setMax(((coDoIntArr *)((coDoSet *)((coDoSet *)set->getElement(0))->getElement(6))->getElement(0))->getAddress()[1]); //get the number of data values
            coModule::sendInfo("Slider max = %d", (int)(((coDoIntArr *)((coDoSet *)((coDoSet *)set->getElement(0))->getElement(6))->getElement(0))->getAddress()[1]));
        }

        for (int i = 0; i < numSetElements; i++)
        {
            tmpDoSet = (coDoSet *)((coDoSet *)set->getElement(i))->getElement(5); //get "record6"
            tmpIntArr = (coDoIntArr *)tmpDoSet->getElement(0);
            funcType = tmpIntArr->getAddress()[0];

            functionTypeSets[funcType].push_back(i);

            if (functionTypeNamesMap.find(funcType) == functionTypeNamesMap.end())
            {
                functionTypeNamesMap[funcType] = (char *)functionNames[funcType];
            }
        }
    }
    else if (stricmp(set->getAttribute("Type"), "Dataset82") == 0)
    {
        coDoIntArr *traceNodeArr = NULL;
        //coDoIntArr	*intArr			= NULL;
        coDoText *idLine = NULL;

        coDoSet *actSet = NULL;
        sTraceLine *tmp = NULL;

        for (int i = 0; i < set->getNumElements(); i++)
        {
            actSet = (coDoSet *)set->getElement(i);

            traceNodeArr = (coDoIntArr *)actSet->getElement(0);
            //intArr			= (coDoIntArr*)actSet->getElement(1);
            idLine = (coDoText *)actSet->getElement(2);

            tmp = new sTraceLine;
            char *idPtr = NULL;
            idLine->getAddress(&idPtr);
            strcpy_s(tmp->name, 80 * sizeof(char), idPtr);
            tmp->indices = NULL;
            tmp->numLines = 0;
            tmp->numVertices = traceNodeArr->getSize() - 1;

            tmp->indices = new int[tmp->numVertices];

            int temp = 0;

            int offset1 = -1;

            if (traceNodeArr->getAddress()[0] == 0)
                offset1 = 0;

            for (unsigned int v = 0; v < tmp->numVertices; v++)
            {

                temp = traceNodeArr->getAddress()[v];
                tmp->indices[v] = temp;

                if (traceNodeArr->getAddress()[v] == 0 && v > 0)
                {
                    int t = v - offset1 - 2;
                    if (t == 0) //if just one number is given, a traceline should be drawed anyway
                    {
                        t = 1;
                        offset1 = v;
                        continue;
                    }
                    else if (t < 0) // if two zeroes are behind each other t is less than 0
                    {
                        offset1 = v;
                        continue;
                    }

                    tmp->numLines += t;
                    offset1 = v;
                    temp = 0;
                }
            }

            if (temp != 0) //dataset82 doesn't end up with a 0
            {
                tmp->numLines += tmp->numVertices - offset1 - 2;
            }

            traceLineMap[i] = tmp;
        }
    }
    else if (stricmp(set->getAttribute("Type"), "Dataset2411") == 0)
    {
        coDoIntArr *nodes_labels = (coDoIntArr *)set->getElement(0);
        //coDoIntArr *nodes_expcosysnums	= (coDoIntArr*)set->getElement(1);
        //coDoIntArr *nodes_discosysnums	= (coDoIntArr*)set->getElement(2);
        //coDoIntArr *nodes_colors		= (coDoIntArr*)set->getElement(3);
        coDoPoints *nodes_points = (coDoPoints *)set->getElement(4);
        float *x;
        float *y;
        float *z;

        numCoordinates = nodes_points->getNumPoints();

        coordinates = new float *[3];
        coordinates[0] = new float[numCoordinates];
        coordinates[1] = new float[numCoordinates];
        coordinates[2] = new float[numCoordinates];

        int substPos = 0;

        nodes_points->getAddresses(&x, &y, &z);

        for (int i = 0; i < nodes_points->getNumPoints(); i++)
        {
            /*			nodeMap[nodes_labels->getAddress()[i]] = new sPoint(nodes_expcosysnums->getAddress()[i],
				nodes_discosysnums->getAddress()[i],
				nodes_colors->getAddress()[i],
				x[i], y[i], z[i]);	
*/
            // create a node label substitution map, eg. point 10002 in the file has position 0 in the coordinate array
            if (substituteMap.find(nodes_labels->getAddress()[i]) == substituteMap.end())
            {
                coordinates[0][substPos] = x[i];
                coordinates[1][substPos] = y[i];
                coordinates[2][substPos] = z[i];
                substituteMap[nodes_labels->getAddress()[i]] = substPos;
                substPos++;
            }
        }
    }
    else
    {
        coModule::sendWarning("no compatible dataset found");
    }
}

list<unsigned int> TraceLineExtractor::getSelectedTracelines()
{
    list<unsigned int> tracelineList;
    char *buf = (char *)traceLineSelection->getValString();
    char delim1[] = ";,";
    char delim2[] = "-";
    char *token = NULL;
    char *token2 = NULL;
#ifdef WIN32
    char *nextToken = NULL;
#endif
    char *nextToken2 = NULL;

    token = strtok_s(buf, delim1, &nextToken);

    int pos1 = 0;
    int pos2 = 0;

    while (token != NULL)
    {
        if ((token2 = strtok_s(token, delim2, &nextToken2)) != 0) // geht so nicht unter linux TODO fix it
        {
            if (nextToken2 && strcmp(nextToken2, "") != 0)
            {
                pos1 = atoi(token2);
                pos2 = atoi(nextToken2);

                for (int i = pos1; i <= pos2; i++)
                    tracelineList.push_back(i);
            }
            else
            {
                tracelineList.push_back(atoi(token2));
            }
        }

        token = strtok_s(NULL, delim1, &nextToken);
    }

    return tracelineList;
}

int TraceLineExtractor::compute(const char *port)
{
    (void)port;

    const coDistributedObject *in_nodesPtr = nodes->getCurrentObject();
    const coDistributedObject *in_traceLineIndicesPtr = traceLineIndices->getCurrentObject();
    //const coDistributedObject *in_datasetFileheaderPtr = datasetFileheader->getCurrentObject();
    //const coDistributedObject *in_datasetUnitDescPtr = datasetUnitDesc->getCurrentObject();
    const coDistributedObject *in_DOFsPtr = DOFs->getCurrentObject();

    choice_traceLineID->setActive(0);

#ifdef WRITE_TO_FILE_ENABLE
    //write to file
    if (choice_writeToFile->getValue())
    {

        if (in_datasetFileheaderPtr && in_datasetFileheaderPtr->isType("SETELE") && stricmp(in_datasetFileheaderPtr->getAttribute("Type"), "dataset151") == 0 && in_datasetUnitDescPtr && in_datasetUnitDescPtr->isType("SETELE") && stricmp(in_datasetUnitDescPtr->getAttribute("Type"), "dataset164") == 0 && in_DOFsPtr && in_DOFsPtr->isType("SETELE") && stricmp(in_DOFsPtr->getAttribute("Type"), "dataset58") == 0)
        {

            if (choice_functionType->getNumChoices() <= 0 || stricmp(choice_functionType->getActLabel(), "none") == 0)
            {
                coModule::sendInfo("No function type choices are made. Maybe a pipeline reexecute solves this.");
                return FAIL;
            }

            map<short, char *>::iterator funcTypeNamesIt;
            short selectedFunc = -1;
            char *funcNameSel = (char *)choice_functionType->getActLabel();

            for (int i = 0; i < 5; i++)
            {
                if (funcNameSel[i] == 127)
                    funcNameSel[i] = 32;
            }

            //find the active selected function
            for (funcTypeNamesIt = functionTypeNamesMap.begin(); funcTypeNamesIt != functionTypeNamesMap.end(); funcTypeNamesIt++)
            {
                if (strnicmp(funcNameSel, funcTypeNamesIt->second, 5) == 0)
                {
                    selectedFunc = funcTypeNamesIt->first;
                    break;
                }
            }

            if (selectedFunc == -1)
            {
                coModule::sendError("Function name not found. Cancel writing...");
                return FAIL;
            }

            FILE *f = NULL;
            const char *fileName = file_writeToFileName->getValue();
            errno_t errnum = 0;

            if ((errnum = fopen_s(&f, fileName, "w")) != 0)
            {

                coModule::sendError("Can't open file to write in (Reason: %s)", strerror(errnum));
                return FAIL;
            }

            coModule::sendInfo("Write to file");

            FortranData::setFile(f);

            writeDatasetToFile(f, (coDoSet *)in_datasetFileheaderPtr);
            writeDatasetToFile(f, (coDoSet *)in_datasetUnitDescPtr);

            //map<short, list<unsigned int>>::iterator funcTypeSetsIt;

            list<unsigned int>::iterator listIt;

            for (listIt = functionTypeSets[selectedFunc].begin(); listIt != functionTypeSets[selectedFunc].end(); listIt++)
            {
                writeDatasetToFile(f, (const coDoSet *)((const coDoSet *)DOFs->getCurrentObject())->getElement(*listIt));
            }

            coModule::sendInfo("File written");

            fflush(f);
            fclose(f);
        }
        else
        {
            coModule::sendError("Dataset information do not match, or not all inputs set");
            return FAIL;
        }
    }
    //output data
    else
    {
#endif
        //IMPORTANT: add a file param to read and evaluate .map files
        //IMPORTANT: check runtime errors, which occures, when loading new files, and executing
        coDoUnstructuredGrid *grid = NULL;
        coDoSet *data = NULL;

        bool nodesInput = false;
        bool tracelineInput = false;
        bool dofsInput = false;

        //Process Nodes input
        coDoSet *nodesSetPtr = NULL;
        if (in_nodesPtr && in_nodesPtr->isType("SETELE"))
        {
            nodesSetPtr = (coDoSet *)in_nodesPtr;
            if (stricmp(nodesSetPtr->getAttribute("Type"), "dataset15") != 0 && stricmp(nodesSetPtr->getAttribute("Type"), "dataset2411") != 0)
            {
                coModule::sendError("Nodes input has wrong format (Dataset 15 or dataset 2411 required)");
                return FAIL;
            }

            char *nodesHaveChanged = (char *)nodesSetPtr->getAttribute("HasChanged");

            if (stricmp(nodesHaveChanged, "true") == 0 || updateNodesInput)
            {

                nodesSetPtr->addAttribute("HasChanged", "false");

                Clean(NODES);

                UnpackDataset(nodesSetPtr);

                updateNodesInput = false;
            }

            nodesInput = true;
        }
        else //no nodes input
        {
        }

        //Process traceLine input
        coDoSet *traceSetPtr = NULL;

        if (in_traceLineIndicesPtr && in_traceLineIndicesPtr->isType("SETELE"))
        {

            traceSetPtr = (coDoSet *)in_traceLineIndicesPtr;
            if (stricmp(traceSetPtr->getAttribute("Type"), "dataset82") != 0)
            {
                coModule::sendError("Traceline indices input has wrong format (Dataset 82 required)");
                return FAIL;
            }

            char *tracesHaveChanged = (char *)traceSetPtr->getAttribute("HasChanged");

            if (stricmp(tracesHaveChanged, "true") == 0 || updateTraceLineInput)
            {

                traceSetPtr->addAttribute("HasChanged", "false");

                Clean(TRACELINES);

                UnpackDataset(traceSetPtr);

                //--------------------------------------------------

                numChoice_traceID = (int)traceLineMap.size() + 1; //+1 for "----all tracelines----"

                choice_traceIDText = new char *[numChoice_traceID];

                map<unsigned int, sTraceLine *>::iterator tlMit;
                int textPos = 0;
                char *textPtr = NULL;

                for (tlMit = traceLineMap.begin(); tlMit != traceLineMap.end(); tlMit++)
                {
                    choice_traceIDText[textPos] = new char[80];
                    ((coDoText *)((coDoSet *)traceSetPtr->getElement(textPos))->getElement(2))->getAddress(&textPtr);
//					strcpy_s(choice_traceIDText[textPos++], 80*sizeof(char), textPtr );
#ifdef WIN32
                    sprintf_s(choice_traceIDText[textPos++], 80 * sizeof(char), "%d. %s", textPos - 1, textPtr);
#else
                snprintf(choice_traceIDText[textPos], 80 * sizeof(char), "%d. %s", textPos, textPtr);
                textPos++;
#endif
                }

                choice_traceIDText[textPos] = new char[80];
                strcpy_s(choice_traceIDText[textPos], 80 * sizeof(char), "----all tracelines----");

                choice_traceLineID->setValue(numChoice_traceID, choice_traceIDText, 0);

                //--------------------------------------------------

                updateTraceLineInput = false;
            }

            tracelineInput = true;
        }
        else //no traceline indices input
        {
        }

        //Process DOF input
        coDoSet *dofsSetPtr = NULL;

        if (in_DOFsPtr && in_DOFsPtr->isType("SETELE"))
        {
            dofsSetPtr = (coDoSet *)in_DOFsPtr;

            if (stricmp(dofsSetPtr->getAttribute("Type"), "dataset58") != 0)
            {
                coModule::sendError("DOFs input has wrong format(Dataset58  required)");
                return FAIL;
            }

            char *dofsHaveChanged = (char *)dofsSetPtr->getAttribute("HasChanged");

            if (stricmp(dofsHaveChanged, "true") == 0 || updateDOFsInput)
            {

                dofsSetPtr->addAttribute("HasChanged", "false");

                Clean(DOFS);

                UnpackDataset(dofsSetPtr);
                //--------------------------------------------------------

                numChoice_functionTypeText = (int)functionTypeNamesMap.size() + 1; //+1 for "none"

                map<short, char *>::iterator fTNit; //function type names iterator

                choice_functionTypeText = new char *[numChoice_functionTypeText];

                int textPos = 0;

                //add the function names to the choice list
                for (fTNit = functionTypeNamesMap.begin(); fTNit != functionTypeNamesMap.end(); fTNit++)
                {
                    choice_functionTypeText[textPos] = new char[80];
                    strcpy_s(choice_functionTypeText[textPos], 80 * sizeof(char), fTNit->second);
                    textPos++;
                }

                choice_functionTypeText[textPos] = new char[80];
                strcpy_s(choice_functionTypeText[textPos], 80 * sizeof(char), "none");

                choice_functionType->updateValue(numChoice_functionTypeText, choice_functionTypeText, 0);
                choice_functionType->show();

                //--------------------------------------------------------

                updateDOFsInput = false;
            }

            dofsInput = true;
        }

        delete cornerList;
        delete typeList;
        delete elementList;

        cornerList = NULL;
        typeList = NULL;
        elementList = NULL;

        numCorners = 0;
        numElements = 0;
        numTypes = 0;

        //Compute dofs vectors for animation

        int numAnimationSteps = 0;
        float **vectors = NULL; //array which contains the direction vectors
        float **refVectors = NULL;

        if (dofsInput)
        {
            char *functionSel = (char *)choice_functionType->getActLabel();
            short selectedFunc = -1;

            if (stricmp(functionSel, "none") != 0) //find the chosen function
            {
                for (int i = 0; i < 5; i++)
                {
                    if (functionSel[i] == 127)
                        functionSel[i] = 32;
                }
                map<short, char *>::iterator funcTypeNamesIt;

                //find the active selected function
                for (funcTypeNamesIt = functionTypeNamesMap.begin(); funcTypeNamesIt != functionTypeNamesMap.end(); funcTypeNamesIt++)
                {
                    if (strnicmp(functionSel, funcTypeNamesIt->second, 5) == 0)
                    {
                        selectedFunc = funcTypeNamesIt->first;
                        break;
                    }
                }
            }

            //no function selected? if so, just show the selected line
            if (selectedFunc == -1)
            {
            }
            //else calculate the animation vectors
            else
            {

                //int numDOFSets = functionTypeSets[selectedFunc].size();		//size of the list for the chosen function
                numAnimationSteps = (int)animationStepsSlider->getValue();

                vectors = new float *[3];

                vectors[0] = new float[numCoordinates];
                vectors[1] = new float[numCoordinates];
                vectors[2] = new float[numCoordinates];

                memset(vectors[0], 0, numCoordinates * sizeof(float));
                memset(vectors[1], 0, numCoordinates * sizeof(float));
                memset(vectors[2], 0, numCoordinates * sizeof(float));

                list<unsigned int>::iterator listIt;

                int actNode = 0;
                short actDir = 0; //the direction of the node
                float *actData = NULL;
                long actDataPos = dataPosSlider->getValue();

                //TODO: replace multiplier by some logarithmic function
                float multiplier = log10((float)actDataPos) * multiplierSlider->getValue();

                coModule::sendInfo("Multiplier %f", multiplier);

                //TODO: adjust for other dof types. This is just for FRF data
                float quot = (float)(-(4 * PI * PI * (actDataPos / multiplier) * (actDataPos / multiplier)));

                //TODO: switch the type to find out, how many data is stored in one line, and change the 2
                actDataPos *= 2; //2 values / data (datapair)

                double actValue = 0;

                //TODO: adjust data access for several abscissa spacing (now, its assumed that data is ordered RX1 IX1 RX2 IX2 ...)
                for (listIt = functionTypeSets[selectedFunc].begin(); listIt != functionTypeSets[selectedFunc].end(); listIt++)
                {
                    //get a dataset58 of the specified function -> get record6 -> get the intArr -> get Value 5(response node)
                    actNode = ((coDoIntArr *)((coDoSet *)((coDoSet *)dofsSetPtr->getElement(*listIt))->getElement(5))->getElement(0))->getAddress()[4];

                    //there's no entry in the substitute map
                    if (substituteMap.find(actNode) == substituteMap.end())
                    {
                        coModule::sendWarning("Cannot find node %d for animation", actNode);
                        continue;
                    }

                    //get a dataset58 of the specified function -> get record6 -> get the intArr -> get Value 6(response direction)
                    actDir = ((coDoIntArr *)((coDoSet *)((coDoSet *)dofsSetPtr->getElement(*listIt))->getElement(5))->getElement(0))->getAddress()[5];

                    coDoSet *actSet = (coDoSet *)dofsSetPtr->getElement(*listIt); //get actual dataset58
                    coDoFloat *record = (coDoFloat *)actSet->getElement(11); //get record 12 (data)

                    record->getAddress(&actData);

                    actValue = (actData[actDataPos] + actData[actDataPos + 1]) / quot;

                    switch (actDir)
                    {
                    case 0: //Scalar
                    {
                        break;
                    }
                    case -1: //-X Translation
                    {
                        if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 1)
                            vectors[0][substituteMap[actNode]] = (float)-actValue;
                        break;
                    }
                    case 1: //X Translation
                    {
                        if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 1)
                            vectors[0][substituteMap[actNode]] = (float)actValue;
                        break;
                    }
                    case -2: //-Y Translation
                    {
                        if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 2)
                            vectors[1][substituteMap[actNode]] = (float)-actValue;
                        break;
                    }
                    case 2: //Y Translation
                    {
                        if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 2)
                            vectors[1][substituteMap[actNode]] = (float)actValue;
                        break;
                    }
                    case -3: //-Z Translation
                    {
                        if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 3)
                            vectors[2][substituteMap[actNode]] = (float)-actValue;
                        break;
                    }
                    case 3: //Z Translation
                    {
                        if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 3)
                            vectors[2][substituteMap[actNode]] = (float)actValue;
                        break;
                    }
                    case -4: //-X Rotation
                    {
                        break;
                    }
                    case 4: //X Rotation
                    {
                        break;
                    }
                    case -5: //-Y Rotation
                    {
                        break;
                    }
                    case 5: //Y Rotation
                    {
                        break;
                    }
                    case -6: //-Z Rotation
                    {
                        break;
                    }
                    case 6: //Z Rotation
                    {
                        break;
                    }
                    }
                }
                //-----------------------testing-IMPORTANT: ADD REFERENCE NODES------------------------------

                refVectors = new float *[3]; //vector of the reference node

                refVectors[0] = new float[numCoordinates];
                refVectors[1] = new float[numCoordinates];
                refVectors[2] = new float[numCoordinates];

                memset(refVectors[0], 0, numCoordinates * sizeof(float));
                memset(refVectors[1], 0, numCoordinates * sizeof(float));
                memset(refVectors[2], 0, numCoordinates * sizeof(float));

                //				list<unsigned int>::iterator listIt;

                int refNode = 0;
                short refDir = 0; //the direction of the node
                //int sign = 1;

                for (listIt = functionTypeSets[selectedFunc].begin(); listIt != functionTypeSets[selectedFunc].end(); listIt++)
                {
                    //get a dataset58 of the specified function -> get record6 -> get the intArr -> get Value 7(reference node)
                    refNode = ((coDoIntArr *)((coDoSet *)((coDoSet *)dofsSetPtr->getElement(*listIt))->getElement(5))->getElement(0))->getAddress()[6];
                    actNode = ((coDoIntArr *)((coDoSet *)((coDoSet *)dofsSetPtr->getElement(*listIt))->getElement(5))->getElement(0))->getAddress()[4];

                    //there's no entry in the substitute map
                    if (substituteMap.find(refNode) == substituteMap.end())
                    {
                        coModule::sendWarning("Cannot find node %d for animation", refNode);
                        continue;
                    }

                    //get a dataset58 of the specified function -> get record6 -> get the intArr -> get Value 8(reference direction)
                    refDir = ((coDoIntArr *)((coDoSet *)((coDoSet *)dofsSetPtr->getElement(*listIt))->getElement(5))->getElement(0))->getAddress()[7];
                    //get a dataset58 of the specified function -> get record6 -> get the intArr -> get Value 6(response direction)
                    actDir = ((coDoIntArr *)((coDoSet *)((coDoSet *)dofsSetPtr->getElement(*listIt))->getElement(5))->getElement(0))->getAddress()[5];

                    /*
					coDoSet *actSet = (coDoSet*)dofsSetPtr->getElement(*listIt);		//get actual dataset58
					coDoFloat *record = (coDoFloat*)actSet->getElement(11);				//get record 12 (data)

					record->getAddress(&actData);										

					actValue = (actData[actDataPos]+actData[actDataPos+1])/quot;
					*/

                    if (refDir == actDir && refNode == actNode)
                        continue;

                    //					if (refDir < 0)
                    //						sign = -1;
                    //					else
                    //						sign = 1;

                    refVectors[ ::abs(actDir) - 1][substituteMap[actNode]] = vectors[ ::abs(refDir) - 1][substituteMap[refNode]];

                    /*

					switch (actDir)
					{
					case 0:								//Scalar
						{
							break;
						}
					case -1:							//-X Translation
						{

//							if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 1)		
							refVectors[0][substituteMap[actNode]] = -vectors[0][substituteMap[refNode]];
							break;
						}
					case 1:								//X Translation
						{
//							if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 1)		
							refVectors[0][substituteMap[actNode]] = vectors[0][substituteMap[refNode]];
							break;
						}
					case -2:							//-Y Translation
						{
//							if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 2)		
							refVectors[1][substituteMap[actNode]] = -vectors[1][substituteMap[refNode]];
							break;
						}
					case 2:								//Y Translation
						{
//							if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 2)		
							refVectors[1][substituteMap[actNode]] = vectors[1][substituteMap[refNode]];
							break;
						}
					case -3:							//-Z Translation
						{
//							if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 3)		
							refVectors[2][substituteMap[actNode]] = -vectors[2][substituteMap[refNode]];
							break;
						}
					case 3:								//Z Translation
						{
//							if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 3)		
							refVectors[2][substituteMap[actNode]] = vectors[2][substituteMap[refNode]];
							break;
						}
					case -4:							//-X Rotation
						{
							break;
						}
					case 4:								//X Rotation
						{
							break;
						}
					case -5:							//-Y Rotation
						{
							break;
						}
					case 5:								//Y Rotation
						{
							break;
						}
					case -6:							//-Z Rotation
						{
							break;
						}
					case 6:								//Z Rotation
						{
							break;
						}
					}   */
                }
                //---------------------------------------------------------------------------------------------
            }
        }

        //Compute output
        if (nodesInput)
        {
            if (tracelineInput) //nodes, tracelines
            {

                //int actTracelineChoice = choice_traceLineID->getValue();
                //int traceLineChoice = actTracelineChoice;
                int numLines = 0;

                list<unsigned int>::iterator selTraceIt;

                coModule::sendInfo("%s", traceLineSelection->getValue());
                list<unsigned int> selTrace = getSelectedTracelines();
                list<unsigned int> traceList;

                for (selTraceIt = selTrace.begin(); selTraceIt != selTrace.end(); selTraceIt++)
                {
                    if (traceLineMap.find(*selTraceIt) != traceLineMap.end()) //filter out no existing traceline numbers
                    {
                        numLines += traceLineMap[*selTraceIt]->numLines;
                        traceList.push_back(*selTraceIt);
                    }
                }

                selTrace.clear();

                numTypes = numLines;
                typeList = new int[numTypes];

                numElements = numLines;
                elementList = new int[numElements];

                for (int i = 0; i < numTypes; i++)
                {
                    typeList[i] = TYPE_BAR;
                    elementList[i] = i * 2;
                }

                numCorners = numLines * 2;
                cornerList = new int[numCorners];

                int lastVertex = 0;
                //int actVertex = 0;
                int cornerPos = 0;
                sTraceLine *actTraceLine;
                //bool lineComputed = false;

                int numObjects = 0;
                coDistributedObject **outLines = new coDistributedObject *[traceList.size() + 1];
                //generate the tracelines
                for (selTraceIt = traceList.begin(); selTraceIt != traceList.end(); selTraceIt++)
                {
                    actTraceLine = traceLineMap[*selTraceIt];
                    lastVertex = 0;
                    while (actTraceLine->indices[lastVertex] == 0)
                        lastVertex++;

                    for (unsigned int i = lastVertex + 1; i < actTraceLine->numVertices; i++)
                    {
                        if (actTraceLine->indices[i] != 0 && actTraceLine->indices[lastVertex] == 0)
                        {
                            //case does not occure
                        }
                        else if (actTraceLine->indices[i] != 0 && actTraceLine->indices[lastVertex] != 0)
                        {
                            cornerList[cornerPos] = substituteMap[actTraceLine->indices[lastVertex]];
                            cornerList[cornerPos + 1] = substituteMap[actTraceLine->indices[i]];
                            cornerPos += 2;
                            //lineComputed = true;
                        }
                        else if (actTraceLine->indices[i] == 0 && actTraceLine->indices[lastVertex] != 0)
                        {
                            //if (!lineComputed)
                            //{
                            //	cornerList[cornerPos] = substituteMap[actTraceLine->indices[lastVertex]];
                            //	cornerList[cornerPos+1] = substituteMap[actTraceLine->indices[lastVertex]];
                            //	cornerPos += 2;
                            //}

                            while (actTraceLine->indices[i] == 0)
                                i++;

                            //lineComputed = false;
                        }
                        else if (actTraceLine->indices[i] == 0 && actTraceLine->indices[lastVertex] == 0)
                        {
                            while (actTraceLine->indices[i] == 0)
                                i++;
                        }

                        lastVertex = i;
                    }
                }

                if (numAnimationSteps == 0)
                {
                    char *objName = new char[strlen(traceLine->getObjName()) + 100];
                    sprintf(objName, "%s_%d", traceLine->getObjName(), numObjects);
                    grid = new coDoUnstructuredGrid(objName, numElements, numCorners, numCoordinates,
                                                    elementList, cornerList,
                                                    coordinates[0], coordinates[1], coordinates[2],
                                                    typeList);
                    grid->addAttribute("POINTSIZE", "2");
                    outLines[numObjects] = grid;
                    numObjects++;
                    outLines[numObjects] = NULL;
                    delete[] objName;

                    coModule::sendInfo("nodes, traceline, no anmiation");
                }
                else
                {

                    //-------------------------------------------------
                    char name[80];

                    coDoUnstructuredGrid **anim = new coDoUnstructuredGrid *[numAnimationSteps];
                    coDoVec3 **animData = new coDoVec3 *[numAnimationSteps];

                    sprintf_s(name, 80 * sizeof(char), "%s_%d", traceLine->getObjName(), 0);

                    float **tmpCoords = new float *[3];
                    tmpCoords[0] = new float[numCoordinates];
                    tmpCoords[1] = new float[numCoordinates];
                    tmpCoords[2] = new float[numCoordinates];

                    memset(tmpCoords[0], 0, numCoordinates * sizeof(float));
                    memset(tmpCoords[1], 0, numCoordinates * sizeof(float));
                    memset(tmpCoords[2], 0, numCoordinates * sizeof(float));

                    float **tmpData = new float *[3];
                    tmpData[0] = new float[numCoordinates];
                    tmpData[1] = new float[numCoordinates];
                    tmpData[2] = new float[numCoordinates];

                    memset(tmpData[0], 0, numCoordinates * sizeof(float));
                    memset(tmpData[1], 0, numCoordinates * sizeof(float));
                    memset(tmpData[2], 0, numCoordinates * sizeof(float));

                    double timeDiff = 2 * PI / numAnimationSteps; //divide 2*PI into numAnimationSteps steps
                    double time = 0; //actual "time"

                    for (int i = 0; i < numAnimationSteps; i++)
                    {

                        for (unsigned int j = 0; j < numCoordinates; j++)
                        {
                            //tmpData[0][j] = (sin(time)*vectors[0][j]);
                            //tmpData[1][j] = (sin(time)*vectors[1][j]);
                            //tmpData[2][j] = (sin(time)*vectors[2][j]);

                            tmpData[0][j] = (float)(sin(time) * (vectors[0][j] + refVectors[0][j]));
                            tmpData[1][j] = (float)(sin(time) * (vectors[1][j] + refVectors[1][j]));
                            tmpData[2][j] = (float)(sin(time) * (vectors[2][j] + refVectors[2][j]));

                            //tmpData[0][j] = (sin(time)*refVectors[0][j]);
                            //tmpData[1][j] = (sin(time)*refVectors[1][j]);
                            //tmpData[2][j] = (sin(time)*refVectors[2][j]);

                            tmpCoords[0][j] = coordinates[0][j] + tmpData[0][j];
                            tmpCoords[1][j] = coordinates[1][j] + tmpData[1][j];
                            tmpCoords[2][j] = coordinates[2][j] + tmpData[2][j];
                        }

                        time += timeDiff;

                        sprintf_s(name, 80 * sizeof(char), "%s_%d", traceLine->getObjName(), i);

                        anim[i] = new coDoUnstructuredGrid(name,
                                                           numElements, numCorners, numCoordinates, elementList, cornerList,
                                                           tmpCoords[0], tmpCoords[1], tmpCoords[2], typeList);

                        animData[i] = new coDoVec3(name, numCoordinates, tmpData[0], tmpData[1], tmpData[2]);

                        anim[i]->addAttribute("POINTSIZE", "2");
                    }

                    char *objName = new char[strlen(traceLine->getObjName()) + 100];
                    sprintf(objName, "%s_%d", traceLine->getObjName(), numObjects);
                    coDoSet *grid = new coDoSet(objName, numAnimationSteps, (coDistributedObject * const *)anim);
                    data = new coDoSet(traceLineData->getObjName(), numAnimationSteps, (coDistributedObject * const *)animData);
                    delete[] objName;
                    sprintf_s(name, 80 * sizeof(char), "1 %d", numAnimationSteps);
                    grid->addAttribute("TIMESTEP", name);
                    outLines[numObjects] = grid;
                    numObjects++;
                    outLines[numObjects] = NULL;

                    //-------------------------------------------------
                }

                coDoSet *tmp = new coDoSet(traceLine->getObjName(), numObjects, outLines);

                traceLine->setCurrentObject(tmp);
            }
            else //nodes, no tracelines
            {
                numElements = numCoordinates;
                elementList = new int[numElements];

                for (int i = 0; i < numElements; i++)
                    elementList[i] = i;

                numCorners = 0;
                cornerList = NULL;

                numTypes = numElements;
                typeList = new int[numTypes];

                for (int i = 0; i < numTypes; i++)
                    typeList[i] = TYPE_POINT;

                if (numAnimationSteps == 0)
                {
                    grid = new coDoUnstructuredGrid(traceLine->getObjName(), numElements, numCorners, numCoordinates,
                                                    elementList, cornerList,
                                                    coordinates[0], coordinates[1], coordinates[2],
                                                    typeList);
                    grid->addAttribute("POINTSIZE", "2");

                    coModule::sendInfo("nodes, no traceline, no anmiation");
                }
                else
                {

                    //-------------------------------------------------
                    char name[80];

                    coDoUnstructuredGrid **anim = new coDoUnstructuredGrid *[numAnimationSteps];
                    coDoVec3 **animData = new coDoVec3 *[numAnimationSteps];

                    sprintf_s(name, 80 * sizeof(char), "%s_%d", traceLine->getObjName(), 0);

                    float **tmpCoords = new float *[3];
                    tmpCoords[0] = new float[numCoordinates];
                    tmpCoords[1] = new float[numCoordinates];
                    tmpCoords[2] = new float[numCoordinates];

                    memset(tmpCoords[0], 0, numCoordinates * sizeof(float));
                    memset(tmpCoords[1], 0, numCoordinates * sizeof(float));
                    memset(tmpCoords[2], 0, numCoordinates * sizeof(float));

                    float **tmpData = new float *[3];
                    tmpData[0] = new float[numCoordinates];
                    tmpData[1] = new float[numCoordinates];
                    tmpData[2] = new float[numCoordinates];

                    memset(tmpData[0], 0, numCoordinates * sizeof(float));
                    memset(tmpData[1], 0, numCoordinates * sizeof(float));
                    memset(tmpData[2], 0, numCoordinates * sizeof(float));

                    double timeDiff = 2 * PI / numAnimationSteps; //divide 2*PI into numAnimationSteps steps
                    double time = 0; //actual "time"

                    for (int i = 0; i < numAnimationSteps; i++)
                    {

                        for (unsigned int j = 0; j < numCoordinates; j++)
                        {
                            //tmpData[0][j] = (sin(time)*vectors[0][j]);
                            //tmpData[1][j] = (sin(time)*vectors[1][j]);
                            //tmpData[2][j] = (sin(time)*vectors[2][j]);

                            tmpData[0][j] = (float)(sin(time) * (vectors[0][j] + refVectors[0][j]));
                            tmpData[1][j] = (float)(sin(time) * (vectors[1][j] + refVectors[1][j]));
                            tmpData[2][j] = (float)(sin(time) * (vectors[2][j] + refVectors[2][j]));

                            //tmpData[0][j] = (sin(time)*refVectors[0][j]);
                            //tmpData[1][j] = (sin(time)*refVectors[1][j]);
                            //tmpData[2][j] = (sin(time)*refVectors[2][j]);

                            tmpCoords[0][j] = coordinates[0][j] + tmpData[0][j];
                            tmpCoords[1][j] = coordinates[1][j] + tmpData[1][j];
                            tmpCoords[2][j] = coordinates[2][j] + tmpData[2][j];
                        }

                        time += timeDiff;

                        sprintf_s(name, 80 * sizeof(char), "%s_%d", traceLine->getObjName(), i);

                        anim[i] = new coDoUnstructuredGrid(name,
                                                           numElements, numCorners, numCoordinates, elementList, cornerList,
                                                           tmpCoords[0], tmpCoords[1], tmpCoords[2], typeList);

                        animData[i] = new coDoVec3(name, numCoordinates, tmpData[0], tmpData[1], tmpData[2]);

                        anim[i]->addAttribute("POINTSIZE", "2");
                    }

                    coDoSet *grid = new coDoSet(traceLine->getObjName(), numAnimationSteps, (coDistributedObject * const *)anim);
                    data = new coDoSet(traceLineData->getObjName(), numAnimationSteps, (coDistributedObject * const *)animData);

                    sprintf_s(name, 80 * sizeof(char), "1 %d", numAnimationSteps);
                    grid->addAttribute("TIMESTEP", name);

                    //-------------------------------------------------
                }
            }
        }
        else //NO nodes
        {
            coModule::sendInfo("Nothing to show");
        }

        /*
		coDoSet *dofsSetPtr = NULL;

		if (in_nodesPtr && in_nodesPtr->isType("SETELE"))
		{
			//is there a traceline input?
			if (in_traceLineIndicesPtr && in_traceLineIndicesPtr->isType("SETELE"))	//TODO: add output even if no tracelines are available
			{

				const coDistributedObject *in_dofsPtr = DOFs->getCurrentObject();

				//is there an dof input?
				if (in_dofsPtr && in_dofsPtr->isType("SETELE"))
				{

					dofsSetPtr = (coDoSet*)in_dofsPtr;

					if (stricmp(dofsSetPtr->getAttribute("Type"), "dataset58") != 0)
					{
						coModule::sendError("DOFs input has wrong format(Dataset58  required), and will be ignored");
						dofsSetPtr = NULL;
					}
				}

				coDoSet *nodesSetPtr = (coDoSet*)in_nodesPtr;
				coDoSet *traceSetPtr = (coDoSet*)in_traceLineIndicesPtr;


				//output data
				if (nodes->isConnected() &&
					stricmp(nodesSetPtr->getAttribute("Type"), "dataset15") != 0 )
				{
					coModule::sendError("Nodes input has wrong format (Dataset 15 required)");
					return FAIL;
				}

				if (traceLineIndices->isConnected() &&
					stricmp(traceSetPtr->getAttribute("Type"), "dataset82") != 0)
				{
					coModule::sendError("Traceline indices input has wrong format (Dataset 82 required)");
					return FAIL;
				}

				char *nodesHaveChanged = (char*)nodesSetPtr->getAttribute("HasChanged");
				char *tracesHaveChanged = (char*)traceSetPtr->getAttribute("HasChanged");

				//get the function names
				if ( dofsSetPtr )
				{
					char *dofsHaveChanged = (char*)dofsSetPtr->getAttribute("HasChanged");

					if (stricmp(dofsHaveChanged, "true") == 0 || 
						updateDOFsInput)
					{
						dofsSetPtr->addAttribute("HasChanged", "false");

						functionTypeNamesMap.clear();
						map<short, list<unsigned int>>::iterator funcTypeSetsIt;

						for (funcTypeSetsIt = functionTypeSets.begin(); funcTypeSetsIt != functionTypeSets.end(); funcTypeSetsIt++)
						{
							funcTypeSetsIt->second.clear();
						}

						functionTypeSets.clear();


						UnpackDataset(dofsSetPtr);


						//TODO: add coordinate direction choices

						if (choice_functionTypeText)
						{
							for (int i = 0; i < numChoice_functionTypeText; i++)
							{
								delete choice_functionTypeText[i];
								choice_functionTypeText[i] = NULL;
							}
							delete choice_functionTypeText;
							choice_functionTypeText = NULL;
						}

						numChoice_functionTypeText = functionTypeNamesMap.size()+1;		 //+1 for "none"

						map<short, char*>::iterator fTNit;			//function type names iterator

						choice_functionTypeText = new char*[numChoice_functionTypeText]; 

						int textPos = 0;

						//add the function names to the choice list
						for (fTNit = functionTypeNamesMap.begin(); fTNit != functionTypeNamesMap.end(); fTNit++)
						{
							choice_functionTypeText[textPos] = new char[80];
							strcpy_s(choice_functionTypeText[textPos], 80*sizeof(char), fTNit->second);
							textPos++;
						}

						choice_functionTypeText[textPos] = new char[80];
						strcpy_s(choice_functionTypeText[textPos], 80*sizeof(char), "none");

						choice_functionType->setValue(numChoice_functionTypeText, choice_functionTypeText, 0);
						choice_functionType->show();

						updateDOFsInput = false;

					}
				}

				if (stricmp(nodesHaveChanged, "true") == 0 ||		//the file has changed, so update
					stricmp(tracesHaveChanged, "true") == 0 ||
					updateNodesInput )
				{


					nodesSetPtr->addAttribute("HasChanged", "false");
					traceSetPtr->addAttribute("HasChanged", "false");

					Clean();

					UnpackDataset(nodesSetPtr);
					UnpackDataset(traceSetPtr);


					if (choice_traceIDText)
					{
						for (int i = 0; i < numChoice_traceID; i++)
						{
							delete choice_traceIDText[i];
							choice_traceIDText[i] = NULL;
						}

						delete choice_traceIDText;
						choice_traceIDText = NULL;
					}

					numChoice_traceID = traceLineMap.size()+1; //+1 for "----all tracelines----"

					choice_traceIDText = new char*[numChoice_traceID];

					map<unsigned int, sTraceLine*>::iterator tlMit;
					int textPos = 0;
					char* textPtr = NULL;

					for (tlMit = traceLineMap.begin(); tlMit != traceLineMap.end(); tlMit++)
					{
						choice_traceIDText[textPos] = new char[80];
						((coDoText*)((coDoSet*)traceSetPtr->getElement(textPos))->getElement(2))->getAddress(&textPtr);
						strcpy_s(choice_traceIDText[textPos++], 80*sizeof(char), textPtr );
					}

					choice_traceIDText[textPos] = new char[80];
					strcpy_s(choice_traceIDText[textPos], 80*sizeof(char), "----all tracelines----");

					choice_traceLineID->setValue(numChoice_traceID, choice_traceIDText, 0);

					updateNodesInput = false;
				}


				delete lineList;
				delete vertList;
				lineList = NULL;
				vertList = NULL;

				int actTracelineChoice = choice_traceLineID->getValue();

				int numVertices = 0;
				int numLines	= 0;		//TODO: obsolete

				//for unstructured grid
				int numConns = 0;			//size of connectivity list
				int numElements = 0;		//size of element list
				int numTypes = 0;			//size of type list

				int numCoordinates = nodeMap.size();
				int numChoices = choice_traceLineID->getNumChoices();


				if (actTracelineChoice == choice_traceLineID->getNumChoices()-2) //"all tracelines" selected
				{
					map<unsigned int, sTraceLine*>::iterator tLMit;

					for (tLMit = traceLineMap.begin(); tLMit != traceLineMap.end(); tLMit++)
					{
						numVertices += tLMit->second->numVertices;
						numLines += tLMit->second->numLines;				//TODO: obsolete
					}

					vertList = new int[numVertices]; //TODO: obsolete
					lineList = new int[numLines];	//TODO: obsolete



					int actVert = 0;
					int actLine = 0;
					int vertOffset = 0;

					for (tLMit = traceLineMap.begin(); tLMit != traceLineMap.end(); tLMit++)
					{
						for (int i = 0; i < tLMit->second->numVertices; i++)
						{
							vertList[actVert++] = substituteMap[tLMit->second->vertList[i]];
						}

						for (int i = 0; i < tLMit->second->numLines; i++)
						{
							lineList[actLine++] = tLMit->second->indices[i] + vertOffset;
						}

						vertOffset += tLMit->second->numVertices;
					}

				}
				else
				{
					numVertices = traceLineMap[actTracelineChoice]->numVertices;
					numLines	= traceLineMap[actTracelineChoice]->numLines;

					vertList = new int[numVertices];
					lineList = new int[numLines];


					for (int i = 0; i < numVertices; i++)
					{
						vertList[i] = substituteMap[traceLineMap[actTracelineChoice]->vertList[i]];
					}

					for (int i = 0; i < numLines; i++)
					{
						lineList[i] = traceLineMap[actTracelineChoice]->indices[i];
					}

				}




				if (dofsSetPtr)		//a DOF input has been detected, so animate the tracelines
				{

					int numAnimationSteps = (int)animationStepsSlider->getValue();
					coDoFloat *dofValues = NULL;
					int numDOFSets = 0;
					char *functionSel = (char*)choice_functionType->getActLabel();
					short selectedFunc = -1;

					if (stricmp(functionSel, "none") != 0)		//find the chosen function
					{
						for (int i = 0; i < 5; i++)
						{
							if (functionSel[i] == 127)
								functionSel[i] = 32;
						}
						map<short, char*>::iterator funcTypeNamesIt;

						//find the active selected function
						for (funcTypeNamesIt = functionTypeNamesMap.begin(); funcTypeNamesIt != functionTypeNamesMap.end(); funcTypeNamesIt++)
						{
							if (strnicmp(functionSel, funcTypeNamesIt->second, 5) == 0)
							{
								selectedFunc = funcTypeNamesIt->first;
								break;
							}
						}
					}

					//no function selected? if so, just show the selected line
					if ( selectedFunc == -1)
					{
						coDoLines *activeTraceline = new coDoLines( traceLine->getObjName(), 
							numCoordinates,
							coordinates[0], coordinates[1], coordinates[2],
							numVertices, vertList,
							numLines, lineList	);

						traceLine->setCurrentObject( activeTraceline );
					}
					//else animate the traceline
					else
					{

						numDOFSets = functionTypeSets[selectedFunc].size();		//size of the list for the chosen function


						float **vectors = new float*[3];						//array which contains the direction vectors
						vectors[0] = new float[numCoordinates];
						vectors[1] = new float[numCoordinates];
						vectors[2] = new float[numCoordinates];

						memset(vectors[0], 0, numCoordinates*sizeof(float));
						memset(vectors[1], 0, numCoordinates*sizeof(float));
						memset(vectors[2], 0, numCoordinates*sizeof(float));

						list<unsigned int>::iterator listIt;

						int actNode = 0;
						short actDir = 0;		//the direction of the node
						float *actData = NULL;
						long actDataPos = dataPosSlider->getValue();
						float multiplier = multiplierSlider->getValue();

						//TODO: adjust for other dof types. This is just for FRF data
						float quot = -(4*PI*PI*(actDataPos/multiplier)*(actDataPos/multiplier));

						//TODO: switch the type to find out, how many data is stored in one line, and change the 2
						actDataPos *= 2;  //2 values / data (datapair)

						double actValue = 0;


						//TODO: adjust data access for several abscissa spacing (now, its assumed that data is ordered RX1 IX1 RX2 IX2 ...)
						for (listIt = functionTypeSets[selectedFunc].begin(); listIt != functionTypeSets[selectedFunc].end(); listIt++)
						{
							//get a dataset58 of the specified function -> get record6 -> get the intArr -> get Value 5(response node)
							actNode = ((coDoIntArr*)((coDoSet*)((coDoSet*)dofsSetPtr->getElement(*listIt))->getElement(5))->getElement(0))->getAddress()[4]; 
							//get a dataset58 of the specified function -> get record6 -> get the intArr -> get Value 6(response direction)
							actDir = ((coDoIntArr*)((coDoSet*)((coDoSet*)dofsSetPtr->getElement(*listIt))->getElement(5))->getElement(0))->getAddress()[5];

							coDoSet *actSet = (coDoSet*)dofsSetPtr->getElement(*listIt);		//get actual dataset58
							coDoFloat *record = (coDoFloat*)actSet->getElement(11);				//get record 12 (data)

							record->getAddress(&actData);										

							actValue = (actData[actDataPos]+actData[actDataPos+1])/quot;

							switch (actDir)
							{
							case 0:								//Scalar
								{
									break;
								}
							case -1:							//-X Translation
								{
									if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 1)		
										vectors[0][substituteMap[actNode]] = -actValue;
									break;
								}
							case 1:								//X Translation
								{
									if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 1)		
										vectors[0][substituteMap[actNode]] = actValue;
									break;
								}
							case -2:							//-Y Translation
								{
									if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 2)		
										vectors[1][substituteMap[actNode]] = -actValue;
									break;
								}
							case 2:								//Y Translation
								{
									if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 2)		
										vectors[1][substituteMap[actNode]] = actValue;
									break;
								}
							case -3:							//-Z Translation
								{
									if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 3)		
										vectors[2][substituteMap[actNode]] = -actValue;
									break;
								}
							case 3:								//Z Translation
								{
									if (choice_coordDir->getValue() == 0 || choice_coordDir->getValue() == 3)		
										vectors[2][substituteMap[actNode]] = actValue;
									break;
								}
							case -4:							//-X Rotation
								{
									break;
								}
							case 4:								//X Rotation
								{
									break;
								}
							case -5:							//-Y Rotation
								{
									break;
								}
							case 5:								//Y Rotation
								{
									break;
								}
							case -6:							//-Z Rotation
								{
									break;
								}
							case 6:								//Z Rotation
								{
									break;
								}
							}
						}

						char name[80];

						coDoLines **anim = new coDoLines*[numAnimationSteps];

						sprintf_s(name, 80*sizeof(char), "%s_%d", traceLine->getObjName(), 0);


						float **tmpCoords = new float*[3];
						tmpCoords[0] = new float[numCoordinates];
						tmpCoords[1] = new float[numCoordinates];
						tmpCoords[2] = new float[numCoordinates];

						memset(tmpCoords[0], 0, numCoordinates*sizeof(float));
						memset(tmpCoords[1], 0, numCoordinates*sizeof(float));
						memset(tmpCoords[2], 0, numCoordinates*sizeof(float));

						double timeDiff = 2*PI / numAnimationSteps;				//divide 2*PI into numAnimationSteps steps
						double time = 0;										//actual "time"

						for (int i = 0; i < numAnimationSteps; i++)
						{

							for (int j = 0; j < numCoordinates; j++)
							{
								tmpCoords[0][j] = coordinates[0][j] + (sin(time)*vectors[0][j]);
								tmpCoords[1][j] = coordinates[1][j] + (sin(time)*vectors[1][j]);
								tmpCoords[2][j] = coordinates[2][j] + (sin(time)*vectors[2][j]);
							}

							time += timeDiff;

							sprintf_s(name, 80*sizeof(char), "%s_%d", traceLine->getObjName(), i);
							anim[i] = new coDoLines( name, 
								numCoordinates,
								tmpCoords[0], tmpCoords[1], tmpCoords[2],
								numVertices, vertList,
								numLines, lineList	);
						}

						coDoSet *outSet = new coDoSet(traceLine->getObjName(), numAnimationSteps, (coDistributedObject* const*) anim);


						outSet->addAttribute("TIMESTEP","1 100");

						traceLine->setCurrentObject( outSet );
					}


				}
				else				//no dofs, just show the traceline, which is selected
				{
						

						coDoUnstructuredGrid *grid = new coDoUnstructuredGrid(traceLine->getObjName(), numElements, numConns, numCoordinates,
						elementList, cornerList, 
						coordinates[0], coordinates[1], coordinates[2], 
						typeList);



					coDoLines *activeTraceline = new coDoLines( traceLine->getObjName(), 
					numCoordinates,
					coordinates[0], coordinates[1], coordinates[2],
					numVertices, vertList,
					numLines, lineList	);
					


				}
			}

		}
	*/
        coFeedback fb("ModalPlugin");
        fb.addPara(multiplierSlider);
        if (data)
        {
            coModule::sendInfo("starting Feedback with data");
            fb.apply(data);

            traceLineData->setCurrentObject(data);
        }
        else if (grid)
        {
            coModule::sendInfo("starting Feedback with grid");
            fb.apply(grid);

            traceLine->setCurrentObject(grid);
        }

        delete cornerList;
        cornerList = NULL;
        numCorners = 0;
        delete typeList;
        typeList = NULL;
        numTypes = 0;
        delete elementList;
        elementList = NULL;
        numElements = 0;

#ifdef WRITE_TO_FILE_ENABLE
    }
#endif

    return SUCCESS;
}

void TraceLineExtractor::Clean(int what)
{
    if ((what & NODES) != 0)
    {
        //map<unsigned int, sPoint*>::iterator nMit;			//node map iterator

        /*		for (nMit = nodeMap.begin(); nMit != nodeMap.end(); nMit++)
			delete nMit->second;

		nodeMap.clear();
*/
        substituteMap.clear();

        if (coordinates)
        {
            delete coordinates[0];
            delete coordinates[1];
            delete coordinates[2];

            delete coordinates;
            coordinates = NULL;
        }

        if (vertList)
        {
            delete vertList;
            vertList = NULL;
        }

        updateNodesInput = true;
    }

    if ((what & TRACELINES) != 0)
    {
        map<unsigned int, sTraceLine *>::iterator tLMit; //traceline map iterator
        for (tLMit = traceLineMap.begin(); tLMit != traceLineMap.end(); tLMit++)
        {
            delete tLMit->second->indices;
            //			delete tLMit->second->vertList;
            delete tLMit->second;
        }

        traceLineMap.clear();

        if (lineList)
        {
            delete lineList;
            lineList = NULL;
        }

        updateTraceLineInput = true;

        if (choice_traceIDText)
        {
            for (int i = 0; i < numChoice_traceID; i++)
                delete choice_traceIDText[i];

            delete choice_traceIDText;
            choice_traceIDText = NULL;

            numChoice_traceID = 0;
        }
    }

    if ((what & DOFS) != 0)
    {
        map<short, char *>::iterator functionTypeNamesMapIt; //map for function type names at nodal DOFs dataset
        map<short, list<unsigned int> >::iterator functionTypeSetsIt; //map which contains a list, in which are all dataset numbers (of dataset58) of

        for (functionTypeNamesMapIt = functionTypeNamesMap.begin(); functionTypeNamesMapIt != functionTypeNamesMap.end(); functionTypeNamesMapIt++)
        {
            delete functionTypeNamesMapIt->second;
        }

        functionTypeNamesMap.clear();

        for (functionTypeSetsIt = functionTypeSets.begin(); functionTypeSetsIt != functionTypeSets.end(); functionTypeSetsIt++)
        {
            functionTypeSetsIt->second.clear();
        }

        functionTypeSets.clear();

        if (choice_functionTypeText)
        {
            for (int i = 0; i < numChoice_functionTypeText; i++)
                delete choice_functionTypeText[i];

            delete choice_functionTypeText;
            choice_functionTypeText = NULL;

            numChoice_functionTypeText = 0;
        }

        updateDOFsInput = true;
    }

    //	use_mapFile = false;
}

void TraceLineExtractor::quit(/* const char* */)
{
    Clean(ALL);
}

TraceLineExtractor::~TraceLineExtractor()
{
}

void TraceLineExtractor::param(const char *name, bool /*inMapLoading*/)
{
    if (stricmp(name, "Function at nodal DOFs") == 0)
    {
        updateDOFsInput = true;

        if (DOFs->isConnected())
        {
            choice_functionType->enable();
            choice_coordDir->enable();
            dataPosSlider->enable();
            animationStepsSlider->enable();
            multiplierSlider->enable();

#ifdef WRITE_TO_FILE_ENABLE
            if (datasetFileheader->isConnected() && datasetUnitDesc->isConnected())
            {
                choice_writeToFile->enable();
                file_writeToFileName->enable();
            }
            else
            {
                choice_writeToFile->disable();
                file_writeToFileName->disable();
            }
#endif
        }
        else
        {
            choice_functionType->disable();
            choice_coordDir->disable();
            dataPosSlider->disable();
            animationStepsSlider->disable();
            multiplierSlider->disable();

#ifdef WRITE_TO_FILE_ENABLE
            choice_writeToFile->disable();
            file_writeToFileName->disable();
#endif
        }
    }
    else if (stricmp(name, "Use map file") == 0)
    {
        use_mapFile = choice_useMapFile->getValue();
        if (use_mapFile)
            file_mapFile->enable();
        else
            file_mapFile->disable();
    }
    else if (stricmp(name, "Map file") == 0 && use_mapFile)
    {
        readMapFile(file_mapFile->getValue());
    }
    else if (stricmp(name, "Traceline ID") == 0)
    {
        int val = choice_traceLineID->getValue();
        char buf[80];

        //		coModule::sendInfo("Traceline %d of %d selected ", val,choice_traceLineID->getNumChoices() );

        if (val == choice_traceLineID->getNumChoices() - 2) //all tracelines selected
        {
            sprintf_s(buf, 80 * sizeof(char), "0-%d", val - 1);
        }
        else
        {
            sprintf_s(buf, 80 * sizeof(char), "%d", val);
        }

        traceLineSelection->setValue((const char *)buf);
    }
    else if (stricmp(name, "Traceline indices") == 0)
    {
        updateTraceLineInput = true;

        int n = nodes->isConnected();
        int t = traceLineIndices->isConnected();

        if (n && t)
        {
            choice_traceLineID->enable();
            traceLineSelection->enable();
            choice_useMapFile->enable();
            if (use_mapFile)
                file_mapFile->enable();
            else
                file_mapFile->disable();
        }
        else
        {
            choice_traceLineID->disable();
            choice_useMapFile->disable();
            traceLineSelection->disable();
            file_mapFile->disable();
        }
    }
    else if (stricmp(name, "Nodes") == 0)
    {
        updateNodesInput = true;

        if (nodes->isConnected() && traceLineIndices->isConnected())
        {
            choice_traceLineID->enable();
            traceLineSelection->enable();
            choice_useMapFile->enable();
            if (use_mapFile)
                file_mapFile->enable();
            else
                file_mapFile->disable();
        }
        else
        {
            choice_traceLineID->disable();
            choice_useMapFile->disable();
            traceLineSelection->disable();
            file_mapFile->disable();
        }
    }
    else
    {
    }
}

void TraceLineExtractor::writeDatasetToFile(FILE *f, coDoSet *dataset)
{
    fprintf(f, "    -1\n"); //write delimiter

    const char *datasetType = dataset->getAttribute("Type");

    if (datasetType)
    {
        if (stricmp(datasetType, "Dataset15") == 0)
        {
        }
        else if (stricmp(datasetType, "Dataset58") == 0)
        {
            fprintf(f, "    58\n");

            char *textPtr[2];
            int *intArr = NULL;
            float *floatArr[3];

            int ordinateDataType = 0;
            int abscissaSpacing = 0;

            //record 1
            ((coDoText *)dataset->getElement(0))->getAddress(&textPtr[0]);
            FortranData::WriteFortranDataFormat("1A80", textPtr[0]);

            //record 2
            ((coDoText *)dataset->getElement(1))->getAddress(&textPtr[0]);
            FortranData::WriteFortranDataFormat("1A80", textPtr[0]);

            //record 3
            ((coDoText *)dataset->getElement(2))->getAddress(&textPtr[0]);
            FortranData::WriteFortranDataFormat("1A80", textPtr[0]);

            //record 4
            ((coDoText *)dataset->getElement(3))->getAddress(&textPtr[0]);
            FortranData::WriteFortranDataFormat("1A80", textPtr[0]);

            //record 5
            ((coDoText *)dataset->getElement(4))->getAddress(&textPtr[0]);
            FortranData::WriteFortranDataFormat("1A80", textPtr[0]);

            //record 6 Format(2(I5,I10),2(1X,10A1,I10,I4))
            ((coDoIntArr *)((coDoSet *)dataset->getElement(5))->getElement(0))->getAddress(&intArr);
            ((coDoText *)((coDoSet *)dataset->getElement(5))->getElement(1))->getAddress(&textPtr[0]);
            ((coDoText *)((coDoSet *)dataset->getElement(5))->getElement(2))->getAddress(&textPtr[1]);

            FortranData::WriteFortranDataFormat("1I5,1I10,1I5,1I10,1X,1A10,1I10,1I4,1X,1A10,1I10,1I4",
                                                &intArr[0], &intArr[1], &intArr[2], &intArr[3],
                                                textPtr[0], &intArr[4], &intArr[5],
                                                textPtr[1], &intArr[6], &intArr[7]);
            //record 7

            ((coDoIntArr *)((coDoSet *)dataset->getElement(6))->getElement(0))->getAddress(&intArr);
            ((coDoFloat *)((coDoSet *)dataset->getElement(6))->getElement(1))->getAddress(&floatArr[0]);
            ((coDoFloat *)((coDoSet *)dataset->getElement(6))->getElement(2))->getAddress(&floatArr[1]);
            ((coDoFloat *)((coDoSet *)dataset->getElement(6))->getElement(3))->getAddress(&floatArr[2]);

            FortranData::WriteFortranDataFormat("3I10,3e13.5",
                                                &intArr[0], &intArr[1], &intArr[2],
                                                floatArr[0], floatArr[1], floatArr[2]);

            ordinateDataType = intArr[0]; //used for the switch instruction in record 12
            abscissaSpacing = intArr[2];

            //record 8 Format(I10,3I5,2(1X,20A1))
            ((coDoIntArr *)((coDoSet *)dataset->getElement(7))->getElement(0))->getAddress(&intArr);
            ((coDoText *)((coDoSet *)dataset->getElement(7))->getElement(1))->getAddress(&textPtr[0]);
            ((coDoText *)((coDoSet *)dataset->getElement(7))->getElement(2))->getAddress(&textPtr[1]);

            FortranData::WriteFortranDataFormat("1I10,3I5,1X,1A20,1X,1A20",
                                                &intArr[0], &intArr[1], &intArr[2], &intArr[3],
                                                textPtr[0], textPtr[1]);

            //record 9 Format(I10,3I5,2(1X,20A1))
            ((coDoIntArr *)((coDoSet *)dataset->getElement(8))->getElement(0))->getAddress(&intArr);
            ((coDoText *)((coDoSet *)dataset->getElement(8))->getElement(1))->getAddress(&textPtr[0]);
            ((coDoText *)((coDoSet *)dataset->getElement(8))->getElement(2))->getAddress(&textPtr[1]);

            FortranData::WriteFortranDataFormat("1I10,3I5,1X,1A20,1X,1A20",
                                                &intArr[0], &intArr[1], &intArr[2], &intArr[3],
                                                textPtr[0], textPtr[1]);

            //record 10 Format(I10,3I5,2(1X,20A1))
            ((coDoIntArr *)((coDoSet *)dataset->getElement(9))->getElement(0))->getAddress(&intArr);
            ((coDoText *)((coDoSet *)dataset->getElement(9))->getElement(1))->getAddress(&textPtr[0]);
            ((coDoText *)((coDoSet *)dataset->getElement(9))->getElement(2))->getAddress(&textPtr[1]);

            FortranData::WriteFortranDataFormat("1I10,3I5,1X,1A20,1X,1A20",
                                                &intArr[0], &intArr[1], &intArr[2], &intArr[3],
                                                textPtr[0], textPtr[1]);

            //record 11 Format(I10,3I5,2(1X,20A1))
            ((coDoIntArr *)((coDoSet *)dataset->getElement(10))->getElement(0))->getAddress(&intArr);
            ((coDoText *)((coDoSet *)dataset->getElement(10))->getElement(1))->getAddress(&textPtr[0]);
            ((coDoText *)((coDoSet *)dataset->getElement(10))->getElement(2))->getAddress(&textPtr[1]);

            FortranData::WriteFortranDataFormat("1I10,3I5,1X,1A20,1X,1A20",
                                                &intArr[0], &intArr[1], &intArr[2], &intArr[3],
                                                textPtr[0], textPtr[1]);

            //record 12

            ((coDoFloat *)dataset->getElement(11))->getAddress(&floatArr[0]);
            int floatArrSize = ((coDoFloat *)dataset->getElement(11))->getNumPoints();
            int dataWritten = 0;

            while (dataWritten < floatArrSize)
            {
                switch (abscissaSpacing)
                {
                case 0: //uneven abscissa spacing
                {
                    switch (ordinateDataType)
                    {
                    case 2: //real, single, uneven
                    {
                        FortranData::WriteFortranDataFormat("6e13.5",
                                                            &floatArr[0][dataWritten + dataWritten + 0], &floatArr[0][dataWritten + dataWritten + 1], &floatArr[0][dataWritten + dataWritten + 2], &floatArr[0][dataWritten + 3], &floatArr[0][dataWritten + 4], &floatArr[0][dataWritten + 5]);

                        dataWritten += 6;

                        break;
                    }
                    case 4: //real, double, uneven
                    {
                        double tmp[4];

                        tmp[0] = (double)floatArr[0][dataWritten + 0];
                        tmp[1] = (double)floatArr[0][dataWritten + 1];
                        tmp[2] = (double)floatArr[0][dataWritten + 2];
                        tmp[3] = (double)floatArr[0][dataWritten + 3];

                        FortranData::WriteFortranDataFormat("4e20.12",
                                                            &tmp[0], &tmp[1], &tmp[2], &tmp[3]);

                        dataWritten += 4;
                        break;
                    }
                    case 5: //complex, single, uneven
                    {

                        FortranData::WriteFortranDataFormat("6e13.5",
                                                            &floatArr[0][dataWritten + 0], &floatArr[0][dataWritten + 1], &floatArr[0][dataWritten + 2], &floatArr[0][dataWritten + 3], &floatArr[0][dataWritten + 4], &floatArr[0][dataWritten + 5]);

                        dataWritten += 6;
                        break;
                    }
                    case 6: //complex, double, uneven			//TODO: special case
                    {
                        double tmp[3];

                        tmp[0] = (double)floatArr[0][dataWritten + 0];
                        tmp[1] = (double)floatArr[0][dataWritten + 1];
                        tmp[2] = (double)floatArr[0][dataWritten + 2];

                        FortranData::WriteFortranDataFormat("3e20.12",
                                                            &tmp[0], &tmp[1], &tmp[2]);

                        dataWritten += 3; //format is "E13.5, 2E20.12", but we read all values as double

                        break;
                    }
                    default:
                    {
                        coModule::sendError("Wrong \"ordinate data type\" in dataset 58");
                        return;
                    }
                    }

                    break;
                }
                case 1: //even abscissa spacing
                {
                    switch (ordinateDataType)
                    {
                    case 2: //real, single, even
                    {
                        FortranData::WriteFortranDataFormat("6e13.5",
                                                            &floatArr[0][dataWritten + 0], &floatArr[0][dataWritten + 1], &floatArr[0][dataWritten + 2], &floatArr[0][dataWritten + 3], &floatArr[0][dataWritten + 4], &floatArr[0][dataWritten + 5]);

                        dataWritten += 6;
                        break;
                    }
                    case 4: //real, double, even
                    {

                        double tmp[4];

                        tmp[0] = (double)floatArr[0][dataWritten + 0];
                        tmp[1] = (double)floatArr[0][dataWritten + 1];
                        tmp[2] = (double)floatArr[0][dataWritten + 2];
                        tmp[3] = (double)floatArr[0][dataWritten + 3];

                        FortranData::WriteFortranDataFormat("4e20.12",
                                                            &tmp[0], &tmp[1], &tmp[2], &tmp[3]);

                        dataWritten += 4;
                        break;
                    }
                    case 5: //complex, single, even
                    {

                        FortranData::WriteFortranDataFormat("6e13.5",
                                                            &floatArr[0][dataWritten + 0], &floatArr[0][dataWritten + 1], &floatArr[0][dataWritten + 2], &floatArr[0][dataWritten + 3], &floatArr[0][dataWritten + 4], &floatArr[0][dataWritten + 5]);

                        dataWritten += 6;
                        break;
                    }
                    case 6: //complex, double, even
                    {
                        double tmp[4];

                        tmp[0] = (double)floatArr[0][dataWritten + 0];
                        tmp[1] = (double)floatArr[0][dataWritten + 1];
                        tmp[2] = (double)floatArr[0][dataWritten + 2];
                        tmp[3] = (double)floatArr[0][dataWritten + 3];

                        FortranData::WriteFortranDataFormat("4e20.12",
                                                            &tmp[0], &tmp[1], &tmp[2], &tmp[3]);

                        dataWritten += 4;
                        break;
                    }
                    default:
                    {
                        coModule::sendError("Wrong \"ordinate data type\" in dataset 58");
                    }
                    }

                    break;
                }

                default:
                {
                    coModule::sendError("Wrong \"abscissa spacing\" in dataset 58");
                }
                }
            }
        }
        else if (stricmp(datasetType, "Dataset151") == 0)
        {
            fprintf(f, "   151\n");

            char *textPtr[10];

            int *intArr = ((coDoIntArr *)dataset->getElement(10))->getAddress();

            ((coDoText *)dataset->getElement(0))->getAddress(&textPtr[0]);
            ((coDoText *)dataset->getElement(1))->getAddress(&textPtr[1]);
            ((coDoText *)dataset->getElement(2))->getAddress(&textPtr[2]);
            ((coDoText *)dataset->getElement(3))->getAddress(&textPtr[3]);
            ((coDoText *)dataset->getElement(4))->getAddress(&textPtr[4]);
            ((coDoText *)dataset->getElement(5))->getAddress(&textPtr[5]);
            ((coDoText *)dataset->getElement(6))->getAddress(&textPtr[6]);
            ((coDoText *)dataset->getElement(7))->getAddress(&textPtr[7]);
            ((coDoText *)dataset->getElement(8))->getAddress(&textPtr[8]);
            ((coDoText *)dataset->getElement(9))->getAddress(&textPtr[9]);

            FortranData::WriteFortranDataFormat("1A80", textPtr[0]);
            FortranData::WriteFortranDataFormat("1A80", textPtr[1]);
            FortranData::WriteFortranDataFormat("1A80", textPtr[2]);

            FortranData::WriteFortranDataFormat("1A10,1A10,3I10", textPtr[3], textPtr[4], &intArr[0], &intArr[1], &intArr[2]);
            FortranData::WriteFortranDataFormat("1A10,1A10", textPtr[5], textPtr[6]);
            FortranData::WriteFortranDataFormat("1A80", textPtr[7]);

            FortranData::WriteFortranDataFormat("1A10,1A10,5I5", textPtr[8], textPtr[9], &intArr[3], &intArr[4], &intArr[5], &intArr[6], &intArr[7]);
        }
        else if (stricmp(datasetType, "Dataset164") == 0)
        {
            fprintf(f, "   164\n");

            char *textPtr;
            int *intArr = NULL;
            float *floatArr = NULL;
            int floatArrSize = 0;
            double *doubleArr = NULL;

            ((coDoText *)dataset->getElement(0))->getAddress(&textPtr);
            ((coDoIntArr *)dataset->getElement(1))->getAddress(&intArr);
            ((coDoFloat *)dataset->getElement(2))->getAddress(&floatArr);
            floatArrSize = ((coDoFloat *)dataset->getElement(2))->getNumPoints();

            doubleArr = new double[floatArrSize];

            for (int i = 0; i < floatArrSize; i++)
            {
                doubleArr[i] = (double)floatArr[i];
            }

            FortranData::WriteFortranDataFormat("1I10,1A20,1I10", &intArr[0], textPtr, &intArr[1]);
            FortranData::WriteFortranDataFormat("3D25.17", &doubleArr[0], &doubleArr[1], &doubleArr[2]);
            FortranData::WriteFortranDataFormat("1D25.17", &doubleArr[3]);
        }
        else
        {
            coModule::sendInfo("writeDatasetToFile has found an unknown dataset dataset type");
        }
    }

    //	FortranData::WriteFortranDataFormat("1I10,3D25.17,1A20");
    fprintf(f, "    -1\n");
}

int TraceLineExtractor::readMapFile(const char *filename)
{
    FILE *file = NULL;

#ifdef _MSC_VER
    errno_t err = fopen_s(&file, filename, "rt");
    if (err != 0)
    {
        coModule::sendError("Cannot open file %d\n", err);
        return -1;
    }
#else
    file = fopen(filename, "r");
    if (file == NULL)
    {
        coModule::sendError("Cannot open file %s %d\n", filename, errno);
        return -1;
    }
#endif

    int n1 = -1;
    int n2 = -1;
    char string[80];
    char tmp[80];

    memset(string, 0, 80 * sizeof(char));
    memset(tmp, 0, 80 * sizeof(char));

    char seps[] = " :\n\r";
    char *token = NULL;
#ifdef WIN32
    char *next_token = NULL;
#endif

    while (!feof(file))
    {

        if (!fgets(string, 80 * sizeof(char), file))
            sendError("fgets failed in readMapFile\n");
        //		printf("input: %s", string);

        token = strtok_s(string, seps, &next_token);

        if (stricmp(token, "#") == 0)
        {
            //			printf("Comment\n");
        }
        else
        {

            //			printf("Token: ");
            while (token != NULL)
            {
                if (n1 == -1)
                {
                    sscanf_s(token, "%d", &n1);
                }
                else if (n2 == -1)
                {
                    sscanf_s(token, "%d", &n2);
                    if (n2 == -1)
                    {
                        strcpy_s(tmp, 80 * sizeof(char), token);
                    }
                }
                //				printf("%s, ", token);
                token = strtok_s(NULL, seps, &next_token);
            }

            if (n1 != -1 && n2 != -1)
            {
                //				printf("\nNUMBERS: %d, %d\n", n1, n2);
                //				printf("STRING: %s\n", tmp);

                if (substituteMap.find(n1) == substituteMap.end())
                {
                    coModule::sendWarning("readMapFile: Node %d - no corresponding node found", n1);
                }
                else
                {
                    substituteMap[n2] = substituteMap[n1];
                }
                n1 = -1;
                n2 = -1;
                memset(tmp, 0, 80 * sizeof(char));
            }
        }
    }

    use_mapFile = true;

    fclose(file);
    return 0;
}

/*
(Re)calculates a transformation matrix. 
mat = [ 0  1  2  3 
		4  5  6	 7
		8  9 10 11
	   12 13 14 15]
tx,ty,tz : Translations
rx,ry,rz : Rotations in radians

*/
/*
void TraceLineExtractor::calcMatrix(float **mat, float tx, float ty, float tz, float rx, float ry, float rz)
{
	*mat[3] += tx;
	*mat[7] += ty;
	*mat[11] += tz;

	*mat[15] = 1;
	
	float tmp[9];

	if (rx != 0 || ry != 8 || rz != 0)
	{
		float m0 = *mat[0];
		float m1 = *mat[1];
		float m2 = *mat[2];
		float m4 = *mat[4];
		float m5 = *mat[5];
		float m6 = *mat[6];

		if (rx != 0)
		{
			float c = cos(rx);
			float s = sin(rx);

			tmp[0] = 1;
			tmp[1] = 0;
			tmp[2] = 0;
			tmp[3] = 0;
			tmp[4] = c;
			tmp[5] = -s;
			tmp[6] = 0;
			tmp[7] = s;
			tmp[8] = c;

			*mat[0]  = tmp[0]*(*mat[0]) + tmp[1]*(*mat[4]) + tmp[2]*(*mat[8]);
			*mat[1]  = tmp[0]*(*mat[1]) + tmp[1]*(*mat[5]) + tmp[2]*(*mat[9]);
			*mat[2]  = tmp[0]*(*mat[2]) + tmp[1]*(*mat[6]) + tmp[2]*(*mat[10]);

			*mat[4]  = tmp[3]*(m0) + tmp[4]*(*mat[4]) + tmp[5]*(*mat[8]);
			*mat[5]  = tmp[3]*(m1) + tmp[4]*(*mat[5]) + tmp[5]*(*mat[9]);
			*mat[6]  = tmp[3]*(m2) + tmp[4]*(*mat[6]) + tmp[5]*(*mat[10]);

			*mat[8]  = tmp[6]*(m0) + tmp[7]*(m4) + tmp[8]*(*mat[8]);
			*mat[9]  = tmp[6]*(m1) + tmp[7]*(m5) + tmp[8]*(*mat[9]);
			*mat[10] = tmp[6]*(m2) + tmp[7]*(m6) + tmp[8]*(*mat[10]);
		}

		if (ry != 0)
		{

			float c = cos(ry);
			float s = sin(ry);

			tmp[0] = c;
			tmp[1] = 0;
			tmp[2] = s;
			tmp[3] = 0;
			tmp[4] = 1;
			tmp[5] = 0;
			tmp[6] = -s;
			tmp[7] = 0;
			tmp[8] = c;

			*mat[0]  = tmp[0]*(*mat[0]) + tmp[1]*(*mat[4]) + tmp[2]*(*mat[8]);
			*mat[1]  = tmp[0]*(*mat[1]) + tmp[1]*(*mat[5]) + tmp[2]*(*mat[9]);
			*mat[2]  = tmp[0]*(*mat[2]) + tmp[1]*(*mat[6]) + tmp[2]*(*mat[10]);

			*mat[4]  = tmp[3]*(m0) + tmp[4]*(*mat[4]) + tmp[5]*(*mat[8]);
			*mat[5]  = tmp[3]*(m1) + tmp[4]*(*mat[5]) + tmp[5]*(*mat[9]);
			*mat[6]  = tmp[3]*(m2) + tmp[4]*(*mat[6]) + tmp[5]*(*mat[10]);

			*mat[8]  = tmp[6]*(m0) + tmp[7]*(m4) + tmp[8]*(*mat[8]);
			*mat[9]  = tmp[6]*(m1) + tmp[7]*(m5) + tmp[8]*(*mat[9]);
			*mat[10] = tmp[6]*(m2) + tmp[7]*(m6) + tmp[8]*(*mat[10]);


		}

		if (rz != 0)
		{
			float c = cos(rz);
			float s = sin(rz);

			tmp[0] = c;
			tmp[1] = -s;
			tmp[2] = 0;
			tmp[3] = s;
			tmp[4] = c;
			tmp[5] = 0;
			tmp[6] = 0;
			tmp[7] = 0;
			tmp[8] = 1;

			*mat[0]  = tmp[0]*(*mat[0]) + tmp[1]*(*mat[4]) + tmp[2]*(*mat[8]);
			*mat[1]  = tmp[0]*(*mat[1]) + tmp[1]*(*mat[5]) + tmp[2]*(*mat[9]);
			*mat[2]  = tmp[0]*(*mat[2]) + tmp[1]*(*mat[6]) + tmp[2]*(*mat[10]);

			*mat[4]  = tmp[3]*(m0) + tmp[4]*(*mat[4]) + tmp[5]*(*mat[8]);
			*mat[5]  = tmp[3]*(m1) + tmp[4]*(*mat[5]) + tmp[5]*(*mat[9]);
			*mat[6]  = tmp[3]*(m2) + tmp[4]*(*mat[6]) + tmp[5]*(*mat[10]);

			*mat[8]  = tmp[6]*(m0) + tmp[7]*(m4) + tmp[8]*(*mat[8]);
			*mat[9]  = tmp[6]*(m1) + tmp[7]*(m5) + tmp[8]*(*mat[9]);
			*mat[10] = tmp[6]*(m2) + tmp[7]*(m6) + tmp[8]*(*mat[10]);


		}   
	}

}
*/

MODULE_MAIN(Tools, TraceLineExtractor)
