/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2007 CHE  ++
// ++ Reader for universal file format								      ++
// ++                                                                     ++
// ++ Author:      Andreas Funke                                          ++
// ++                                                                     ++
// ++																      ++
// ++																	  ++
// ++																	  ++
// ++				                                                      ++
// ++						                                              ++
// ++ Date:	16.04.2007						                              ++
// ++**********************************************************************/
#include "ReadUFF.h"
#include <util/common.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoText.h>
#include <do/coDoPoints.h>
//#define OLD_COVISE

#ifdef OLD_COVISE
#define addAttribute setAttribute
#endif
#ifndef _MSC_VER
#define sscanf_s sscanf
#define sprintf_s snprintf
#define strcpy_s(a, b, c) strncpy(a, c, b)
#define strncpy_s(a, b, c, d) strncpy(a, c, min(d, b))
#define stricmp strcasecmp
#endif
/*
read a line and search for delimiter, which is -1
returns only true if delimiter is on the fifth position

Input:	resetPosition	if true, then the position in the file is stored, and, if no delimiter is found, set back

NOTE:	resetPosition should be used, if GetDelimiter is used as a loop condition, e.g. in a while loop
*/
bool ReadUFF::GetDelimiter(bool resetPosition)
{
    char inputString[256];
    char c;
    unsigned int inputStringLength = 0;
    fpos_t pos;

    if (resetPosition)
        fgetpos(uffFile, &pos);

    while ((c = fgetc(uffFile)) != '\n' && inputStringLength < 256 && !feof(uffFile))
        inputString[inputStringLength++] = c;

    if (feof(uffFile))
        return false;

    inputString[inputStringLength++] = '\n';

    if (resetPosition)
        fsetpos(uffFile, &pos); //reset the position

    if (inputString[4] == '-' && inputString[5] == '1')
    {
        return true;
    }
    return false;
}

/*
return a dataset number
*/
int ReadUFF::GetDatasetNr()
{
    char inputString[256];
    char c;
    unsigned int inputStringLength = 0;
    int number = 0;

    while ((c = fgetc(uffFile)) != '\n' && inputStringLength < 256 && !feof(uffFile))
        inputString[inputStringLength++] = c;

    if (feof(uffFile))
        return 0;

    inputString[inputStringLength++] = '\n';

    sscanf_s(inputString, "%d", &number);

    return number;
}

ReadUFF::ReadUFF(int argc, char *argv[])
    : coModule(argc, argv, "Reader for universal files")
{
    char name[20];
    const char *none[] = { "----                                                                              " };
    fileName = addFileBrowserParam("File", "Location to the universal file");
    fileName->setValue("D:/Modalanalyse", "*.unv");
    fileName->show();

    for (int i = 0; i < NUM_PORTS; i++)
    {
        sprintf_s(name, 20 * sizeof(char), "DataPort%d", i);
        outputPorts[i] = addOutputPort(name, "Dataset", "specific datasets");
    }

    for (int i = 0; i < NUM_PORTS; i++)
    {
        sprintf_s(name, 20 * sizeof(char), "Port%d", i);
        portChoices[i] = addChoiceParam(name, "Choose dataset");
        portChoices[i]->setValue(1, none, 0);
    }

    forceUpdate = true;
    strcpy(loadedFile, "");
}

ReadUFF::~ReadUFF()
{
    coModule::sendInfo("Destructor called");
    Clean();
}

void ReadUFF::postInst()
{
    memset(&fileHeader, 0, sizeof(fileHeader));
    memset(&units, 0, sizeof(units));

    uffFile = NULL;
    traceLines = NULL;

    numNodes = 0;
    numTraceLines = 0;
}

int ReadUFF::compute(const char *port)
{
    (void)port;

    //if (!fileLoaded) ReadFile("D:/Dokumente und Einstellungen/apollon/Desktop/Modalanalyse/Geometrie.unv");

    if (strcmp(loadedFile, "") != 0)
    {
        for (int i = 0; i < NUM_PORTS; i++)
        {
            if (outputPorts[i]->isConnected())
            {
                coDoSet *params = PackDataset(portChoices[i]->getActLabel(), outputPorts[i]->getObjName());

                if (forceUpdate)
                    params->addAttribute("HasChanged", "true");
                else
                    params->addAttribute("HasChanged", "false");

                outputPorts[i]->setCurrentObject(params);
            }
        }

        forceUpdate = false;
        return SUCCESS;
    }
    else
    {
        coModule::sendError("Please open a file, before executing");
        return FAIL;
    }

    return SUCCESS;
}

/*
 * 
 */
void ReadUFF::param(const char *paramName, bool /*inMapLoading*/)
{

    if (stricmp(paramName, fileName->getName()) == 0)
    {
        if (stricmp(loadedFile, fileName->getValue()) != 0)
        {

            if (ReadFileHeaders(fileName->getValue()) != 0)
                return;

            forceUpdate = true;
            strcpy(loadedFile, fileName->getValue());

            if (choices.size() > 0)
            {
                char **choiceList = new char *[choices.size()];
                map<unsigned int, char *>::iterator it;
                int i = 0;

                for (it = choices.begin(); it != choices.end(); it++)
                {
                    choiceList[i] = new char[64];
                    strcpy_s(choiceList[i], 64 * sizeof(char), it->second);
                    i++;
                }

                for (i = 0; i < NUM_PORTS; i++)
                {
                    portChoices[i]->updateValue((int)choices.size(), (const char *const *)choiceList, 0);
                    coModule::sendInfo("%s", portChoices[i]->getValString());
                }
            }
        }
    }
}

coDoSet *ReadUFF::PackDataset(const char *datasetName, const char *portName)
{
    coDoSet *tmp = NULL;

    coObjInfo::setBaseName(portName);
    char *tok = strtok((char *)datasetName, " (");
    char token[20];
    strncpy_s(token, 20 * sizeof(char), tok, strlen(tok) - 1);
    token[strlen(tok) - 1] = '\0';

    if (stricmp(token, "dataset15") == 0)
    {
        /*
		struct dataset15
		{
		0 long label;				//node label
		1 long defcosysnum;		//definition coordinate system number
		2 long discosysnum;		//displacement coordinate system number
		3 int color;				//should also be _int64 but probably not necessary
		0 float p[3];			//coordinate p[0] = x, p[1] = y, p[2] = z
		};

		*/
        //
        // wer braucht das? coDoPoints *points = new coDoPoints( coObjInfo("points"), old_nodes.size() );

        vector<float> x;
        vector<float> y;
        vector<float> z;

        vector<int> labels;
        vector<int> defcosysnums;
        vector<int> discosysnums;
        vector<int> colors;

        coDistributedObject **doPtr = new coDistributedObject *[5];

        int arrSize = old_nodes.size();

        for (int i = 0; i < old_nodes.size(); i++)
        {
            x.push_back(old_nodes[i].record1.p[0]);
            y.push_back(old_nodes[i].record1.p[1]);
            z.push_back(old_nodes[i].record1.p[2]);

            labels.push_back(old_nodes[i].record1.label);
            defcosysnums.push_back(old_nodes[i].record1.defcosysnum);
            discosysnums.push_back(old_nodes[i].record1.discosysnum);
            colors.push_back(old_nodes[i].record1.color);
        }

        float *px = &x[0];
        float *py = &y[0];
        float *pz = &z[0];

        int *plabels = &labels[0];
        int *pdefcosysnums = &defcosysnums[0];
        int *pdiscosysnums = &discosysnums[0];
        int *pcolors = &colors[0];

        doPtr[0] = new coDoIntArr(coObjInfo("labels"), 1, &arrSize, plabels);
        doPtr[1] = new coDoIntArr(coObjInfo("defcosysnums"), 1, &arrSize, pdefcosysnums);
        doPtr[2] = new coDoIntArr(coObjInfo("discosysnums"), 1, &arrSize, pdiscosysnums);
        doPtr[3] = new coDoIntArr(coObjInfo("colors"), 1, &arrSize, pcolors);
        doPtr[4] = new coDoPoints(coObjInfo("points"), arrSize, px, py, pz);

        tmp = new coDoSet(coObjInfo(""), 5, doPtr);
        tmp->addAttribute("Type", "Dataset15");
    }
    else if (stricmp(token, "dataset55") == 0)
    {
        //TODO: PackDataset for dataset 55
    }

    else if (stricmp(token, "dataset58") == 0)
    {
        coDistributedObject **doPtr = new coDistributedObject *[12];
        coDistributedObject **tmpDO;

        coDoSet **container = new coDoSet *[DOFs.size()];
        int arraySize = 0;

        char setName[32];

        vector<int> tmpArr;
        int *ptmpArr = NULL;

        dataset58 *pDOFs = &DOFs[0];

        for (int i = 0; i < DOFs.size(); i++)
        {

            //record 1
            doPtr[0] = new coDoText(coObjInfo("idLine1"), 80, pDOFs[i].record1.idLine);

            //record 2
            doPtr[1] = new coDoText(coObjInfo("idLine2"), 80, pDOFs[i].record2.idLine);

            //record 3
            doPtr[2] = new coDoText(coObjInfo("idLine3"), 80, pDOFs[i].record3.idLine);

            //record 4
            doPtr[3] = new coDoText(coObjInfo("idLine4"), 80, pDOFs[i].record4.idLine);

            //record 5
            doPtr[4] = new coDoText(coObjInfo("idLine5"), 80, pDOFs[i].record5.idLine);

            //record 6

            tmpDO = new coDistributedObject *[3];

            tmpArr.push_back(pDOFs[i].record6.functionType);
            tmpArr.push_back(pDOFs[i].record6.functionID);
            tmpArr.push_back(pDOFs[i].record6.versionNumber);
            tmpArr.push_back(pDOFs[i].record6.loadCaseIdendificationNumber);
            tmpArr.push_back(pDOFs[i].record6.responseNode);
            tmpArr.push_back(pDOFs[i].record6.responseDirection);
            tmpArr.push_back(pDOFs[i].record6.referenceNode);
            tmpArr.push_back(pDOFs[i].record6.referenceDirection);

            arraySize = tmpArr.size();
            ptmpArr = &tmpArr[0];

            tmpDO[0] = new coDoIntArr(coObjInfo("intArray"), 1, &arraySize, ptmpArr);
            tmpDO[1] = new coDoText(coObjInfo("responseEntityName"), 10, pDOFs[i].record6.responseEntityName);
            tmpDO[2] = new coDoText(coObjInfo("referenceEntityName"), 10, pDOFs[i].record6.referenceEntityName);

            doPtr[5] = new coDoSet(coObjInfo("record6"), 3, tmpDO);

            //record 7

            tmpDO = new coDistributedObject *[4];

            tmpArr.clear();
            tmpArr.push_back(pDOFs[i].record7.ordinateDataType);
            tmpArr.push_back(pDOFs[i].record7.numDataPairs);
            tmpArr.push_back(pDOFs[i].record7.abscissaSpacing);

            arraySize = tmpArr.size();
            ptmpArr = &tmpArr[0];

            tmpDO[0] = new coDoIntArr(coObjInfo("intArray"), 1, &arraySize, ptmpArr);
            tmpDO[1] = new coDoFloat(coObjInfo("abscissaMininum"), 1, &pDOFs[i].record7.abscissaMinimum);
            tmpDO[2] = new coDoFloat(coObjInfo("abscissaIncrement"), 1, &pDOFs[i].record7.abscissaIncrement);
            tmpDO[3] = new coDoFloat(coObjInfo("zAxisValue"), 1, &pDOFs[i].record7.zAxisValue);

            doPtr[6] = new coDoSet(coObjInfo("record7"), 4, tmpDO);

            //record 8

            tmpDO = new coDistributedObject *[3];

            tmpArr.clear();
            tmpArr.push_back(pDOFs[i].record8.specificDataType);
            tmpArr.push_back(pDOFs[i].record8.lengthUnitsExponent);
            tmpArr.push_back(pDOFs[i].record8.forceUnitsExponent);
            tmpArr.push_back(pDOFs[i].record8.temperatureUnitsExponent);

            arraySize = tmpArr.size();
            ptmpArr = &tmpArr[0];

            tmpDO[0] = new coDoIntArr(coObjInfo("intArray"), 1, &arraySize, ptmpArr);
            tmpDO[1] = new coDoText(coObjInfo("axisLabel"), 20, pDOFs[i].record8.axisLabel);
            tmpDO[2] = new coDoText(coObjInfo("axisUnitsLabel"), 20, pDOFs[i].record8.axisUnitsLabel);

            doPtr[7] = new coDoSet(coObjInfo("record8"), 3, tmpDO);

            //record 9

            tmpDO = new coDistributedObject *[3];

            tmpArr.clear();

            tmpArr.push_back(pDOFs[i].record9.specificDataType);
            tmpArr.push_back(pDOFs[i].record9.lengthUnitsExponent);
            tmpArr.push_back(pDOFs[i].record9.forceUnitsExponent);
            tmpArr.push_back(pDOFs[i].record9.temperatureUnitsExponent);

            arraySize = tmpArr.size();
            ptmpArr = &tmpArr[0];

            tmpDO[0] = new coDoIntArr(coObjInfo("intArray"), 1, &arraySize, ptmpArr);
            tmpDO[1] = new coDoText(coObjInfo("axisLabel"), 20, pDOFs[i].record9.axisLabel);
            tmpDO[2] = new coDoText(coObjInfo("axisUnitsLabel"), 20, pDOFs[i].record9.axisUnitsLabel);

            doPtr[8] = new coDoSet(coObjInfo("record9"), 3, tmpDO);

            //record 10
            tmpDO = new coDistributedObject *[3];

            tmpArr.clear();

            tmpArr.push_back(pDOFs[i].record10.specificDataType);
            tmpArr.push_back(pDOFs[i].record10.lengthUnitsExponent);
            tmpArr.push_back(pDOFs[i].record10.forceUnitsExponent);
            tmpArr.push_back(pDOFs[i].record10.temperatureUnitsExponent);

            arraySize = tmpArr.size();
            ptmpArr = &tmpArr[0];

            tmpDO[0] = new coDoIntArr(coObjInfo("intArray"), 1, &arraySize, ptmpArr);
            tmpDO[1] = new coDoText(coObjInfo("axisLabel"), 20, pDOFs[i].record10.axisLabel);
            tmpDO[2] = new coDoText(coObjInfo("axisUnitsLabel"), 20, pDOFs[i].record10.axisUnitsLabel);

            doPtr[9] = new coDoSet(coObjInfo("record10"), 3, tmpDO);

            //record 11
            tmpDO = new coDistributedObject *[3];

            tmpArr.clear();

            tmpArr.push_back(pDOFs[i].record11.specificDataType);
            tmpArr.push_back(pDOFs[i].record11.lengthUnitsExponent);
            tmpArr.push_back(pDOFs[i].record11.forceUnitsExponent);
            tmpArr.push_back(pDOFs[i].record11.temperatureUnitsExponent);

            arraySize = tmpArr.size();
            ptmpArr = &tmpArr[0];

            tmpDO[0] = new coDoIntArr(coObjInfo("intArray"), 1, &arraySize, ptmpArr);
            tmpDO[1] = new coDoText(coObjInfo("axisLabel"), 20, pDOFs[i].record11.axisLabel);
            tmpDO[2] = new coDoText(coObjInfo("axisUnitsLabel"), 20, pDOFs[i].record11.axisUnitsLabel);

            doPtr[10] = new coDoSet(coObjInfo("record11"), 3, tmpDO);

            //record 12

            doPtr[11] = new coDoFloat(coObjInfo("data"), pDOFs[i].record12.num, pDOFs[i].record12.dataf);

            sprintf_s(setName, 32 * sizeof(char), "Dataset58_%d", i);

            container[i] = new coDoSet(coObjInfo(setName), 12, doPtr);
            container[i]->addAttribute("Type", "Dataset58");

            tmpArr.clear();
            arraySize = 0;
        }

        tmp = new coDoSet(coObjInfo(""), DOFs.size(), (coDistributedObject * const *)container);
        tmp->addAttribute("Type", "Dataset58");
    }
    else if (stricmp(token, "dataset82") == 0)
    {
        /*
		struct dataset82
		{						//long should be __int64 because 10-digit numbers should be possible to load, but __int64 is microsoft specific
		long traceLineNumber;	//Traceline number
		int	 numNodes;			//number of nodes defining traceline
		long color;				
		char idLine[80];		//Identification line
		long *traceNodes;		//nodes defining trace line
		};
		*/
        coDistributedObject **doPtr = new coDistributedObject *[3];
        coDistributedObject **doSetPtr = new coDistributedObject *[numTraceLines]; /*because we have multiple tracelines, 
																						  and each one is packed in a coDoSet*/
        vector<int> array;
        int arraySize = 0;
        int *parray = NULL;

        char lineName[80];

        for (int i = 0; i < numTraceLines; i++)
        {
            array.clear();

            array.push_back(traceLines[i].record1.traceLineNumber);
            array.push_back(traceLines[i].record1.numNodes);
            array.push_back(traceLines[i].record1.color);

            arraySize = array.size();
            parray = &array[0];

            doPtr[0] = new coDoIntArr(coObjInfo("traceNodes"), 1, &traceLines[i].record1.numNodes, traceLines[i].record3.traceNodes);
            doPtr[1] = new coDoIntArr(coObjInfo("intArray"), 1, &arraySize, parray);
            doPtr[2] = new coDoText(coObjInfo("idLine"), 80, traceLines[i].record2.idLine);

            memcpy(lineName, traceLines[i].record2.idLine, 80 * sizeof(char));

            doSetPtr[i] = new coDoSet(coObjInfo(lineName), 3, doPtr);
        }

        tmp = new coDoSet(coObjInfo(""), numTraceLines, doSetPtr);
        //		int i = tmp->getNumElements();
        tmp->addAttribute("Type", "dataset82");
    }
    else if (stricmp(token, "dataset151") == 0)
    {
        coDistributedObject **doPtr = new coDistributedObject *[11];

        /*	
		pos in int array
		0   unsigned int	DBVersion;				//database version
		1	unsigned int	DBSubversion;			//database subversion
		2	unsigned short	fileType;				//File type =0  Universal, =1  Archive, =2  Other
		3	unsigned int	release;				//Release which wrote universal file
		4	unsigned int	version;				//Version number
		5	unsigned int	hostID;					//Host ID MS1.  1-Vax/VMS 2-SGI, 3-HP7xx,HP-UX, 4-RS/6000, 5-Alp/VMS, 6-Sun, 7-Sony, 8-NEC, 9-Alp/OSF
		6	unsigned int	testID;					//Test ID
		7	unsigned int	releaseCounterPerHost;	//Release counter per host
		*/

        vector<int> array;
        array.push_back(fileHeader.record4.DBVersion);
        array.push_back(fileHeader.record4.DBSubversion);
        array.push_back(fileHeader.record4.fileType);
        array.push_back(fileHeader.record7.release);
        array.push_back(fileHeader.record7.version);
        array.push_back(fileHeader.record7.hostID);
        array.push_back(fileHeader.record7.testID);
        array.push_back(fileHeader.record7.releaseCounterPerHost);

        int matSize = array.size();
        int *parray = &array[0];

        coDoIntArr *intArr = new coDoIntArr(coObjInfo("intArray"), 1, &matSize, parray);

        doPtr[0] = new coDoText(coObjInfo("modelname"), 80, fileHeader.record1.modelName); //the models name
        doPtr[1] = new coDoText(coObjInfo("modelFileDesc"), 80, fileHeader.record2.modelFileDesc); //model file description
        doPtr[2] = new coDoText(coObjInfo("DBProgram"), 80, fileHeader.record3.DBProgram); //program which created DB;
        doPtr[3] = new coDoText(coObjInfo("DBCreateDate"), 10, fileHeader.record4.DBCreateDate); //date database created (DD-MMM-YY);
        doPtr[4] = new coDoText(coObjInfo("DBCreateTime"), 10, fileHeader.record4.DBCreateTime); //time		-"-			(HH:MM:SS);
        doPtr[5] = new coDoText(coObjInfo("DBLastSavedDate"), 10, fileHeader.record5.DBLastSavedDate); //date database last saved (DD-MMM-YY);
        doPtr[6] = new coDoText(coObjInfo("DBLastSavedTime"), 10, fileHeader.record5.DBLastSavedTime); //time database last saved (HH:MM:SS);
        doPtr[7] = new coDoText(coObjInfo("UFProgramName"), 80, fileHeader.record6.UFProgramName); //program which created universal file;
        doPtr[8] = new coDoText(coObjInfo("UFWrittenDate"), 10, fileHeader.record7.UFWrittenDate); //date universal file written (DD-MMM-YY);
        doPtr[9] = new coDoText(coObjInfo("UFWrittenTime"), 10, fileHeader.record7.UFWrittenTime); //time universal file written (HH:MM:SS);
        doPtr[10] = intArr;

        tmp = new coDoSet(coObjInfo(""), 11, doPtr);
        tmp->addAttribute("Type", "Dataset151");
    }
    else if (stricmp(token, "dataset164") == 0)
    {

        /*	
		struct dataset164
		{
		//actual double has not the required precision, but there is no data type with higher precision
		char unitsDesc[20];
		0 long unitsCode;
		1 long tempMode;

		0 double facForce;
		1 double facLength;
		2 double facTemp;
		3 double facTempOff;
		};
		*/
        coDistributedObject **doPtr = new coDistributedObject *[3];

        doPtr[0] = new coDoText(coObjInfo("unitsDesc"), 20, units.record1.unitsDesc);

        vector<int> intArray;
        intArray.push_back(units.record1.unitsCode);
        intArray.push_back(units.record1.tempMode);

        int matSize = intArray.size();
        int *pintArray = &intArray[0];

        doPtr[1] = new coDoIntArr(coObjInfo("intArray"), 1, &matSize, pintArray);

        vector<float> floatArray;
        floatArray.push_back(units.record2.facLength);
        floatArray.push_back(units.record2.facForce);
        floatArray.push_back(units.record2.facTemp);
        floatArray.push_back(units.record2.facTempOff);

        float *pfloatArray = &floatArray[0];
        doPtr[2] = new coDoFloat(coObjInfo("floatArray"), floatArray.size(), pfloatArray);

        tmp = new coDoSet(coObjInfo(""), 3, doPtr);
        tmp->addAttribute("Type", "Dataset164");
    }
    else if (stricmp(token, "dataset2411") == 0)
    {
        /*
		struct dataset2411
		{
			struct
			{
				long nodeLabel;
				long exportCoordSysNum;
				long dispCoordSysNum;
				long color;
			} record1; 

			struct
			{
				double coords[3];
			} record2; 
		};

		*/

        // wer braucht das? so geht das nicht coDoPoints *points = new coDoPoints( coObjInfo("points"), nodes.size() );

        vector<float> x;
        vector<float> y;
        vector<float> z;

        vector<int> labels;
        vector<int> exportCoordSysNums;
        vector<int> dispCoordSysNums;
        vector<int> colors;

        coDistributedObject **doPtr = new coDistributedObject *[5];

        int arrSize = nodes.size();

        for (int i = 0; i < nodes.size(); i++)
        {
            x.push_back((float)nodes[i].record2.coords[0]);
            y.push_back((float)nodes[i].record2.coords[1]);
            z.push_back((float)nodes[i].record2.coords[2]);

            labels.push_back(nodes[i].record1.nodeLabel);
            exportCoordSysNums.push_back(nodes[i].record1.exportCoordSysNum);
            dispCoordSysNums.push_back(nodes[i].record1.dispCoordSysNum);
            colors.push_back(nodes[i].record1.color);
        }

        float *px = &x[0];
        float *py = &y[0];
        float *pz = &z[0];

        int *plabels = &labels[0];
        int *pexpcosysnums = &exportCoordSysNums[0];
        int *pdiscosysnums = &dispCoordSysNums[0];
        int *pcolors = &colors[0];

        doPtr[0] = new coDoIntArr(coObjInfo("labels"), 1, &arrSize, plabels);
        doPtr[1] = new coDoIntArr(coObjInfo("exportCoordSysNums"), 1, &arrSize, pexpcosysnums);
        doPtr[2] = new coDoIntArr(coObjInfo("dispCoordSysNums"), 1, &arrSize, pdiscosysnums);
        doPtr[3] = new coDoIntArr(coObjInfo("colors"), 1, &arrSize, pcolors);
        doPtr[4] = new coDoPoints(coObjInfo("coords"), arrSize, px, py, pz);

        tmp = new coDoSet(coObjInfo(""), 5, doPtr);
        tmp->addAttribute("Type", "Dataset2411");
    }

    return tmp;
}

/*
 *  Just get the files dataset numbers, and only input the file, when executing
 *	the whole pipeline.
 *	Return values:
 *	0				everything ok
 *	1				error while opening file
 *	2				maybe the file is corrupted
 *
 */

int ReadUFF::ReadFileHeaders(const char *name)
{
    bool delimiter = false;
    unsigned int datasetNr = 0;

    choices.clear();
    Clean();

    fprintf(stdout, "ReadFileHeaders: Try to open %s\n", name);
    uffFile = fopen(name, "r");
    if (uffFile == NULL)
    {
        coModule::sendError("Unable to open file %s", name);
        return 1;
    }
    else
        fprintf(stdout, "ReadFileHeaders: %s opened\n", name);

    FortranData::setFile(uffFile);

    while (!feof(uffFile))
    {
        delimiter = false;
        while (!feof(uffFile))
        {
            delimiter = GetDelimiter(); //search for delimiter
            if (delimiter) //search for delimiter
                break;
        }

        if (!delimiter) //if eof is reached, before delimiter is found
        {
            if (!feof(uffFile))
            {
                coModule::sendError("Delimiter expected. Maybe file corrupt.");
                return 2;
            }
            else
                break;
        }

        datasetNr = GetDatasetNr();

        coModule::sendInfo("Dataset no %d found", datasetNr);

        switch (datasetNr)
        {
        case 15:
            choices[datasetNr] = new char[64];
            strncpy(choices[datasetNr], "dataset15 (Nodes)", 64 * sizeof(char));
            break;
        case 55:
            choices[datasetNr] = new char[64];
            strncpy(choices[datasetNr], "dataset55 (Data at nodes)", 64 * sizeof(char));
            break;
        case 58:
            choices[datasetNr] = new char[64];
            strncpy(choices[datasetNr], "dataset58 (Function at Nodal DOF)", 64 * sizeof(char));
            break;
        case 82:
            choices[datasetNr] = new char[64];
            strncpy(choices[datasetNr], "dataset82 (Tracelines)", 64 * sizeof(char));
            break;
        case 151:
            choices[datasetNr] = new char[64];
            strncpy(choices[datasetNr], "dataset151 (Fileheader)", 64 * sizeof(char));
            break;
        case 164:
            choices[datasetNr] = new char[64];
            strncpy(choices[datasetNr], "dataset164 (Unit description)", 64 * sizeof(char));
            break;
        case 2411:
            choices[datasetNr] = new char[64];
            strncpy(choices[datasetNr], "dataset2411 (Nodes)", 64 * sizeof(char));
            break;
        default:
            choices[datasetNr] = new char[64];
            sprintf_s(choices[datasetNr], 64 * sizeof(char), "dataset%d (unknown)", datasetNr);
            break;
        }

        delimiter = false;
        while (!feof(uffFile))
        {
            delimiter = GetDelimiter(); //search for delimiter
            if (delimiter)
                break;
        }

        if (!delimiter) //if eof is reached, before delimiter is found
        {
            if (!feof(uffFile))
            {
                coModule::sendError("Delimiter expected. Maybe file corrupt.");
                return 2;
            }
            else
                break;
        }
    }

    fclose(uffFile);
    coModule::sendInfo("File succesfully read");

    return 0;
}

int ReadUFF::ReadFile(const char *name)
{
    int err = 0;
    bool delimiter = false;
    unsigned int datasetNr = 0;

    choices.clear();
    Clean();

#ifndef _MSC_VER
    if ((uffFile = fopen(name, "r")) == NULL)
#else
    if ((err = fopen_s(&uffFile, name, "rt")) != 0)
#endif
    {
        coModule::sendError("Unable to open file %s", name);
        return err;
    }

    FortranData::setFile(uffFile);

    while (!feof(uffFile))
    {
        delimiter = false;
        while (!feof(uffFile))
        {
            delimiter = GetDelimiter(); //search for delimiter
            if (delimiter)
                break;
        }

        if (!delimiter) //if eof is reached, before delimiter is found
        {
            if (!feof(uffFile))
            {
                coModule::sendError("Delimiter expected. Maybe file corrupt.");
                return -1;
            }
            else
                break;
        }

        datasetNr = GetDatasetNr();

        if (datasetNr == 15)
        {
            coModule::sendInfo("Dataset no %d found", datasetNr);

            while (!GetDelimiter(true))
            {
                //				nodes = (dataset15*)realloc(nodes, (++numNodes)*sizeof(dataset15));
                dataset15 tmp;
                memset(&tmp, 0, sizeof(dataset15));

                if (FortranData::ReadFortranDataFormat("4I10,3E13.5", &tmp.record1.label,
                                                       &tmp.record1.defcosysnum,
                                                       &tmp.record1.discosysnum,
                                                       &tmp.record1.color,
                                                       &tmp.record1.p[0],
                                                       &tmp.record1.p[1],
                                                       &tmp.record1.p[2]) == 127) //the returned number is 1111111 -> all 7 variables could be assigned
                {
                    coModule::sendInfo("Node %d read", ++numNodes);
                }
                else
                {
                    coModule::sendWarning("Node %d incompletely read", ++numNodes);
                }

                old_nodes.push_back(tmp);
            }

            coModule::sendInfo("Dataset no %d read", datasetNr);

            choices[datasetNr] = new char[64];
            strcpy_s(choices[datasetNr], 64 * sizeof(char), "dataset15 (Nodes) obsolete");
        }
        else if (datasetNr == 55)
        {
            coModule::sendInfo("Dataset no %d found", datasetNr);

            dataset55 tmp;
            memset(&tmp, 0, sizeof(dataset55));

            FortranData::ReadFortranDataFormat("80A1", tmp.record1.idLine);
            FortranData::ReadFortranDataFormat("80A1", tmp.record2.idLine);
            FortranData::ReadFortranDataFormat("80A1", tmp.record3.idLine);
            FortranData::ReadFortranDataFormat("80A1", tmp.record4.idLine);
            FortranData::ReadFortranDataFormat("80A1", tmp.record5.idLine);

            FortranData::ReadFortranDataFormat("6I10", &tmp.record6.modelType,
                                               &tmp.record6.analysisType,
                                               &tmp.record6.dataCharacteristic,
                                               &tmp.record6.specificDataType,
                                               &tmp.record6.dataType,
                                               &tmp.record6.numberOfDataValuesPerNode);

            switch (tmp.record6.analysisType)
            {
            case 0:
            case 1: //Unknown, Static
            {
                FortranData::ReadFortranDataFormat("3I10", &tmp.record7.numberOfIntegerDataValues,
                                                   &tmp.record7.numberOfRealDataValues,
                                                   &tmp.record7.iParams[0]);

                FortranData::ReadFortranDataFormat("1E13.5", &tmp.record8.fParams[0]);
                break;
            }

            case 2: //Normal mode
            {
                FortranData::ReadFortranDataFormat("4I10", &tmp.record7.numberOfIntegerDataValues,
                                                   &tmp.record7.numberOfRealDataValues,
                                                   &tmp.record7.iParams[0],
                                                   &tmp.record7.iParams[1]);

                FortranData::ReadFortranDataFormat("4E13.5", &tmp.record8.fParams[0],
                                                   &tmp.record8.fParams[1],
                                                   &tmp.record8.fParams[2],
                                                   &tmp.record8.fParams[3]);
                break;
            }

            case 3: //Normal mode
            {
                FortranData::ReadFortranDataFormat("4I10", &tmp.record7.numberOfIntegerDataValues,
                                                   &tmp.record7.numberOfRealDataValues,
                                                   &tmp.record7.iParams[0],
                                                   &tmp.record7.iParams[1]);

                FortranData::ReadFortranDataFormat("6E13.5", &tmp.record8.fParams[0],
                                                   &tmp.record8.fParams[1],
                                                   &tmp.record8.fParams[2],
                                                   &tmp.record8.fParams[3],
                                                   &tmp.record8.fParams[4],
                                                   &tmp.record8.fParams[5]);
                break;
            }
            case 4:
            case 5: //Complex Eigenvalue, Frequency Response
            {
                FortranData::ReadFortranDataFormat("3I10", &tmp.record7.numberOfIntegerDataValues,
                                                   &tmp.record7.numberOfRealDataValues,
                                                   &tmp.record7.iParams[0]);

                FortranData::ReadFortranDataFormat("1E13.5", &tmp.record8.fParams[0]);
                break;
            }

            case 6: //Buckling
            {
                FortranData::ReadFortranDataFormat("4I10", &tmp.record7.numberOfIntegerDataValues,
                                                   &tmp.record7.numberOfRealDataValues,
                                                   &tmp.record7.iParams[0],
                                                   &tmp.record7.iParams[1]);

                FortranData::ReadFortranDataFormat("1E13.5", &tmp.record8.fParams[0]);
                break;
            }

            default:
            {
                coModule::sendError("Unknown analysis type");
                return -1;
            }
            }

            tmp.record9.nodeNumber = new int[old_nodes.size()];
            tmp.record10.dataValues = new float *[old_nodes.size()];

            tmp.record10.numDataValues = tmp.record6.numberOfDataValuesPerNode * ((tmp.record6.dataType == 5) ? 2 : 1); //complex or real data
            //TODO: how many nodes are in the dataset? The count of vector<dataset15> nodes?

            /*			char formatStr[10];
			sprintf_s(formatStr, 10*sizeof(char), "%dE13.5", tmp.record10.numDataValues
*/
            for (int i = 0; i < (int)old_nodes.size(); i++)
            {
                tmp.record10.dataValues[i] = new float[tmp.record10.numDataValues];
                FortranData::ReadFortranDataFormat("I10", &tmp.record9.nodeNumber[i]);

                switch (tmp.record10.numDataValues)
                {
                case 1:
                {
                    FortranData::ReadFortranDataFormat("1E13.5", &tmp.record10.dataValues[i][0]);
                    break;
                }
                case 2:
                {
                    FortranData::ReadFortranDataFormat("2E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1]);
                    break;
                }
                case 3:
                {
                    FortranData::ReadFortranDataFormat("3E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2]);
                    break;
                }
                case 4:
                {
                    FortranData::ReadFortranDataFormat("4E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2],
                                                       &tmp.record10.dataValues[i][3]);
                    break;
                }
                case 5:
                {
                    FortranData::ReadFortranDataFormat("5E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2],
                                                       &tmp.record10.dataValues[i][3],
                                                       &tmp.record10.dataValues[i][4]);
                    break;
                }

                case 6:
                {
                    FortranData::ReadFortranDataFormat("6E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2],
                                                       &tmp.record10.dataValues[i][3],
                                                       &tmp.record10.dataValues[i][4],
                                                       &tmp.record10.dataValues[i][5]);
                    break;
                }
                case 7:
                {
                    FortranData::ReadFortranDataFormat("7E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2],
                                                       &tmp.record10.dataValues[i][3],
                                                       &tmp.record10.dataValues[i][4],
                                                       &tmp.record10.dataValues[i][5],
                                                       &tmp.record10.dataValues[i][6]);
                    break;
                }
                case 8:
                {
                    FortranData::ReadFortranDataFormat("8E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2],
                                                       &tmp.record10.dataValues[i][3],
                                                       &tmp.record10.dataValues[i][4],
                                                       &tmp.record10.dataValues[i][5],
                                                       &tmp.record10.dataValues[i][6],
                                                       &tmp.record10.dataValues[i][7]);
                    break;
                }
                case 9:
                {
                    FortranData::ReadFortranDataFormat("9E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2],
                                                       &tmp.record10.dataValues[i][3],
                                                       &tmp.record10.dataValues[i][4],
                                                       &tmp.record10.dataValues[i][5],
                                                       &tmp.record10.dataValues[i][6],
                                                       &tmp.record10.dataValues[i][7],
                                                       &tmp.record10.dataValues[i][8]);
                    break;
                }
                case 10:
                {
                    FortranData::ReadFortranDataFormat("10E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2],
                                                       &tmp.record10.dataValues[i][3],
                                                       &tmp.record10.dataValues[i][4],
                                                       &tmp.record10.dataValues[i][5],
                                                       &tmp.record10.dataValues[i][6],
                                                       &tmp.record10.dataValues[i][7],
                                                       &tmp.record10.dataValues[i][8],
                                                       &tmp.record10.dataValues[i][9]);
                    break;
                }
                case 11:
                {
                    FortranData::ReadFortranDataFormat("11E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2],
                                                       &tmp.record10.dataValues[i][3],
                                                       &tmp.record10.dataValues[i][4],
                                                       &tmp.record10.dataValues[i][5],
                                                       &tmp.record10.dataValues[i][6],
                                                       &tmp.record10.dataValues[i][7],
                                                       &tmp.record10.dataValues[i][8],
                                                       &tmp.record10.dataValues[i][9],
                                                       &tmp.record10.dataValues[i][10]);
                    break;
                }
                case 12:
                {
                    FortranData::ReadFortranDataFormat("12E13.5", &tmp.record10.dataValues[i][0],
                                                       &tmp.record10.dataValues[i][1],
                                                       &tmp.record10.dataValues[i][2],
                                                       &tmp.record10.dataValues[i][3],
                                                       &tmp.record10.dataValues[i][4],
                                                       &tmp.record10.dataValues[i][5],
                                                       &tmp.record10.dataValues[i][6],
                                                       &tmp.record10.dataValues[i][7],
                                                       &tmp.record10.dataValues[i][8],
                                                       &tmp.record10.dataValues[i][9],
                                                       &tmp.record10.dataValues[i][10],
                                                       &tmp.record10.dataValues[i][11]);
                    break;
                }

                default:
                {
                    coModule::sendError("unsupported number of data values");
                    return -1;
                }

                    //TODO: extend the ReadFortranData, for supporting pointer, eg. for int *i = new int[4]; ReadFortranData("4I10", i);
                }
            }

            coModule::sendInfo("Dataset no %d read", datasetNr);

            choices[datasetNr] = new char[64];
            strcpy_s(choices[datasetNr], 64 * sizeof(char), "dataset55 (Data at nodes)");
        }
        else if (datasetNr == 58)
        {

            coModule::sendInfo("Dataset no %d found", datasetNr);

            dataset58 tmp;

            memset(&tmp, 0, sizeof(dataset58));

            FortranData::ReadFortranDataFormat("80A1", tmp.record1.idLine);
            FortranData::ReadFortranDataFormat("80A1", tmp.record2.idLine);
            FortranData::ReadFortranDataFormat("80A1", tmp.record3.idLine);
            FortranData::ReadFortranDataFormat("80A1", tmp.record4.idLine);
            FortranData::ReadFortranDataFormat("80A1", tmp.record5.idLine);

            //(2(I5,I10),2(1X,10A1,I10,I4))
            FortranData::ReadFortranDataFormat("1I5, 1I10, 1I5, 1I10, 1X, 10A1, 1I10, 1I4, 1X, 10A1, 1I10, 1I4",
                                               &tmp.record6.functionType,
                                               &tmp.record6.functionID,
                                               &tmp.record6.versionNumber,
                                               &tmp.record6.loadCaseIdendificationNumber,
                                               tmp.record6.responseEntityName,
                                               &tmp.record6.responseNode,
                                               &tmp.record6.responseDirection,
                                               tmp.record6.referenceEntityName,
                                               &tmp.record6.referenceNode,
                                               &tmp.record6.referenceDirection);

            //(3I10,3E13.5)
            FortranData::ReadFortranDataFormat("3I10,3E13.5",
                                               &tmp.record7.ordinateDataType,
                                               &tmp.record7.numDataPairs,
                                               &tmp.record7.abscissaSpacing,
                                               &tmp.record7.abscissaMinimum,
                                               &tmp.record7.abscissaIncrement,
                                               &tmp.record7.zAxisValue);

            //(I10,3I5,2(1X,20A1))
            FortranData::ReadFortranDataFormat("1I10, 3I5, 1X, 20A1, 1X, 20A1",
                                               &tmp.record8.specificDataType,
                                               &tmp.record8.lengthUnitsExponent,
                                               &tmp.record8.forceUnitsExponent,
                                               &tmp.record8.temperatureUnitsExponent,
                                               tmp.record8.axisLabel,
                                               tmp.record8.axisUnitsLabel);

            FortranData::ReadFortranDataFormat("1I10, 3I5, 1X, 20A1, 1X, 20A1",
                                               &tmp.record9.specificDataType,
                                               &tmp.record9.lengthUnitsExponent,
                                               &tmp.record9.forceUnitsExponent,
                                               &tmp.record9.temperatureUnitsExponent,
                                               tmp.record9.axisLabel,
                                               tmp.record9.axisUnitsLabel);

            FortranData::ReadFortranDataFormat("1I10, 3I5, 1X, 20A1, 1X, 20A1",
                                               &tmp.record10.specificDataType,
                                               &tmp.record10.lengthUnitsExponent,
                                               &tmp.record10.forceUnitsExponent,
                                               &tmp.record10.temperatureUnitsExponent,
                                               tmp.record10.axisLabel,
                                               tmp.record10.axisUnitsLabel);

            FortranData::ReadFortranDataFormat("1I10, 3I5, 1X, 20A1, 1X, 20A1",
                                               &tmp.record11.specificDataType,
                                               &tmp.record11.lengthUnitsExponent,
                                               &tmp.record11.forceUnitsExponent,
                                               &tmp.record11.temperatureUnitsExponent,
                                               tmp.record11.axisLabel,
                                               tmp.record11.axisUnitsLabel);

            tmp.record12.num = tmp.record7.numDataPairs * ((tmp.record7.ordinateDataType == 5 || tmp.record7.ordinateDataType == 6) ? 2 : 1);

            if (tmp.record7.ordinateDataType == 2 || tmp.record7.ordinateDataType == 5) //single precision
            {
                tmp.record12.dataf = new float[tmp.record12.num];
                memset(tmp.record12.dataf, 0, tmp.record12.num * sizeof(float));
            }
            else if (tmp.record7.ordinateDataType == 5 || tmp.record7.ordinateDataType == 6) //double precision
            {
                tmp.record12.datad = new double[tmp.record12.num];
                memset(tmp.record12.datad, 0, tmp.record12.num * sizeof(double));
            }
            else
            {
                coModule::sendError("Wrong \"ordinate data type\" in dataset 58");
                return -1;
            }

            int actDataPos = 0;

            //TODO: add a choice to read single data from the file, not all data, at one time

            while (!GetDelimiter(true))
            {

                switch (tmp.record7.abscissaSpacing)
                {
                case 0: //uneven abscissa spacing
                {
                    switch (tmp.record7.ordinateDataType)
                    {
                    case 2: //real, single, uneven
                    {
                        if (FortranData::ReadFortranDataFormat("6E13.5",
                                                               &tmp.record12.dataf[actDataPos + 0],
                                                               &tmp.record12.dataf[actDataPos + 1],
                                                               &tmp.record12.dataf[actDataPos + 2],
                                                               &tmp.record12.dataf[actDataPos + 3],
                                                               &tmp.record12.dataf[actDataPos + 4],
                                                               &tmp.record12.dataf[actDataPos + 5]) != 63)
                        {
                            coModule::sendWarning("Could not assign all data values in set %d line %d", (int)DOFs.size(), (int)(actDataPos / 6));
                        }

                        actDataPos += 6;

                        break;
                    }
                    case 4: //real, double, uneven
                    {

                        if (FortranData::ReadFortranDataFormat("4E20.12",
                                                               &tmp.record12.datad[actDataPos + 0],
                                                               &tmp.record12.datad[actDataPos + 1],
                                                               &tmp.record12.datad[actDataPos + 2],
                                                               &tmp.record12.datad[actDataPos + 3]) != 15)
                        {
                            coModule::sendWarning("Could not assign all data values in set %d line %d", (int)DOFs.size(), (int)(actDataPos / 6));
                        }

                        actDataPos += 4;
                        break;
                    }
                    case 5: //complex, single, uneven
                    {

                        if (FortranData::ReadFortranDataFormat("6E13.5",
                                                               &tmp.record12.dataf[actDataPos + 0],
                                                               &tmp.record12.dataf[actDataPos + 1],
                                                               &tmp.record12.dataf[actDataPos + 2],
                                                               &tmp.record12.dataf[actDataPos + 3],
                                                               &tmp.record12.dataf[actDataPos + 4],
                                                               &tmp.record12.dataf[actDataPos + 5]) != 63)
                        {
                            coModule::sendWarning("Could not assign all data values in set %d line %d", (int)DOFs.size(), (int)(actDataPos / 6));
                        }

                        actDataPos += 6;

                        break;
                    }
                    case 6: //complex, double, uneven			//TODO: special case
                    {

                        if (FortranData::ReadFortranDataFormat("3E20.12",
                                                               &tmp.record12.datad[actDataPos + 0],
                                                               &tmp.record12.datad[actDataPos + 1],
                                                               &tmp.record12.datad[actDataPos + 2]) != 7)
                        {
                            coModule::sendWarning("Could not assign all data values in set %d line %d", (int)DOFs.size(), (int)(actDataPos / 6));
                        }

                        actDataPos += 3;

                        break;
                    }
                    default:
                    {
                        coModule::sendError("Wrong \"ordinate data type\" in dataset 58");
                        return -1;
                    }
                    }

                    break;
                }
                case 1: //even abscissa spacing
                {
                    switch (tmp.record7.ordinateDataType)
                    {
                    case 2: //real, single, even
                    {

                        if (FortranData::ReadFortranDataFormat("6E13.5",
                                                               &tmp.record12.dataf[actDataPos + 0],
                                                               &tmp.record12.dataf[actDataPos + 1],
                                                               &tmp.record12.dataf[actDataPos + 2],
                                                               &tmp.record12.dataf[actDataPos + 3],
                                                               &tmp.record12.dataf[actDataPos + 4],
                                                               &tmp.record12.dataf[actDataPos + 5]) != 63)
                        {
                            coModule::sendWarning("Could not assign all data values in set %d line %d", (int)DOFs.size(), (int)(actDataPos / 6));
                        }

                        actDataPos += 6;

                        break;
                    }
                    case 4: //real, double, even
                    {

                        if (FortranData::ReadFortranDataFormat("4E20.12",
                                                               &tmp.record12.datad[actDataPos + 0],
                                                               &tmp.record12.datad[actDataPos + 1],
                                                               &tmp.record12.datad[actDataPos + 2],
                                                               &tmp.record12.datad[actDataPos + 3]) != 15)
                        {
                            coModule::sendWarning("Could not assign all data values in set %d line %d", (int)DOFs.size(), (int)(actDataPos / 6));
                        }

                        actDataPos += 4;

                        break;
                    }
                    case 5: //complex, single, even
                    {

                        if (FortranData::ReadFortranDataFormat("6E13.5",
                                                               &tmp.record12.dataf[actDataPos + 0],
                                                               &tmp.record12.dataf[actDataPos + 1],
                                                               &tmp.record12.dataf[actDataPos + 2],
                                                               &tmp.record12.dataf[actDataPos + 3],
                                                               &tmp.record12.dataf[actDataPos + 4],
                                                               &tmp.record12.dataf[actDataPos + 5]) != 63)
                        {
                            coModule::sendWarning("Could not assign all data values in set %d line %d", (int)DOFs.size(), (int)(actDataPos / 6));
                        }

                        actDataPos += 6;

                        break;
                    }
                    case 6: //complex, double, even
                    {

                        if (FortranData::ReadFortranDataFormat("4E20.12",
                                                               &tmp.record12.datad[actDataPos + 0],
                                                               &tmp.record12.datad[actDataPos + 1],
                                                               &tmp.record12.datad[actDataPos + 2],
                                                               &tmp.record12.datad[actDataPos + 3]) != 15)
                        {
                            coModule::sendWarning("Could not assign all data values in set %d line %d", (int)DOFs.size(), (int)(actDataPos / 6));
                        }

                        actDataPos += 4;

                        break;
                    }
                    default:
                    {
                        coModule::sendError("Wrong \"ordinate data type\" in dataset 58");
                        return -1;
                    }
                    }

                    break;
                }

                default:
                {
                    coModule::sendError("Wrong \"abscissa spacing\" in dataset 58");
                    return -1;
                }
                }
            }

            coModule::sendInfo("Dataset no %d read", datasetNr);

            DOFs.push_back(tmp);

            choices[datasetNr] = new char[64];
            strcpy_s(choices[datasetNr], 64 * sizeof(char), "dataset58 (Function at Nodal DOF)");
        }
        else if (datasetNr == 82)
        {
            coModule::sendInfo("Dataset no %d found", datasetNr);

            traceLines = (dataset82 *)realloc(traceLines, (++numTraceLines) * sizeof(dataset82));
            memset(&traceLines[numTraceLines - 1], 0, sizeof(dataset82));

            FortranData::ReadFortranDataFormat("3I10", &traceLines[numTraceLines - 1].record1.traceLineNumber,
                                               &traceLines[numTraceLines - 1].record1.numNodes,
                                               &traceLines[numTraceLines - 1].record1.color);
            FortranData::ReadFortranDataFormat("80A1", traceLines[numTraceLines - 1].record2.idLine);

            unsigned int numLines = (int)ceilf(float(traceLines[numTraceLines - 1].record1.numNodes) / 8);

            traceLines[numTraceLines - 1].record3.traceNodes = (int *)malloc((numLines * 8) * sizeof(int));

            for (int i = 0; i < numLines; i++)
            {
                if (FortranData::ReadFortranDataFormat("8I10", &traceLines[numTraceLines - 1].record3.traceNodes[8 * i + 0],
                                                       &traceLines[numTraceLines - 1].record3.traceNodes[8 * i + 1],
                                                       &traceLines[numTraceLines - 1].record3.traceNodes[8 * i + 2],
                                                       &traceLines[numTraceLines - 1].record3.traceNodes[8 * i + 3],
                                                       &traceLines[numTraceLines - 1].record3.traceNodes[8 * i + 4],
                                                       &traceLines[numTraceLines - 1].record3.traceNodes[8 * i + 5],
                                                       &traceLines[numTraceLines - 1].record3.traceNodes[8 * i + 6],
                                                       &traceLines[numTraceLines - 1].record3.traceNodes[8 * i + 7]) != 255) //255 = 11111111b
                {
                    coModule::sendWarning("Traceline %d incompletely read", i);
                }
            }

            coModule::sendInfo("Traceline %d successfully read", numTraceLines);

            coModule::sendInfo("Dataset no %d read", datasetNr);

            choices[datasetNr] = new char[64];
            strcpy_s(choices[datasetNr], 64 * sizeof(char), "dataset82 (Tracelines)");
        }
        else if (datasetNr == 151) //Header dataset
        {
            coModule::sendInfo("Dataset no %d found", datasetNr);

            memset(&fileHeader, 0, sizeof(dataset151));

            FortranData::ReadFortranDataFormat("80A1", fileHeader.record1.modelName);
            FortranData::ReadFortranDataFormat("80A1", fileHeader.record2.modelFileDesc);
            FortranData::ReadFortranDataFormat("80A1", fileHeader.record3.DBProgram);
            FortranData::ReadFortranDataFormat("10A1,10A1,3I10", fileHeader.record4.DBCreateDate,
                                               fileHeader.record4.DBCreateTime,
                                               &fileHeader.record4.DBVersion,
                                               &fileHeader.record4.DBSubversion,
                                               &fileHeader.record4.fileType);
            FortranData::ReadFortranDataFormat("10A1,10A1", fileHeader.record5.DBLastSavedDate,
                                               fileHeader.record5.DBLastSavedTime);
            FortranData::ReadFortranDataFormat("80A1", fileHeader.record6.UFProgramName);
            FortranData::ReadFortranDataFormat("10A1,10A1,5I5", fileHeader.record7.UFWrittenDate,
                                               fileHeader.record7.UFWrittenTime,
                                               &fileHeader.record7.release,
                                               &fileHeader.record7.version,
                                               &fileHeader.record7.hostID,
                                               &fileHeader.record7.testID,
                                               &fileHeader.record7.releaseCounterPerHost);

            coModule::sendInfo("Header dataset read");

            coModule::sendInfo("Dataset no %d read", datasetNr);

            choices[datasetNr] = new char[64];
            strcpy_s(choices[datasetNr], 64 * sizeof(char), "dataset151 (Fileheader)");
        }
        else if (datasetNr == 164) //Units dataset
        {
            coModule::sendInfo("Dataset no %d found", datasetNr);

            memset(&units, 0, sizeof(dataset164));

            FortranData::ReadFortranDataFormat("1I10,20A1,1I10", &units.record1.unitsCode,
                                               units.record1.unitsDesc,
                                               &units.record1.tempMode);

            FortranData::ReadFortranDataFormat("3D25.17", &units.record2.facLength,
                                               &units.record2.facForce,
                                               &units.record2.facTemp);
            FortranData::ReadFortranDataFormat("1D25.17", &units.record2.facTempOff);

            coModule::sendInfo("Unit dataset read");

            coModule::sendInfo("Dataset no %d read", datasetNr);

            choices[datasetNr] = new char[64];
            strcpy_s(choices[datasetNr], 64 * sizeof(char), "dataset164 (Unit description)");
        }
        else if (datasetNr == 2411)
        {
            coModule::sendInfo("Dataset no %d found", datasetNr);

            while (!GetDelimiter(true))
            {
                //				nodes = (dataset15*)realloc(nodes, (++numNodes)*sizeof(dataset15));
                dataset2411 tmp;
                memset(&tmp, 0, sizeof(dataset2411));

                if (FortranData::ReadFortranDataFormat("4I10", &tmp.record1.nodeLabel,
                                                       &tmp.record1.exportCoordSysNum,
                                                       &tmp.record1.dispCoordSysNum,
                                                       &tmp.record1.color) == 15) //the returned number is 1111 -> all 4 variables could be assigned
                {
                    coModule::sendInfo("Node %d read", ++numNodes);
                }
                else
                {
                    coModule::sendWarning("Node %d incompletely read", ++numNodes);
                }

                if (FortranData::ReadFortranDataFormat("3D25.16",
                                                       &tmp.record2.coords[0],
                                                       &tmp.record2.coords[1],
                                                       &tmp.record2.coords[2]) != 7) //the returned number is 111 -> all 3 variables could be assigned
                {
                    coModule::sendInfo("Coordinates read");
                }
                else
                {
                    coModule::sendWarning("Coordinates incompletely read");
                }

                nodes.push_back(tmp);
            }

            coModule::sendInfo("Dataset no %d read", datasetNr);

            choices[datasetNr] = new char[64];
            strcpy_s(choices[datasetNr], 64 * sizeof(char), "dataset2411 (Nodes)");
        }

        /*else 
		if (datasetNr == XXX)
		{
			coModule::sendInfo("Dataset no %d found", datasetNr);

			//TODO: insert your own dataset reading here

			coModule::sendInfo("Dataset no %d read", datasetNr);

			choices[datasetNr] = new char[64];
			strcpy_s(choices[datasetNr], 64*sizeof(char), "datasetXXX (Tracelines)");

		}
		*/
        else
        {
            if (!feof(uffFile))
                coModule::sendWarning("Unknown dataset found. Dataset no %d", datasetNr);
        }

        delimiter = false;
        while (!feof(uffFile))
        {
            delimiter = GetDelimiter(); //search for delimiter
            if (delimiter)
                break;
        }

        if (!delimiter) //if eof is reached, before delimiter is found
        {
            if (!feof(uffFile))
            {
                coModule::sendError("Delimiter expected. Maybe file corrupt.");
                return -1;
            }
            else
                break;
        }
    }

    fclose(uffFile);
    coModule::sendInfo("File succesfully read");

    return 0;
}

void ReadUFF::Clean()
{
    old_nodes.clear();
    numNodes = 0;

    for (int i = 0; i < numTraceLines; i++)
    {
        free(traceLines[i].record3.traceNodes);
    }

    DOFs.clear();

    map<unsigned int, char *>::iterator it;

    for (it = choices.begin(); it != choices.end(); it++)
        delete it->second;

    free(traceLines);
    traceLines = NULL;
    numTraceLines = 0;
}

MODULE_MAIN(IO, ReadUFF)
