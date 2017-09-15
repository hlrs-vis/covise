/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParserAscii.h
 * a results file parser for ASCII files.
 */

#include "ResultsFileParserAscii.h" // a results file parser for ASCII files.
#include "errorinfo.h" // a container for error data.
#include <do/coDoData.h>

ResultsFileParserAscii::ResultsFileParserAscii(OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
    , _noOfTimeSteps(0)
    , _currentDataTypeNo(0)
    , _xData(0)
    , _yData(0)
    , _zData(0)
{
    int i;
    for (i = 0; i < NUMRES; i++)
    {
        _names[i] = 0;
    }
}

void ResultsFileParserAscii::parseResultsFile(std::string filename,
                                              const int &noOfTimeStepToSkip,
                                              const int &noOfTimeStepsToParse,
                                              MeshDataTrans *meshDataTrans,
                                              std::string baseName)
{
    _maxNodeNo = meshDataTrans->getMaxNodeNo(0);

    _file = fopen(filename.c_str(), "r");
    if (!_file)
    {
        ERROR1("cannot open results file ", filename, _outputHandler);
    }

    _displacementNo.clear();
    _xData.reserve(_maxNodeNo); // _numPoints
    _yData.reserve(_maxNodeNo);
    _zData.reserve(_maxNodeNo);

    _noOfTimeSteps = 0;
    _skipped = 0;
    _pushedBack = false;

    //_haveDisplacements=false;
    while (!feof(_file))
    {
        getLine();
        if (strncmp(_c, "SUBCASE", 7) == 0)
        {
            // new _line node
            _displacementNo.push_back(-1);
            _currentDataTypeNo = 0;
            char buf[1000];
            if (_skipped < noOfTimeStepToSkip) //_p_skip->getValue()
            {
                _skipped++;
                _doSkip = true;
                sprintf(buf, "Skipping %d", _skipped);
                _outputHandler->displayString(buf);
            }
            else
            {
                _skipped = 0;
                _doSkip = false;
                sprintf(buf, "Reading TimeStep %d", _noOfTimeSteps);
                _outputHandler->displayString(buf);
            }
            while (!feof(_file))
            {
                getLine();
                if (strncmp(_c, "RESULTS", 7) == 0)
                {
                    _names[_currentDataTypeNo] = new char[strlen(_c + 8) + 1];
                    strcpy(_names[_currentDataTypeNo], _c + 8);
                    getLine();

                    char buf[1000];
                    sprintf(buf, "%s_%d_%d", baseName.c_str(), _currentDataTypeNo, _noOfTimeSteps);
                    //sprintf(buf,"%d_%d", _currentDataTypeNo, _noOfTimeSteps);
                    if (_c[0] == 'V')
                    {
                        pushBack();
                        _dataObjects[_currentDataTypeNo][_noOfTimeSteps] = readVec(buf, _noOfTimeSteps);
                        _dimension[_currentDataTypeNo][_noOfTimeSteps] = ResultsFileData::VECTOR;
                    }
                    else if (_c[0] == 'S')
                    {
                        pushBack();
                        _dataObjects[_currentDataTypeNo][_noOfTimeSteps] = readScal(buf);
                        _dimension[_currentDataTypeNo][_noOfTimeSteps] = ResultsFileData::SCALAR;
                    }
                    else if (strncmp(_c, "DISPLACE", 8) == 0) // vector
                    {
                        pushBack();
                        _dataObjects[_currentDataTypeNo][_noOfTimeSteps] = readVec(buf, _noOfTimeSteps);
                        _dimension[_currentDataTypeNo][_noOfTimeSteps] = ResultsFileData::VECTOR;
                    }
                    else if (strncmp(_c, "NSTRESS", 7) == 0) // scalar
                    {
                        pushBack();
                        _dataObjects[_currentDataTypeNo][_noOfTimeSteps] = readScal(buf);
                        _dimension[_currentDataTypeNo][_noOfTimeSteps] = ResultsFileData::SCALAR;
                    }
                    _currentDataTypeNo++;
                }
                else if (strncmp(_c, "SUBCASE", 7) == 0) // new timestep, so quit this loop
                {
                    pushBack();
                    break;
                }
            }
            if (_dataObjects[0][_noOfTimeSteps]) // if we actually have data for this timestep go to next timestep
            {
                _noOfTimeSteps++;
            }
        }
        if (_noOfTimeSteps >= noOfTimeStepsToParse) //_p_numt->getValue()
        {
            break;
        }
    }
    fclose(_file);
    debug();
}

// --------------------------------------------------------------------------------------------------------------
//                                          parsing subroutines
// --------------------------------------------------------------------------------------------------------------

void ResultsFileParserAscii::getLine()
{
    if (!_pushedBack)
    {
        if (fgets(_line, LINE_SIZE, _file) == NULL)
        {
            //   fprintf(stderr, "ReadFamu::getLine(): fgets failed\n" );
        }
    }
    else
    {
        _pushedBack = false;
    }
    _c = _line;
    while (*_c != '\0' && isspace(*_c))
    {
        _c++;
    }
}

void ResultsFileParserAscii::pushBack()
{
    _pushedBack = true;
}

coDistributedObject *ResultsFileParserAscii::readVec(const char *name,
                                                     int timeStepNo)
{
    while (!feof(_file))
    {
        getLine();
        if (_c[0] == 'V') // vector
        {
            int nodeNum;
            float xc, yc, zc;
            int iret = sscanf(_c + 1, "%d %f %f %f", &nodeNum, &xc, &yc, &zc);
            if (iret != 4)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            _xData[nodeNum - 1] = xc;
            _yData[nodeNum - 1] = yc;
            _zData[nodeNum - 1] = zc;
        }
        else if (strncmp(_c, "DISPLACE", 8) == 0) // vector
        {
            int nodeNum;
            float xc, yc, zc;
            int iret = sscanf(_c + 8, "%d %f %f %f", &nodeNum, &xc, &yc, &zc);
            if (iret != 4)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            _xData[nodeNum - 1] = xc;
            _yData[nodeNum - 1] = yc;
            _zData[nodeNum - 1] = zc;
        }
        else
        {
            break;
        }
    }
    if (_doSkip)
    {
        return NULL;
    }
    ASSERT0(_xData.size() >= _maxNodeNo, "sorry, and internal error occured.", _outputHandler);
    coDoVec3 *dataObj = new coDoVec3(name, _xData.size(), &_xData[0], &_yData[0], &_zData[0]);
    if (strncmp(_names[_currentDataTypeNo], "Displace", 8) == 0)
    {
        //_displacementDataTypeNo = _currentDataTypeNo;
        _displacementNo[timeStepNo] = _currentDataTypeNo;
        //_haveDisplacements = true;
    }
    return dataObj;
}

coDistributedObject *ResultsFileParserAscii::readScal(const char *name)
{
    while (!feof(_file))
    {
        getLine();
        if (_c[0] == 'S') // scalar
        {

            int nodeNum;
            float xc;
            int iret = sscanf(_c + 1, "%d %f", &nodeNum, &xc);
            if (iret != 2)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            _xData[nodeNum - 1] = xc;
        }
        else if (strncmp(_c, "NSTRESS", 7) == 0) // scalar
        {

            int nodeNum;
            float xc;
            int iret = sscanf(_c + 7, "%d %f", &nodeNum, &xc);
            if (iret != 2)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            _xData[nodeNum - 1] = xc;
        }
        else
        {
            break;
        }
    }
    if (_doSkip)
    {
        return NULL;
    }
    ASSERT0(_xData.size() >= _maxNodeNo, "sorry, and internal error occured.", _outputHandler);
    coDoFloat *dataObj = new coDoFloat(name, _xData.size(), &_xData[0]);
    return dataObj;
}

// --------------------------------------------------------------------------------------------------------------
//                                          getter methods
// --------------------------------------------------------------------------------------------------------------

int ResultsFileParserAscii::getNoOfTimeSteps(void) const
{
    return _noOfTimeSteps;
}

std::string ResultsFileParserAscii::getDataTypeName(int timeStepNo, int dataTypeNo)
{
    (void)timeStepNo;
    return std::string(_names[dataTypeNo]);
}

int ResultsFileParserAscii::getNoOfDataTypes(int timeStepNo) const
{
    (void)timeStepNo;
    return _currentDataTypeNo;
}

coDistributedObject *ResultsFileParserAscii::getDataObject(int dataTypeNo, int timeStepNo)
{
    coDistributedObject *retval = _dataObjects[dataTypeNo][timeStepNo];
    return retval;
}

ResultsFileData::EDimension ResultsFileParserAscii::getDimension(int timeStepNo, int dataTypeNo)
{
    ASSERT(timeStepNo < _noOfTimeSteps, _outputHandler);
    ResultsFileData::EDimension retval = _dimension[dataTypeNo][timeStepNo];
    return retval;
}

bool ResultsFileParserAscii::getDisplacements(int timeStepNo,
                                              float *dx[], float *dy[], float *dz[])
{
    bool retval = false;
    *dx = *dy = *dz = NULL;
    if (timeStepNo < _noOfTimeSteps)
    {
        int dataTypeNo = _displacementNo[timeStepNo];
        if (dataTypeNo != -1)
        {
            coDoVec3 *dataObj = (coDoVec3 *)_dataObjects[dataTypeNo][timeStepNo];
            dataObj->getAddresses(dx, dy, dz);
            retval = true;
        }
    }
    return retval;
}

void ResultsFileParserAscii::debug(void)
{
    std::ofstream f("content_ascii.txt");
    f << "_maxNodeNo = " << _maxNodeNo << "\n";
    f << "_noOfTimeSteps = " << _noOfTimeSteps << "\n";
    f << "_currentDataTypeNo = " << _currentDataTypeNo << "\n";

    INT i;
    for (i = 0; i < _noOfTimeSteps; i++)
    {
        f << "----------------- " << i << " ----------------------\n";
        INT j;
        for (j = 0; j < _currentDataTypeNo; j++)
        {
            if (_dimension[j][i] == ResultsFileParserAscii::SCALAR)
            {
                coDoFloat *dataObj = (coDoFloat *)_dataObjects[j][i];
                float *valuesArr;
                dataObj->getAddress(&valuesArr);
                int noOfPoints = dataObj->getNumPoints();

                f << "_name = " << _names[j] << "\n";
                INT k;
                for (k = 0; k < 5; k++)
                {
                    f << k << "\t" << valuesArr[k] << "\n";
                }
                f << "...\n";
                int z = 0;
                for (k = 0; k < noOfPoints - 5; k++)
                {
                    if (fabs(valuesArr[k]) > 1e-6 && z < 5)
                    {
                        f << k << "\t" << valuesArr[k] << "\n";
                        z++;
                    }
                }
                f << "...\n";
                for (k = noOfPoints - 5; k < noOfPoints; k++)
                {
                    f << k << "\t" << valuesArr[k] << "\n";
                }
            }
            else
            {
                coDoVec3 *dataObj = (coDoVec3 *)_dataObjects[j][i];
                float *x, *y, *z;
                dataObj->getAddresses(&x, &y, &z);
                int noOfPoints = dataObj->getNumPoints();

                f << "_name = " << _names[j] << "\n";
                INT k;
                for (k = 0; k < 5; k++)
                {
                    f << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
                }
                f << "...\n";
                int n = 0;
                for (k = 0; k < noOfPoints - 5; k++)
                {
                    if (fabs(x[k]) > 1e-6 && n < 5)
                    {
                        f << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
                        n++;
                    }
                }
                f << "...\n";
                for (k = noOfPoints - 5; k < noOfPoints; k++)
                {
                    f << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
                }
            }
        }
    }
    f.close();
}
