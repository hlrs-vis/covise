// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include "vvperformancetest.h"
#include <virvo/vvtoolshed.h>
#include <virvo/vvvecmath.h>
#include <virvo/vvvirvo.h>

#include <virvo/private/vvgltools.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <float.h>

using std::cerr;
using std::endl;

#define MAX_LINE_LEN 4096

vvTestResult::vvTestResult()
{

}

vvTestResult::~vvTestResult()
{

}

void vvTestResult::setDiffTimes(const std::vector<float> diffTimes)
{
  _diffTimes = diffTimes;
}

void vvTestResult::setModelViewMatrices(const std::vector<vvMatrix> modelViewMatrices)
{
  _modelViewMatrices = modelViewMatrices;
}

std::vector<float> vvTestResult::getDiffTimes() const
{
  return _diffTimes;
}

std::vector<vvMatrix> vvTestResult::getModelViewMatrices() const
{
  return _modelViewMatrices;
}

float vvTestResult::getTotalTime() const
{
  return _totalTime;
}

float vvTestResult::getAvgTime() const
{
  return _avgTime;
}

float vvTestResult::getVariance() const
{
  return _variance;
}

float vvTestResult::getMaxTime() const
{
  return _maxTime;
}

float vvTestResult::getMinTime() const
{
  return _minTime;
}

void vvTestResult::calc()
{
  _minTime = FLT_MAX;
  _maxTime = -FLT_MAX;
  _totalTime = 0.0f;

  std::vector<float>::const_iterator it;
  for (it = _diffTimes.begin(); it != _diffTimes.end(); ++it)
  {
    const float t = *it;

    if (t > _maxTime)
    {
      _maxTime = t;
    }
    else if (t < _minTime)
    {
      _minTime = t;
    }

    _totalTime += t;
  }
  _avgTime = _totalTime / static_cast<float>(_diffTimes.size());

  // Calc variance.
  _variance = 0.0f;

  for (it = _diffTimes.begin(); it != _diffTimes.end(); ++it)
  {
    float t = *it;

    _variance += (t - _avgTime) * (t - _avgTime);
  }
  _variance /= static_cast<float>(_diffTimes.size());
}

vvPerformanceTest::vvPerformanceTest()
{
  _outputType = VV_DETAILED;
  _datasetName = "";
  _verbose = true;
  _testResult = new vvTestResult();
  _iterations = 1;
  _frames = 90;
  _quality = 1.0f;
  _testAnimation = VV_ROT_Y;
  _projectionType = vvObjView::PERSPECTIVE;
  _brickDims[0] = _brickDims[1] = _brickDims[2] = 64;
}

vvPerformanceTest::~vvPerformanceTest()
{
  delete _testResult;
}

void vvPerformanceTest::writeResultFiles()
{
#if !defined(_WIN32)
  _testResult->calc();
  if ((_outputType == VV_SUMMARY) || (_outputType == VV_DETAILED))
  {
    // Text file with summary.
    char* summaryFile = new char[80];
    time_t now = time(NULL);
    struct tm  *ts;

    ts = localtime(&now);
    strftime(summaryFile, 80, "%Y-%m-%d_%H:%M:%S_%Z_summary.txt", ts);

    FILE* handle = fopen(summaryFile, "w");

    if (handle != NULL)
    {
      vvGLTools::GLInfo glInfo = vvGLTools::getGLInfo();
      char* dateStr = new char[80];
      strftime(dateStr, 80, "%Y-%m-%d, %H:%M:%S %Z", ts);
      fprintf(handle, "************************* Summary test %i *************************\n", _id);
      fprintf(handle, "Test performed at:....................%s\n", dateStr);
      fprintf(handle, "Virvo version:........................%s\n", virvo::version());
      fprintf(handle, "OpenGL vendor string:.................%s\n", glInfo.vendor);
      fprintf(handle, "OpenGL renderer string:...............%s\n", glInfo.renderer);
      fprintf(handle, "OpenGL version string:................%s\n", glInfo.version);
      fprintf(handle, "Total profiling time:.................%f\n", _testResult->getTotalTime());
      fprintf(handle, "Average time per frame:...............%f\n", _testResult->getAvgTime());
      fprintf(handle, "Variance:.............................%f\n", _testResult->getVariance());
      fprintf(handle, "Max rendering time:...................%f\n", _testResult->getMaxTime());
      fprintf(handle, "Min rendering time:...................%f\n", _testResult->getMinTime());
      fclose(handle);
    }

    if (_outputType == VV_DETAILED)
    {
      // Csv file simply with the diff times.
      char* csvFile = new char[80];
      strftime(csvFile, 80, "%Y-%m-%d_%H:%M:%S_%Z_times.csv", ts);

      handle = fopen(csvFile, "w");

      if (handle != NULL)
      {
        std::vector<float> times = _testResult->getDiffTimes();
        std::vector<vvMatrix> matrices = _testResult->getModelViewMatrices();
        std::vector<float>::const_iterator it;

        fprintf(handle, "\"TIME\",\"MODELVIEW_MATRIX\"\n");
        size_t i = 0;
        for (it = times.begin(); it != times.end(); ++it)
        {
          fprintf(handle, "[%f],", *it);
          fprintf(handle, "[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]",
                 matrices[i](0, 0), matrices[i](0, 1), matrices[i](0, 2), matrices[i](0, 3),
                 matrices[i](1, 0), matrices[i](1, 1), matrices[i](1, 2), matrices[i](1, 3),
                 matrices[i](2, 0), matrices[i](2, 1), matrices[i](2, 2), matrices[i](2, 3),
                 matrices[i](3, 0), matrices[i](3, 1), matrices[i](3, 2), matrices[i](3, 3));
          fprintf(handle, "\n");
          ++i;
        }
        fclose(handle);
      }
    }
  }
  else if (_outputType == VV_BRICKSIZES)
  {
    // Csv file with avg time for  the given brick size.
    const int HOST_NAME_LEN = 80;
    char  localHost[HOST_NAME_LEN];
  #ifdef _WIN32
    strcpy(localHost, "n/a");
  #else
    if (gethostname(localHost, HOST_NAME_LEN-1))
    {
      strcpy(localHost, "n/a");
    }
  #endif

    char brickFile[HOST_NAME_LEN + 4];
    sprintf(brickFile, "%s.csv", localHost);
    FILE* handle;
    const bool fileExists = (handle = fopen(brickFile, "r"));
    if (fileExists)
    {
      fclose(handle);
    }
    handle = fopen(brickFile, "a+");

    // Write header if file didn't exist until now.
    if (!fileExists)
    {
      fprintf(handle, "HOSTNAME,DATASET_NAME,BRICKSIZE_X,BRICKSIZE_Y,BRICKSIZE_Z,AVG_TIME\n");
    }

    // Append timing result.
    fprintf(handle, "%s,%s,%i,%i,%i,%f\n", localHost, _datasetName,
            static_cast<int>(_brickDims[0]),
            static_cast<int>(_brickDims[1]),
            static_cast<int>(_brickDims[2]),
            _testResult->getAvgTime());
    fclose(handle);
  }
#endif
}

void vvPerformanceTest::setId(const int id)
{
  _id = id;
}

void vvPerformanceTest::setOutputType(const OutputType outputType)
{
  _outputType = outputType;
}

void vvPerformanceTest::setDatasetName(const char* datasetName)
{
  _datasetName = datasetName;
}

void vvPerformanceTest::setIterations(const int iterations)
{
  _iterations = iterations;
}

void vvPerformanceTest::setVerbose(const bool verbose)
{
  _verbose = verbose;
}

void vvPerformanceTest::setQuality(const float quality)
{
  _quality = quality;
}

void vvPerformanceTest::setBrickDims(const vvVector3& brickDims)
{
  _brickDims = brickDims;
}

void vvPerformanceTest::setBrickDimX(const float brickDimX)
{
  _brickDims[0] = brickDimX;
}

void vvPerformanceTest::setBrickDimY(const float brickDimY)
{
  _brickDims[1] = brickDimY;
}

void vvPerformanceTest::setBrickDimZ(const float brickDimZ)
{
  _brickDims[2] = brickDimZ;
}

void vvPerformanceTest::setFrames(const int frames)
{
  _frames = frames;
}

void vvPerformanceTest::setTestAnimation(const TestAnimation testAnimation)
{
  _testAnimation = testAnimation;
}

void vvPerformanceTest::setProjectionType(const vvObjView::ProjectionType projectionType)
{
  _projectionType = projectionType;
}

int vvPerformanceTest::getId() const
{
  return _id;
}

vvPerformanceTest::OutputType vvPerformanceTest::getOutputType() const
{
  return _outputType;
}

const char* vvPerformanceTest::getDatasetName() const
{
  return _datasetName;
}

int vvPerformanceTest::getIterations() const
{
  return _iterations;
}

bool vvPerformanceTest::getVerbose() const
{
  return _verbose;
}

float vvPerformanceTest::getQuality() const
{
  return _quality;
}

vvVector3 vvPerformanceTest::getBrickDims() const
{
  return _brickDims;
}

int vvPerformanceTest::getFrames() const
{
  return _frames;
}

vvPerformanceTest::TestAnimation vvPerformanceTest::getTestAnimation() const
{
  return _testAnimation;
}

vvObjView::ProjectionType vvPerformanceTest::getProjectionType() const
{
  return _projectionType;
}

vvTestResult* vvPerformanceTest::getTestResult() const
{
  return _testResult;
}

vvTestSuite::vvTestSuite(const char* pathToFile)
    : _pathToFile(pathToFile)
{
  _initialized = false;

  if (_pathToFile)
  {
      init();
  }
}

vvTestSuite::~vvTestSuite()
{

}

bool vvTestSuite::isInitialized() const
{
  return _initialized;
}

std::vector<vvPerformanceTest*> vvTestSuite::getTests() const
{
  return _tests;
}

void vvTestSuite::init()
{
  _tests.clear();
  initColumnHeaders();

  FILE* handle = fopen(_pathToFile, "r");

  if (handle)
  {
    int lineCount = 0;
    char rawline[MAX_LINE_LEN];
    while (fgets(rawline, MAX_LINE_LEN, handle))
    {
      char *line = stripSpace(rawline);
      if(!line || !strcmp(line, ""))
      {
        // skip empty lines
        continue;
      }

      // One test per line (except the first one, which contains the header mapping).
      vvPerformanceTest* test = NULL;
      vvPerformanceTest* previousTest = NULL;
      if (!_tests.empty())
      {
        previousTest = _tests.back();
      }
      bool testSaved = false;

      int itemCount = 0;
      char* item = stripQuotes(stripSpace(strtok(line, ",")));
      if(item)
      {
        test = new vvPerformanceTest();
        if(previousTest)
          *test = *previousTest;
      }
      while(item)
      {
        if (lineCount == 0)
        {
          initHeader(item, itemCount);
        }
        else
        {
          initValue(test, item, itemCount, previousTest);
        }
        ++itemCount;

        if ((test != NULL) && (itemCount == NUM_COL_HEADERS))
        {
          // Thus and so that the ids are more legible, they are 1-based.
          test->setId(lineCount);
          _tests.push_back(test);
          testSaved = true;
        }
        item = stripQuotes(stripSpace(strtok(NULL, ",")));
      }

      if ((lineCount > 0) && (!testSaved))
      {
        test->setId(lineCount);
        _tests.push_back(test);
      }
      ++lineCount;
    }

    fclose(handle);
    _initialized = true;
  }
  else
  {
    _initialized = false;
  }
}

void vvTestSuite::initColumnHeaders()
{
  _columnHeaders[0] = "BRICKSIZE_X";    _headerPos[0] = 0;
  _columnHeaders[1] = "BRICKSIZE_Y";    _headerPos[1] = 1;
  _columnHeaders[2] = "BRICKSIZE_Z";    _headerPos[2] = 2;
  _columnHeaders[3] = "ITERATIONS";     _headerPos[3] = 3;
  _columnHeaders[4] = "QUALITY";        _headerPos[4] = 4;
  _columnHeaders[5] = "GEOMTYPE";       _headerPos[5] = 5;
  _columnHeaders[6] = "VOXELTYPE";      _headerPos[6] = 6;
  _columnHeaders[7] = "FRAMES";         _headerPos[7] = 7;
  _columnHeaders[8] = "TESTANIMATION";  _headerPos[8] = 8;
  _columnHeaders[9] = "PROJECTIONTYPE"; _headerPos[9] = 9;
  _columnHeaders[10] = "OUTPUTTYPE";    _headerPos[10] = 10;
}

void vvTestSuite::initHeader(char* str, const int col)
{
  toUpper(str);
  if (isHeader(str))
  {
    setHeaderPos(str, col);
  }
}

void vvTestSuite::initValue(vvPerformanceTest* test, char* str, const char* headerName)
{
#define atof(x) static_cast<float>(atof(x))
  if (strcmp(headerName, "BRICKSIZE_X") == 0)
  {
    test->setBrickDimX(atof(str));
  }
  else if (strcmp(headerName, "BRICKSIZE_Y") == 0)
  {
    test->setBrickDimY(atof(str));
  }
  else if (strcmp(headerName, "BRICKSIZE_Z") == 0)
  {
    test->setBrickDimZ(atof(str));
  }
  else if (strcmp(headerName, "ITERATIONS") == 0)
  {
    test->setIterations(atoi(str));
  }
  else if (strcmp(headerName, "QUALITY") == 0)
  {
    test->setQuality(atof(str));
#undef atof
  }
  else if (strcmp(headerName, "GEOMTYPE") == 0)
  {
    std::cerr << "only planar 3D textures are supported" << std::endl;
  }
  else if (strcmp(headerName, "FRAMES") == 0)
  {
    test->setFrames(atoi(str));
  }
  else if (strcmp(headerName, "TESTANIMATION") == 0)
  {
    if (strcmp(str, "VV_ROT_X") == 0)
    {
      test->setTestAnimation(vvPerformanceTest::VV_ROT_X);
    }
    else if (strcmp(str, "VV_ROT_Y") == 0)
    {
      test->setTestAnimation(vvPerformanceTest::VV_ROT_Y);
    }
    else if (strcmp(str, "VV_ROT_Z") == 0)
    {
      test->setTestAnimation(vvPerformanceTest::VV_ROT_Z);
    }
    else if (strcmp(str, "VV_ROT_RAND") == 0)
    {
      test->setTestAnimation(vvPerformanceTest::VV_ROT_RAND);
    }
  }
  else if (strcmp(headerName, "PROJECTIONTYPE") == 0)
  {
    if (strcmp(str, "ORTHO") == 0)
    {
      test->setProjectionType(vvObjView::ORTHO);
    }
    else if (strcmp(str, "PERSPECTIVE") == 0)
    {
      test->setProjectionType(vvObjView::PERSPECTIVE);
    }
  }
  else if (strcmp(headerName, "OUTPUTTYPE") == 0)
  {
    if (strcmp(str, "VV_BRICKSIZES") == 0)
    {
      test->setOutputType(vvPerformanceTest::VV_BRICKSIZES);
    }
    else if (strcmp(str, "VV_DETAILED") == 0)
    {
      test->setOutputType(vvPerformanceTest::VV_DETAILED);
    }
    else if (strcmp(str, "VV_NONE") == 0)
    {
      test->setOutputType(vvPerformanceTest::VV_NONE);
    }
    else if (strcmp(str, "VV_SUMMARY") == 0)
    {
      test->setOutputType(vvPerformanceTest::VV_SUMMARY);
    }
  }
}

void vvTestSuite::initValue(vvPerformanceTest* test, char* str, const int col,
                            vvPerformanceTest* previousTest)
{
  toUpper(str);

  const char* headerName = getHeaderName(col);

  // TODO: fix the \n hack... .
  if ((strcmp(str, "*") == 0) || (strcmp(str, "*\n") == 0))
  {
    // * means: take the value from the previous text.
    initFromPreviousValue(test, headerName, previousTest);
  }
  else
  {
    initValue(test, str, headerName);
  }
}

void vvTestSuite::initFromPreviousValue(vvPerformanceTest* test, const char* headerName,
                                        vvPerformanceTest* previousTest)
{
  char* str = new char[256];
  if (strcmp(headerName, "BRICKSIZE_X") == 0)
  {
    sprintf(str, "%i", (int)previousTest->getBrickDims()[0]);
  }
  else if (strcmp(headerName, "BRICKSIZE_Y") == 0)
  {
    sprintf(str, "%i", (int)previousTest->getBrickDims()[1]);
  }
  else if (strcmp(headerName, "BRICKSIZE_Z") == 0)
  {
    sprintf(str, "%i", (int)previousTest->getBrickDims()[2]);
  }
  else if (strcmp(headerName, "ITERATIONS") == 0)
  {
    sprintf(str, "%i", previousTest->getIterations());
  }
  else if (strcmp(headerName, "QUALITY") == 0)
  {
    sprintf(str, "%f", previousTest->getQuality());
  }
  else if (strcmp(headerName, "GEOMTYPE") == 0)
  {
    std::cerr << "only planar 3D textures are supported" << std::endl;
  }
  else if (strcmp(headerName, "FRAMES") == 0)
  {
    sprintf(str, "%i", previousTest->getFrames());
  }
  else if (strcmp(headerName, "TESTANIMATION") == 0)
  {
    switch (previousTest->getTestAnimation())
    {
    case vvPerformanceTest::VV_ROT_X:
      sprintf(str, "%s", "VV_ROT_X");
      break;
    case vvPerformanceTest::VV_ROT_Y:
      sprintf(str, "%s", "VV_ROT_Y");
      break;
    case vvPerformanceTest::VV_ROT_Z:
      sprintf(str, "%s", "VV_ROT_Z");
      break;
    case vvPerformanceTest::VV_ROT_RAND:
      sprintf(str, "%s", "VV_ROT_RAND");
      break;
    default:
      sprintf(str, "%s", "VV_ROT_Y");
      break;
    }
  }
  else if (strcmp(headerName, "PROJECTIONTYPE") == 0)
  {
    switch (previousTest->getProjectionType())
    {
    case vvObjView::FRUSTUM:
      sprintf(str, "%s", "FRUSTUM");
      break;
    case vvObjView::ORTHO:
      sprintf(str, "%s", "ORTHOG");
      break;
    case vvObjView::PERSPECTIVE:
      sprintf(str, "%s", "PERSPECTIVE");
      break;
    default:
      sprintf(str, "%s", "PERSPECTIVE");
      break;
    }
  }
  else if (strcmp(headerName, "OUTPUTTYPE") == 0)
  {
    switch (previousTest->getOutputType())
    {
    case vvPerformanceTest::VV_BRICKSIZES:
      sprintf(str, "%s", "VV_BRICKSIZES");
      break;
    case vvPerformanceTest::VV_DETAILED:
      sprintf(str, "%s", "VV_DETAILED");
      break;
    case vvPerformanceTest::VV_NONE:
      sprintf(str, "%s", "VV_NONE");
      break;
    case vvPerformanceTest::VV_SUMMARY:
      sprintf(str, "%s", "VV_SUMMARY");
      break;
    default:
      sprintf(str, "%s", "VV_DETAILED");
      break;
    }
  }
  initValue(test, str, headerName);
}

char* vvTestSuite::stripSpace(char* item)
{
  if(!item)
    return NULL;

  // strip white space at head and tail
  while(isspace(*item))
    ++item;
  size_t len = strlen(item);
  if(len > 0)
  {
    for(size_t i=len-1; i>0; --i)
    {
      if(isspace(item[i]))
      {
        item[i] = '\0';
        --len;
      }
    }
  }

  return item;
}

char* vvTestSuite::stripQuotes(char* item)
{
  if(!item)
    return NULL;

  // possibly strip quotation marks
  size_t len = strlen(item);
  if(len >=2 && item[0]=='\"' && item[len-1]=='\"')
  {
    item[len-1] = '\0';
    ++item;
  }

  return item;
}

void vvTestSuite::toUpper(char* str)
{
  size_t len = strlen(str);

  for (size_t i = 0; i < len; ++i)
  {
    if ((str[i] >= 'a') && (str[i] <= 'z'))
    {
      str[i] -= 32;
    }
  }
}

bool vvTestSuite::isHeader(const char* str)
{
  for (int i = 0; i < NUM_COL_HEADERS; ++i)
  {
    if ((strcmp(str, _columnHeaders[i])) == 0)
    {
      return true;
    }
  }
  return false;
}

void vvTestSuite::setHeaderPos(const char* header, const int pos)
{
  for (int i = 0; i < NUM_COL_HEADERS; ++i)
  {
    if (strcmp(header, _columnHeaders[i]) == 0)
    {
      _headerPos[i] = pos;
      break;
    }
  }
}

int vvTestSuite::getHeaderPos(const char* header)
{
  int result = -1;
  for (int i = 0; i < NUM_COL_HEADERS; ++i)
  {
    if (strcmp(header, _columnHeaders[i]) == 0)
    {
      result = _headerPos[i];
      break;
    }
  }
  return result;
}

const char* vvTestSuite::getHeaderName(const int pos)
{
  for (int i = 0; i < NUM_COL_HEADERS; ++i)
  {
    if (_headerPos[i] == pos)
    {
      return _columnHeaders[i];
    }
  }
  return NULL;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
