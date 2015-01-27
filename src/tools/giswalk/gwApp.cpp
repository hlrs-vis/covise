/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>

#include "giswalk.h"
#include "gwApp.h"
#include "gwTier.h"
#include <math.h>
#include <iostream>
#include <string>
using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_TIFF
#include <tiffio.h>
#endif
#ifndef _WIN32
#include <inttypes.h>
#define CALLBACK
#else
#include <OaIdl.h>
#include <sys/types.h>
#include <sys/timeb.h>
#include <windows.h>
#include <OleAuto.h>

#endif

using namespace std;

/*
void setMotionParameters(int type, param....);
void getResults(int id, array posx, array posy);
void getPos(int id, float posx, float posy);*/

// Visual Basic Code:
/*
Declare Sub initialize Lib "libGiswalk.dll" ()
Declare Sub setMap Lib "libGiswalk.dll" (ByRef map() As System.Byte, ByVal ysize As Real)
Declare Sub setGlobalParameters Lib "libGiswalk.dll" (ByRef deathRate() As Real,ByRef transitionMatrix() As Real,ByVal lifetime As Boolean)
Declare Sub setMotionParameters Lib "libGiswalk.dll" (ByVal type As Integer,ByVal directedWalkFromStep As Integer,ByVal directedWalkFromValue As Integer,ByRef maxSpeedArray() As Real, ByVal reorientation As Integer, ByVal scanStartFromStep As Integer, ByVal visionRange As Real,ByVal satisfactoryHabitat As IntegerByRef stopAtBoundaryArray() As Integer,ByVal settlement As Integer)
Declare Sub addAnimal Lib "libGiswalk.dll" (ByVal id As Integer,ByVal type As Integer,ByVal lifetime As Integer, ByVal xpos As Real,ByVal ypos As Real)
Declare Sub singleStep Lib "libGiswalk.dll" ()
Declare Sub compute Lib "libGiswalk.dll" ()
*/
#ifdef WIN32
extern "C" {
void __declspec(dllexport) __stdcall setGlobalParameters(SAFEARRAY **deathRate, SAFEARRAY **transitionMatrix, bool stopEveryTransition)
{
    gwApp *gw = gwApp::instance();

    gw->numHabitatValues = (*deathRate)->rgsabound[0].cElements;
    for (int i = 0; i < gwApp::instance()->numHabitatValues; i++)
    {
        gw->deathRate[i] = ((float *)(*deathRate)->pvData)[i];
    }
    int numx = (*transitionMatrix)->rgsabound[0].cElements;
    int numy = (*transitionMatrix)->rgsabound[1].cElements;
    for (int y = 0; y < numy; y++)
    {
        for (int x = 0; x < numx; x++)
        {
            gw->transitionMatrix[y][x] = ((float *)(*transitionMatrix)->pvData)[(y * numx) + x];
        }
    }
}
void __declspec(dllexport) __stdcall setMotionParameters(int type, int directedWalkFromStep, int directedWalkFromValue, SAFEARRAY **maxSpeedArray, int reorientation, int scanStartFromStep,
                                                         float visionRange, int satisfactoryHabitat, SAFEARRAY **stopAtBoundaryArray, int settlement)
{
    gwParamSet &ps = gwApp::instance()->params[type];
    ps.directedWalkFromStep = directedWalkFromStep;
    ps.directedWalkFromValue = directedWalkFromValue;

    int numSpeeds = (*maxSpeedArray)->rgsabound[0].cElements;
    for (int i = 0; i < numSpeeds; i++)
    {
        ps.maxSpeed[i] = ((float *)(*maxSpeedArray)->pvData)[i];
    }
    ps.reorientation = reorientation;
    ps.scanStartFromStep = scanStartFromStep;
    ps.visionRange = visionRange;
    ps.satisfactoryHabitat = satisfactoryHabitat;
    ps.settlement = settlement;
    int numstopAtBoundary = (*stopAtBoundaryArray)->rgsabound[0].cElements;
    for (int i = 0; i < numstopAtBoundary; i++)
    {
        ps.stopAtBoundary[i] = ((int *)(*stopAtBoundaryArray)->pvData)[i];
    }
}
void __declspec(dllexport) __stdcall compute()
{
    gwApp::instance()->run();
}
void __declspec(dllexport) __stdcall singleStep()
{
    gwApp::instance()->singleStep();
}
void __declspec(dllexport) __stdcall addAnimal(int id, int type, int lifetime, float xpos, float ypos)
{
    gwApp::instance()->addAnimal(id, (gwTier::Typ)type, lifetime, xpos, ypos);
}
void __declspec(dllexport) __stdcall setMap(SAFEARRAY **mapArray, float cellSize)
{
    HRESULT lResult; // return code for OLE functions
    // checking if it is an one-dimensional array
    if ((*mapArray)->cDims != 2)
        return;

    // checking if it is an array of bytes
    if ((*mapArray)->cbElements != 1)
        return;
    int xs = (*mapArray)->rgsabound[0].cElements;
    int ys = (*mapArray)->rgsabound[1].cElements;

    // locking the array before using its elements
    lResult = SafeArrayLock(*mapArray);
    if (lResult)
        return;

    gwApp::instance()->setMap((unsigned char *)(*mapArray)->pvData, xs, ys, cellSize);

    // releasing the array
    lResult = SafeArrayUnlock(*mapArray);
    if (lResult)
        return;
}

// C++ Code:
void __declspec(dllexport) __stdcall initialize()
{
    gwApp::instance()->initialize();
}
}
#endif

gwParamSet::gwParamSet()
{
    directedWalkFromValue = 2.0;
    directedWalkFromStep = 10;
    streightness = 1.0;
    reorientation = 1.0;
    scanStartFromStep = 20;
    visionRange = 2.0;
    satisfactoryHabitat = 3;
    settlement = 1000000000;
    maxSpeed[0] = 5.0;
    maxSpeed[1] = 15.0;
    maxSpeed[2] = 25.0;
    maxSpeed[3] = 35.0;
    maxSpeed[4] = 45.0;
    maxSpeed[5] = 55.0;
    stopAtBoundary[0] = 0;
    stopAtBoundary[1] = 0;
    stopAtBoundary[2] = 0;
    stopAtBoundary[3] = 0;
    stopAtBoundary[4] = 0;
    stopAtBoundary[5] = 0;
    stopAtBoundary[6] = 0;
}

gwApp *gwApp::inst = NULL;

void gwApp::setMap(unsigned char *ma, int xsp, int ysp, float cellSize)
{
    xs = xsp;
    ys = ysp;
    pSize[0] = cellSize;
    pSize[1] = cellSize;
    size[0] = xs * pSize[0];
    size[1] = ys * pSize[1];

    unsigned char *buf = NULL;
    map = new unsigned char[xs * ys * 4];
    for (int n = 0; n < ys; n++)
    {
        for (int m = 0; m < xs; m++)
        {
            map[((n * xs) + m) * 4] = ma[((n * xs) + m)];
        }
    }
}

void gwApp::initialize()
{
    minLifeTime = 90;
    maxLifeTime = 110;
    lastNumHValues = -1;

    G = 100.0;
    mDist = 100.0;

    XLLCorner = 0;
    YLLCorner = 0;
    stopEveryTransition = false;

    pSize[0] = (float)0.1;
    pSize[1] = (float)0.1;
    size[0] = xs * pSize[0];
    size[1] = ys * pSize[1];
    deathRate[0] = -1;
    deathRate[1] = -1;
    deathRate[2] = 5;
    deathRate[3] = 10;
    deathRate[4] = 20;
    deathRate[5] = 99;
    percentColonizers = 50;

    float defaultTM[6][6] = {
        { 100, 90, 20, 10, 3, 0 },
        { 100, 100, 30, 10, 3, 0 },
        { 100, 100, 100, 10, 3, 0 },
        { 100, 100, 100, 100, 3, 0 },
        { 100, 100, 100, 100, 100, 0 },
        { 100, 100, 100, 100, 100, 0 }
    };
    for (int n = 0; n < 6; n++)
    {
        for (int m = 0; m < 6; m++)
        {
            transitionMatrix[n][m] = defaultTM[n][m];
        }
    }
    params[gwTier::Philopatric].directedWalkFromStep = 3;
    params[gwTier::Philopatric].directedWalkFromValue = 2;
    params[gwTier::Philopatric].reorientation = 10;
    params[gwTier::Philopatric].scanStartFromStep = 5;
    params[gwTier::Philopatric].streightness = (float)0.006;
    params[gwTier::Philopatric].visionRange = 3;

    params[gwTier::Colonizers].directedWalkFromStep = 0;
    params[gwTier::Colonizers].directedWalkFromValue = 0;
    params[gwTier::Colonizers].reorientation = 10;
    params[gwTier::Colonizers].scanStartFromStep = 5;
    params[gwTier::Colonizers].streightness = (float)0.006;
    params[gwTier::Colonizers].visionRange = 3;

    double k = G * (2.0 * log(G - 1.0) / (mDist * G));

    for (int i = 0; i < mDist; i++)
    {

        double sigmo = G - (G / (1.0 + (G - 1.0) * exp(-k * (i + 1))));
        int num = (int)sigmo;
        for (int n = 0; n < num; n++)
            sigmoDist.push_back(i);
    }
}

gwApp::gwApp(const char *filename)
{
    minLifeTime = 90;
    maxLifeTime = 110;
    lastNumHValues = -1;

    G = 100.0;
    mDist = 100.0;

    XLLCorner = 0;
    YLLCorner = 0;
    stopEveryTransition = false;

    pSize[0] = (float)0.1;
    pSize[1] = (float)0.1;
    size[0] = xs * pSize[0];
    size[1] = ys * pSize[1];
    deathRate[0] = -1;
    deathRate[1] = -1;
    deathRate[2] = 5;
    deathRate[3] = 10;
    deathRate[4] = 20;
    deathRate[5] = 99;
    percentColonizers = 50;

    float defaultTM[6][6] = {
        { 100, 90, 20, 10, 3, 0 },
        { 100, 100, 30, 10, 3, 0 },
        { 100, 100, 100, 10, 3, 0 },
        { 100, 100, 100, 100, 3, 0 },
        { 100, 100, 100, 100, 100, 0 },
        { 100, 100, 100, 100, 100, 0 }
    };
    for (int n = 0; n < 6; n++)
    {
        for (int m = 0; m < 6; m++)
        {
            transitionMatrix[n][m] = defaultTM[n][m];
        }
    }
    params[gwTier::Philopatric].directedWalkFromStep = 3;
    params[gwTier::Philopatric].directedWalkFromValue = 2;
    params[gwTier::Philopatric].reorientation = 10;
    params[gwTier::Philopatric].scanStartFromStep = 5;
    params[gwTier::Philopatric].streightness = (float)0.006;
    params[gwTier::Philopatric].visionRange = 3;

    params[gwTier::Colonizers].directedWalkFromStep = 0;
    params[gwTier::Colonizers].directedWalkFromValue = 0;
    params[gwTier::Colonizers].reorientation = 10;
    params[gwTier::Colonizers].scanStartFromStep = 5;
    params[gwTier::Colonizers].streightness = (float)0.006;
    params[gwTier::Colonizers].visionRange = 3;
    int nc;
    if (filename)
    {
        std::string fn(filename);
        int pos = fn.rfind(".");
        if (pos != std::string::npos)
        {
            ending = fn.substr(pos);
            base = fn.substr(0, pos);
#ifdef HAVE_TIFF
            if (ending == ".tif")
                map = tifread(filename, &xs, &ys, &nc);
            else
#endif
                if (ending == ".hdr")
            {
                map = readHDR(filename);
            }
            else if (ending == ".txt")
            {
                map = readTXT(filename);
            }
        }
        else
        {
            fprintf(stderr, "could not read map file %s\n", filename);
            fprintf(stderr, "usage: giswalk [mapFile.tif,mapfile.hdr]\n");
        }
    }

    readConfigFile(base + ".xml");

    readStartPoints(base + ".start.txt");
    if (tiere.size() == 0) // generate some if we did not read them from a file
    {
        for (int i = 0; i < 1000; i++)
        {
            gwTier *t = new gwTier(i, gwTier::Philopatric, this);
            t->setPos(((float)rand() / (float)RAND_MAX) * size[0], ((float)rand() / (float)RAND_MAX) * size[1]);
            tiere.push_back(t);
        }
    }
    std::list<gwTier *>::iterator ti;
    if (percentColonizers > 100)
    {
        percentColonizers = 100;
    }

    int numColonizers = tiere.size() * ((float)percentColonizers / 100.0);
    int numPhilopatric = tiere.size() - numColonizers;
    if (percentColonizers > 50)
    {
        int i = 0;
        ti = tiere.begin();
        int stepSize = (tiere.size() / numColonizers) * 2;
        while (i < numColonizers)
        {
            int step = stepSize * ((float)rand() / (float)RAND_MAX);
            while (ti != tiere.end() && step > 0)
            {
                ti++;
                step--;
            }
            if (ti == tiere.end())
                ti = tiere.begin();
            if ((*ti)->getType() == gwTier::Philopatric)
            {
                i++;
                (*ti)->setType(gwTier::Colonizers);
            }
        }
    }
    else
    {
        for (ti = tiere.begin(); ti != tiere.end(); ti++)
        {
            gwTier *t = (*ti);
            t->setType(gwTier::Colonizers);
        }
        int i = 0;
        ti = tiere.begin();
        int stepSize = (tiere.size() / numColonizers) * 2;
        while (i < numPhilopatric)
        {
            int step = stepSize * ((float)rand() / (float)RAND_MAX);
            while (ti != tiere.end() && step > 0)
            {
                ti++;
                step--;
            }
            if (ti == tiere.end())
                ti = tiere.begin();
            if ((*ti)->getType() == gwTier::Colonizers)
            {
                i++;
                (*ti)->setType(gwTier::Philopatric);
            }
        }
    }

    for (ti = tiere.begin(); ti != tiere.end(); ti++)
    {
        gwTier *t = (*ti);
        t->setLifeTime(minLifeTime + ((float)rand() / (float)RAND_MAX) * (maxLifeTime - minLifeTime));
    }

    double k = G * (2.0 * log(G - 1.0) / (mDist * G));

    for (int i = 0; i < mDist; i++)
    {

        double sigmo = G - (G / (1.0 + (G - 1.0) * exp(-k * (i + 1))));
        int num = (int)sigmo;
        for (int n = 0; n < num; n++)
            sigmoDist.push_back(i);
    }
}
void gwApp::singleStep()
{
    std::list<gwTier *>::iterator ti;
    for (ti = tiere.begin(); ti != tiere.end();)
    {
        gwTier::Motion m = (*ti)->move();
        if (m == gwTier::Died)
        {
            toteTiere.push_back(*ti);
            ti = tiere.erase(ti);
        }
        else
        {
            //someAlive=true;
            ti++;
        }
    }
}

void gwApp::run()
{
    gwTier::Motion m;
    //bool someAlive = true;
    while (tiere.size())
    {
        //someAlive=false;
        std::list<gwTier *>::iterator ti;
        for (ti = tiere.begin(); ti != tiere.end();)
        {
            m = (*ti)->move();
            if (m == gwTier::Died)
            {
                toteTiere.push_back(*ti);
                ti = tiere.erase(ti);
            }
            else
            {
                //someAlive=true;
                ti++;
            }
        }
    }
}

void gwApp::writeSVG()
{
    std::string filename = base + ".svg";
    FILE *fp = fopen(filename.c_str(), "w");
    if (fp)
    {
        fprintf(fp, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        fprintf(fp, "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\">\n", xs, ys);
        fprintf(fp, "<g>\n");
        fprintf(fp, "<title>%s</title>\n", filename.c_str());

        for (int n = 0; n < ys; n++)
        {
            for (int m = 0; m < xs; m++)
            {
                //if(map[((n*xs)+m)*4]==6)
                //fprintf(fp,"<rect x=\"%d\" y=\"%d\" width=\"1\" height=\"1\" fill=\"green\" stroke=\"none\" stroke-width=\"0\"/>\n",m,n);
                if (map[((n * xs) + m) * 4] == 1)
                    fprintf(fp, "<rect x=\"%d\" y=\"%d\" width=\"1\" height=\"1\" fill=\"red\" stroke=\"none\" stroke-width=\"0\"/>\n", m, n);
            }
        }

        std::list<gwTier *>::iterator ti;
        for (ti = toteTiere.begin(); ti != toteTiere.end(); ti++)
        {
            (*ti)->writeSVG(fp);
        }

        fprintf(fp, "</g>\n");
        fprintf(fp, "</svg>\n");
    }
}

void gwApp::writeShape()
{
    std::string filename = base + "out.txt";
    FILE *fp = fopen(filename.c_str(), "w");
    if (fp)
    {
        fprintf(fp, "\"ID\"\t\"TAG\"\t\"X\"\t\"Y\"\t\"END\"\t\"TYPE\"\t\"ORIGTYPE\"\n");
        std::list<gwTier *>::iterator ti;
        for (ti = toteTiere.begin(); ti != toteTiere.end(); ti++)
        {
            (*ti)->writeShape(fp);
        }
        fprintf(fp, "END\n");
    }
}

void gwApp::readConfigFile(std::string filename)
{
#ifdef HAVE_XERCESC
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *pMsg = xercesc::XMLString::transcode(toCatch.getMessage());
        xercesc::XMLString::release(&pMsg);
    }
    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(filename.c_str());
    }
    catch (...)
    {
        cerr << "error parsing config file" << filename.c_str() << endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    xercesc::DOMElement *rootElement = NULL;
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    if (rootElement)
    {
        xercesc::DOMNodeList *nodeList = rootElement->getChildNodes();
        for (int i = 0; i < nodeList->getLength(); ++i)
        {
            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (!node)
                continue;
            const char *tagName = xercesc::XMLString::transcode(node->getTagName());
            if (tagName)
            {
                if (strcmp(tagName, "Philopatric") == 0 || strcmp(tagName, "Colonizers") == 0)
                {
                    gwTier::Typ type = gwTier::Colonizers;
                    if (strcmp(tagName, "Philopatric") == 0)
                    {
                        type = gwTier::Philopatric;
                    }
                    const char *directedWalkFromValue = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("directedWalkFromValue")));
                    const char *directedWalkFromStep = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("directedWalkFromStep")));
                    const char *streightness = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("streightness")));
                    const char *reorientation = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("reorientation")));
                    const char *scanStartFromStep = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("scanStartFromStep")));
                    const char *visionRange = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("visionRange")));
                    const char *satisfactoryHabitat = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("satisfactoryHabitat")));
                    const char *settlement = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("settlement")));

                    const char *maxSpeedA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("maxSpeed")));
                    if (maxSpeedA && maxSpeedA[0] != '\0')
                    {
                        const char *c = maxSpeedA;
                        int num = 0;
                        while (*c != '\0')
                        {
                            while (*c != '\0' && *c == ' ')
                                c++;
                            sscanf(c, "%f", params[type].maxSpeed + num);
                            while (*c != '\0' && *c != ' ')
                                c++;
                            while (*c != '\0' && *c == ' ')
                                c++;
                            num++;
                        }
                        if (lastNumHValues != -1 && lastNumHValues != num)
                        {
                            fprintf(stderr, "number of habitat values in config file is inconsistent\n");
                            fprintf(stderr, "was %d now %d\n", lastNumHValues, num);
                            exit(-1);
                        }
                        lastNumHValues = num;
                    }
                    const char *stopAtBoundaryA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("stopAtBoundary")));
                    if (stopAtBoundaryA && stopAtBoundaryA[0] != '\0')
                    {
                        const char *c = stopAtBoundaryA;
                        int num = 0;
                        while (*c != '\0')
                        {
                            while (*c != '\0' && *c == ' ')
                                c++;
                            sscanf(c, "%d", params[type].stopAtBoundary + num);
                            while (*c != '\0' && *c != ' ')
                                c++;
                            while (*c != '\0' && *c == ' ')
                                c++;
                            num++;
                        }
                        if (lastNumHValues != -1 && lastNumHValues != num)
                        {
                            fprintf(stderr, "number of habitat values in config file is inconsistent stopAtBoundary\n");
                            fprintf(stderr, "was %d now %d\n", lastNumHValues, num);
                            exit(-1);
                        }
                        lastNumHValues = num;
                    }

                    if (directedWalkFromValue && directedWalkFromValue[0] != '\0')
                        params[type].directedWalkFromValue = atoi(directedWalkFromValue) - 1;
                    if (directedWalkFromStep && directedWalkFromStep[0] != '\0')
                        params[type].directedWalkFromStep = atoi(directedWalkFromStep);
                    if (streightness && streightness[0] != '\0')
                        params[type].streightness = atof(streightness);
                    if (reorientation && reorientation[0] != '\0')
                        params[type].reorientation = atoi(reorientation);
                    if (scanStartFromStep && scanStartFromStep[0] != '\0')
                        params[type].scanStartFromStep = atoi(scanStartFromStep);
                    if (visionRange && visionRange[0] != '\0')
                        params[type].visionRange = atof(visionRange);
                    if (satisfactoryHabitat && satisfactoryHabitat[0] != '\0')
                        params[type].satisfactoryHabitat = atoi(satisfactoryHabitat) - 1;
                    if (settlement && settlement[0] != '\0')
                        params[type].settlement = atoi(settlement);
                }
                else if (strcmp(tagName, "Both") == 0)
                {

                    const char *GA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("G")));
                    const char *mDistA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("mDist")));

                    const char *numHabitatValuesA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("numHabitatValues")));

                    numHabitatValues = lastNumHValues;
                    if (numHabitatValuesA && numHabitatValuesA[0] != '\0')
                        numHabitatValues = atof(numHabitatValuesA);

                    if (lastNumHValues != -1 && lastNumHValues != numHabitatValues)
                    {
                        fprintf(stderr, "number of habitat values in config file is inconsistent\n");
                        fprintf(stderr, "was %d now %d\n", lastNumHValues, numHabitatValues);
                        exit(-1);
                    }
                    lastNumHValues = numHabitatValues;

                    if (GA && GA[0] != '\0')
                        G = atof(GA);
                    if (mDistA && mDistA[0] != '\0')
                        mDist = atof(mDistA);

                    const char *stopEveryTransitionA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("stopEveryTransition")));

                    if (stopEveryTransitionA)
                    {
                        if (strcasecmp(stopEveryTransitionA, "true") == 0)
                            stopEveryTransition = true;
                        else
                            stopEveryTransition = false;
                    }

                    const char *minLifeTimeA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("minLifeTime")));
                    const char *maxLifeTimeA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("maxLifeTime")));

                    if (minLifeTimeA && minLifeTimeA[0] != '\0')
                        minLifeTime = atoi(minLifeTimeA);
                    if (maxLifeTimeA && maxLifeTimeA[0] != '\0')
                        maxLifeTime = atoi(maxLifeTimeA);

                    const char *deathRateA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("deathRate")));

                    if (deathRateA && deathRateA[0] != '\0')
                    {
                        const char *c = deathRateA;
                        int num = 0;
                        while (*c != '\0')
                        {
                            while (*c != '\0' && *c == ' ')
                                c++;
                            sscanf(c, "%f", deathRate + num);
                            while (*c != '\0' && *c != ' ')
                                c++;
                            while (*c != '\0' && *c == ' ')
                                c++;
                            num++;
                        }
                        if (lastNumHValues != -1 && lastNumHValues != num)
                        {
                            fprintf(stderr, "number of habitat values in config file is inconsistent\n");
                            fprintf(stderr, "was %d now %d\n", lastNumHValues, num);
                            exit(-1);
                        }
                        lastNumHValues = num;
                    }

                    const char *transitionMatrixA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("transitionMatrix")));

                    if (transitionMatrixA && transitionMatrixA[0] != '\0')
                    {
                        const char *tmp = transitionMatrixA;

                        for (int n = 0; n < numHabitatValues; n++)
                        {
                            while (*tmp != '\0' && *tmp != '{')
                                tmp++;
                            if (*tmp == '{')
                                tmp++;
                            else
                                break;
                            for (int m = 0; m < numHabitatValues; m++)
                            {
                                while (*tmp != '\0' && *tmp == ' ')
                                    tmp++;
                                sscanf(tmp, "%f", &transitionMatrix[n][m]);
                                while (*tmp != '\0' && *tmp != ',')
                                    tmp++;
                                if (*tmp == ',')
                                    tmp++;
                                else
                                    break;
                            }
                        }
                    }

                    const char *percentColonizersA = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("percentColonizers")));

                    if (percentColonizersA && percentColonizersA[0] != '\0')
                    {
                        percentColonizers = atof(percentColonizersA);
                    }
                }
                /* else if(strcmp(tagName,"fragmentProgram")==0)
            {
               const char *code=xercesc::XMLString::transcode(node->getTextContent());
               if(code && code[0]!='\0')
                  fragmentShader = new osg::Shader( osg::Shader::FRAGMENT, code );

            }*/
            }
        }
    }
#endif
}

unsigned char *gwApp::readHDR(std::string filename)
{
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp)
    {
        int nCols = 0;
        int nRows = 0;
        bool xCenter = false;
        bool yCenter = false;
        float XLLCenter = 0;
        float YLLCenter = 0;
        float CellSize = 0;
        float NODataValue = 0;

        while (!feof(fp))
        {
            char line[500];
            if (!fgets(line, 500, fp))
            {
                fprintf(stderr, "Premature end 3 while reading %s\n", filename.c_str());
                break;
            }
            if (strncasecmp(line, "XLLCENTER", 9) == 0)
            {
                xCenter = true;
                sscanf(line + 10, "%f", &XLLCenter);
            }
            if (strncasecmp(line, "XLLCORNER", 9) == 0)
            {
                xCenter = false;
                sscanf(line + 10, "%f", &XLLCorner);
            }
            if (strncasecmp(line, "YLLCENTER", 9) == 0)
            {
                yCenter = true;
                sscanf(line + 10, "%f", &YLLCenter);
            }
            if (strncasecmp(line, "YLLCORNER", 9) == 0)
            {
                yCenter = false;
                sscanf(line + 10, "%f", &YLLCorner);
            }
            if (strncasecmp(line, "NCOLS", 5) == 0)
            {
                sscanf(line + 6, "%d", &nCols);
            }
            if (strncasecmp(line, "NROWS", 5) == 0)
            {
                sscanf(line + 6, "%d", &nRows);
            }
            if (strncasecmp(line, "CELLSIZE", 8) == 0)
            {
                sscanf(line + 9, "%f", &CellSize);
            }
            if (strncasecmp(line, "NODATA_VALUE", 12) == 0)
            {
                sscanf(line + 13, "%f", &NODataValue);
            }
            if (strncasecmp(line, "BYTEORDER", 10) == 0)
            {
                if (strstr(line, "LSBFIRST") != NULL)
                {
                    fprintf(stderr, "Byteswapping not implemented yet!\n");
                }
            }
        }

        xs = nCols;
        ys = nRows;
        pSize[0] = CellSize;
        pSize[1] = CellSize;
        size[0] = xs * pSize[0];
        size[1] = ys * pSize[1];

        unsigned char *buf = NULL;
        float *fbuf = new float[nCols * nRows];
        std::string fltFile = filename + ".flt";
        FILE *fp2 = fopen(fltFile.c_str(), "rb");
        if (fp2)
        {
            if (!fread(fbuf, sizeof(float), nCols * nRows, fp2))
            {
                std::cerr << "read error in " << filename << std::endl;
            }
            fclose(fp2);
            buf = new unsigned char[nCols * nRows * 4];
            for (int n = 0; n < nRows; n++)
            {
                for (int m = 0; m < nCols; m++)
                {
                    buf[((n * nCols) + m) * 4] = (unsigned char)fbuf[((nRows - n - 1) * nCols) + m];
                }
            }
        }
        else
        {
            fprintf(stderr, "Could not open FLT file %s\n", fltFile.c_str());
        }
        if (xCenter)
            XLLCorner = XLLCenter - (nCols / 2.0 * CellSize);
        if (yCenter)
            YLLCorner = YLLCenter - (nRows / 2.0 * CellSize);

        delete[] fbuf;
        return buf;
    }
    else
    {
        fprintf(stderr, "Could not open HDR file %s\n", filename.c_str());
    }
    return NULL;
}

unsigned char *gwApp::readTXT(std::string filename)
{
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp)
    {
        int nCols = 0;
        int nRows = 0;
        bool xCenter = false;
        bool yCenter = false;
        float XLLCenter = 0;
        float YLLCenter = 0;
        float CellSize = 0;
        float NODataValue = 0;

        while (!feof(fp))
        {
            char line[500];
            if (!fgets(line, 500, fp))
            {
                fprintf(stderr, "Premature end 2 while reading %s\n", filename.c_str());
            }
            if (strncasecmp(line, "XLLCENTER", 9) == 0)
            {
                xCenter = true;
                sscanf(line + 10, "%f", &XLLCenter);
            }
            if (strncasecmp(line, "XLLCORNER", 9) == 0)
            {
                xCenter = false;
                sscanf(line + 10, "%f", &XLLCorner);
            }
            if (strncasecmp(line, "YLLCENTER", 9) == 0)
            {
                yCenter = true;
                sscanf(line + 10, "%f", &YLLCenter);
            }
            if (strncasecmp(line, "YLLCORNER", 9) == 0)
            {
                yCenter = false;
                sscanf(line + 10, "%f", &YLLCorner);
            }
            if (strncasecmp(line, "NCOLS", 5) == 0)
            {
                sscanf(line + 6, "%d", &nCols);
            }
            if (strncasecmp(line, "NROWS", 5) == 0)
            {
                sscanf(line + 6, "%d", &nRows);
            }
            if (strncasecmp(line, "CELLSIZE", 8) == 0)
            {
                sscanf(line + 9, "%f", &CellSize);
            }
            if (strncasecmp(line, "NODATA_VALUE", 12) == 0)
            {
                sscanf(line + 13, "%f", &NODataValue);
                break;
            }
            if (strncasecmp(line, "BYTEORDER", 10) == 0)
            {
                if (strstr(line, "LSBFIRST") != NULL)
                {
                    fprintf(stderr, "Byteswapping not implemented yet!\n");
                }
            }
        }

        xs = nCols;
        ys = nRows;
        pSize[0] = CellSize;
        pSize[1] = CellSize;
        size[0] = xs * pSize[0];
        size[1] = ys * pSize[1];

        unsigned char *buf = NULL;
        buf = new unsigned char[nCols * nRows * 4];
        for (int n = 0; n < nRows; n++)
        {
            for (int m = 0; m < nCols; m++)
            {
                int hq = 0;
                if (fscanf(fp, "%d", &hq) != 1)
                {
                    std::cerr << "error reading int" << std::endl;
                }
                buf[(((nRows - n - 1) * nCols) + m) * 4] = (unsigned char)hq - 1;
            }
        }
        if (xCenter)
            XLLCorner = XLLCenter - (nCols / 2.0 * CellSize);
        if (yCenter)
            YLLCorner = YLLCenter - (nRows / 2.0 * CellSize);

        return buf;
    }
    else
    {
        fprintf(stderr, "Could not open txt file %s\n", filename.c_str());
    }
    return NULL;
}

void gwApp::readStartPoints(std::string filename)
{
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp)
    {
        char line[500];
        while (!feof(fp))
        {
            if (!fgets(line, 500, fp))
            {
                fprintf(stderr, "Premature end 1 while reading %s\n", filename.c_str());
                break;
            }
            if (strncmp(line, "END", 3) == 0)
            {
                break;
            }
            int id, group = 0;
            float x, y;
            int numValues = sscanf(line, "%d %f %f", &id /*,&group*/, &x, &y);
            if (numValues == 3)
            {
                gwTier *t = new gwTier(id, gwTier::Philopatric, this, group);
                tiere.push_back(t);
                t->setPos(x - XLLCorner, y - YLLCorner);
            }
            else
            {
                break;
            }
        }
    }
    else
    {
        fprintf(stderr, "Could not open StartShape file %s\n", filename.c_str());
    }
}

void gwApp::addAnimal(int id, gwTier::Typ type, int lifetime, float xpos, float ypos)
{
    gwTier *t = new gwTier(id, type, this);
    tiere.push_back(t);
    t->setPos(xpos, ypos);
    t->setLifeTime(lifetime);
}

#ifdef HAVE_TIFF

void myWarn(const char *c, const char *c2, va_list list)
{
    (void)c;
    (void)c2;
    (void)list;
}
unsigned char *tifread(const char *url, int *w, int *h, int *nc)
{
    static int firstTime = 1;
    TIFF *tif;
    if (firstTime)
    {
        firstTime = 0;
        TIFFSetWarningHandler(myWarn);
    }

/*tiffDoRGBA = 1;*/
#ifdef _WIN32
    tif = TIFFOpen(url, "r");
#else
    tif = TIFFOpen(url, "r");
#endif
    if (tif)
    {
        size_t npixels;
        size_t widthbytes;
        int i;
        uint32 *raster;
        unsigned char *raster2;
        unsigned char *image;
        int samples;
        samples = 4;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, h);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
        npixels = *w * *h;
        raster = (uint32 *)malloc(npixels * sizeof(uint32));
        if (raster != NULL)
        {
            if (TIFFReadRGBAImage(tif, *w, *h, raster, 0))
            {
                *nc = 4;
                if (samples < 4)
                {
                    /* ugly hack by Uwe for grey scale/b/w images */
                    *nc = 40;
                }

                raster2 = (unsigned char *)malloc(npixels * sizeof(uint32));
                image = (unsigned char *)raster;
                widthbytes = *w * sizeof(uint32);
                for (i = 0; i < *h; i++)
                {
                    memcpy(raster2 + (npixels * sizeof(uint32)) - ((i + 1) * widthbytes), image + (i * widthbytes), widthbytes);
                }
                free(raster);

/* We have to byteswap on SGI ! */
#ifdef BIG_ENDIAN
                {
                    int i;
                    uint32_t *iPtr = (uint32_t *)raster2;
                    for (i = 0; i < npixels; i++, iPtr++)
                        *iPtr = ((*iPtr & 0x000000ff) << 24) | ((*iPtr & 0x0000ff00) << 8) | ((*iPtr & 0x00ff0000) >> 8) | ((*iPtr & 0xff000000) >> 24);
                }
#endif

                /*TIFFClose(tif);*/
                return (unsigned char *)raster2;
            }
        }
    }
    return NULL;
}
#endif
