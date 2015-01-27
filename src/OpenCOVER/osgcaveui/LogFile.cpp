/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <fstream>

// Virvo:
#include <virvo/vvclock.h>

// CUI:
#include "Dial.h"

// Local:
#include "LogFile.h"

using namespace cui;
using namespace osg;
using namespace std;

LogFile::LogFile(const char *filename)
{
    _fp = fopen(filename, "wb");
    if (_fp == NULL)
    {
        cerr << "Failed creating log file " << filename << endl;
    }
    else
    {
        cerr << "Log file created: " << filename << endl;
    }
    _sw = new vvStopwatch();
    _sw->start();
    logStart();
}

LogFile::~LogFile()
{
    logStop();
    delete _sw;
    fclose(_fp);
}

void LogFile::addDialChangedLog(Dial *dial)
{
    fprintf(_fp, "%d\tDIALCHANGE\t'%s'\t%f\n", int(_sw->getTime() * 1000.0f), dial->getText().c_str(), dial->getValue());
}

void LogFile::addCardClickedLog(Card *card, int button, int newState)
{
    fprintf(_fp, "%d\tCARDCLICK\t'%s'\t%d\t%d\n", int(_sw->getTime() * 1000.0f), card->getText().c_str(), button, newState);
}

void LogFile::addButtonStateLog(int device, int num, int newState)
{
    fprintf(_fp, "%d\tBUTTONSTATE\t%d\t%d\t%d\n", int(_sw->getTime() * 1000.0f), device, num, newState);
}

/** @param box
    @param interactionType 1=moved with mouse button down, 2=tumbled with trackball
*/
void LogFile::addPickBoxMovedLog(PickBox *box, MoveType interactionType)
{
    Vec3 pos, up;
    Matrix box2w = box->getMatrix();
    pos = box2w.getTrans();
    up = Vec3(0.0f, 1.0f, 0.0f) * box2w;
    fprintf(_fp, "%d\tPICKBOXMOVE\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n", int(_sw->getTime() * 1000.0f),
            int(interactionType), pos[0], pos[1], pos[2], up[0], up[1], up[2]);
}

void LogFile::addMarkerLog(int index, Marker *marker)
{
    Vec3 pos;
    Matrix marker2v = marker->getMatrix();
    pos = marker2v.getTrans();
    fprintf(_fp, "%d\tMARKER\t%d\t%f\t%f\t%f\n", int(_sw->getTime() * 1000.0f),
            index, pos[0], pos[1], pos[2]);
}

void LogFile::addLog(const char *message)
{
    fprintf(_fp, "%f\t'%s'\n", (_sw->getTime() * 1000.0f), message);
    printf("%f\t'%s'\n", (_sw->getTime() * 1000.0f), message);
}

void LogFile::logStart()
{
    time_t rawTime;
    time(&rawTime);
    fprintf(_fp, "%d\tStarting log at %s", int(_sw->getTime() * 1000.0f), ctime(&rawTime));
}

void LogFile::logStop()
{
    time_t rawTime;
    time(&rawTime);
    fprintf(_fp, "%d\tStopped log at %s", int(_sw->getTime() * 1000.0f), ctime(&rawTime));
}

/** Log head to world and input device to world matrices.
*/
void LogFile::logInputDevices(Matrix &h2w, Matrix &i2w)
{
    Vec3 pos;
    double *mat;
    //float fMat;
    int i;
    int timeStamp;

    timeStamp = int(_sw->getTime() * 1000.0f);

    fprintf(_fp, "%d\tHEADMATRIX", timeStamp);
    pos = h2w.getTrans();
    for (i = 0; i < 3; ++i)
    { //cerr << float(pos[i]) << " ";
        fprintf(_fp, "\t%f", float(pos[i]));
    }
    //cerr << endl;
    mat = h2w.ptr();
    for (i = 8; i < 11; ++i)
    {
        fprintf(_fp, "\t%f", float(mat[i]));
    }
    fprintf(_fp, "\n");

    fprintf(_fp, "%d\tWANDMATRIX", timeStamp);
    pos = i2w.getTrans();
    for (i = 0; i < 3; ++i)
    {
        fprintf(_fp, "\t%f", float(pos[i]));
    }
    mat = i2w.ptr();
    for (i = 8; i < 11; ++i)
    {
        fprintf(_fp, "\t%f", float(mat[i]));
    }
    fprintf(_fp, "\n");
}
