/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "manualAction.h"
#include <stream.h>
void manualAction::hostConfigItem(
    const char *nameOfHost,
    int sharedMemory, const char *strSharedMemory,
    int executionMode, const char *strExecutionMode,
    int timeout, const char *hostname)
{
    cout << "HOSTCONFIGITEM "
         << nameOfHost << " "
         << strSharedMemory << " "
         << strExecutionMode << " "
         << timeout << " "
         << hostname << endl;
}

void manualAction::windowConfigItem(
    int winNo,
    const char *name,
    int softPipeNo,
    int origX,
    int origY,
    int sizeX,
    int sizeY,
    const char *hostname)
{
    cout << "WINDOWCONFIGITEM "
         << winNo << " "
         << name << " "
         << softPipeNo << " "
         << origX << " "
         << origY << " "
         << sizeX << " "
         << sizeY << " "
         << hostname << endl;
}

void manualAction::screenConfigItem(
    int screenNo,
    const char *screenName,
    int hSize,
    int vSize,
    int origX,
    int origY,
    int origZ,
    double head,
    double pitch,
    double roll,
    const char *hostname)
{
    cout << "SCREENCONFIGITEM "
         << screenNo << " "
         << screenName << " "
         << hSize << " "
         << vSize << " "
         << origX << " "
         << origY << " "
         << origZ << " "
         << head << " "
         << pitch << " "
         << roll << " "
         << hostname << endl;
}

void manualAction::channelConfigItem(
    int channelNo,
    const char *channelName,
    int winNo,
    double viewPortXMin,
    double viewPortYMin,
    double viewPortXMax,
    double viewPortYMax,
    const char *hostname)
{
    cout << "CHANNELCONFIGITEM "
         << channelNo << " "
         << channelName << " "
         << winNo << " "
         << viewPortXMin << " "
         << viewPortYMin << " "
         << viewPortXMax << " "
         << viewPortYMax << " "
         << hostname << endl;
}

void manualAction::pipeConfigItem(
    char *hostname,
    int softPipe,
    int hardPipe,
    const char *display)
{
    cout << "PIPECONFIGITEM "
         << hostname << " "
         << softPipe << " "
         << hardPipe << " "
         << display << endl;
}

void manualAction::licenseItem(
    const char *key,
    const char *name,
    const char *date,
    const char *hostname)
{
    cout << "LICENSEITEM "
         << key << " "
         << name << " "
         << date << " "
         << hostname << endl;
}

void manualAction::vrviewpointsItem(
    const char *viewpointName,
    float scale,
    float x,
    float y,
    float z,
    float h,
    float p,
    float r,
    const char *hostname)
{
    cout << "VRVIEWPOINTSITEM "
         << viewpointName << " "
         << scale << " "
         << x << " "
         << y << " "
         << z << " "
         << h << " "
         << p << " "
         << r << " "
         << hostname << endl;
}

void manualAction::buttonConfigMAP(char *hostname, int number, int buttonTypeNumber, const char *button)
{
    cout << "buttonConfigMAP "
         << hostname << " "
         << number << " "
         << button << endl;
}

void manualAction::createStringList(const char *string)
{
    cout << "global stringList; set stringList {" << string << "}" << endl;
}

void manualAction::addToStringList(const char *string)
{
    cout << "global stringList; lappend stringList " << string << endl;
}

void manualAction::CoverWelcome(const char *hostname)
{
    cout << "global stringList; global ListForSection;set ListForSection(COVERConfig,WELCOME_MESSAGE," << hostname << ") $stringList" << endl;
}

void manualAction::UIShortCuts(const char *hostname)
{
    cout << "global stringList; UIShortCuts $stringList " << hostname << endl;
}

void manualAction::colormapsitem(const char *name)
{
    cout << "global ColormapList; lappend ColormapList " << name << endl;
}

void manualAction::colormapRGB(const char *hostname, const char *colorMapName, int x, int y, int z)
{
    /* We will use this later
      cout << "set Colormap(RGB," << colorMapName << ",x," << hostname << ")" << x << endl;
      cout << "set Colormap(RGB," << colorMapName << ",y," << hostname << ")" << y << endl;
      cout << "set Colormap(RGB," << colorMapName << ",z," << hostname << ")" << z << endl;
   */
    cout << "lappend ColorMapList(" << colorMapName << "," << hostname << ") {RGB " << x << " " << y << " " << z << "}" << endl;
}

void manualAction::colormapRGBA(const char *hostname, const char *colorMapName, double x, double y, double z, double w)
{
    /* We will use this later
      cout << "set Colormap(RGB," << colorMapName << ",x," << hostname << ")" << x << endl;
      cout << "set Colormap(RGB," << colorMapName << ",y," << hostname << ")" << y << endl;
      cout << "set Colormap(RGB," << colorMapName << ",z," << hostname << ")" << z << endl;
      cout << "set Colormap(RGB," << colorMapName << ",w," << hostname << ")" << w << endl;
   */
    cout << "lappend ColorMapList(" << colorMapName << "," << hostname << ") {RGBA " << x << " " << y << " " << z << " " << w << "}" << endl;
}

void manualAction::colormapRGBAX(const char *hostname, const char *colorMapName, double x, double y, double z, double v, double w)
{
    /* We will use this later
      cout << "set Colormap(RGBAX," << colorMapName << ",x," << hostname << ")" << x << endl;
      cout << "set Colormap(RGBAX," << colorMapName << ",y," << hostname << ")" << y << endl;
      cout << "set Colormap(RGBAX," << colorMapName << ",z," << hostname << ")" << z << endl;
      cout << "set Colormap(RGBAX," << colorMapName << ",w," << hostname << ")" << w << endl;
      cout << "set Colormap(RGBAX," << colorMapName << ",v," << hostname << ")" << v << endl;
   */
    cout << "lappend ColorMapList(" << colorMapName << "," << hostname << ") {RGBAX "
         << x << " " << y << " "
         << z << " " << v << " "
         << w << "}" << endl;
}

void manualAction::COVERModule(const char *hostname, const char *modules)
{
    cout << "global ListForSection ; "
         << "lappend ListForSection(COVERConfig,MODULE," << hostname << ") "
         << modules << endl;
}
