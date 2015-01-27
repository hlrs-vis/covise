/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CALIBRATE_PLUGIN_H
#define _CALIBRATE_PLUGIN_H
/****************************************************************************\ 
**                                                          (C)20006 HLRS   **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRPlugin.h>
#include <cover/coVRTui.h>
#include <cover/coTabletUI.h>
#include <osg/Switch>

struct ori
{
    float o[9];
};

struct pos
{
    float x;
    float y;
    float z;
};

class Calibrate : public coVRPlugin, public coTUIListener
{
private:
    float getX(int i)
    {
        return ((max[0] - min[0]) / (nx - 1)) * i + min[0];
    };
    float getY(int i)
    {
        return ((max[1] - min[1]) / (ny - 1)) * -i + max[1];
    };
    float getZ(int i)
    {
        return ((max[2] - min[2]) / (nz - 1)) * i + min[2];
    };

    // create all Buttons
    void createMenuEntry();

    // remove the menu items
    void removeMenuEntry();

    // Calibration file
    FILE *fp;

    // x,y and z dimension of Calibration grid
    int nx, ny, nz;

    ori *orientation;
    pos *position; // tracker Values

    float min[3]; // x,y,z min
    float max[3]; // x,y,z max

    float CubeMin[3]; // size of the cube (place, where lines/text are displayd
    float CubeMax[3];

    int cI; // current index
    int cJ;
    int cK;

    osg::Matrix textMat;
    osg::ref_ptr<osg::Switch> marker;
    osg::ref_ptr<osg::RefMatrix> markers[20];

    coTUITab *calibTab;
    coTUILabel *currentIndex;
    coTUILabel *currentSoll;
    coTUILabel *currentIst;
    coTUILabel *statusLine;
    coTUIButton *Next;
    coTUIButton *StepX;
    coTUIButton *StepY;
    coTUIButton *StepZ;
    coTUIButton *Save;
    coTUIButton *Load;
    coTUIButton *Capture;

    void tabletPressEvent(coTUIElement *);

public:
    Calibrate();
    ~Calibrate();

    void readFile();

    // returns true, if init was successful
    int isOK()
    {
        return (nx == 0 || ny == 0 || nz == 0);
    };

    // update position Display on the screen
    void updateDisplay();

    // generate GridLines on the screens
    void makeDisplay();

    // remove GridLines on the screens
    void removeDisplay();

    // does button checking
    void preFrame();

    void save();

    void step();
    void stepDown(); // from top do bottom
    void stepX();
    void stepY();
    void stepZ();
    void stepNext();
};
#endif
