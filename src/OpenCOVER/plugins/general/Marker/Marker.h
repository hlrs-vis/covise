/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MARKER_H
#define _MARKER_H
#include <util/common.h>

#include <osg/MatrixTransform>
#include <osg/Billboard>
#include <osg/Group>
#include <osg/Switch>
#include <osg/Material>
#include <osgText/Text>
#include <osg/AlphaFunc>
#include <osg/BlendFunc>

#include <PluginUtil/coSensor.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coPopupHandle.h>

namespace vrui
{
class coButtonMenuItem;
class coCheckboxMenuItem;
class coTrackerButtonInteraction;
class coNavInteraction;
class coMouseButtonInteraction;
class coPotiMenuItem;
class coSubMenuItem;
class coRowMenu;
class coFrame;
class coPanel;
}

using namespace vrui;
using namespace opencover;

struct MarkerMessage
{
    int token;
    int id;
    float color;
    osg::Matrix mat;
    char host[20];
    char selecthost[20];
};

class Mark;

class MarkerSensor : public coPickSensor
{
private:
    Mark *myMark;

public:
    MarkerSensor(Mark *m, osg::Node *n);
    ~MarkerSensor();
    // this method is called if intersection just started
    // and should be overloaded
    virtual void activate();

    // should be overloaded, is called if intersection finishes
    virtual void disactivate();
};

class Mark
{
private:
    int id;
    osg::MatrixTransform *pos;
    osg::MatrixTransform *scale;
    osg::Material *mat;
    osg::Geode *geo;
    osg::Billboard *billText;
    osg::Cone *cone;
    string hname;
    MarkerSensor *mySensor;
    static float basevalue;
    string _selectedhname;
    float _hue;
    float _scaleVal;

public:
    // use index of vector to determine unique id
    Mark(int id, string, osg::Group *, float);
    ~Mark();
    void setSelectedHost(const char *);
    void setMat(osg::Matrix &mat);
    void setScale(float);
    float getScale();
    void setBaseSize(float);
    void setColor(float);
    float getColor();
    void setAmbient(osg::Vec4);
    void resetColor();
    //float getDist(osg::Vec3 &a);
    void getMat(osg::Matrix &);
    void setVisible(bool);
    bool isSelectable(string);
    bool matches(string);
    int getID();
    string getHost();
    osg::Matrix getPos();
};

class Marker : public coVRPlugin, public coMenuListener, public coButtonActor, public coValuePotiActor
{
    friend class Mark;

private:
    coSensorList sensorList;
    void menuEvent(coMenuItem *);
    Mark *currentMarker;
    Mark *previousMarker;
    string myHost;
    osg::Matrix invStarHand;
    vector<Mark *> marker;

    void createMenuEntry();

    // remove the menu item
    void removeMenuEntry();
    bool moving;
    float scale;
    int selectedMarkerId;
    coCheckboxMenuItem *markerMenuCheckbox;
    coSubMenuItem *markerMenuItem;
    coPotiMenuItem *scaleMenuPoti;
    coPotiMenuItem *headMenuPoti;
    coCheckboxMenuItem *hideMenuCheckbox;
    coButtonMenuItem *deleteAllButton;
    coRowMenu *markerMenu;
    // marker gui
    coPopupHandle *markerHandle;
    coFrame *markerFrame;
    coPanel *markerPanel;
    coLabel *markerLabel;
    coButton *markerPlayButton;
    coButton *markerRecordButton;
    coButton *markerDeleteButton;
    coValuePoti *colorPoti;
    // button interaction
    coNavInteraction *interactionA; ///< interaction for first button
    coNavInteraction *interactionC; ///< interaction for third button

    // Group node for attaching markers
    osg::ref_ptr<osg::Group> mainNode;

    void setScaleAll(float);
    void buttonEvent(coButton *);
    void setVisible(bool);
    string getMyHost();
    void setMMString(char *, string);
    int getLowestUnusedMarkerID();
    bool isIDInUse(int);

protected:
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);

public:
    static Marker *plugin;

    Marker();
    virtual ~Marker();
    bool init();

    // this will be called in PreFrame
    void preFrame();
    void message(int toWhom, int type, int len, const void *buf);
    void setCurrentMarker(Mark *m);
    int menuSelected; // TRUE if menu itme "Cube" was selected
    //void setMarkers(string);
    void deleteAllMarkers();

    static vrml::Player *player;
};
#endif
