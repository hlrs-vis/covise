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
#include <osgFX/Outline>

#include <cover/coVRPluginSupport.h>
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

struct SchweissbrennerMessage
{
    int token;
    int id;
    float color;
    osg::Matrix mat;
    char host[20];
    char selecthost[20];
};

class Schweissbrenner;

class SchweissbrennerSensor : public coPickSensor
{
private:
    Schweissbrenner *mySchweissbrenner;

public:
    SchweissbrennerSensor(Schweissbrenner *m, osg::Node *n);
    ~SchweissbrennerSensor();
    // this method is called if intersection just started
    // and should be overloaded
    virtual void activate();

    // should be overloaded, is called if intersection finishes
    virtual void disactivate();
};

class Schweissbrenner
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
    SchweissbrennerSensor *mySensor;
    static float basevalue;
    string _selectedhname;
    float _hue;
    float _scaleVal;
    osgFX::Outline *out;

public:
    // use index of vector to determine unique id
    Schweissbrenner(int id, string, osg::Group *, float);
    ~Schweissbrenner();
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

class Schweissen : public coVRPlugin, public coMenuListener, public coButtonActor, public coValuePotiActor
{
    friend class vrui::coNavInteraction;
    friend class ListenerCover;
    friend class SystemCover;

private:
    void menuEvent(coMenuItem *);
    Schweissbrenner *currentSchweissbrenner;
    Schweissbrenner *previousSchweissbrenner;
    string myHost;
    osg::Matrix invStarHand;
    vector<Schweissbrenner *> schweissbrenner;

    void createMenuEntry();

    // remove the menu item
    void removeMenuEntry();
    bool moving;
    float scale;
    int selectedSchweissenId;
    coCheckboxMenuItem *schweissbrennerMenuCheckbox;
    coSubMenuItem *schweissbrennerMenuItem;
    coPotiMenuItem *scaleMenuPoti;
    coPotiMenuItem *headMenuPoti;
    coCheckboxMenuItem *hideMenuCheckbox;
    coButtonMenuItem *deleteAllButton;
    coRowMenu *schweissbrennerMenu;
    // schweissbrenner gui
    coPopupHandle *schweissbrennerHandle;
    coFrame *schweissbrennerFrame;
    coPanel *schweissbrennerPanel;
    coLabel *schweissbrennerLabel;
    coButton *schweissbrennerPlayButton;
    coButton *schweissbrennerRecordButton;
    coButton *schweissbrennerDeleteButton;
    coValuePoti *colorPoti;
    osg::MatrixTransform *handTransform;

    // Group node for attaching schweissbrenners
    osg::ref_ptr<osg::Group> mainNode;

    void setScaleAll(float);
    void buttonEvent(coButton *);
    void setVisible(bool);
    string getMyHost();
    void setMMString(char *, string);
    int getLowestUnusedSchweissenID();
    bool isIDInUse(int);

protected:
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);

public:
    static Schweissen *plugin;

    osg::ref_ptr<osg::Node> SchweissbrennerNode;

    // button interaction
    coNavInteraction *interactionA; ///< interaction for first button
    coNavInteraction *interactionC; ///< interaction for third button

    Schweissen();
    virtual ~Schweissen();
    bool init();

    // this will be called in PreFrame
    void preFrame();
    void message(int type, int len, const void *buf);
    void setcurrentSchweissbrenner(Schweissbrenner *m);
    int menuSelected; // TRUE if menu itme "Cube" was selected
    //void setSchweissens(string);
    void deleteAllSchweissens();

    static vrml::Player *player;
};
#endif
