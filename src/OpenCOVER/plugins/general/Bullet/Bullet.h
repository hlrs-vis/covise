/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Bullet_H
#define _Bullet_H
#include <util/common.h>

#include <osg/MatrixTransform>
#include <osg/Billboard>
#include <osg/Group>
#include <osg/Switch>
#include <osg/Material>
#include <osgText/Text>
#include <osg/AlphaFunc>
#include <osg/BlendFunc>
#include <cover/coVRFileManager.h>

#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coPopupHandle.h>
#include <PluginUtil/coSensor.h>

#include <xercesc/dom/DOMImplementation.hpp>

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

namespace opencover
{
class SystemCover;
}

using namespace vrui;
using namespace opencover;

struct BulletMessage
{
    int token;
    int id;
    float color;
    osg::Matrix mat;
    char host[20];
    char selecthost[20];
};

class BulletProbe;

class BulletSensor : public coPickSensor
{
private:
    BulletProbe *myBulletProbe;

public:
    BulletSensor(BulletProbe *m, osg::Node *n);
    ~BulletSensor();
    // this method is called if intersection just started
    // and should be overloaded
    virtual void activate();

    // should be overloaded, is called if intersection finishes
    virtual void disactivate();
};

class BulletProbe
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
    BulletSensor *mySensor;
    static float basevalue;
    string _selectedhname;
    float _hue;
    float _scaleVal;

public:
    int loadBullet(const char *filename, osg::Group *loadParent);
    static int sloadBullet(const char *filename, osg::Group *loadParent, const char *);
    static int unloadBullet(const char *filename, const char *);
    // use index of vector to determine unique id
    BulletProbe(int id, string, osg::Group *, float);
    ~BulletProbe();
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

    void HSBtoRGB(float h, float s, float v, float *r, float *g, float *b);
};

class Bullet : public coVRPlugin, public coMenuListener, public coButtonActor, public coValuePotiActor
{
    friend class BulletSensor;
private:
    coSensorList sensorList;
    std::string fileName;
    void menuEvent(coMenuItem *);
    BulletProbe *currentBullet;
    BulletProbe *previousBullet;
    string myHost;
    osg::Matrix invStarHand;
    std::vector<BulletProbe *> bullet;

    int loadBullet(const char *filename, osg::Group *loadParent);

    void createMenuEntry();

    // remove the menu item
    void removeMenuEntry();
    bool moving;
    float scale;
    int selectedBulletId;
    coCheckboxMenuItem *BulletMenuCheckbox;
    coSubMenuItem *BulletMenuItem;
    coPotiMenuItem *scaleMenuPoti;
    coPotiMenuItem *headMenuPoti;
    coCheckboxMenuItem *hideMenuCheckbox;
    coButtonMenuItem *deleteAllButton;
    coButtonMenuItem *SaveButton;

    coRowMenu *BulletMenu;
    // Bullet gui
    coPopupHandle *BulletHandle;
    coFrame *BulletFrame;
    coPanel *BulletPanel;
    coLabel *BulletLabel;
    coButton *BulletPlayButton;
    coButton *BulletRecordButton;
    coButton *BulletDeleteButton;
    coValuePoti *colorPoti;
    // button interaction
    coNavInteraction *interactionA; ///< interaction for first button
    coNavInteraction *interactionC; ///< interaction for third button

    // Group node for attaching Bullets
    osg::ref_ptr<osg::Group> mainNode;

    void setScaleAll(float);
    void buttonEvent(coButton *);
    void setVisible(bool);
    string getMyHost();
    void setMMString(char *, string);
    int getLowestUnusedBulletID();
    bool isIDInUse(int);

protected:
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);

public:
    static Bullet *plugin;

    static int sloadBullet(const char *filename, osg::Group *loadParent, const char *);
    static int unloadBullet(const char *filename, const char *);

    Bullet();
    virtual ~Bullet();
    bool init();

    // this will be called in PreFrame
    void preFrame();
    void message(int toWhom, int type, int len, const void *buf);
    void setCurrentBullet(BulletProbe *m);
    int menuSelected; // TRUE if menu itme "Cube" was selected
    //void setBullets(string);
    void deleteAllBullets();

    static vrml::Player *player;
    void save();
    void load();
    xercesc::DOMImplementation *impl;
    static Bullet *instance();
};
#endif
