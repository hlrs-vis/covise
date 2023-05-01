/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HIGHSCORE_PLUGIN
#define HIGHSCORE_PLUGIN

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Highscore Plugin collects lap times                         **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** March-08  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>

class Highscore;

using namespace covise;
using namespace opencover;
using namespace vrml;

class VrmlNodeHighscore : public VrmlNodeChild
{

public:
    // Define the fields of Timesteps nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeHighscore(VrmlScene *scene = 0);
    VrmlNodeHighscore(const VrmlNodeHighscore &n);
    virtual ~VrmlNodeHighscore();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeHighscore *toHighscore() const;

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);
    const VrmlField *getField(const char *fieldName) const;

private:
};
class Highscore;

class HSEntry
{
public:
    HSEntry(Highscore *);
    ~HSEntry();
    void setStartTime(double t);
    void setInterimTime(double t);
    void setEndTime(double t);
    double getLapTime();
    double getInterimTimeDiff();
    int getPos()
    {
        return pos;
    }
    double getStartTime()
    {
        return startTime;
    }
    double getEndTime()
    {
        return endTime;
    }
    double getInterimTime()
    {
        return interimTime;
    }
    std::string getName()
    {
        return name;
    }
    void setName(std::string &n);
    void setPos(int p);
    void reset();

private:
    double startTime;
    double interimTime;
    double endTime;
    std::string name;
    int pos;
    Highscore *hs;

    coTUILabel *PosLabel;
    coTUILabel *NameLabel;
    coTUILabel *LapLabel;
    coTUILabel *InterimLabel;
};

class Highscore : public coVRPlugin, public coTUIListener
{
public:
    Highscore();
    bool init();
    virtual ~Highscore();
    coTUITab *HighscoreTab;
    coTUILabel *HSL;
    coTUILabel *DL;
    coTUIEditField *DriverName;
    HSEntry *currentEntry;
    std::list<HSEntry *> hsEntries;
    void tabletPressEvent(coTUIElement *);
    void tabletEvent(coTUIElement *);

    void setStartTime(double time);
    void setInterimTime(double time);
    void setResetTime(double time);
    bool passedInterim;
    static Highscore *instance()
    {
        if (myInstance == NULL)
            myInstance = new Highscore();
        return myInstance;
    }

private:
    void save();
    void load();
    static Highscore *myInstance;
    xercesc::DOMImplementation *impl;
};
#endif
