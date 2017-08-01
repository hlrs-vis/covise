/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PedestrianFactory_h
#define PedestrianFactory_h

#include "Pedestrian.h"
#include "PedestrianManager.h"
#include "PedestrianGeometry.h"
#include <osgCal/Model>
#include <osgCal/CoreModel>
#include <xercesc/dom/DOM.hpp>
#include <string>

struct TRAFFICSIMULATIONEXPORT PedestrianSettings
{
    PedestrianSettings( // <ped> //
        std::string _id = "",
        std::string _name = "",
        std::string _range = "",
        std::string _debugLvl = "",

        // <geometry> //
        std::string _modelFile = "",
        std::string _scale = "",
        std::string _heading = "",

        // <start> //
        std::string _startRoadId = "",
        std::string _startLane = "",
        std::string _startDir = "",
        std::string _startSOff = "",
        std::string _startVOff = "",
        std::string _startVel = "",
        std::string _startAcc = "",

        // <animations> //
        std::string _idleIdx = "",
        std::string _idleVel = "",
        std::string _slowIdx = "",
        std::string _slowVel = "",
        std::string _walkIdx = "",
        std::string _walkVel = "",
        std::string _jogIdx = "",
        std::string _jogVel = "",
        std::string _lookIdx = "",
        std::string _waveIdx = "")
        : id(_id)
        , name(_name)
        , rangeLOD(_range)
        , debugLvl(_debugLvl)
        , modelFile(_modelFile)
        , scale(_scale)
        , heading(_heading)
        , startRoadId(_startRoadId)
        , startLane(_startLane)
        , startDir(_startDir)
        , startSOff(_startSOff)
        , startVOff(_startVOff)
        , startVel(_startVel)
        , startAcc(_startAcc)
        , idleIdx(_idleIdx)
        , idleVel(_idleVel)
        , slowIdx(_slowIdx)
        , slowVel(_slowVel)
        , walkIdx(_walkIdx)
        , walkVel(_walkVel)
        , jogIdx(_jogIdx)
        , jogVel(_jogVel)
        , lookIdx(_lookIdx)
        , waveIdx(_waveIdx)
    {
    }

    // Pedestrian Info
    std::string id;
    std::string name;
    std::string rangeLOD;
    std::string debugLvl;

    // Geometry
    std::string modelFile;
    std::string scale;
    std::string heading;

    // Start position
    std::string startRoadId;
    std::string startLane;
    std::string startDir;
    std::string startSOff;
    std::string startVOff;
    std::string startVel;
    std::string startAcc;

    // Animation mappings
    std::string idleIdx;
    std::string idleVel;
    std::string slowIdx;
    std::string slowVel;
    std::string walkIdx;
    std::string walkVel;
    std::string jogIdx;
    std::string jogVel;
    std::string lookIdx;
    std::string waveIdx;
};

class Pedestrian;
class TRAFFICSIMULATIONEXPORT PedestrianFactory
{
public:
    static PedestrianFactory *Instance();
    static void Destroy();

    void deletePedestrian(Pedestrian *p);
    Pedestrian *createPedestrian(const std::string &name, const std::string &tmpl, const std::string &r, const int l, const int dir, const double pos, const double vOff, const double vel, const double acc);
    Pedestrian *createPedestrian(PedestrianSettings ps);

    int maxPeds() const
    {
        return maximumPeds;
    }

    osgCal::CoreModel *getCoreModel(const std::string &modelFile);
    void parseOpenDrive(xercesc::DOMElement *, const std::string & = ".");

protected:
    PedestrianFactory();
    static PedestrianFactory *__instance;

    std::map<std::string, osg::ref_ptr<osgCal::CoreModel> > coreModelMap;

private:
    void applyTemplateToSettings(std::string tmpl, PedestrianSettings *pedSettings);
    void parseElementForSettings(xercesc::DOMElement *element, PedestrianSettings *pedSettings);

    bool hasAttr(xercesc::DOMElement *element, const char *tag);
    bool tagsMatch(xercesc::DOMElement *element, const char *tag);
    std::string getValOfAttr(xercesc::DOMElement *element, const char *attr);

    std::string xodrDir;
    PedestrianSettings pedDefaults;
    std::list<PedestrianSettings> pedTemplatesList;
    std::list<PedestrianSettings> pedInstancesList;
    int maximumPeds;

    osg::ref_ptr<osg::Group> pedestrianGroup;
};

#endif
