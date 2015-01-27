/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RoadSystem_h
#define RoadSystem_h

#include <string>
#include <map>
#include <vector>
#include <ostream>

#include "Element.h"
#include "Road.h"
#include "Controller.h"
#include "Junction.h"
#include "Fiddleyard.h"
#include <xercesc/dom/DOM.hpp>
#if _XERCES_VERSION >= 30001
#include <xercesc/dom/DOMLSSerializer.hpp>
#else
#include <xercesc/dom/DOMWriter.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>

#define _tile_width 20

struct RoadSystemHeader
{
    RoadSystemHeader()
        : north(0.0)
        , east(0.0)
        , south(0.0)
        , west(0.0)
        , xoffset(0.0)
        , yoffset(0.0)
    {
    }

    std::string date;
    std::string name;
    double north;
    double east;
    double south;
    double west;
    int revMajor;
    int revMinor;
    double version;
    double xoffset;
    double yoffset;
};

//Typ für Rasterung
class VEHICLEUTILEXPORT RoadLineSegment
{
public:
    RoadLineSegment();
    Road *getRoad();
    void setRoad(Road *);
    double get_smax();
    double get_smin();
    void set_smax(double);
    void set_smin(double);
    void check_s(double);

private:
    Road *road;
    double smin, smax;
};

class VEHICLEUTILEXPORT RoadSystem
{
public:
    static RoadSystem *Instance();
    static void Destroy();

    const RoadSystemHeader &getHeader();

    void addRoad(Road *);
    void addController(Controller *);
    void addJunction(Junction *);
    void addFiddleyard(Fiddleyard *);
    void addPedFiddleyard(PedFiddleyard *);
    void addRoadSignal(RoadSignal *);
    void addRoadSensor(RoadSensor *);

    Road *getRoad(int);
    Road *getRoad(std::string);
    int getNumRoads();

    Controller *getController(int);
    int getNumControllers();

    Junction *getJunction(int);
    int getNumJunctions();

    Fiddleyard *getFiddleyard(int);
    int getNumFiddleyards();

    PedFiddleyard *getPedFiddleyard(int);
    int getNumPedFiddleyards();
    void clearPedFiddleyards();
    std::string getRoadId(Road *road);

    RoadSignal *getRoadSignal(int);
    int getNumRoadSignals();

    RoadSensor *getRoadSensor(int);
    int getNumRoadSensors();

    void parseOpenDrive(xercesc::DOMElement *);
    void parseOpenDrive(std::string);

    void writeOpenDrive(std::string);

    void parseCinema4dXml(std::string);

    void parseLandXml(std::string);

    void parseIntermapRoad(const std::string &, const std::string & = "+proj=latlong +datum=WGS84", const std::string & = "+proj=merc");

    Vector2D searchPosition(const Vector3D &, Road *&, double &);
    Vector2D searchPositionFollowingRoad(const Vector3D &, Road *&, double &);

    void analyzeForCrossingJunctionPaths();

    void update(const double &);

    void scanStreets(void);
    bool check_position(int, int);

    std::list<RoadLineSegment *> getRLS_List(int x, int y);

    osg::Vec2d get_tile(double x, double y);

    int current_tile_x;
    int current_tile_y;

    static int _tiles_x;
    static int _tiles_y;
    static float dSpace_v;

protected:
    RoadSystem();

    std::vector<Road *> roadVector;
    std::vector<Controller *> controllerVector;
    std::vector<Junction *> junctionVector;
    std::vector<Fiddleyard *> fiddleyardVector;
    std::vector<PedFiddleyard *> pedFiddleyardVector;
    std::map<std::string, Road *> roadIdMap;
    std::map<std::string, Controller *> controllerIdMap;
    std::map<std::string, Junction *> junctionIdMap;
    std::map<std::string, Fiddleyard *> fiddleyardIdMap;

    std::vector<RoadSignal *> signalVector;
    std::map<std::string, RoadSignal *> signalIdMap;

    std::vector<RoadSensor *> sensorVector;
    std::map<std::string, RoadSensor *> sensorIdMap;

    RoadSystemHeader header;

    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double tile_width;
    double tile_height;

    //Vektor für die Rasterung des Straßenenetzes
    std::vector<std::vector<std::list<RoadLineSegment *> > > rls_vector;
    //vec<vec>: x , vec<rls>: y

private:
    static RoadSystem *__instance;

    int getLineLength(std::vector<double> &XVector, std::vector<double> &YVector, int startIndex, int endIndex, double delta);
};

VEHICLEUTILEXPORT std::ostream &operator<<(std::ostream &, RoadSystem *);

#endif
