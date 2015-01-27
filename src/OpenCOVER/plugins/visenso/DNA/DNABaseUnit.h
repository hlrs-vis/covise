/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DNA_BASE_UNIT_H
#define _DNA_BASE_UNIT_H

// class for phosphat, zucker, basen
#include <PluginUtil/coVR3DTransRotInteractor.h>

class DNABaseUnitConnectionPoint;

class DNABaseUnit : public opencover::coVR3DTransRotInteractor
{
protected:
    // distance for docking (length of the connection direction vector)
    float radius_;
    float size_;
    std::string geofilename_;
    int num_;
    bool dontMove_;
    bool movedByInteraction_; /// flag if a baseUnit is allready moved because of the interaction
    bool isMoving_;
    bool visible_;
    bool registered_;
    bool presentationMode_;
    char *name_;
    // list of possible connections
    //      /
    //     o -
    //     |
    //      -o
    //      /
    std::list<DNABaseUnitConnectionPoint *> connections_;

    osg::Quat rot180z, rot180y;
    osg::Matrix rotMat10, rotMatm10;

    virtual void createGeometry();
    void sendMatrixToGUI();
    const char *getNameforMessage();
    osg::Matrix calcTransform(osg::Matrix, DNABaseUnitConnectionPoint *cp);
    // set transformation of unit because of connectionPoint cp
    virtual void transform(osg::Matrix m, DNABaseUnitConnectionPoint *cp);
    virtual void sendTransform(osg::Matrix m, DNABaseUnitConnectionPoint *cp);
    virtual void registerAtGui();

public:
    // filename: name of geomtry file, radius for connections and intersection, list of connection names and vectors
    DNABaseUnit(osg::Matrix m, float size, const char *interactorName,
                std::string geofilename, float boundingRadius, int num = 0);

    virtual ~DNABaseUnit();

    // add name, normal, direction
    void addConnectionPoint(DNABaseUnitConnectionPoint *connection);
    virtual DNABaseUnitConnectionPoint *getConnectionPoint(std::string name);
    virtual std::list<DNABaseUnitConnectionPoint *> getAllConnectionPoints()
    {
        return connections_;
    };

    // reposition unit to dock to the other unit
    virtual bool connectTo(DNABaseUnit *otherUnit, DNABaseUnitConnectionPoint *myConnectionPoint = NULL, DNABaseUnitConnectionPoint *otherConnectionPoint = NULL);
    // check if connectionpoint is free (need for adenin, thymin, cytosin, guanin)
    virtual bool isConnectionPossible(std::string)
    {
        return true;
    };

    // check if this BaseUnit is connected to otherUnit
    virtual bool isConnectedTo(DNABaseUnit *otherUnit);

    virtual bool disconnectAll();
    virtual bool isConnected();
    virtual bool isVisible()
    {
        return visible_;
    };
    virtual void setVisible(bool b);

    // set Connection form Gui
    virtual void setConnection(std::string nameConnPoint, std::string nameConnPoint2, bool connected, bool enabled, DNABaseUnit *connObj, bool sendBack = true);
    // send state of connection
    void sendConnectionToGUI(std::string conn1, std::string conn2, int connected, int enabled, std::string connObj);

    // get position, base class returns only matrix
    osg::Vec3 getPosition()
    {
        return getMatrix().getTrans();
    };
    // get ObjectName
    std::string getObjectName();

    // get radius
    float getRadius()
    {
        return radius_;
    };

    // start the interaction (grab pointer, set selected hl, store dcsmat)
    virtual void startInteraction();
    virtual void stopInteraction();
    virtual void doInteraction();

    // update position not interactively but from outside and update positions of connected units
    virtual void sendUpdateTransform(osg::Matrix m, bool sendBack, bool recursive);
    virtual void updateTransform(osg::Matrix m);

    // get Transform matrix
    const osg::MatrixTransform *getMoveTransform()
    {
        return moveTransform.get();
    };

    // returns if baseUnit is moving (connected to an interactor which is running)
    bool isMoving()
    {
        return isMoving_;
    };
    void setMoving(bool b);

    // set MoveByInteraction recursivly to false
    void unsetMoveByInteraction();

    virtual void showConnectionGeom(bool b, std::string connName) = 0;
};
#endif
