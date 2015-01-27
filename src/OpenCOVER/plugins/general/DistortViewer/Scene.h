/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include "Screen.h"
#include "Projector.h"

#include <boost/shared_ptr.hpp>

typedef boost::shared_ptr<Projector> Projector_ptr;

class Scene
{
public:
    Scene(bool load = false, bool visStatus = false);
    virtual ~Scene(void) = 0;
    void init();
    void loadFromXML();
    void saveToXML();
    void resetSceneContent();

    /** Setzt die Screengeometrie
	*
	* @param shapeName Name der neuen Screengeometrie
	*/
    bool setScreenShape(std::string shapeName);

    static Screen *getScreen()
    {
        return screen;
    };
    Projector *getProjector(int no);
    virtual osg::Group *getSceneGroup(int num = 0) = 0;
    virtual void updateScene(int num = 0) = 0;

    int getNumProjectors();
    static bool getVisStatus()
    {
        return visStatus;
    };
    static void setVisStatus(bool status)
    {
        visStatus = status;
    };

    void makeNewProjector(void);
    void deleteProjector(int numProj);

protected:
    static bool visStatus; //Soll Scene zu visualisierung erstellt werden?
    static Screen *screen;
    static std::vector<Projector_ptr> projectors;
};
