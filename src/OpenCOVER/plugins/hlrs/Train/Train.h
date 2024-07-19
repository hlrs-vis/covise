/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Train OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

//#include <cover/coVRPlugin.h>
//
//class Train : public opencover::coVRPlugin
//{
//public:
//    Train();
//    ~Train();
//    osg::ref_ptr<osg::MatrixTransform> carPosition; 
//    virtual bool update();
//    osg::Matrix newPosition;
//};
//#endif

#ifndef TRAIN_H
#define TRAIN_H

#include <cover/coVRPlugin.h>
#include <osg/MatrixTransform>
#include <osg/Vec3>
#include <vector>
#include <string>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>


class Train : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    Train(); // Constructor
    virtual ~Train(); // Destructor
    bool init();
    virtual bool update(); // Update method to move the train along the trajectory

    // Function to parse the .wrl file and extract points
    void parseTrajectoryFile(const std::string& filename, std::vector<osg::Vec3>& trajectoryPoints);
    
    void findDivergePoint(const std::vector<osg::Vec3>& trajectory1, const std::vector<osg::Vec3>& trajectory2);

    //void switchTrack(bool useTrack1);
    bool reachEnd = false;
    int divergePointIndex;


    void trainMoving(bool& checkReachEnd, const std::vector<osg::Vec3>& trajectory, float& distance, size_t& currentPoint);

    float DistanceSinceLastTime = 0;
    float speed = 20;

private:
    osg::ref_ptr<osg::MatrixTransform> carPosition;

    std::vector<osg::Vec3> trajectoryPoints1; // Vectors representing the trajectory
    std::vector<osg::Vec3> trajectoryPoints2;

    size_t currentPointIndex=0; 

    opencover::ui::Menu *trainMenu;
    opencover::ui::Action *restartButton;
    opencover::ui::Action* switchTrackButton;
    opencover::ui::Slider *speedSlider;
    bool useTrack1 = true;

};

#endif // TRAIN_H
