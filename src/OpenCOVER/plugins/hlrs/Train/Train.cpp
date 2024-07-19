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

#include "Train.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cmath>
#include <osg/Vec3>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace opencover;

Train::Train()
    : coVRPlugin(COVER_PLUGIN_NAME), ui::Owner("Train", cover->ui), currentPointIndex(0)
{
}

Train::~Train()
{
    cover->getObjectsRoot()->removeChild(carPosition);
}

// Define the overload for the << operator for osg::Vec3
std::ostream& operator<<(std::ostream& os, const osg::Vec3& vec) {
    os << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")";
    return os;
}


bool Train::init()
{
    //Restart Button
    trainMenu = new ui::Menu("Train", this);
    trainMenu->setText("Train");

    //Restart Button
    restartButton = new ui::Action(trainMenu, "Restart");
    restartButton->setText("restart");
    restartButton->setCallback([this]() {currentPointIndex = 0; });

    //Switch track
    SwitchTrack = new ui::Action(trainMenu, "SwitchTrack");
    SwitchTrack->setText("SwitchTrack");
    SwitchTrack->setCallback([this]() {useTrack1 = !useTrack1; });

    //Reset at Index 50
    ResetAt50 = new ui::Action(trainMenu, "ResetAt50");
    ResetAt50->setText("ResetAt50");
    ResetAt50->setCallback([this]() {currentPointIndex = 50; });

    //Reset at Index 2480
    ResetAt2500 = new ui::Action(trainMenu, "ResetAt2500");
    ResetAt2500->setText("ResetAt2500");
    ResetAt2500->setCallback([this]() {currentPointIndex = 2500; });

    //Speed Slider
    speedSlider = new ui::Slider(trainMenu, "Speed");
    speedSlider->setText("Speed (km/h)");

    speedSlider->setBounds(0.0, 1000.0);
    speedSlider->setValue(speed);
    speedSlider->setCallback([this](ui::Slider::ValueType value, bool released) {
        speed = value / 3.6;
    });

    //CarPosition SetUp
    carPosition = new osg::MatrixTransform();
    carPosition->setName("carPosition");
    cover->getObjectsRoot()->addChild(carPosition);

    if (!coVRFileManager::instance()->loadFile("C:\\data\\Suedlink\\out2024\\Triebwagen_Vorne.wrl", nullptr, carPosition))
    {
        std::cerr << "Error: Failed to load file 'Triebwagen_Vorne.wrl'" << std::endl;
    }

    // Load Trajectory points
    parseTrajectoryFile("C:\\data\\Suedlink\\out2024\\Trajectory.wrl", trajectoryPoints1);
    parseTrajectoryFile("C:\\data\\Suedlink\\out2024\\Trajectory2.wrl", trajectoryPoints2);

    if (trajectoryPoints1.empty() || trajectoryPoints2.empty()) {
        std::cerr << "No points found in the trajectory file or failed to parse the file." << std::endl;
    }

    //findDivergePoint(trajectoryPoints1, trajectoryPoints2);

    return true;
}



bool Train::update()
{
    if (trajectoryPoints1.empty() ||trajectoryPoints2.empty()) {
        return false; // No points or distances to move along
    }

    double elapsedTime = cover->frameDuration();
    DistanceSinceLastTime += speed * elapsedTime;

    //std::cout << "\n" << "\n" << "2nd Print: Speed now =" << speed << "m/s" << std::endl;
    std::cout << "DistanceSinceLastTime =" << DistanceSinceLastTime << "m" << std::endl;
    //std::cout << "reachEnd or not: " << reachEnd << std::endl;
    std::cout << "currentPointIndex: " << currentPointIndex << std::endl;
 
    //if (useTrack1) {
    //    trainMoving(reachEnd, trajectoryPoints1, DistanceSinceLastTime, currentPointIndex);
    //}
    //else {
    //    trainMoving(reachEnd, trajectoryPoints2, DistanceSinceLastTime, currentPointIndex);
    //}

    const auto& trajectory = useTrack1 ? trajectoryPoints1 : trajectoryPoints2;
    trainMoving(reachEnd, trajectory, DistanceSinceLastTime, currentPointIndex);
    std::cout << "trajectory Point: " << trajectory[currentPointIndex] << "\n" << "\n" << std::endl;

    return true;
}




void Train::parseTrajectoryFile(const std::string& filename, std::vector<osg::Vec3>& trajectoryPoints) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
        return;
    }

    osg::Matrix rotate = osg::Matrix::rotate(M_PI_2, osg::Vec3(1, 0, 0));

    std::string line;
    while (std::getline(file, line)) {
        // Find the line that starts with 'point'
        if (line.find("point [") != std::string::npos) {
            while (std::getline(file, line) && line.find("]") == std::string::npos) {
                osg::Vec3 point;
                if (sscanf(line.c_str(), "%f %f %f,", &point[0], &point[1], &point[2]) == 3) {
                    
                    point = osg::Matrix::rotate(-1.90072, osg::Vec3(0, -1, 0)).preMult(point);
                    point = osg::Matrix::translate(2479.97, -29.969, 932.408).preMult(point);
                   
                    point = rotate.preMult(point);
                    trajectoryPoints.push_back(point);
                }
            }
            break; // Exit after parsing points
        }
    }

    file.close();
}




void Train::findDivergePoint(const std::vector<osg::Vec3>& trajectory1, const std::vector<osg::Vec3>& trajectory2) {

    for (int i = 0; i < 5225; i++) {

        osg::Vec3 vec1 = trajectory1[i + 1] - trajectory1[i];
        osg::Vec3 vec2 = trajectory2[i + 1] - trajectory2[i];

        float magnitudeProduct = vec1.length() * vec2.length();

        float dotProduct = vec1 * vec2;

        if (std::abs(magnitudeProduct - dotProduct) > 1e-6) {   // edited by GPT: Allow for floating-point error
            divergePointIndex = i;
            std::cout << "The Train Diverges Here: " << trajectory1[i].x() << ", " << trajectory1[i].y() << ", " << trajectory1[i].z() << std::endl;
            return;
        }
    }

    return;
}


float t;
osg::Vec3 interpolatedPosition;
osg::ref_ptr<osg::MatrixTransform> carPosition;
osg::Vec3 directionVectorXY;
osg::Matrix rotationMatrix;


void Train::trainMoving(bool &checkReachEnd, const std::vector<osg::Vec3>& trajectory, float& distance, size_t& currentPoint) {
    
    if (!checkReachEnd) { // FORWARD!!!!!!!!!!

        float IntervalDistance = (trajectory[currentPoint + 1] - trajectory[currentPoint]).length();
        std::cout << "Interval Distance = " << IntervalDistance << "m" << std::endl;

        while (distance >= IntervalDistance)
        {
            distance -= IntervalDistance;
            if (currentPoint != trajectory.size() - 2) {
                currentPoint++;
            }
            else {
                checkReachEnd = true;
                speed = -speed;
                distance = 0;
                currentPoint = trajectory.size() - 1;
                std::cout << "\n" << "\n" << "\n" << "Reached Far End!" << "Speed now =" << speed << "m/s" << "\n" << "\n" << "\n" << std::endl;
                return;
            }
        }

        t = distance / (trajectory[currentPoint + 1] - trajectory[currentPoint]).length();
        interpolatedPosition = trajectory[currentPoint] * (1.0f - t) + trajectory[(currentPoint + 1)] * t;

    }
    else {                // BACKWARD!!!!!!!!!!   

        float IntervalDistance = (trajectory[currentPoint] - trajectory[currentPoint - 1]).length();
        std::cout << "Interval Distance = " << IntervalDistance << "m; " << "Now at: " << trajectory[currentPoint] << "; Last at: " << trajectory[currentPoint - 1] << std::endl;

        while (std::abs(distance) > IntervalDistance)
        {
            distance += IntervalDistance;
            if (currentPoint != 1) {
                currentPoint--;
            }
            else {
                interpolatedPosition = trajectory[0] * t + trajectory[1] * (1.0f - t);
                checkReachEnd = false;
                speed = -speed;
                std::cout << "\n" << "\n" << "\n" << "Reached Start!" << "Speed now =" << speed << "m/s" << "\n" << "\n" << "\n" << std::endl;
                currentPoint = 0;
                return;
            }
        }
        t = std::abs(distance) / (trajectory[currentPoint] - trajectory[currentPoint - 1]).length();
        std::cout  << "t =" << t << std::endl;
        interpolatedPosition = trajectory[(currentPoint - 1)] * t + trajectory[currentPoint] * (1.0f - t); 
        std::cout << "interpolatedPosition: " << interpolatedPosition << std::endl;

    }

    //Direction Angle
    if (currentPoint > 0) {
        directionVectorXY = trajectory[currentPoint] - trajectory[currentPoint - 1];
    }
    else {
        directionVectorXY = trajectory[1] - trajectory[0];
    }

    directionVectorXY[2] = 0;
    directionVectorXY.normalize();

    angle = atan2(directionVectorXY.y(), directionVectorXY.x()) - M_PI_2;
    rotationMatrix = osg::Matrix::rotate(angle, osg::Vec3(0, 0, 1));

    osg::Matrix matrix;
    matrix.makeTranslate(interpolatedPosition);

    matrix = rotationMatrix * matrix;
    carPosition->setMatrix(matrix);

}


COVERPLUGIN(Train)