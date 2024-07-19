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
    //Restart Buttom
    trainMenu = new ui::Menu("Train", this);
    trainMenu->setText("Train");

    restartButton = new ui::Action(trainMenu, "Restart");
    restartButton->setText("restart");
    restartButton->setCallback([this]() {currentPointIndex = 0; });

    //Speed Slider
    speedSlider = new ui::Slider(trainMenu, "Speed");
    speedSlider->setText("Speed (km/h)");

    speedSlider->setBounds(0.0, 250.0);
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

    for (int i = 0; i < 16; i++) {
        std::cout << "trajectoryPoints1 Point" << i << ":" << trajectoryPoints1[i] << std::endl;
    }

    for (int i = 0; i < 16; i++) {
        std::cout << "trajectoryPoints2 Point" << i << ":" << trajectoryPoints2[i] << std::endl;
    }

    if (trajectoryPoints1.empty() || trajectoryPoints2.empty()) {
        std::cerr << "No points found in the trajectory file or failed to parse the file." << std::endl;
    }

    //findDivergePoint(trajectoryPoints1, trajectoryPoints2);

    return true;
}



//osg::Vec3 lastPosition;
osg::Vec3 directionVectorXY;
//osg::Vec3 TrainDirection;
osg::Matrix rotationMatrix;





bool Train::update()
{
    if (trajectoryPoints1.empty() ||trajectoryPoints2.empty()) {
        return false; // No points or distances to move along
    }

    //Interpolation
    double elapsedTime = cover->frameDuration();
    DistanceSinceLastTime += speed * elapsedTime;
 
    if (useTrack1) {
        trainMoving(reachEnd, trajectoryPoints1, DistanceSinceLastTime, currentPointIndex);
    }
    else {
        trainMoving(reachEnd, trajectoryPoints2, DistanceSinceLastTime, currentPointIndex);
    }

    //std::cout << "elapsedTime: " << elapsedTime << std::endl;
    //std::cout << "DistanceSinceLastTime: " << DistanceSinceLastTime << std::endl;
    //std::cout << "t: " << t << std::endl;
    //std::cout << "interpolatedPosition: " << interpolatedPosition << std::endl;

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

        if (magnitudeProduct != dotProduct) {
            divergePointIndex = i;
            std::cout << "The Train Diverges Here: "  << trajectory1[divergePointIndex];
            return;
        }
    }

    return;
}











//Intropolation
float t;
osg::Vec3 interpolatedPosition;
osg::ref_ptr<osg::MatrixTransform> carPosition;


void Train::trainMoving(bool& checkReachEnd, const std::vector<osg::Vec3>& trajectory, float& distance, size_t& currentPoint) {

    if (checkReachEnd == false) {
        while (distance > (trajectory[currentPoint + 1] - trajectory[currentPoint]).length())
        {
            distance -= (trajectory[currentPoint + 1] - trajectory[currentPoint]).length();
            currentPoint++;
            t = distance / (trajectory[currentPoint + 1] - trajectory[currentPoint]).length();
            interpolatedPosition = trajectory[currentPoint] * (1.0f - t) + trajectory[(currentPoint + 1) % trajectory.size()] * t;


            if (currentPoint == trajectory.size() - 1) {
                checkReachEnd = true;
            }

        }
    }
    else {
        while (distance > (trajectory[currentPoint] - trajectory[currentPoint - 1]).length())
        {
            distance -= (trajectory[currentPoint] - trajectory[currentPoint - 1]).length();
            currentPoint--;
            t = distance / (trajectory[currentPoint] - trajectory[currentPoint - 1]).length();

            if (currentPoint == 1) {
                checkReachEnd = false;
            }
        }
    }

    // Calculate direction vector
    float angle = 0;
    if (currentPoint > 0) {
        directionVectorXY = trajectory[currentPoint] - trajectory[currentPoint - 1];
        directionVectorXY[2] = 0;

        // Normalize the direction vector
        directionVectorXY.normalize();

        // Calculate the angle to rotate based on the direction vector
        angle = atan2(directionVectorXY.y(), directionVectorXY.x());
        angle -= M_PI_2;

    }

    rotationMatrix = osg::Matrix::rotate(angle, osg::Vec3(0, 0, 1));

    osg::Matrix matrix;
    matrix.makeTranslate(interpolatedPosition);
    matrix = rotationMatrix * matrix;
    carPosition->setMatrix(matrix);

}



//void Train::switchTrack(bool useTrack1) {
//    this->useTrack1 = useTrack1;
//}



COVERPLUGIN(Train)