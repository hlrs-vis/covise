/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
/****************************************************************************\
 **                                                            (C)2020 HLRS  **
 **                                                                          **
 ** Description: Camera position and orientation optimization                **
 **                                                                          **
 **                                                                          **
 ** Author: Matthias Epple	                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** April 2020  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <vector>
#include <memory>


#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Owner.h>
class Camera
{
public:
    void calcVisibilityMatrix()
    {
        std::cout<<"calc"<<std::endl;
    }

private:
    std::vector<int> visMat;

};
class SafetyZone
{

};


typedef std::unique_ptr<Camera> upCamera;
typedef std::unique_ptr<SafetyZone> upSafetyZone;

//Singleton Class
class Data
{
public:
    Data(const Data& other) = delete;
    Data operator=(const Data& other) = delete;
    
    static Data& GetInstance()
    {
        static Data instance;
        return instance;
    }

    static const std::vector<upCamera>& GetCameras(){return GetInstance().m_Cameras;}
    static const std::vector<upSafetyZone>& GetSafetyZones(){return GetInstance().m_SafetyZones;}

    static void AddCamera(upCamera camera)
    {
        GetInstance().m_Cameras.push_back(std::move(camera));
    }
    static void AddSafetyZone(upSafetyZone safetyZone)
    {
        GetInstance().m_SafetyZones.push_back(std::move(safetyZone));
    }

private:
    Data(){}

    std::vector<upCamera>& IGetCameras(){return m_Cameras;}
    std::vector<upSafetyZone>& IGetSafetyZones(){return m_SafetyZones;}

    std::vector<upCamera> m_Cameras;
    std::vector<upSafetyZone> m_SafetyZones;
};
class SensorPlacementPlugin :public opencover::coVRPlugin, public opencover::ui::Owner
{
    public:
    SensorPlacementPlugin();
    bool init() override;
    void preFrame() override;

    private:
    
};