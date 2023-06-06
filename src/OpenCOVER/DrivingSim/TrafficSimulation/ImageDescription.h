#ifndef SUMO_IMAGE_DESCRIPTION_H
#define SUMO_IMAGE_DESCRIPTION_H

#include <osg/MatrixTransform>
#include <string>
#include <fstream>

#include <util/coExport.h>

class TRAFFICSIMULATIONEXPORT ImageDescriptor
{
public:
void update();
void registerRoadUser(int id, const std::string& name, osg::MatrixTransform* transorm);
void unregisterRoadUser(osg::MatrixTransform* transorm);
void open(const std::string& filename);
private:
    std::fstream m_file;
    size_t m_frame_id = 0;
    
    struct RoadUser
    {
        int id;
        std::string name;
        osg::MatrixTransform *transform;
    };
    std::vector<RoadUser> m_roadUsers;

    void writeTimestepHeader();
    void writeTimestepFooter();
    bool writeObject(const RoadUser &roadUser, const osg::Matrix& objToScreen);
   
};

#endif // SUMO_IMAGE_DESCRIPTION_H