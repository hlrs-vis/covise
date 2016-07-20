#include "PointReader.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace visionaray;

PointReader::PointReader(){

}

std::vector<std::string> splitString(const std::string& s, char separator){
    std::vector<std::string> retVal;
    std::string::size_type prev_pos = 0, pos = 0;

    while((pos = s.find(separator, pos)) != std::string::npos){
        std::string substring(s.substr(prev_pos, pos - prev_pos));
        if(substring != "") retVal.push_back(substring);
        prev_pos = ++pos;
    }

    std::string substring(s.substr(prev_pos, pos - prev_pos));
    if(substring != "") retVal.push_back(substring);

    return retVal;
}

void addPoint(point_vector& points,
              color_vector& colors,
              std::vector<std::string>& tokens,
              aabb& bbox,
              int count,
              bool cutUTMdata){

    if(tokens.size() != 7){
        std::cout << "PointReader::addPoint ERROR: size of stringlist != 7" << std::endl;
        return;
    }

    if(cutUTMdata){
        tokens[0] = tokens[0].substr(4,std::string::npos);
        tokens[1] = tokens[1].substr(4,std::string::npos);
    }

    float x = std::stof(tokens[0]);
    float y = std::stof(tokens[1]);
    float z = std::stof(tokens[2]);

    int r = std::stoi(tokens[4]);
    int g = std::stoi(tokens[5]);
    int b = std::stoi(tokens[6]);

//        for(int i = 0; i < tokens.size(); i++){
//            std::cout << "token " << i << " : " << tokens[i] << "  ";
//        }
//        std::cout << std::endl;
//        std::cout << "x: " << x << "  y: " << y << "  z: " << z << "    r: " << r << "  g: " << g << "  b: " << b << std::endl << std::endl;

    sphere_type sp;
    sp.radius = 0.01f;
    sp.center = vec3(x,y,z);
    sp.prim_id = count;
    sp.geom_id = 0;
    points.push_back(sp);
    colors.emplace_back((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);


    if(x < bbox.min.x) bbox.min.x = x; else if(x > bbox.max.x) bbox.max.x = x;
    if(y < bbox.min.y) bbox.min.y = y; else if(y > bbox.max.y) bbox.max.y = y;
    if(z < bbox.min.z) bbox.min.z = z; else if(z > bbox.max.z) bbox.max.z = z;
}


bool PointReader::readFile(std::string filename, float pointSize,
                           point_vector &points,
                           color_vector& colors,
                           aabb &bbox,
                           bool cutUTMdata){

    //open the file
    FILE * f = fopen(filename.c_str(),"r");
    if(f == NULL){
        std::cerr << "PointReader::readFile() Could not open file: " << filename.c_str() << std::endl;
        return false;
    }

    //data storage for ascii lines
    char line[200];

    //read the first line. Sometimes it contains the total number of points in the file
    if(!fgets(line,200,f)) {
        std::cout << "Could not read first line from file " << filename.c_str() << std::endl;
        return false;
    }

    int numPoints = 0;
    int count = 0;
    std::vector<std::string> tokens = splitString(line, ' ');

    if(tokens.size() != 7){
        numPoints = std::stoi(tokens[0]);
        if(numPoints != 0) std::cout << "PointReader::readFile() reading " << numPoints << "points" << std::endl;
    } else {
        addPoint(points,colors,tokens,bbox,count,cutUTMdata);
        count++;
    }

    while(fgets(line,200,f)){

        tokens = splitString(line,' ');
        if(tokens.size() != 7) {
            std::cout << "PointReader::readFile() ERROR: number of tokens not 7 in line " << count << " of file " << filename.c_str() << std::endl;
            break;
        }

        addPoint(points,colors,tokens,bbox,count,cutUTMdata);
        count++;

        if(count % 100000 == 0) std::cout << "PointReader::readFile() reading line " << count << std::endl;
    }


    /*
    //filename = "/data/KleinAltendorf/ausschnitte/test_UTM_KLA2506_2015.pts";
    std::ifstream stream;
    stream.open(filename.c_str());

    if(!stream.is_open())
    {
        std::cerr << "PointReader::readFile() Could not open file: " << filename.c_str() << std::endl;
        return false;
    }
    std::cout << "PointReader::readFile() reading file: " << filename.c_str() << std::endl;

    std::string line;

    int numPoints = 0;
    std::getline(stream, line);
    std::sscanf(line.c_str(),"%d",&numPoints);
    std::cout << "PointReader::readFile() reading " << numPoints << "points" << std::endl;

    int i = 0;
    while(std::getline(stream, line)){

        float x,y,z;
        int r,g,b;
        int ignore;
        std::sscanf(line.c_str(),"%20f %20f %20f %i %i %i %i", &x, &y, &z, &ignore, &r, &g, &b);

        sphere_type sp;
        sp.radius = 0.01f;
        sp.center = vec3(x,y,z);
        sp.prim_id = i;
        //sp.geom_id = current_geom;
        points.push_back(sp);
        colors.emplace_back((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);


        if(x < bbox.min.x) bbox.min.x = x; else if(x > bbox.max.x) bbox.max.x = x;
        if(y < bbox.min.y) bbox.min.y = y; else if(y > bbox.max.y) bbox.max.y = y;
        if(z < bbox.min.z) bbox.min.z = z; else if(z > bbox.max.z) bbox.max.z = z;
        i++;

        if(i > 20) break;
    }

    stream.close();
    std::cout << "done reading" << std::endl;
    */

    return true;
}
