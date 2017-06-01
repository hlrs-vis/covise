#include "PointReader.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>

#include <visionaray/detail/color_conversion.h>
#include "rply.h"
#include "rplycallbacks.inl"

#include "rply.h"
#include "rplycallbacks.inl"

using namespace visionaray;

PointReader *
PointReader::instance()
{
    static PointReader *singleton = NULL;
    if (!singleton)
        singleton = new PointReader;
    return singleton;
}

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

void addPtsPoint(point_vector& points,
              std::vector<std::string>& tokens,
              aabb& bbox,
              float pointSize,
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
    sp.radius() = pointSize;
    sp.center() = vec3(x,y,z);
    sp.color() = visionaray::vector<3, visionaray::unorm<8>>((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
    points.push_back(sp);

    if(x < bbox.min.x) bbox.min.x = x; else if(x > bbox.max.x) bbox.max.x = x;
    if(y < bbox.min.y) bbox.min.y = y; else if(y > bbox.max.y) bbox.max.y = y;
    if(z < bbox.min.z) bbox.min.z = z; else if(z > bbox.max.z) bbox.max.z = z;
}

void addXyzPoint(point_vector& points,
              std::vector<std::string>& tokens,
              aabb& bbox,
              float pointSize,
              int count){

    if(tokens.size() != 6){
        std::cout << "PointReader::addPoint ERROR: size of stringlist != 7" << std::endl;
        return;
    }

    float x = std::stof(tokens[0]);
    float y = std::stof(tokens[1]);
    float z = std::stof(tokens[2]);

    int r = std::stoi(tokens[3]);
    int g = std::stoi(tokens[4]);
    int b = std::stoi(tokens[5]);

    sphere_type sp;
    sp.radius() = pointSize;
    sp.center() = vec3(x,y,z);
    sp.color() = visionaray::vector<3, visionaray::unorm<8>>((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
    points.push_back(sp);

    if(x < bbox.min.x) bbox.min.x = x; else if(x > bbox.max.x) bbox.max.x = x;
    if(y < bbox.min.y) bbox.min.y = y; else if(y > bbox.max.y) bbox.max.y = y;
    if(z < bbox.min.z) bbox.min.z = z; else if(z > bbox.max.z) bbox.max.z = z;
}

void add3DPoint(point_vector& points,
                std::vector<std::string>& tokens,
                aabb& bbox,
                float pointSize,
                int count){

    if(tokens.size() != 4){
        std::cout << "PointReader::addPoint ERROR: size of stringlist != 7" << std::endl;
        return;
    }

    float x = std::stof(tokens[0]);
    float y = std::stof(tokens[1]);
    float z = std::stof(tokens[2]);

    float val = std::stof(tokens[3]);
    auto rgb = hue_to_rgb((val + 45.0) / 75.7 ); // plot max. 120 ray interactions..

    sphere_type sp;
    sp.radius() = pointSize;
    sp.center() = vec3(x,y,z);
    sp.color() = visionaray::vector<3, visionaray::unorm<8>>(rgb.x, rgb.y, rgb.z);
    points.push_back(sp);

    if(x < bbox.min.x) bbox.min.x = x; else if(x > bbox.max.x) bbox.max.x = x;
    if(y < bbox.min.y) bbox.min.y = y; else if(y > bbox.max.y) bbox.max.y = y;
    if(z < bbox.min.z) bbox.min.z = z; else if(z > bbox.max.z) bbox.max.z = z;
}


bool PointReader::readFile(std::string filename,
                           float pointSize,
                           std::vector<host_bvh_type> &bvh_vector,
                           aabb &bbox,
                           bool useCache,
                           bool cutUTMdata){

    //optionally load data from cache
    bool binaryLoaded = false;

    //path to binary cache file
    static const std::string cacheDir = "/var/tmp/";
    boost::filesystem::path p(filename);
    std::string basename = p.stem().string();
    std::string binaryPath = cacheDir + basename;

    if(useCache && boost::filesystem::exists(binaryPath)){
        std::cout << "Load binary data from " << binaryPath << '\n';
        host_bvh_type bvh;
        binaryLoaded = loadBvh(binaryPath, bvh);
        if(binaryLoaded) bvh_vector.push_back(bvh);
        std::cout << "Ready\n";
    }

    if (!binaryLoaded)
    {
        std::cout << "PointReader::readFile() reading file " << filename.c_str() << std::endl;

        std::string extension = p.extension().string();
        point_vector points;

        if(extension.compare(".ply") == 0) {
            std::cout << "PointReader::readfile() reading a ply file" << std::endl;

            long nVertices = 0;

            p_ply ply = ply_open(filename.c_str(), NULL, 0, NULL);
            if(!ply)
            {
                std::cout << "Could not open file: " << filename.c_str() << std::endl;
                return false;
            }

            if(!ply_read_header(ply))
            {
                std::cout << "Could not read file header of file: " << filename.c_str() << std::endl;
                return false;
            }

            nVertices = ply_set_read_cb(ply, "vertex", "x", &readVertexCallback, &points, 0);
            ply_set_read_cb(ply, "vertex", "y", &readVertexCallback, &points, 1);
            ply_set_read_cb(ply, "vertex", "z", &readVertexCallback, &points, 2);

            ply_set_read_cb(ply, "vertex", "red", &readColorCallback, &points, 0);
            ply_set_read_cb(ply, "vertex", "green", &readColorCallback, &points, 1);
            ply_set_read_cb(ply, "vertex", "blue", &readColorCallback, &points, 2);

            //m_vertexArray.reserve(nVertices + oldVertexArraySize);
            //m_colorArray.reserve(nVertices + oldVertexArraySize);
            for(long i = 0; i < nVertices; i++){
                sphere_type sp;
                sp.radius() = pointSize;
                sp.center() = visionaray::vec3(0.0f,0.0f,0.0f);
                sp.color() = visionaray::vector<3, visionaray::unorm<8>>(1.0f, 1.0f, 1.0f);
                points.push_back(sp);
            }


            if(!ply_read(ply))
            {
                std::cout << "Could not read file: " << filename.c_str() << std::endl;
                return false;
            }
            //log(QString("Read %1 vertices").arg(nVertices));
            ply_close(ply);

        } else {

            //open the file
            FILE * f = fopen(filename.c_str(),"r");
            if(f == NULL){
                std::cerr << "PointReader::readFile() Could not open file: " << filename.c_str() << std::endl;
                return false;
            }

            //data storage for ascii lines
            char line[200];
            int numPoints = 0;
            int count = 0;

            if(extension.compare(".pts") == 0)
            {
                //read the first line. Sometimes it contains the total number of points in the file
                if(!fgets(line,200,f)) {
                    std::cout << "PointReader::readFile() Could not read first line from file " << filename.c_str() << std::endl;
                    fclose(f);
                    return false;
                }

                std::vector<std::string> tokens = splitString(line, ' ');

                if(tokens.size() != 7){
                    numPoints = std::stoi(tokens[0]);
                    if(numPoints != 0) std::cout << "PointReader::readFile() reading " << numPoints << " points" << std::endl;
                } else {
                    addPtsPoint(points,tokens,bbox,pointSize,count,cutUTMdata);
                    count++;
                }

                while(fgets(line,200,f)){

                    tokens = splitString(line,' ');
                    if(tokens.size() != 7) {
                        std::cout << "PointReader::readFile() ERROR: number of tokens not 7 in line " << count << " of file " << filename.c_str() << std::endl;
                        break;
                    }
                    addPtsPoint(points,tokens,bbox,pointSize,count,cutUTMdata);
                    count++;

                    if(count % 100000 == 0) std::cout << "PointReader::readFile() reading line " << count << std::endl;
                }
            }
            else if(extension.compare(".xyz") == 0)
            {

                while(fgets(line,200,f)){

                    std::vector<std::string> tokens = splitString(line,' ');
                    if(tokens.size() != 6) {
                        std::cout << "PointReader::readFile() ERROR: number of tokens not 6 in line " << count << " of file " << filename.c_str() << std::endl;
                        break;
                    }

                    addXyzPoint(points,tokens,bbox,pointSize,count);
                    count++;

                    if(count % 100000 == 0) std::cout << "PointReader::readFile() reading line " << count << std::endl;
                }
            }
            else if(extension.compare(".3d") == 0)
            {

                double max = -1000000.0;
                double min = 1000000.0;
                double val = 0.0;


                while(fgets(line,200,f)){

                    std::vector<std::string> tokens = splitString(line, ' ');
                    if(tokens.size() != 4) {
                        std::cout << "PointReader::readFile() ERROR: number of tokens not 4 in line " << count << " of file " << filename.c_str() << std::endl;
                        break;
                    }

                    val = std::stod(tokens[3]);
                    if(val < min) min = val;
                    else if(val > max) max = val;

                    add3DPoint(points,tokens,bbox,pointSize,count);
                    count++;

                    if(count % 100000 == 0) std::cout << "PointReader::readFile() reading line " << count << std::endl;
                }

                std::cout << "min: "  << min << "   max: " << max << std::endl;
            }
            fclose(f);
        }

        std::cout << "PointReader::readFile() building bvh for file " << filename.c_str() << std::endl;
        bvh_vector.emplace_back(visionaray::build<host_bvh_type>(points.data(), points.size()));

        if (useCache && !boost::filesystem::exists(binaryPath)) //don't overwrite..
        {
            std::cout << "Storing binary file to " << binaryPath << "...\n";
            storeBvh(binaryPath, bvh_vector.back());
            std::cout << "Ready\n";
        }

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

bool PointReader::loadBvh(std::string filename, host_bvh_type &bvh)
{

    std::ifstream stream;
    stream.open(filename, std::ios::in | std::ios::binary);

    if(!stream.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    uint64_t num_primitives = 0;
    uint64_t num_indices = 0;
    uint64_t num_nodes = 0;    

    stream.read((char*)&num_primitives, sizeof(num_primitives));
    stream.read((char*)&num_indices, sizeof(num_indices));
    stream.read((char*)&num_nodes, sizeof(num_nodes));    

    bvh.primitives().resize(num_primitives);
    bvh.indices().resize(num_indices);
    bvh.nodes().resize(num_nodes);    

    stream.read((char*)bvh.primitives().data(), num_primitives * sizeof(host_bvh_type::primitive_type));
    stream.read((char*)bvh.indices().data(), num_indices * sizeof(int));
    stream.read((char*)bvh.nodes().data(), num_nodes * sizeof(bvh_node));    

    return true;
}

bool PointReader::storeBvh(std::string filename, host_bvh_type &bvh)
{

    std::ofstream stream;
    stream.open(filename, std::ios::out | std::ios::binary);

    if(!stream.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    uint64_t num_primitives = bvh.primitives().size();
    uint64_t num_indices = bvh.indices().size();
    uint64_t num_nodes = bvh.nodes().size();    

    stream.write((char const*)&num_primitives, sizeof(num_primitives));
    stream.write((char const*)&num_indices, sizeof(num_indices));
    stream.write((char const*)&num_nodes, sizeof(num_nodes));

    stream.write((char const*)bvh.primitives().data(), num_primitives * sizeof(host_bvh_type::primitive_type));
    stream.write((char const*)bvh.indices().data(), num_indices * sizeof(int));
    stream.write((char const*)bvh.nodes().data(), num_nodes * sizeof(bvh_node));

    return true;
}
