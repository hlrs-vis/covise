/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
 **                                                   	   (C)2016 UKoeln  **
 **                                                                        **
 ** Description: Simple Reader for Volumes of Interest (VOIs)              **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: D. Wickeroth                                                   **
 **                                                                        **
 ** History:                                                               **
 ** June 2016        v1                                                    **
 **                                                                        **
 **                                                                        **
\**************************************************************************/

#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <utility>

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#include <boost/lexical_cast.hpp>

#include "ReadVois.h"
#include "VoisGlobal.h"
#include <do/coDoPolygons.h>


//global variables and constants
const int LINE_SIZE = 8192;
const int Z_TABLE_SIZE = 200;
double zTable[Z_TABLE_SIZE];
double globalTraMatrix[4][4];
const double pixelSize = 1.0;
const int resolution = 512;


//free functions
vector<xyz>::iterator getNearestPoint(xyz v1, vector<xyz>* points){
    //cerr << "ReadVois::getNearestPoint" << endl;

    double shortestDist = std::numeric_limits<double>::max();
    vector<xyz>::iterator retVal = points->end();

    for(vector<xyz>::iterator iter = points->begin(); iter != points->end(); iter++){
        double dist = (v1 - *iter).squaredLength();
        if(dist < shortestDist){
            shortestDist = dist;
            retVal = iter;
        }
    }

    return retVal;
}

triangle_t makeTriangle(int index1, int index2, int index3){
    //cerr << "ReadVois::makeTriangle" << endl;

    triangle_t t;
    t.vertex1 = index1;
    t.vertex2 = index2;
    t.vertex3 = index3;
    return t;
}

std::pair<triangle_t, triangle_t> makeQuad(xyz lowerleft, xyz upperleft, xyz lowerright, xyz upperright){
    //cerr << "ReadVois::makeQuad" << endl;

    if( (lowerleft-upperright).squaredLength() < (upperleft-lowerright).squaredLength() ){
        return std::make_pair(
            makeTriangle(lowerleft.index, upperright.index, upperleft.index),
            makeTriangle(lowerleft.index, lowerright.index, upperright.index)
            );
    } else {
        return std::make_pair(
            makeTriangle(upperleft.index,lowerleft.index,lowerright.index),
            makeTriangle(lowerright.index,upperright.index,upperleft.index)
            );
    }
}

bool getCircle(xyz v1, xyz v2, xyz v3, xyz& center, double& squaredRadius){
    //cerr << "ReadVois::getCircle" << endl;

    if(v1.z != v2.z || v1.z != v3.z){
        cerr << "getCircle ERROR: the z-coordinates are not equal" << endl;
        return false;
    }

    //remember z for return Value
    int zVal = v1.z;

    //homogenisierung
    v1.z = 1.0;
    v2.z = 1.0;
    v3.z = 1.0;

    //Mittelsenkrechte1
    xyz a1 = (v1 + v2) / 2.0;
    xyz a2 = a1 + xyz(-(v2-v1).y,(v2-v1).x,0);
    xyz ms1 = cross(a1,a2);

    //Mittelsenkrechte2
    xyz b1 = (v2 + v3) / 2.0;
    xyz b2 = b1 + xyz(-(v3-v2).y,(v3-v2).x,0);
    xyz ms2 = cross(b1,b2);

    //Mittelpunkt = Schnittpunkt der Mittelsenkrechten
    xyz mp = cross(ms1,ms2);

    mp.homogen();
    if(mp.z == 0){
        squaredRadius = 0.0;
        return false;
    }

    squaredRadius = (mp - v1).squaredLength();
    mp.z = zVal;
    center = mp;
    return true;
}

bool isDelaunay(contour_t& contour, vector<xyz>::iterator startIter){
    //cerr << "ReadVois::isDelaunay" << endl;

    vector<xyz>::iterator midIter = contour.getNextPoint(startIter);
    vector<xyz>::iterator endIter = contour.getNextPoint(midIter);

    xyz intersection(0,0,0);
    double squaredRadius;

    if(!getCircle(*startIter,*midIter,*endIter,intersection,squaredRadius)){
        //cerr << "Not a circle start:" << *startIter << "  mid:" << *midIter << "  end:" <<  *endIter << "  intersection:" <<  intersection << "  squaredRadius:" <<  squaredRadius << endl;
        return false;
    }

    vector<xyz>::iterator iter = contour.getNextPoint(endIter);
    while(iter != startIter){
        if((*iter - intersection).squaredLength() < squaredRadius) return false;
        iter = contour.getNextPoint(iter);
    }

    return true;
}

void triangulatePoly(contour_t contour, std::vector<triangle_t>& triangles){
    //cerr << "ReadVois::triangulatePoly" << endl;

    if(contour.points.size() == 0) {
        cerr << "ERROR: triangulatePoly size = 0" << endl;
        return;
    } else if(contour.points.size() == 1) {
        cerr << "ERROR: triangulatePoly size = 1" << endl;
        return;
    } else if(contour.points.size() == 2) {
        cerr << "ERROR: triangulatePoly size = 2" << endl;
        return;
    } else if(contour.points.size() == 3) {
        triangles.push_back(makeTriangle(contour.points[0].index, contour.points[1].index, contour.points[2].index));
        return;
    } else {

        bool startFound = false;
        vector<xyz>::iterator startIter = contour.points.begin();
        vector<xyz>::iterator midIter, endIter;

        while(!startFound){
            midIter = contour.getNextPoint(startIter);
            endIter = contour.getNextPoint(midIter);

            xyz start = *startIter;
            xyz mid = *midIter;
            xyz end = *endIter;

            xyz startNormal = cross((mid-start),xyz(0,0,1));
            startNormal.normalize();
            xyz endMid = end - mid;
            endMid.normalize();
            double angle = dot(startNormal, endMid);

            if(angle > 0) {
                startIter = contour.getNextPoint(startIter);
                continue;
            }

            if(!isDelaunay(contour,startIter)){
                startIter = contour.getNextPoint(startIter);
                continue;
            }

            startFound = true;
            triangles.push_back(makeTriangle(startIter->index, midIter->index, endIter->index));
        }

        contour_t newContour;
        for(vector<xyz>::iterator newIter = contour.points.begin(); newIter != contour.points.end(); newIter++){
            if(newIter != midIter) newContour.points.push_back(*newIter);
        }
        triangulatePoly(newContour, triangles);

    }
}

void writePly(coBooleanParam **voiActive,
              const std::vector<voi_t> &voiVector,
              const std::vector<triangle_t> *triangles){
    std::string filename = "/home/zellmans/vois.ply";

    // open ply file
    FILE *file = NULL;
    if ((file = fopen(filename.c_str(), "w")) == NULL)
    {
        cerr << "Opening " <<filename.c_str() << " failed!";
        return;
    }

    //get total num of vertices
    int numVertices = 0;
    int numTriangles = 0;
    int voi = 0;
    for(vector<voi_t>::const_iterator voi_iter = voiVector.begin(); voi_iter != voiVector.end(); ++voi_iter){
        if(voiActive[voi]->getValue()){
            for(vector<contour_t>::const_iterator con_iter = voi_iter->contours.begin(); con_iter != voi_iter->contours.end(); con_iter++){
                numVertices += con_iter->points.size();
            }
            numTriangles += triangles[voi].size();
        }
        ++voi;
    }

    fprintf(file, "ply\n");
    fprintf(file, "format ascii 1.0\n");
    fprintf(file, "comment generated by Daniel\n");
    fprintf(file, "element vertex %i\n", numVertices);
    fprintf(file, "property float x\n");
    fprintf(file, "property float y\n");
    fprintf(file, "property float z\n");
    fprintf(file, "property uchar red\n");
    fprintf(file, "property uchar green\n");
    fprintf(file, "property uchar blue\n");
    fprintf(file, "element face %i\n", numTriangles);
    fprintf(file, "property list uchar int vertex_indices\n");
    fprintf(file, "end_header\n");

    int colors[6][3]= {
        {255,0,0},
        {255,255,0},
        {0,255,0},
        {0,255,255},
        {0,0,255},
        {255,0,255}
    };

    //write vertices
    int count = 0;
    voi = 0;
    for(vector<voi_t>::const_iterator voi_iter = voiVector.begin(); voi_iter != voiVector.end(); ++voi_iter){
        if(voiActive[voi]->getValue()){
            for(vector<contour_t>::const_iterator con_iter = voi_iter->contours.begin(); con_iter != voi_iter->contours.end(); con_iter++){
                int colorIndex = count % 6;
                for(vector<xyz>::const_iterator point_iter = con_iter->points.begin(); point_iter != con_iter->points.end(); point_iter++) {
                    fprintf(file, "%f %f %f %i %i %i\n", point_iter->x, point_iter->y, point_iter->z, colors[colorIndex][0], colors[colorIndex][1], colors[colorIndex][2]);
                }
                count++;
            }
        }
        ++voi;
    }

    count = 0;
    voi = 0;
    for (; voi < MAXVOIS; ++voi){
        if(voiActive[voi]){
            for(vector<triangle_t>::const_iterator tri_iter = triangles[voi].begin(); tri_iter != triangles[voi].end(); tri_iter++) {
                fprintf(file, "3 %i %i %i\n", tri_iter->vertex1, tri_iter->vertex2, tri_iter->vertex3);
            }
        }
    }

    fclose(file);
}

ReadVois::ReadVois(int argc, char *argv[])
    : coModule(argc, argv, "Simple VOI Reader")
{
    // the output port
    m_polygonPort = addOutputPort("polygons", "Polygons", "geometry polygons");

    // select the vois-file name with a file browser
    m_voiFileParam = addFileBrowserParam("VOIsFile", "vois file");
    m_voiFileParam->setValue("/data/stereotaxie/mr01.vois", "*.vois");

    for (int i = 0; i < MAXVOIS; ++i)
    {
        std::string str1 = "UseVOI_" + boost::lexical_cast<std::string>(i);
        std::string str2 = "VOI " + boost::lexical_cast<std::string>(i) + " active";
        m_voiActive[i] = addBooleanParam(str1.c_str(), str2.c_str());
        m_voiActive[i]->setValue(i);
    }
}

ReadVois::~ReadVois()
{
}

void ReadVois::quit()
{
}

int ReadVois::compute(const char *port)
{
    (void)port;

    // get the file name
    m_filename = m_voiFileParam->getValue();

    if (m_filename != NULL)
    {
        // open the files
        if (openFile())
        {
            sendInfo("File %s open", m_filename);

            //init some global variables
            for(int i = 0; i < Z_TABLE_SIZE; i++){
                zTable[i] = i;
            }

            for(int i = 0; i < 4; i++){
                for(int j = 0; j < 4; j++){
                    globalTraMatrix[i][j] = 0.0;
                }
            }
            globalTraMatrix[0][0] = 1.0;
            globalTraMatrix[1][1] = 1.0;
            globalTraMatrix[2][2] = 1.0;
            globalTraMatrix[3][3] = 1.0;

            //delete all old data that may be left over from last run
            int voi = 0;
            for(vector<voi_t>::iterator voi_iter = voiVector.begin(); voi_iter != voiVector.end(); ++voi_iter){
                triangles[voi].clear();
                for(vector<contour_t>::iterator con_iter = voi_iter->contours.begin(); con_iter != voi_iter->contours.end(); con_iter++){
                    con_iter->points.clear();
                }
                voi_iter->contours.clear();
                ++voi;
            }
            voiVector.clear();

            // read the file, create the vois
            if(readFile()){
                //printVois();
                if(triangulate()){
                    //writePly(m_voiActive, voiVector, triangles);    //write vertices and triangles to an ascii coded ply-file
                    sendDataToCovise();
                }
                else
                {
                    sendError("Error triangulating VOIs from file %s", m_filename);
                    return FAIL;
                }
            }
            else
            {
                sendError("Error parsing file %s", m_filename);
                return FAIL;
            }
        }
        else
        {
            sendError("Error opening file %s", m_filename);
            return FAIL;
        }
    }
    else
    {
        sendError("ERROR: fileName is NULL");
        return FAIL;
    }
    return SUCCESS;
}

bool ReadVois::openFile()
{
    //sendInfo("ReadVois::Opening file %s", filename);

    // open the vois file
    if ((m_file = Covise::fopen((char *)m_filename, "rb")) == NULL)
    {
        sendError("ERROR: Can't open file >> %s", m_filename);
        return false;
    }
    else
    {
        return true;
    }
}

bool ReadVois::readFile()
{
    //std::cerr << "ReadVois::readFile()" << std::endl;

    // read file header
    int  version_number = 0;
    int  voi_header     = 0;
    char patient_name[200];
    char dummy_name[80];
    int  total_no_vois  = 0;
    int  no_slices      = 0;
    size_t bytes_read = 0;

    bytes_read += fread(&version_number,   4,  1, m_file);   // 340
    bytes_read += fread(&voi_header,       4,  1, m_file);   // 2048
    bytes_read += fread(patient_name,      1, 80, m_file);
    bytes_read += fread(dummy_name,        1, 80, m_file);   // empty
    bytes_read += fread(&total_no_vois,    4,  1, m_file);   // 20
    bytes_read += fread(&no_slices,        4,  1, m_file);

    if(version_number != 340){
        sendError("ERROR: file has wrong version number. Should be '340', but it is '%i'", version_number);
        return false;
    }

    // jump to end of head / beginning of data
    fseek(m_file, voi_header, SEEK_SET);

    // read the description of each voi
    for (int voiIndex = 0; voiIndex < total_no_vois; voiIndex++)
    {
        char      name[40];
        int       property;
        int       first_slice;
        int       last_slice;
        int       color;

        bytes_read += fread(&property,      4,  1, m_file);
        bytes_read += fread( name,          1, 40, m_file);
        bytes_read += fread(&first_slice,   4,  1, m_file);
        bytes_read += fread(&last_slice,    4,  1, m_file);
        bytes_read += fread(&color,         4,  1, m_file);

#if 0
        std::cout << "property: " << property << '\n'; 
        std::cout << "name: " << name << '\n'; 
        std::cout << "first_slice" << first_slice << '\n'; 
        std::cout << "last_slice " << last_slice << '\n'; 
        std::cout << "color " << color << '\n';
        std::cout << "no slices: " << no_slices << '\n';
#endif

        if(voiIsValid(property)){
            voi_t voi;

            voi.voi_name = name;
            voi.voi_property = property;
            voi.voi_first_slice = first_slice;
            voi.voi_last_slice = last_slice;
            voi.voi_color = color;
            voi.voi_index = voiIndex;

            voiVector.push_back(voi);
        }
    }

    // loop for all slices
    for (int slice = 0; slice < no_slices; slice++)
    {
        // loop for all VOI's
        for (int voi = 0; voi < MAXVOIS; voi++)
        {
            int no_contours = 0;
            bytes_read += fread(&no_contours, 4, 1, m_file);

            if (no_contours != 0)
            {
                int no_points = 0;
                bytes_read += fread(&no_points, 4, 1, m_file);

                //We have to read the points no matter what
                vector<xy> pointVector;
                for(int j = 0; j < no_points; j++){
                    xy point_xy;
                    bytes_read += fread(&point_xy, sizeof(xy), 1, m_file);
                    pointVector.push_back(point_xy);
                }

                //if the VOI exists, copy points to VOI (only valid VOIs exist)
                vector<voi_t>::iterator voi_iter = getVoiWithIndex(voiVector,voi);
                if(voi_iter != voiVector.end()){

                    contour_t contour;
                    contour.no_slice = slice + 1;
                    double g_factor = resolution * pixelSize / 1023.0;

                    int count = 0;
                    for(vector<xy>::iterator point_iter = pointVector.begin(); point_iter != pointVector.end(); point_iter++) {

                        // convert from generalized image to image coordinates
                        double value_x   = (511.5 - point_iter->x) * g_factor;
                        double value_y   = (511.5 - point_iter->y) * g_factor;
                        double value_z   = zTable[slice];

                        //convert from image coordinates to stereotactic coordinate system
                        xyz point_xyz;
                        point_xyz.x =
                            value_x * globalTraMatrix[0][0] +
                            value_y * globalTraMatrix[1][0] +
                            value_z * globalTraMatrix[2][0] +
                            globalTraMatrix[3][0];

                        point_xyz.y =
                            value_x * globalTraMatrix[0][1] +
                            value_y * globalTraMatrix[1][1] +
                            value_z * globalTraMatrix[2][1] +
                            globalTraMatrix[3][1];

                        point_xyz.z =
                            value_x * globalTraMatrix[0][2] +
                            value_y * globalTraMatrix[1][2] +
                            value_z * globalTraMatrix[2][2] +
                            globalTraMatrix[3][2];

                        point_xyz.index = -1;

                        //std::cout << point_xyz << '\n';

                        //add coordinates to voi-datastructure
                        contour.points.push_back(point_xyz);
                        count++;
                    }

                    //std::cout << '\n';

                    voi_iter->contours.push_back(contour);

                } else {
                    sendInfo("Could not find Voi with index %i in voiVector. VOI invalid?",voi);
                    //return false;
                }

            }  // no_contours != 0
        }  // loop for all VOI's
    } // loop for all slices

    // close voi file
    fclose(m_file);

    //give each point/xyz a unique index
    int index = 0;
    for(int voiIndex = 0; voiIndex < voiVector.size(); voiIndex ++) {
        if(m_voiActive[voiIndex]->getValue()){
            for(int contourIndex = 0; contourIndex < voiVector[voiIndex].contours.size(); contourIndex++) {
                for(int pointIndex = 0; pointIndex < voiVector[voiIndex].contours[contourIndex].points.size(); pointIndex++) {
                    voiVector[voiIndex].contours[contourIndex].points[pointIndex].index = index;
                    index++;
                }
            }
        }
    }

    return true;
}

void ReadVois::printVois(){
    //std::cerr << "ReadVois::printVois()" << std::endl;

    for(vector<voi_t>::iterator voi_iter = voiVector.begin(); voi_iter != voiVector.end(); ++voi_iter){
        cerr << endl << "voi index: " << voi_iter->voi_index << endl;
        cerr << "voi_name: " << voi_iter->voi_name << endl;
        cerr << "voi_property: " << voi_iter->voi_property << endl;
        cerr << "voi_first_slice: " << voi_iter->voi_first_slice << endl;
        cerr << "voi_last_slice: " << voi_iter->voi_last_slice << endl;
        cerr << "voi_color: " << voi_iter->voi_color << endl;

        for(vector<contour_t>::iterator con_iter = voi_iter->contours.begin(); con_iter != voi_iter->contours.end(); con_iter++){
            cerr << "slice : " << con_iter->no_slice << endl;
            for(vector<xyz>::iterator point_iter = con_iter->points.begin(); point_iter != con_iter->points.end(); point_iter++) {
                cerr << point_iter->x << " " << point_iter->y << " " << point_iter->z << endl;
            }
        }
    }
}

bool ReadVois::triangulate()
{
    //std::cerr << "ReadVois::triangulate()" << std::endl;

    int voi = 0;

    //iterate over all VOIs
    for(vector<voi_t>::iterator voiIter = voiVector.begin(); voiIter != voiVector.end(); voiIter++){
        triangles[voi].clear();
        if(m_voiActive[voi]->getValue()){
            cerr << "ReadVois::triangulate() triangulating voi " << voi + 1 << endl;

            //iterate over all contours
            for(vector<contour_t>::iterator contourIter = voiIter->contours.begin(); contourIter != voiIter->contours.end(); contourIter++){
                //cerr << "ReadVois::triangulate() triangulating contour " << std::distance(voiIter->contours.begin(), contourIter) + 1 << endl;

                //traingulate top and bottom contour
                if(contourIter == voiIter->contours.begin() || contourIter == (voiIter->contours.end() - 1)){
                    //cerr << "ReadVois::triangulate() triangulating top or bottom" << endl;
                    triangulatePoly(*contourIter, triangles[voi]);
                }

                //traingulate two contours, but not the last one...
                if(contourIter != (voiIter->contours.end() - 1)){
                    //cerr << "ReadVois::triangulate() triangulating two contours" << endl;

                    //get the next contour
                    vector<contour_t>::iterator lowerContour = contourIter;
                    vector<contour_t>::iterator upperContour = contourIter + 1;

                    //find a good start, where the upper and lower start vertices both are nearest to each other
                    vector<xyz>::iterator lowerVertex = lowerContour->points.begin();
                    vector<xyz>::iterator upperVertex;

                    for(; lowerVertex < lowerContour->points.end(); lowerVertex++ ){
                        upperVertex = getNearestPoint(*lowerVertex, &(upperContour->points));
                        if(lowerVertex == getNearestPoint(*upperVertex, &(lowerContour->points))){
                            break; //this pair is good start. lower is the nearest to upper and upper is the nearest to lower.
                        }
                        if(lowerVertex == (lowerContour->points.end() - 1))
                        {
                            cerr << "ERROR: no good start found for contours. let's hope it works anyways' " << endl;
                        }
                    }

                    //remember where we started
                    vector<xyz>::iterator lowerStart = lowerVertex;
                    vector<xyz>::iterator upperStart = upperVertex;

                    //loop over lower and upper contour at the same time
                    do {
                        xyz lower1 = *lowerVertex;
                        xyz lower2 = *(lowerContour->getNextPoint(lowerVertex));
                        xyz upper1 = *upperVertex;
                        xyz upper2 = *(upperContour->getNextPoint(upperVertex));

                        bool advanceLower = false;
                        bool advanceUpper = false;

                        double dist_l1u2 = (lower1 - upper2).squaredLength();
                        double dist_u1l2 = (upper1 - lower2).squaredLength();
                        double dist_l2u2 = (lower2 - upper2).squaredLength();

                        if(        dist_l1u2 <= dist_u1l2 && dist_l1u2 <= dist_l2u2) {
                            //cerr << "dist_l1u2 is the smallest" << endl;
                            triangles[voi].push_back(makeTriangle(lower1.index, upper2.index, upper1.index));
                            advanceUpper = true;

                        } else if (dist_u1l2 <= dist_l1u2 && dist_u1l2 <= dist_l2u2) {
                            //cerr << "dist_u1l2 is the smallest" << endl;
                            triangles[voi].push_back(makeTriangle(upper1.index, lower1.index, lower2.index));
                            advanceLower = true;

                        } else if (dist_l2u2 <= dist_l1u2 && dist_l2u2 <= dist_u1l2) {
                            //cerr << "dist_l2u2 is the smallest" << endl;
                            std::pair<triangle_t, triangle_t> qs = makeQuad(lower1,upper1,lower2,upper2);
                            triangles[voi].push_back(qs.first);
                            triangles[voi].push_back(qs.second);
                            advanceLower = true;
                            advanceUpper = true;

                        } else {
                            cerr << "ERROR: no smallest l1u2: " << dist_l1u2 << "  u1l2: " << dist_u1l2 << "  l2u2: " << dist_l2u2 << endl;
                            return false;
                        }

                        if(advanceLower) lowerVertex = lowerContour->getNextPoint(lowerVertex);
                        if(advanceUpper) upperVertex = upperContour->getNextPoint(upperVertex);

                    } while ((lowerVertex != lowerStart || upperVertex != upperStart));
                }
            }//iterate over all contours
        }//if void active
        ++voi;
    }//iterate over all VOIs

    return true;
}

void ReadVois::sendDataToCovise()
{
    std::cerr << "ReadVois::sendDataToCoise()" << std::endl;

    //get the total number of vertices
    int numVertices = 0;
    int voi = 0;
    for(vector<voi_t>::iterator voi_iter = voiVector.begin(); voi_iter != voiVector.end(); ++voi_iter){
        if(m_voiActive[voi]->getValue()){
            for(vector<contour_t>::iterator con_iter = voi_iter->contours.begin(); con_iter != voi_iter->contours.end(); con_iter++){
                numVertices += con_iter->points.size();
            }
        }
        ++voi;
    }

    //create three float arrays to store vertex coordinates temporarily
    std::vector<float> x_c(numVertices);
    std::vector<float> y_c(numVertices);
    std::vector<float> z_c(numVertices);

    //copy vertex data from vois to arrays
    int idx = 0;
    voi = 0;
    for(vector<voi_t>::iterator voi_iter = voiVector.begin(); voi_iter != voiVector.end(); ++voi_iter){
        if(m_voiActive[voi]->getValue()){
            for(vector<contour_t>::iterator con_iter = voi_iter->contours.begin(); con_iter != voi_iter->contours.end(); con_iter++){
                for(vector<xyz>::iterator point_iter = con_iter->points.begin(); point_iter != con_iter->points.end(); point_iter++) {
                    x_c[idx] = point_iter->x;
                    y_c[idx] = point_iter->y;
                    z_c[idx] = point_iter->z;
                    idx++;
                }
            }
        }
        ++voi;
    }

    //get the total number of triangles
    int numTriangles = 0;
    voi = 0;
    for ( ; voi < MAXVOIS; ++voi)
        if (m_voiActive[voi])
            numTriangles += triangles[voi].size();

    //create two int arrays to store triangle info temporarily
    std::vector<int> v_l(numTriangles * 3); //for each triangle 3 integer indices into x_c/y_c/z_c
    std::vector<int> pol_l(numTriangles); //for each triangle one index into v_l

    //copy triangle info to arrays
    voi = 0;
    idx = 0;
    for ( ; voi < MAXVOIS; ++voi)
    {
        if(m_voiActive[voi]->getValue()){
            for(std::vector<triangle_t>::iterator iter = triangles[voi].begin(); iter != triangles[voi].end(); iter++){
                pol_l[idx] = idx*3;  // = 0 3 6 9 ...
                v_l[3*idx + 0] = iter->vertex1;
                v_l[3*idx + 1] = iter->vertex2;
                v_l[3*idx + 2] = iter->vertex3;
                idx++;
            }
        }
    }

    sendInfo("ReadVois:: found %d vertices, created %d triangles", numVertices, numTriangles);

    const char *polygonObjectName; // output object name assigned by the controller
    polygonObjectName = m_polygonPort->getObjName(); // get the COVISE output object name from the controller

    // create the COVISE output object
    coDoPolygons *polygonObject; // output object
    polygonObject = new coDoPolygons(polygonObjectName, numVertices, &x_c[0], &y_c[0], &z_c[0], numTriangles * 3, &v_l[0], numTriangles, &pol_l[0]);
    m_polygonPort->setCurrentObject(polygonObject);

    // set the vertex order for twosided lighting in the renderer
    polygonObject->addAttribute("vertexOrder", "2");
}

MODULE_MAIN(IO, ReadVois)
