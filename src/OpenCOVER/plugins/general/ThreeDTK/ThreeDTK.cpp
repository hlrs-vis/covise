/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2013 HLRS  **
 **                                                                          **
 ** Description: ThreeDTK Plugin (loads and renders PointCloud)              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Nov-01  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#ifdef WIN32
#include <WinSock2.h>
#include <Windows.h>
#endif
#include "ThreeDTK.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>
#include "slam6d/point_type.h"
#include "show/display.h"
#include "show/show_Boctree.h"
#include "show/show.h"

#include "slam6d/globals.icc"

/* This vector contains the pointer to a vertex array for
 * all colors (inner vector) and all scans (outer vector)
 */
vector<vector<vertexArray *> > vvertexArrayList;

/**
 * Storing of AlgoType for all frames
 */
vector<vector<Scan::AlgoType> > MetaAlgoType;

/**
 * Storing of all transformation (frames for animation) of all scans
 */
vector<vector<double *> > MetaMatrix;

ThreeDTK *plugin = NULL;

// Defines for Point Semantic
#define TYPE_UNKNOWN 0x0000
#define TYPE_OBJECT 0x0001
#define TYPE_GROUND 0x0002
#define TYPE_CEILING 0x0003

FileHandler fileHandler[] = {
    { NULL,
      ThreeDTK::loadFile,
      ThreeDTK::unloadFile,
      "oct" }
};

int ThreeDTK::loadFile(const char *fn, osg::Group *parent)
{
    if (plugin)
        return plugin->loadFile(fn);

    return -1;
}

int ThreeDTK::unloadFile(const char *fn)
{
    if (plugin)
    {
        //plugin->unloadData(fn);
        return 0;
    }

    return -1;
}

ThreeDTK::ThreeDTK()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "ThreeDTK::ThreeDTK\n");

    drawable = new co3dtkDrawable();
    drawable->setUseDisplayList(false);

    mtl = new osg::Material();
    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    geoState = new osg::StateSet();
    geoState->setGlobalDefaults();
    geoState->setAttributeAndModes(mtl.get(), osg::StateAttribute::ON);
    geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    geoState->setMode(GL_BLEND, osg::StateAttribute::ON);
    geoState->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    geoState->setNestRenderBins(false);

    node = new osg::Geode();
    node->setStateSet(geoState.get());
    node->addDrawable(drawable.get());
    node->setNodeMask(node->getNodeMask() & ~Isect::Intersection);
    cover->getObjectsRoot()->addChild(node.get());

    coVRFileManager::instance()->registerFileHandler(&fileHandler[0]);
    plugin = this;
}

// this is called if the plugin is removed at runtime
ThreeDTK::~ThreeDTK()
{
    fprintf(stderr, "ThreeDTK::~ThreeDTK\n");
    cover->getObjectsRoot()->removeChild(node.get());
}

void
ThreeDTK::preFrame()
{
    /*
  long time = GetCurrentTimeInMilliSec();
  double min = 0.000000001;
  double max = 1.0;
  LevelOfDetail *= 1.0 + adaption_rate*(lastfps - idealfps)/idealfps;
  if (LevelOfDetail > max) LevelOfDetail = max;
  else if (LevelOfDetail < min) LevelOfDetail = min;

  
  if (pointmode == 1 ) {
    fullydisplayed = true;
  } else {
    unsigned long td = (GetCurrentTimeInMilliSec() - time);
    if (td > 0)
      lastfps =  1000.0/td;
    else
      lastfps = 1000.0;
    fullydisplayed = false;
  }
  */
}

/*
 * A function that read the .frame files created by slam6D
 *
 * @param dir the directory
 * @param start starting at scan number 'start'
 * @param end stopping at scan number 'end'
 * @param read a file containing a initial transformation matrix and apply it
 */
int ThreeDTK::readFrames(string dir, int start, int end, bool readInitial, reader_type &type)
{

    double initialTransform[16];
    if (readInitial)
    {
        cout << "Initial Transform:" << endl;
        string initialTransformFileName = dir + "initital.frame";
        ifstream initial_in(initialTransformFileName.c_str());
        if (!initial_in.good())
        {
            cout << "Error opening " << initialTransformFileName << endl;
            exit(-1);
        }
        initial_in >> initialTransform;
        cout << initialTransform << endl;
    }

    ifstream frame_in;
    int fileCounter = start;
    string frameFileName;
    for (;;)
    {
        if (end > -1 && fileCounter > end)
            break; // 'nuf read
        frameFileName = dir + "scan" + to_string(fileCounter++, 3) + ".frames";

        frame_in.open(frameFileName.c_str());

        // read 3D scan
        if (!frame_in.good())
            break; // no more files in the directory

        cout << "Reading Frames for 3D Scan " << frameFileName << "...";
        vector<double *> Matrices;
        vector<Scan::AlgoType> algoTypes;
        int frameCounter = 0;

        while (frame_in.good())
        {
            frameCounter++;
            double *transMatOpenGL = new double[16];
            int algoTypeInt;
            Scan::AlgoType algoType;
            try
            {
                double transMat[16];
                frame_in >> transMat >> algoTypeInt;
                algoType = (Scan::AlgoType)algoTypeInt;

                // convert to OpenGL coordinate system
                double mirror[16];
                M4identity(mirror);
                mirror[10] = -1.0;
                if (readInitial)
                {
                    double tempxf[16];
                    MMult(mirror, initialTransform, tempxf);
                    memcpy(mirror, tempxf, sizeof(tempxf));
                }
                //@@@
                //	   memcpy(transMatOpenGL, transMat, 16*sizeof(double));
                MMult(mirror, transMat, transMatOpenGL);
            }
            catch (const exception & /* e */)
            {
                break;
            }
            Matrices.push_back(transMatOpenGL);
            algoTypes.push_back(algoType);
        }
        MetaAlgoType.push_back(algoTypes);

        MetaMatrix.push_back(Matrices);

        if ((type == UOS_MAP || type == UOS_MAP_FRAMES || type == RTS_MAP) && fileCounter == start + 1)
        {
            MetaAlgoType.push_back(algoTypes);
            MetaMatrix.push_back(Matrices);
        }

        frame_in.close();
        frame_in.clear();
        cout << MetaMatrix.back().size() << " done." << endl;
        drawable->currentFrame = MetaMatrix.back().size() - 1;
    }
    if (MetaMatrix.size() == 0)
    {
        cerr << "*****************************************" << endl;
        cerr << "** ERROR: No .frames could be found!   **" << endl;
        cerr << "*****************************************" << endl;
        cerr << " ERROR: Missing or empty directory: " << dir << endl << endl;
        return -1;
    }
    return 0;
}

void ThreeDTK::generateFrames(int start, int end, bool identity)
{
    if (identity)
    {
        cout << "using Identity for frames " << endl;
    }
    else
    {
        cout << "using pose information for frames " << endl;
    }
    int fileCounter = start;
    int index = 0;
    for (;;)
    {
        if (fileCounter > end)
            break; // 'nuf read
        fileCounter++;

        vector<double *> Matrices;
        vector<Scan::AlgoType> algoTypes;

        for (int i = 0; i < 3; i++)
        {
            double *transMat = new double[16];

            if (identity)
            {
                M4identity(transMat);
                transMat[10] = -1.0;
            }
            else
            {
                EulerToMatrix4(Scan::allScans[index]->get_rPos(), Scan::allScans[index]->get_rPosTheta(), transMat);
            }

            Matrices.push_back(transMat);
            algoTypes.push_back(Scan::ICP);
        }
        index++;
        MetaAlgoType.push_back(algoTypes);
        MetaMatrix.push_back(Matrices);
    }
}

/*
 * create display lists
 * @to do general framework for color & type definitions
 */
void ThreeDTK::createDisplayLists(bool reduced)
{
    for (unsigned int i = 0; i < Scan::allScans.size(); i++)
    {

        // count points
        int color1 = 0, color2 = 0;
        if (!reduced)
        {
            for (unsigned int jterator = 0; jterator < Scan::allScans[i]->get_points()->size(); jterator++)
            {
                if (Scan::allScans[i]->get_points()->at(jterator).type & TYPE_GROUND)
                {
                    color1++;
                }
                else
                {
                    color2++;
                }
            }
        }
        else
        {
            color2 = 3 * Scan::allScans[i]->get_points_red_size();
        }

        // allocate memory
        vertexArray *myvertexArray1 = new vertexArray(color1);
        vertexArray *myvertexArray2 = new vertexArray(color2);

        // fill points
        color1 = 0, color2 = 0;
        if (reduced)
        {
            for (int jterator = 0; jterator < Scan::allScans[i]->get_points_red_size(); jterator++)
            {
                myvertexArray2->array[color2] = Scan::allScans[i]->get_points_red()[jterator][0];
                myvertexArray2->array[color2 + 1] = Scan::allScans[i]->get_points_red()[jterator][1];
                myvertexArray2->array[color2 + 2] = Scan::allScans[i]->get_points_red()[jterator][2];
                color2 += 3;
            }
        }
        else
        {
            for (unsigned int jterator = 0; jterator < Scan::allScans[i]->get_points()->size(); jterator++)
            {
                if (Scan::allScans[i]->get_points()->at(jterator).type & TYPE_GROUND)
                {
                    myvertexArray1->array[color1] = Scan::allScans[i]->get_points()->at(jterator).x;
                    myvertexArray1->array[color1 + 1] = Scan::allScans[i]->get_points()->at(jterator).y;
                    myvertexArray1->array[color1 + 2] = Scan::allScans[i]->get_points()->at(jterator).z;
                    color1 += 3;
                }
                else
                {
                    myvertexArray2->array[color2] = Scan::allScans[i]->get_points()->at(jterator).x;
                    myvertexArray2->array[color2 + 1] = Scan::allScans[i]->get_points()->at(jterator).y;
                    myvertexArray2->array[color2 + 2] = Scan::allScans[i]->get_points()->at(jterator).z;
                    color2 += 3;
                }
            }
        }

        glNewList(myvertexArray1->name, GL_COMPILE);
        //@
        //glColor4d(0.44, 0.44, 0.44, 1.0);
        //glColor4d(0.66, 0.66, 0.66, 1.0);
        glVertexPointer(3, GL_FLOAT, 0, myvertexArray1->array);
        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_POINTS, 0, myvertexArray1->numPointsToRender);
        glDisableClientState(GL_VERTEX_ARRAY);
        glEndList();

        glNewList(myvertexArray2->name, GL_COMPILE);
        //glColor4d(1.0, 1.0, 1.0, 1.0);
        //glColor4d(0.0, 0.0, 0.0, 1.0);
        glVertexPointer(3, GL_FLOAT, 0, myvertexArray2->array);
        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_POINTS, 0, myvertexArray2->numPointsToRender);
        glDisableClientState(GL_VERTEX_ARRAY);
        glEndList();

        // append to vector
        vector<vertexArray *> vvertexArray;
        vvertexArray.push_back(myvertexArray1);
        vvertexArray.push_back(myvertexArray2);
        vvertexArrayList.push_back(vvertexArray);
    }
}

void ThreeDTK::cycleLOD()
{
    drawable->LevelOfDetail = 0.00001;
    for (unsigned int i = 0; i < drawable->octpts.size(); i++)
        drawable->octpts[i]->cycleLOD();
}

int ThreeDTK::loadFile(std::string filename)
{
    int start = 0;
    int end = -1;
    std::string directory;
    double red = -1.0;

    int octree = 0;
    bool loadOct = false;
    PointType pointtype;
    reader_type type = UOS;
    bool readInitial = false;
    int maxDist = -1;
    int minDist = -1;

    size_t pos = filename.find_last_of("\\/");
    directory = (std::string::npos == pos)
                    ? ""
                    : filename.substr(0, pos);

    directory += "/";

    if (type == OCT)
    {
        loadOct = true;
    }

    // if we want to load display file get pointtypes from the files first
    if (loadOct)
    {
        string scanFileName = directory + "scan" + to_string(start, 3) + ".oct";
        cout << "Getting point information from " << scanFileName << endl;
        cout << "Attention! All subsequent oct-files must be of the same type!" << endl;

        pointtype = BOctTree<sfloat>::readType(scanFileName);
    }
    std::string scandirectory = directory;

    // read frames first, to get notifyied of missing frames before all scans are read in
    int r = readFrames(directory, start, end, readInitial, type);

    // Get Scans
    if (!loadOct)
    {
        Scan::readScans(type, start, end, directory, maxDist, minDist, 0);
    }
    else
    {
        cout << "Skipping files.." << endl;
    }

    if (!loadOct)
    {
        if (r)
            generateFrames(start, start + Scan::allScans.size() - 1, false);
    }
    else
    {
        if (r)
            generateFrames(start, start + drawable->octpts.size() - 1, true);
    }

    int end_reduction = (int)Scan::allScans.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int iterator = 0; iterator < end_reduction; iterator++)
    {
        // reduction filter for current scan!
        if (red > 0)
        {
            cout << "Reducing Scan No. " << iterator << endl;
            // TODO do another reduction so reflectance values etc are carried over
            Scan::allScans[iterator]->calcReducedPoints(red, octree);
        } // no copying necessary for show!
    }
    drawable->cm = new ScanColorManager(4096, pointtype);

    if (loadOct)
    {
        for (int i = start; i <= end; i++)
        {
            string scanFileName = directory + "scan" + to_string(i, 3) + ".oct";
            cout << "Reading octree " << scanFileName << endl;
#ifdef USE_COMPACT_TREE
            drawable->octpts.push_back(new compactTree(scanFileName, cm));
#else
            drawable->octpts.push_back(new Show_BOctTree<sfloat>(scanFileName, drawable->cm));
#endif
        }
    }
    else
    {
#ifndef USE_GL_POINTS
        createDisplayLists(red > 0);
#elif USE_COMPACT_TREE
        cout << "Creating compact display octrees.." << endl;
        for (int i = 0; i < (int)Scan::allScans.size(); i++)
        {
            compactTree *tree;
            if (red > 0)
            {
                tree = new compactTree(Scan::allScans[i]->get_points_red(), Scan::allScans[i]->get_points_red_size(), voxelSize, pointtype, cm); // TODO remove magic number
            }
            else
            {
                unsigned int nrpts = Scan::allScans[i]->get_points()->size();
                sfloat **pts = new sfloat *[nrpts];
                for (unsigned int jterator = 0; jterator < nrpts; jterator++)
                {
                    pts[jterator] = pointtype.createPoint<sfloat>(Scan::allScans[i]->get_points()->at(jterator));
                }
                Scan::allScans[i]->clearPoints();
                tree = new compactTree(pts, nrpts, voxelSize, pointtype, cm); //TODO remove magic number
                for (unsigned int jterator = 0; jterator < nrpts; jterator++)
                {
                    delete[] pts[jterator];
                }
                delete[] pts;
            }
            if (saveOct)
            {
                string scanFileName = directory + "scan" + to_string(i + start, 3) + ".oct";
                cout << "Saving octree " << scanFileName << endl;
                tree->serialize(scanFileName);
            }
            drawable->octpts.push_back(tree);
            cout << "Scan " << i << " octree finished. Deleting original points.." << endl;
        }
#else
        cout << "Creating display octrees.." << endl;
        for (int i = 0; i < (int)Scan::allScans.size(); i++)
        {
            Show_BOctTree<sfloat> *tree;
            if (red > 0)
            {
                tree = new Show_BOctTree<sfloat>(Scan::allScans[i]->get_points_red(), Scan::allScans[i]->get_points_red_size(), drawable->voxelSize, pointtype, drawable->cm); // TODO remove magic number
            }
            else
            {
                unsigned int nrpts = Scan::allScans[i]->get_points()->size();
                sfloat **pts = new sfloat *[nrpts];
                for (unsigned int jterator = 0; jterator < nrpts; jterator++)
                {
                    pts[jterator] = pointtype.createPoint<sfloat>(Scan::allScans[i]->get_points()->at(jterator));
                }
                Scan::allScans[i]->clearPoints();
                tree = new Show_BOctTree<sfloat>(pts, nrpts, drawable->voxelSize, pointtype, drawable->cm); //TODO remove magic number
                for (unsigned int jterator = 0; jterator < nrpts; jterator++)
                {
                    delete[] pts[jterator];
                }
                delete[] pts;
            }
            drawable->octpts.push_back(tree);
            /*  if (saveOct) {
        string scanFileName = directory + "scan" + to_string(i+start,3) + ".oct";
        cout << "Saving octree " << scanFileName << endl;
        tree->serialize(scanFileName);
      }*/
            cout << "Scan " << i << " octree finished. Deleting original points.." << endl;
        }
#endif
    }
    osg::BoundingBox bb(-100000, -100000, -100000, 100000, 100000, 100000);
    drawable->setInitialBound(bb);

    drawable->cm->setCurrentType(PointType::USE_HEIGHT);
    //ColorMap cmap;
    //cm->setColorMap(cmap);
    drawable->resetMinMax();

    //selected_points = new set<sfloat*>[octpts.size()];

    return 0;
}

COVERPLUGIN(ThreeDTK)
