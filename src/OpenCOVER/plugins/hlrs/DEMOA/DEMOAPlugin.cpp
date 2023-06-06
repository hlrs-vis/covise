/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\
**                                                            (C)2014 HLRS  **
**                                                                          **
** Description: load DEMOA Simulation files (INSPO)                         **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** 2014  v1	    				       		                                **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "DEMOAPlugin.h"
#define USE_MATH_DEFINES
#include <math.h>
#include <QDir>
#include <config/coConfig.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/LineWidth>

#include <osg/LineSegment>
#include <osg/Matrix>
#include <osg/Vec3>

#include "string_parse.h"
#include "parmblock.h"
#include "primitive.h"
#include "utils.h"

using namespace osg;
using namespace osgUtil;

// miscellanious definitions
const double UNIT = 1.0;
const char DIRSLASH = '/';
const int SEGCOORD = 16;
const int FRCCOORD = 6;

DEMOAPlugin *DEMOAPlugin::thePlugin = NULL;

DEMOAPlugin *DEMOAPlugin::instance()
{
    if (!thePlugin)
        thePlugin = new DEMOAPlugin();
    return thePlugin;
}

DEMOAPlugin::DEMOAPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    //positions=NULL;
    thePlugin = this;
    data = NULL;
    inc = 1;

    // switches
    db = GL_TRUE;
    sm = GL_TRUE;
    lp = GL_FALSE;
    wf = GL_FALSE;
    fw = GL_FALSE;
    gnd = GL_FALSE;
    rc = GL_FALSE;
    notimefact = GL_TRUE;

    // view
    ax = 0.5;
    ngrid_xlo = 5;
    ngrid_xhi = 5;
    ngrid_ylo = 5;
    ngrid_yhi = 5;
    scale = 1.0;
    follow_x = 0.0;
    follow_y = 0.0;
    follow_idx = -1;

    pngbase[0] = '\0';
    pngno = 0;

    // linkage variables
    segs = NULL;
    nSegslinks = 0;
    n_points_per_seg = 0;
}

static FileHandler handlers[] = {
    { NULL,
      DEMOAPlugin::sloadANI,
      DEMOAPlugin::unloadANI,
      "dani" },
    { NULL,
      DEMOAPlugin::sloadANI,
      DEMOAPlugin::unloadANI,
      "dsim" }
};

int DEMOAPlugin::sloadANI(const char *filename, osg::Group *loadParent, const char *)
{

    instance()->loadANI(filename, loadParent);
    return 0;
}

int DEMOAPlugin::loadANI(const char *filename, osg::Group *loadParent)
{

    parentNode = loadParent;
    if (parentNode == NULL)
        parentNode = cover->getObjectsRoot();
    DemoaRoot = new osg::Group();
    parentNode->addChild(DemoaRoot);
    DemoaRoot->setName("DEMOA_Root");
    std::ifstream cfgfile;
    std::string cfgname;
    const char cfgsuffix[] = ".dsim"; // suffix of configuration file
    const char anisuffix[] = ".dani"; // suffix of animation data file
    std::string fn(filename);
    string::size_type found = fn.find_last_of("/\\");
    path = ".";
    if (found != string::npos)
        path = fn.substr(0, found);
    // open configuration file
    if (std::strcmp(filename + std::strlen(filename) - 5, cfgsuffix) != 0)
    {

        if (std::strcmp(filename + std::strlen(filename) - 5, anisuffix) == 0)
        {
            cfgname = std::string(filename).substr(0, std::strlen(filename) - 5) + cfgsuffix;
        }
        else
        {
            cfgname = std::string(filename) + cfgsuffix;
        }
    }
    else
    {
        cfgname = filename;
    }
    cfgfile.open(cfgname.c_str());
    if (!cfgfile)
    {
        std::fprintf(stderr, "%s%s%s", "Configuration file '", cfgname.c_str(),
                     "' not found.");
        return -1;
    }

    // read configuration file into buffer
    string_parse filebuf(cfgfile);
    if (filebuf.empty())
    {
        std::fprintf(stderr, "%s", "Empty configuration file.");
        return -1;
    }
    // delete comments
    filebuf.nocomment("/*", "*/").nocomment("//");

    parmblock cfgblock;
    // read modeldata blockwise from configuration file
    while (true)
    {
        if (!cfgblock.read(filebuf))
        {
            std::cerr << "Error! 'Header' block not found in configuration "
                      << "file '" << cfgname << "'. Exiting.\n";
            return -1;
        }
        else
        {
            if (cfgblock.keyword() == "Header")
            {
                aniFileName = path + "/" + cfgblock.getstring("ModelName") + anisuffix;
                filebuf.rewind();
                break;
            }
        }
    }

    std::ifstream anifile;
    anifile.open(aniFileName.c_str());
    if (!anifile)
    {
        std::cerr << "Animation file '" << aniFileName << "' not found. ";
        return -1;
    }

    std::string *bodyname = NULL, *forcename = NULL;
    while (true)
    {
        std::string word;
        anifile >> word;
        anifile >> word;
        if (word == "DemoaAniFileVersion:")
        {
            anifile >> word;
            // parse word as float here to get version if needed
        }
        else
        {
            std::fprintf(stderr, ".dani-file header corrupt.DemoaAniFileVersion");
            return -1;
        }
        anifile >> word;
        anifile >> word;
        if (word == "ModelName:")
        {
            anifile >> word;
        }
        else
        {
            std::fprintf(stderr, ".dani-file header corrupt.ModelName");
            return -1;
        }
        anifile >> word;
        anifile >> word;
        if (word == "NumberOfBodies:")
        {
            anifile >> word;
            nSegs = atoi(word.c_str());
        }
        else
        {
            std::fprintf(stderr, ".dani-file header corrupt.NumberOfBodies");
            return -1;
        }
        anifile >> word;
        anifile >> word;
        if (word != "NamesOfBodies:")
        {
            std::fprintf(stderr, ".dani-file header corrupt.NamesOfBodies2");
            return -1;
        }
        else
        {
            bodyname = new std::string[nSegs];
            for (int i = 0; i < nSegs; ++i)
            {
                anifile >> bodyname[i];
            }
        }
        anifile >> word;
        anifile >> word;
        if (word == "NumberOfForces:")
        {
            anifile >> word;
            nForces = atoi(word.c_str());
        }
        else
        {
            std::fprintf(stderr, ".dani-file header corrupt.NumberOfForces");
            return -1;
        }
        anifile >> word;
        anifile >> word;
        if (word != "NamesOfForces:")
        {
            std::fprintf(stderr, ".dani-file header corrupt.NamesOfForces2");
            return -1;
        }
        else if (nForces > 0)
        {
            forcename = new std::string[nForces];
            for (int i = 0; i < nForces; ++i)
            {
                anifile >> forcename[i];
            }
        }
        else
        {
            std::fprintf(stderr, "Info()! No force primitives defined!\n");
        }
        break;
    }

    anifile.close();

    while (!cfgblock.read(filebuf).eos())
    {
        if (cfgblock.keyword() == "Primitive")
        {
            if (cfgblock.getstring("Parent") == "world")
            {
                DemoaRoot->addChild(newPrimitive(cfgblock, -1));
                // std::fprintf(stderr, "A world primitive.\n");
            }
        }
    }
    filebuf.rewind();

    // Primitives attached to bodies
    for (int i = 0; i < nSegs; i++)
    {
        while (!cfgblock.read(filebuf).eos())
        {
            if (cfgblock.keyword() == "Primitive")
            {
                if (cfgblock.getstring("Parent") == bodyname[i])
                {
                    DemoaRoot->addChild(newPrimitive(cfgblock, i));
                    // std::fprintf(stderr, "A body primitive.\n");
                }
            }
        }
        filebuf.rewind();
    }
    delete[] bodyname;

    // Primitives attached to forces
    for (int k = 0; k < nForces; ++k)
    {
        while (!cfgblock.read(filebuf).eos())
        {
            if (cfgblock.keyword() == "Primitive")
            {
                if (cfgblock.getstring("Parent") == "force" && cfgblock.getstring("ForceName") == forcename[k])
                {
                    DemoaRoot->addChild(newPrimitive(cfgblock, k));
                    // std::fprintf(stderr, "A force primitive.\n");
                }
            }
        }
        filebuf.rewind();
    }
    delete[] forcename;

    // read time and animation coordinates from data file
    getdata(aniFileName.c_str(), 0);
    if (nLines > coVRAnimationManager::instance()->getNumTimesteps())
    {
        coVRAnimationManager::instance()->setNumTimesteps(nLines);
        coVRAnimationManager::instance()->showAnimMenu(true);
    }

    // average time step length
    dtt = (data[nLines - 1][0] - data[0][0]) / (nLines - 1);

    return 0;
}

// calculate transformation matrix
void DEMOAPlugin::Coord2Matrix(GLfloat **AA, double *t_frame, double **dataptr, int frame)
{
    // time
    *t_frame = dataptr[frame][0];

    // segments
    for (int i = 0; i < nSegs; ++i)
    {
        // translation
        AA[i][12] = (GLfloat)dataptr[frame][4 + i * SEGCOORD];
        AA[i][13] = (GLfloat)dataptr[frame][8 + i * SEGCOORD];
        AA[i][14] = (GLfloat)dataptr[frame][12 + i * SEGCOORD];
        // rotation
        AA[i][0] = (GLfloat)dataptr[frame][1 + i * SEGCOORD];
        AA[i][1] = (GLfloat)dataptr[frame][5 + i * SEGCOORD];
        AA[i][2] = (GLfloat)dataptr[frame][9 + i * SEGCOORD];
        AA[i][4] = (GLfloat)dataptr[frame][2 + i * SEGCOORD];
        AA[i][5] = (GLfloat)dataptr[frame][6 + i * SEGCOORD];
        AA[i][6] = (GLfloat)dataptr[frame][10 + i * SEGCOORD];
        AA[i][8] = (GLfloat)dataptr[frame][3 + i * SEGCOORD];
        AA[i][9] = (GLfloat)dataptr[frame][7 + i * SEGCOORD];
        AA[i][10] = (GLfloat)dataptr[frame][11 + i * SEGCOORD];
    }

    // forces
    for (int i = 0; i < nForces; ++i)
    {
        // translation
        AA[nSegs + i][0] = (GLfloat)dataptr[frame][nSegs * SEGCOORD + 1 + i * FRCCOORD];
        AA[nSegs + i][1] = (GLfloat)dataptr[frame][nSegs * SEGCOORD + 2 + i * FRCCOORD];
        AA[nSegs + i][2] = (GLfloat)dataptr[frame][nSegs * SEGCOORD + 3 + i * FRCCOORD];
        AA[nSegs + i][3] = (GLfloat)dataptr[frame][nSegs * SEGCOORD + 4 + i * FRCCOORD];
        AA[nSegs + i][4] = (GLfloat)dataptr[frame][nSegs * SEGCOORD + 5 + i * FRCCOORD];
        AA[nSegs + i][5] = (GLfloat)dataptr[frame][nSegs * SEGCOORD + 6 + i * FRCCOORD];
        // std::cerr << "Coord2Matrix()" << std::endl;
        // std::cerr << "Info(): AA[nSegs + i][0] = " << AA[nSegs + i][0] << std::endl;
        // std::cerr << "Info(): AA[nSegs + i][1] = " << AA[nSegs + i][1] << std::endl;
        // std::cerr << "Info(): AA[nSegs + i][2] = " << AA[nSegs + i][2] << std::endl;
        // std::cerr << "Info(): AA[nSegs + i][3] = " << AA[nSegs + i][3] << std::endl;
        // std::cerr << "Info(): AA[nSegs + i][4] = " << AA[nSegs + i][4] << std::endl;
        // std::cerr << "Info(): AA[nSegs + i][5] = " << AA[nSegs + i][5] << std::endl;
    }
}

// update follow marker coordinates
void DEMOAPlugin::updatefollow()
{
    follow_x = data[frame][4 + follow_idx * SEGCOORD];
    follow_y = data[frame][8 + follow_idx * SEGCOORD];
}

// clear stdin
void DEMOAPlugin::clear_stdin()
{
    while (std::fgetc(stdin) != '\n')
        ;
}
// read (new) data
void DEMOAPlugin::getdata(const char file[], int initframe)
{
    // memory for filename (case file=NULL)
    std::FILE *stream;
    char cbuf[128];
    std::string fname;

    // ask for file name if file = NULL
    if (file == NULL)
    {
        std::fprintf(stderr, "New filename: ");
        if (std::fscanf(stdin, "%s", cbuf) != 1)
        {
            fprintf(stderr, "Failed to read from stdin...");
        }
        clear_stdin();
        fname = cbuf;
        if (fname[0] == 27)
        {
            fprintf(stderr, "Continuing with old data...");
            return;
        }
    }
    else
    {
        fname = file;
    }

    // open file
    if ((stream = std::fopen(fname.c_str(), "r")) == NULL)
    {
        std::fprintf(stderr, "Filename: %s\n", fname.c_str());
        perror("");
        if (data == 0)
        {
            fprintf(stderr, "Can't continue.");
            return;
        }
        fprintf(stderr, "Continuing with old data...");
        return;
    }

    // free memory of old data
    if (data != 0)
    {
        delete[] data[0];
        delete[] data;
        for (int aux = 0; aux < nSegs + nForces; ++aux)
        {
            delete[] A[aux];
        }
        delete[] A;
    }

    // read data
    data = ReadData(stream, &nLines, &nCols);
    if (data == NULL)
    {
        fprintf(stderr, "Could not read file. Can't continue.");
        return;
    }

    if (nSegs * SEGCOORD + nForces * FRCCOORD != nCols - 1)
    {
        fprintf(stderr, "Missing data column. Can't continue.");
        return;
    }

    // allocate memory for transformation matrices and initialize
    A = new GLfloat *[nSegs + nForces];
    for (int aux = 0; aux < nSegs; ++aux)
    {
        A[aux] = new GLfloat[16];
        // same for all matrices
        A[aux][3] = 0.0;
        A[aux][7] = 0.0;
        A[aux][11] = 0.0;
        A[aux][15] = 1.0;
        // initialize identity transformation for rotational part of all
        // matrices
        A[aux][0] = 1.0;
        A[aux][1] = 0.0;
        A[aux][2] = 0.0;
        A[aux][4] = 0.0;
        A[aux][5] = 1.0;
        A[aux][6] = 0.0;
        A[aux][8] = 0.0;
        A[aux][9] = 0.0;
        A[aux][10] = 1.0;
    }

    // initialize force data array
    for (int aux = nSegs; aux < nSegs + nForces; ++aux)
    {
        A[aux] = new GLfloat[6];
        A[aux][0] = 0.0;
        A[aux][1] = 0.0;
        A[aux][2] = 0.0;
        A[aux][3] = 0.0;
        A[aux][4] = 0.0;
        A[aux][5] = 0.0;
    }

    // set start frame
    frame = initframe;
    // store name of current datafile
    aniFileName = fname;
    // close file
    if (stream)
    {
        std::fclose(stream);
        stream = NULL;
    }
}

// allocate array for data to be stored, read data from file
// and return pointer to data array
double **DEMOAPlugin::ReadData(std::FILE *stream, int *rows, int *cols)
{
    double **dataptr;

    *rows = CountLines(stream, '#');
    *cols = CountColumns(stream, '#');

    dataptr = new double *[*rows];
    dataptr[0] = new double[(*rows) * (*cols)];
    for (int i = 1; i < *rows; ++i)
    {
        dataptr[i] = dataptr[0] + i * (*cols);
    }

    // read time and all segment coordinates from file
    int row = 0;
    while (row < *rows)
    {
        NoComment(stream, '#');
        if (std::fscanf(stream, "%le", &dataptr[row][0]) < 1)
        {
            if (row == 0)
            {
                fprintf(stderr, "Incomplete data. Can't continue.");
                return NULL;
            }
            else
            {
                *rows = row;
            }
            break;
        }

        for (int col = 1; col < *cols; ++col)
        {
            if (std::fscanf(stream, "%le", &dataptr[row][col]) < 1)
            {
                if (row == 0)
                {
                    fprintf(stderr, "Incomplete data. Can't continue.");
                    return NULL;
                }
                else
                {
                    *rows = row;
                }
                break;
            }
            else
            {
                dataptr[row][col] *= UNIT;
            }
        }
        ++row;
    }

    return dataptr;
}

// read new grid dimensions
void DEMOAPlugin::getgriddim()
{
    std::fprintf(stderr, "Dimensions of ground grid [xlo,xhi,ylo,yhi]: ");
    if (std::fscanf(stdin, "%d,%d,%d,%d",
                    &ngrid_xlo, &ngrid_xhi, &ngrid_ylo, &ngrid_yhi) < 4)
    {
        fprintf(stderr, "Wrong input format. Continuing.");
        ngrid_xlo = ngrid_xhi = ngrid_ylo = ngrid_yhi = 5;
        clear_stdin();
    }
    else
    {
        clear_stdin();
    }
}

// read (new) marker to follow
void DEMOAPlugin::getfollow_idx()
{
    std::fprintf(stderr, "Index of segment to follow: ");
    if (std::fscanf(stdin, "%d", &follow_idx) < 1)
    {
        fprintf(stderr, "Wrong input format. Continuing.");
        follow_idx = -1;
        clear_stdin();
        return;
    }
    else
    {
        clear_stdin();
    }

    if (follow_idx < 1 || follow_idx > nSegs)
    {
        fprintf(stderr, "No such segment index. Continuing.");
        follow_idx = -1;
        return;
    }
    --follow_idx;
}

// read (new) basename for PNG-images
void DEMOAPlugin::getpngbase()
{
    pngno = 0;
    if (isrecord())
    {
        toggle_record();
    }

    std::fprintf(stderr, "Basename for PNG image files: ");
    if (std::fscanf(stdin, "%s", pngbase) < 1)
    {
        fprintf(stderr, "Wrong input format. Continuing.");
        pngbase[0] = '\0';
        clear_stdin();
        return;
    }
    else
    {
        clear_stdin();
    }

    if (pngbase[0] == 27)
    {
        pngbase[0] = '\0';
    }
}

osg::MatrixTransform *DEMOAPlugin::newPrimitive(parmblock &block, int idx)
{
    std::string type(block.getstring("Type"));
    std::string parent(block.getstring("Parent"));
    D_Primitive *primitive;

    if (type == "Box")
    {
        primitive = new D_Box(block);
    }
    else if (type == "Sphere")
    {
        primitive = new D_Sphere(block);
    }
    else if (type == "Cylinder")
    {
        primitive = new D_Cylinder(block);
    }
    else if (type == "Cone")
    {
        primitive = new D_Cone(block);
    }
    else if (type == "Axes")
    {
        primitive = new D_Axes(block);
    }
    else if (type == "Extrude")
    {
        primitive = new D_Extrude(block);
    }
    else if (type == "Tetraeder")
    {
        primitive = new D_Tetraeder(block);
    }
    else if (type == "Surface")
    {
        primitive = new D_Surface(block);
    }
    else if (type == "Muscle")
    {
        primitive = new D_Muscle(block);
    }
    else if (type == "IVD")
    {
        primitive = new D_IVD(block);
    }
    else
    {
        std::cerr << "Warning! Unknown Primitive type. Ignoring.\n";
    }

    if (primitive)
    {
        if (parent == "force")
        {
            primitive->set_forceid(idx);
            primitive->set_bodyid(-99);
            // std::cerr << "Warning! Set force primitive id.\n";
        }
        else
        {
            primitive->set_bodyid(idx);
            // std::cerr << "Warning! Set body primitive id.\n";
        }
    }
    primitives.push_back(primitive);
    return primitive->getNode();
}

void DEMOAPlugin::setTimestep(int t)
{
    Coord2Matrix(A, &tt, data, t);
    for (unsigned int i = 0; i < primitives.size(); ++i)
    {
        // translate and rotate primitive into its segment's position
        // and orientation
        if (primitives[i]->bodyid() >= 0)
        {
            osg::Matrix m;
            m.set(A[primitives[i]->bodyid()]);
            primitives[i]->getNode()->setMatrix(m);
        }
        else if (primitives[i]->bodyid() == -99)
        {
            const int index = nSegs + primitives[i]->forceid();
            primitives[i]->set_mytriads(A[index]);
            primitives[i]->Define();
            //glCallList(primitives[i]->listid());
        }
        else
        {
        }
    }
}

int DEMOAPlugin::unloadANI(const char *filename, const char *)
{
    (void)filename;

    return 0;
}

void DEMOAPlugin::deleteColorMap(const std::string &name)
{
    float *mval = mapValues[name];
    mapSize.erase(name);
    mapValues.erase(name);
    delete[] mval;
}

bool DEMOAPlugin::init()
{
    fprintf(stderr, "DEMOAPlugin::DEMOAPlugin\n");

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);

    filename = NULL;

    coConfig *config = coConfig::getInstance();

    // read the name of all colormaps in file
    auto list = config->getVariableList("Colormaps").entries();
    for (const auto &e : list)
        mapNames.insert(e.entry);

    // read the values for each colormap
    for (const auto &mapName : mapNames)
    {
        // get all definition points for the colormap
        std::string cmapname = "Colormaps." + mapName;
        auto variable = config->getVariableList(cmapname).entries();

        mapSize.insert({mapName, variable.size()});
        float *cval = new float[variable.size() * 5];
        mapValues.insert({mapName, cval});

        // read the rgbax values
        int it = 0;
        for (int l = 0; l < variable.size() * 5; l = l + 5)
        {
            std::string tmp = cmapname + ".Point:" + std::to_string(it);
            cval[l] = config->getFloat("x", tmp, -1.0);
            if (cval[l] == -1)
            {
                cval[l] = (1.0 / (variable.size() - 1)) * (l / 5);
            }
            cval[l + 1] = config->getFloat("r", tmp, 1.0);
            cval[l + 2] = config->getFloat("g", tmp, 1.0);
            cval[l + 3] = config->getFloat("b", tmp, 1.0);
            cval[l + 4] = config->getFloat("a", tmp, 1.0);
            it++;
        }
    }
    currentMap = mapNames.begin();
    // read values of local colormap files in .covise
    auto place = coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "colormaps";

    QDir directory(place.c_str());
    if (directory.exists())
    {
        QStringList filters;
        filters << "colormap_*.xml";
        directory.setNameFilters(filters);
        directory.setFilter(QDir::Files);
        QStringList files = directory.entryList();

        // loop over all found colormap xml files
        for (int j = 0; j < files.size(); j++)
        {
            coConfigGroup *colorConfig = new coConfigGroup("ColorMap");
            colorConfig->addConfig(place + "/" + files[j].toStdString(), "local", true);

            // read the name of the colormaps
            auto list = colorConfig->getVariableList("Colormaps").entries();

            // loop over all colormaps in one file
            for (const auto e : list)
            {
                const std::string &entry = e.entry;
                // remove global colormap with same name
                auto index = mapNames.find(entry);
                if (index != mapNames.end())
                {
                    deleteColorMap(entry);
                }

                // get all definition points for the colormap
                std::string cmapname = "Colormaps." + entry;
                auto variable = colorConfig->getVariableList(cmapname).entries();

                mapSize.insert({entry, variable.size()});
                float *cval = new float[variable.size() * 5];
                mapValues.insert({entry, cval});

                // read the rgbax values
                int it = 0;
                for (int l = 0; l < variable.size() * 5; l = l + 5)
                {
                    std::string tmp = cmapname + ".Point:" + std::to_string(it);
                    cval[l] = std::stof(colorConfig->getValue("x", tmp, " -1.0").entry);
                    if (cval[l] == -1)
                    {
                        cval[l] = (1.0 / (variable.size() - 1)) * (l / 5);
                    }
                    cval[l + 1] = std::stof(colorConfig->getValue("r", tmp, "1.0").entry);
                    cval[l + 2] = std::stof(colorConfig->getValue("g", tmp, "1.0").entry);
                    cval[l + 3] = std::stof(colorConfig->getValue("b", tmp, "1.0").entry);
                    cval[l + 4] = std::stof(colorConfig->getValue("a", tmp, "1.0").entry);
                    it++;
                }
            }
            config->removeConfig(place + "/" + files[j].toStdString());
        }
    }

    PathTab = new coTUITab("DEMOA", coVRTui::instance()->mainFolder->getID());
    record = new coTUIToggleButton("Record", PathTab->getID());
    stop = new coTUIButton("Stop", PathTab->getID());

    mapChoice = new coTUIComboBox("mapChoice", PathTab->getID());
    mapChoice->setEventListener(this);
    for (const auto &map : mapNames)
    {
        mapChoice->addEntry(map);
    }
    mapChoice->setSelectedEntry(0); // meant to be current map but current map was never set to anything but 0
    mapChoice->setPos(6, 0);

    geoState = new osg::StateSet();
    linemtl = new osg::Material;
    lineWidth = new osg::LineWidth(2.0);
    linemtl.get()->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    linemtl.get()->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
    linemtl.get()->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0));
    linemtl.get()->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
    linemtl.get()->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
    linemtl.get()->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    geoState->setAttributeAndModes(linemtl.get(), StateAttribute::ON);

    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    geoState->setAttributeAndModes(lineWidth.get(), StateAttribute::ON);

    return true;
}

// this is called if the plugin is removed at runtime
DEMOAPlugin::~DEMOAPlugin()
{
    fprintf(stderr, "DEMOAPlugin::~DEMOAPlugin\n");

    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);

    delete record;
    delete stop;
    delete PathTab;
    delete[] filename;

    if (DemoaRoot->getNumParents() > 0)
    {
        parentNode = DemoaRoot->getParent(0);
        if (parentNode)
            parentNode->removeChild(DemoaRoot.get());
    }
}

void
DEMOAPlugin::preFrame()
{
    if (record->getState())
    {
    }
}

void DEMOAPlugin::tabletEvent(coTUIElement *tUIItem)
{
}
void DEMOAPlugin::save()
{
}

void DEMOAPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
}

void DEMOAPlugin::tabletReleaseEvent(coTUIElement *tUIItem)
{
    (void)tUIItem;
}

osg::Vec4 DEMOAPlugin::getColor(float pos)
{

    osg::Vec4 actCol;
    if (currentMap == mapNames.end())
        return actCol;
    int idx = 0;
    //cerr << "name: " << (const char *)mapNames[currentMap].toAscii() << endl;
    float *map = mapValues[*currentMap];
    int mapS = mapSize[*currentMap];
    if (map == NULL)
    {
        return actCol;
    }
    while (map[(idx + 1) * 5] <= pos)
    {
        idx++;
        if (idx > mapS - 2)
        {
            idx = mapS - 2;
            break;
        }
    }
    double d = (pos - map[idx * 5]) / (map[(idx + 1) * 5] - map[idx * 5]);
    actCol[0] = (float)((1 - d) * map[idx * 5 + 1] + d * map[(idx + 1) * 5 + 1]);
    actCol[1] = (float)((1 - d) * map[idx * 5 + 2] + d * map[(idx + 1) * 5 + 2]);
    actCol[2] = (float)((1 - d) * map[idx * 5 + 3] + d * map[(idx + 1) * 5 + 3]);
    actCol[3] = (float)((1 - d) * map[idx * 5 + 4] + d * map[(idx + 1) * 5 + 4]);

    return actCol;
}

void DEMOAPlugin::toggle_loop()
{
    lp ^= 1;
}

GLboolean DEMOAPlugin::isloop()
{
    return lp;
}

void DEMOAPlugin::toggle_smooth()
{
    sm ^= 1;
}

GLboolean DEMOAPlugin::issmooth()
{
    return sm;
}

void DEMOAPlugin::toggle_wire()
{
    wf ^= 1;
}

GLboolean DEMOAPlugin::iswire()
{
    return wf;
}

void DEMOAPlugin::toggle_follow()
{
    fw ^= 1;
}

GLboolean DEMOAPlugin::isfollow()
{
    return fw;
}

void DEMOAPlugin::toggle_ground()
{
    gnd ^= 1;
}

GLboolean DEMOAPlugin::isground()
{
    return gnd;
}

void DEMOAPlugin::toggle_record()
{
    rc ^= 1;
}

GLboolean DEMOAPlugin::isrecord()
{
    return rc;
}

COVERPLUGIN(DEMOAPlugin)
