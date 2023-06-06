/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
//			Source File
//
// * Description    : MoleculeViewer plugin module for the Cover Covise Renderer
//                    Reads Molecule Structures based on the Jorgensen Model
//                    The data is provided from the Itt / University Stuttgart
//
// * Class(es)      :
//
// * inherited from :
//
// * Author  : Thilo Krueger
//
// * History : started 6.3.2001
//
// **************************************************************************

//lenght of a line
#define LINE_SIZE 512
#define MAX_FRAMES_PER_SEC 25.0
#include <config/CoviseConfig.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRFileManager.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Button.h>
#include <cover/ui/Action.h>

#ifndef _WIN32
#include <sys/dir.h>
#else
#include <stdio.h>
#include <process.h>
#include <io.h>
#include <direct.h>
#endif
#include "VRMoleculeViewer.h"
#ifdef USE_OLD_COVISE
#include <cover/VRCoviseConnection.h>
#endif
#include <cover/RenderObject.h>
#include "Molecule.h"
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <cover/VRSceneGraph.h>

#include <grmsg/coGRKeyWordMsg.h>

VRMoleculeViewer *plugin = NULL;

FileHandler fileHandler[] = {
    { NULL,
      VRMoleculeViewer::loadFile,
      VRMoleculeViewer::unloadFile,
      "via" },
    { NULL,
      VRMoleculeViewer::loadFile,
      VRMoleculeViewer::unloadFile,
      "vim" },
    { NULL,
      VRMoleculeViewer::loadFile,
      VRMoleculeViewer::unloadFile,
      "vis" }
};

int
VRMoleculeViewer::loadFile(const char *fn, osg::Group *parent, const char *)
{
    if (plugin)
        return plugin->loadData(fn, parent);

    return -1;
}

int
VRMoleculeViewer::unloadFile(const char *, const char *)
{
    if (plugin)
    {
        plugin->clearUp();
        return 0;
    }

    return -1;
}

// this function is called for every module at the time COVER loads the module
// coVRInit(coVrModule): init this module
// REQUIRED! (see coVrModuleSupport.h)

//=======================================================================
// CONSTRUCTOR
//=======================================================================

VRMoleculeViewer::VRMoleculeViewer()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("MoleculePlugin", cover->ui)
, DCSList(NULL)
, radius(NULL)
, maxNumberOfMolecules(0)
, numberOfTimesteps(0)
{
}

bool VRMoleculeViewer::init()
{
    if (plugin)
        return false;

    plugin = this;

    maxNumberOfMolecules = 0;
    numberOfTimesteps = 0;
    moving = 0;
    frameIndex = 0;
    structure = NULL;
    animationSpeed = 1;
    m_strFeedbackInfo = ""; // !\n0\n\n dummy feedback info
    m_bNewObject = false;
    m_bDoOnceViewAll = false;

    coVRFileManager::instance()->registerFileHandler(&fileHandler[0]);
    coVRFileManager::instance()->registerFileHandler(&fileHandler[1]);
    coVRFileManager::instance()->registerFileHandler(&fileHandler[2]);

    viewerDCS = NULL;

    m_rootNode = cover->getObjectsRoot();

    sphereRatio = coCoviseConfig::getFloat("COVER.Plugin.Molecules.SphereRatio", 0.f);

    addMenuEntry();

    return true;
} // CONSTRUCTOR

//==========================================================================
// DESTRUCTOR
//==========================================================================

VRMoleculeViewer::~VRMoleculeViewer()
{
    coVRFileManager::instance()->unregisterFileHandler(&fileHandler[2]);
    coVRFileManager::instance()->unregisterFileHandler(&fileHandler[1]);
    coVRFileManager::instance()->unregisterFileHandler(&fileHandler[0]);

    coVRAnimationManager::instance()->removeTimestepProvider(plugin);

    // remove menu entry
    removeMenuEntry();

    if (viewerDCS.valid())
        m_rootNode->removeChild(viewerDCS.get());
} // DESTRUCTOR

// Called before each frame
void VRMoleculeViewer::preFrame()
{
    if (m_bNewObject)
    {
        coVRAnimationManager::instance()->showAnimMenu(true);
        m_bNewObject = false;
    }
    if (m_bDoOnceViewAll)
    {
        // execute view all
        VRSceneGraph::instance()->viewAll();

        m_bDoOnceViewAll = false;
    }

    if (moving)
    {
        stepping();
    }
}

void VRMoleculeViewer::setTimestep(int t)
{
    if (t > numberOfTimesteps)
        t = numberOfTimesteps - 1;

    if (numberOfTimesteps > 0)
    {
        framelist.current()->hide();
        framelist.set(t);
        frameIndex = t + 1;
        framelist.current()->display();
    }
}

void VRMoleculeViewer::addMenuEntry()
{
    uniqueMenu = new ui::ButtonGroup("MoleculeFiles", this);
    uniqueMenu->enableDeselect(true);

#ifdef USE_OLD_COVISE
    if (VRCoviseConnection::covconn)
    {
        cover->addSubmenuButton("Molecules...", NULL, "Molecules", false, (void *)menuCallback, -1, this);
        // generate menu entries for all datafiles in current directory
        readDirectory("Molecules");
    }
    else
#endif
    {
        menu = new ui::Menu("Molecules", this);
        auto load = new ui::Action(menu, "LoadData");
        load->setText("Load data");

#if 0
        cover->addSubmenuButton("Molecules...", NULL, "Molecules", false, menuCallback, -1, this);
        cover->addSubmenuButton("Load Data", "Molecules", "Datafiles", false, menuCallback,
                                cover->createUniqueButtonGroupId(), this);
#endif
#if 0
      cover->addToggleButton( "Animate", "Molecules", false, menuCallback, this );
                                    cover->addFunctionButton( "Step forward", "Molecules",  menuCallback, this );
                                    cover->addFunctionButton( "Step backward", "Molecules", menuCallback, this );
#endif
        auto clear = new ui::Action(menu, "Clear");
        clear->setCallback([this](){
            clearUp();
        });
#if 0
        cover->addFunctionButton("Clear", "Molecules", menuCallback, this);
#endif
        // generate menu entries for all datafiles in current directory
        readDirectory("Datafiles");
        auto sl = new ui::Slider(menu, "SphereRatio");
        sl->setText("Sphere ratio");
        sl->setBounds(0., 1.);
        sl->setValue(sphereRatio);
        sl->setCallback([this](double value, bool released){
            if (structure != NULL)
                structure->setSphereRatio(value);
        });
        //cover->addSliderButton("Sphere Ratio", "Molecules", 0.0f, 1.0f, sphereRatio, sphereCallback, this);
        auto update = new ui::Action(menu, "UpdateRatio");
        update->setText("Update ratio");
        update->setCallback([this](){
            Init();
            loadData(filename.c_str(), m_rootNode.get());
        });
        //cover->addFunctionButton("Update Ratio", "Molecules", updateCallback, this);
        //cover->addSliderButton("Speed", "Molecules", -2.0f, 2.0f, animationSpeed, speederCallback, this);
        auto speed = new ui::Slider(menu, "Speed");
        speed->setBounds(-2., 2.);
        speed->setValue(animationSpeed);
        speed->setCallback([this](double value, bool released){
            animationSpeed = value;

            if (animationSpeed == 0)
                timeBetweenFrames = 0;
            else
                timeBetweenFrames = fabs(1 / MAX_FRAMES_PER_SEC / (double)animationSpeed);
        });
    }
}

void VRMoleculeViewer::removeMenuEntry()
{
#if 0
    cover->removeButton("Clear", "Molecules");
    cover->removeButton("Sphere Ratio", "Molecules");
    cover->removeButton("Speed", "Molecules");
    cover->removeButton("Update Ratio", "Molecules");
    cover->removeButton("Load Data", "Molecules");
    cover->removeButton("Molecules...", NULL);
#endif
}

#if 0
void VRMoleculeViewer::speederCallback(void *viewer, buttonSpecCell *spec)
{
    VRMoleculeViewer *m = (VRMoleculeViewer *)viewer;

    m->animationSpeed = spec->state;

    if (m->animationSpeed == 0)
        m->timeBetweenFrames = 0;
    else
        m->timeBetweenFrames = fabs(1 / MAX_FRAMES_PER_SEC / (double)m->animationSpeed);
}

void VRMoleculeViewer::sphereCallback(void *viewer, buttonSpecCell *spec)
{
    VRMoleculeViewer *m = (VRMoleculeViewer *)viewer;

    m->sphereRatio = spec->state;
    if (m->structure != NULL)
    {
        m->structure->setSphereRatio(spec->state);
        if (cover->debugLevel(3))
            printf("Ratio = %f\n", m->sphereRatio);
    }
}

void VRMoleculeViewer::updateCallback(void *viewer, buttonSpecCell *spec)
{
    (void)spec;
    VRMoleculeViewer *m = (VRMoleculeViewer *)viewer;

    Init(viewer);
    m->loadData(m->filename.c_str(), m->m_rootNode.get());
}
#endif

#if 0
void VRMoleculeViewer::menuCallback(void *viewer, buttonSpecCell *spec)
{
    VRMoleculeViewer *m = (VRMoleculeViewer *)viewer;

    if (strcmp(spec->name, "Load Data") == 0)
    {
        //m->readDirectory();
        //m->loadData();
        //m->show();
    }
#if 0
   if (strcmp(spec->name, "Animate") == 0)
                          {
                                if ( spec->state == 1.0  )
                          {
                                m->move();                               // activate if switch is on

}                                           // if(spec->state)
                                else
                          {
                                printf("stop animation\n");
                                m->NoMove();                             // disactivate if switch is off
}
}
                                if (strcmp(spec->name, "Step forward") == 0)
                          {
                                m->NoMove();
                                m->stepForward();                           // activate if switch is on
}
                                if (strcmp(spec->name, "Step backward") == 0)
                          {
                                m->NoMove();
                                m->stepBackward();                          // activate if switch is on
}
#endif
    if (strcmp(spec->name, "Clear") == 0)
    {
        m->clearUp();
    }

} // end of menucallback
#endif

void VRMoleculeViewer::readDirectory(const char *parent)
{
    // std::list<std::string> vsFiles;
    dirName = coCoviseConfig::getEntry("COVER.Plugin.Molecules.DataDir");
    if (cover->debugLevel(3))
        std::cout << "COVER.Plugin.Molecules.DataDir = " << dirName << std::endl;

#ifdef _WIN32
    char olddir[502];
    int olddrive = 0;
    char newpath[502];
    std::string strDirName;

    int pathLen = 0;
    getcwd(olddir, 500);
    olddrive = _getdrive(); /* Save current drive */

    bool bDataDirFound = false;

    // check if we have one complete relative path
    std::string dataDir = coCoviseConfig::getEntry("COVER.Plugin.Molecules.DataDir");
    std::string strTmp;
    if (dataDir.empty())
        strTmp = std::string(getenv("COVISEDIR")) + std::string("data\\");
    else
        strTmp = (std::string(olddir) + std::string("\\") + std::string(dataDir));
    dirName = strTmp;

    if (chdir(dirName.c_str()) != 0)
    {
        if (!bDataDirFound && !dataDir.empty())
        {
            // check if we have to deal with an absolute path
            strDirName = dataDir;
            dirName = strDirName;
            if (chdir(dirName.c_str()) == 0)
                bDataDirFound = true;
            if (cover->debugLevel(3))
                std::cout << "... test data directory " << dirName << std::endl;
        }

        if (!bDataDirFound && !dataDir.empty())
        {
            // check if we have to deal with a COVISEDIR\data relativ path
            strDirName = std::string(getenv("COVISEDIR")) + std::string("data\\") + std::string(dataDir);
            dirName = strDirName;
            if (chdir(dirName.c_str()) == 0)
                bDataDirFound = true;
            if (cover->debugLevel(3))
                std::cout << "... test data directory " << dirName << std::endl;
        }

        if (!bDataDirFound)
        {
            // last and final trial
            strDirName = std::string(getenv("COVISEDIR")) + std::string("data\\");
            dirName = strDirName;
            if (chdir(dirName.c_str()) == 0)
                bDataDirFound = true;
            if (cover->debugLevel(3))
                std::cout << "... test data directory " << dirName << std::endl;
        }
    }
    else
        bDataDirFound = true;

    if (bDataDirFound)
    {
        chdir(dirName.c_str());
        if (cover->debugLevel(3))
            std::cout << "... valid data directory is " << dirName << std::endl;
        getcwd(newpath, 500);
        struct _finddata_t c_file;
        long hFile;
        pathLen = strlen(newpath);

        if ((hFile = _findfirst("*.vis", &c_file)) != -1L)
        {
            if (!(c_file.attrib & _A_SUBDIR))
            {
                char *name = new char[pathLen + strlen(c_file.name) + 2];
                //            sprintf(name,"%s\\%s",newpath,c_file.name);
                sprintf(name, "%s", c_file.name);
                name[strlen(name) - 4] = '\0';

                vsFiles.push_back(std::string(name) + std::string(".vis"));
                //            cover->addFunctionButton(name, "Datafiles", (void*)fileSelection, this);
            }

            while (_findnext(hFile, &c_file) == 0)
            {
                if (!(c_file.attrib & _A_SUBDIR))
                {
                    char *name = new char[pathLen + strlen(c_file.name) + 2];
                    //               sprintf(name,"%s\\%s",newpath,c_file.name);
                    sprintf(name, "%s", c_file.name);
                    name[strlen(name) - 4] = '\0';

                    vsFiles.push_back(std::string(name) + std::string(".vis"));
                    //               cover->addFunctionButton(name, "Datafiles", (void*)fileSelection, this);
                }
            }
        }
        _findclose(hFile);

        if ((hFile = _findfirst("*.via", &c_file)) != -1L)
        {
            if (!(c_file.attrib & _A_SUBDIR))
            {
                char *name = new char[pathLen + strlen(c_file.name) + 2];
                //            sprintf(name,"%s\\%s",newpath,c_file.name);
                sprintf(name, "%s", c_file.name);
                name[strlen(name) - 4] = '\0';

                vsFiles.push_back(std::string(name) + std::string(".via"));
                //            cover->addFunctionButton(name, "Datafiles", (void*)fileSelection, this);
            }

            while (_findnext(hFile, &c_file) == 0)
            {
                if (!(c_file.attrib & _A_SUBDIR))
                {
                    char *name = new char[pathLen + strlen(c_file.name) + 2];
                    //               sprintf(name,"%s\\%s",newpath,c_file.name);
                    sprintf(name, "%s", c_file.name);
                    name[strlen(name) - 4] = '\0';

                    vsFiles.push_back(std::string(name) + std::string(".via"));
                    //               cover->addFunctionButton(name, "Datafiles", (void*)fileSelection, this);
                }
            }
        }
        _findclose(hFile);

        if ((hFile = _findfirst("*.vim", &c_file)) != -1L)
        {
            if (!(c_file.attrib & _A_SUBDIR))
            {
                char *name = new char[pathLen + strlen(c_file.name) + 2];
                //            sprintf(name,"%s\\%s",newpath,c_file.name);
                sprintf(name, "%s", c_file.name);
                name[strlen(name) - 4] = '\0';

                vsFiles.push_back(std::string(name) + std::string(".vim"));
                //            cover->addFunctionButton(name, "Datafiles", (void*)fileSelection, this);
            }

            while (_findnext(hFile, &c_file) == 0)
            {
                if (!(c_file.attrib & _A_SUBDIR))
                {
                    char *name = new char[pathLen + strlen(c_file.name) + 2];
                    //               sprintf(name,"%s\\%s",newpath,c_file.name);
                    sprintf(name, "%s", c_file.name);
                    name[strlen(name) - 4] = '\0';

                    vsFiles.push_back(std::string(name) + std::string(".vim"));
                    //               cover->addFunctionButton(name, "Datafiles", (void*)fileSelection, this);
                }
            }
        }
        _findclose(hFile);
    }
    else
        std::cout << "... no valid data directory found!" << std::endl;

    _chdrive(olddrive);
    chdir(olddir);

#else
    DIR *dirp;
    struct direct *direntp;
    dirp = opendir(dirName.c_str());
    while (dirp && ((direntp = readdir(dirp)) != NULL))
    {
        int len = strlen(direntp->d_name);
        if (len > 4)
        {
            if (strcmp((direntp->d_name) + len - 4, ".vis") == 0)
            {
                char *name = new char[len + 1];
                strcpy(name, direntp->d_name);
                name[len - 4] = '\0';

                vsFiles.push_back(std::string(name) + std::string(".vis"));
                //            cover->addFunctionButton(name, "Datafiles", (void*)fileSelection, this);
            }
            if (strcmp((direntp->d_name) + len - 4, ".vim") == 0)
            {
                char *name = new char[len + 1];
                strcpy(name, direntp->d_name);
                name[len - 4] = '\0';

                vsFiles.push_back(std::string(name) + std::string(".vim"));
                //            cover->addFunctionButton(name, "Datafiles", (void*)fileSelection, this);
            }
            if (strcmp((direntp->d_name) + len - 4, ".via") == 0)
            {
                char *name = new char[len + 1];
                strcpy(name, direntp->d_name);
                name[len - 4] = '\0';

                vsFiles.push_back(std::string(name) + std::string(".via"));
                //            cover->addFunctionButton(name, "Datafiles", (void*)fileSelection, this);
            }
        }
    }

    if (dirp)
        closedir(dirp);
#endif
    vsFiles.sort();

    // hack for cyberclassroom
    iterFiles = vsFiles.begin();

    std::list<std::string>::iterator itFiles;

    int count = 0;
    for (itFiles = vsFiles.begin(); itFiles != vsFiles.end(); itFiles++)
    {
        auto filename = *itFiles;
        auto b = new ui::Button(menu, "File"+std::to_string(count));
        b->setGroup(uniqueMenu);
        b->setText(filename.substr(0, filename.length()-4));
        b->setCallback([this, filename](bool state){
            if (!state)
                return;
            std::string f = dirName + "/";
            f += filename;
            Init();
            loadData(f.c_str(), m_rootNode.get());
        });

        //std::cout << "VRMoleculeViewer::readDirectory " << *itFiles << std::endl;
        //cover->addFunctionButton(((*itFiles).substr(0, (*itFiles).length() - 4)).c_str(), parent, fileSelection, this); // "Datafiles"

        ++count;
    }
}

#if 0
void VRMoleculeViewer::fileSelection(void *viewer, buttonSpecCell *spec)
{
    fprintf(stderr, "VRMoleculeViewer::fileSelection\n");

    VRMoleculeViewer *m = (VRMoleculeViewer *)viewer;
//   std::string filename;

#ifdef _WIN32
    m->filename = m->dirName + "\\";
#else
    m->filename = m->dirName + "/";
#endif

    std::list<std::string>::iterator itFiles;
    for (itFiles = m->vsFiles.begin(); itFiles != m->vsFiles.end(); itFiles++)
    {
        if (strcmp(((*itFiles).substr(0, (*itFiles).length() - 4)).c_str(), spec->name) == 0)
            m->filename += std::string(*itFiles);
    }

    if (cover->debugLevel(3))
        printf("%s selected to load\n", m->filename.c_str());

#ifdef USE_OLD_COVISE
    if (VRCoviseConnection::covconn)
    {
        if (m->m_strFeedbackInfo != std::string(""))
        {
            // Send Messages to Controller to update the string parameter in the ReadITT module
            CoviseRender::set_feedback_info(m->m_strFeedbackInfo.c_str()); // use the registered ReadITT module

            std::string strbuf;
            strbuf = "Filename\nString\n1\n" + m->filename + "\n";
            CoviseRender::send_feedback_message("PARAM", strbuf.c_str());
            CoviseRender::send_feedback_message("PARAM", strbuf.c_str());

            // Send Message to Controller to execute Network
            CoviseRender::send_feedback_message("EXEC", "");
        }
        else
        {
            printf("No registered ReadITT module available. Feedback message canceled.\n");
            printf("Remember to execute the pipeline including one ReadITT module at least once!\n");
        }
    }
    else
#endif
    {

        Init(viewer);
        m->loadData(const_cast<char *>(m->filename.c_str()), m->m_rootNode.get());
    }
}
#endif

#if 0
void VRMoleculeViewer::sliderCallback(void *viewer, buttonSpecCell *spec)
{
    VRMoleculeViewer *m = (VRMoleculeViewer *)viewer;

    m->slide(spec->state);
}
#endif

void VRMoleculeViewer::Init()
{
    // clear memory if any data has been loaded
    if (numberOfTimesteps)
    {
        if (cover->debugLevel(3))
            printf("clearing memory...\n");
        clearUp();
    }

    // set all back to zero
    maxNumberOfMolecules = 0;
    numberOfTimesteps = 0;
    moving = 0;
}

int
VRMoleculeViewer::loadData(const char *moleculepath, osg::Group *parent)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRMoleculeViewer::loadData for %s\n", moleculepath);

    Init();

    FILE *molecule_fp = fopen(moleculepath, "r");

    if (molecule_fp == NULL)
    {
        fprintf(stderr, "VRMoleculeViewer ERROR: Can't open file \"%s\"\n", moleculepath);
        return -1;
    }

    // read in the data
    // read the molecule structure first

    structure = new MoleculeStructure(molecule_fp, sphereRatio);

    // now we will read in the simulation data
    // we will perform are first scan to determine the
    // number of timesteps and the maximum number of
    // molecules appearing in a timestep

    char buf[512];
    int linenumber = 0;
    while (fgets(buf, LINE_SIZE, molecule_fp) != NULL)
    {
        // skip blank lines and comments
        if (*buf == '\0')
            // read the next line
            continue;

        // store the data according to the parts...
        if (*buf == '!') // this identifies another molecule in the frame
        {
            linenumber++;
        }

        if (*buf == '#') // this identifies a new frame
        {
            char tmps[1000];
            float tmpf;
            if (sscanf(buf, "%s %f", tmps, &tmpf) == 2)
            {
                numberOfTimesteps++;
                if (linenumber > maxNumberOfMolecules)
                    maxNumberOfMolecules = linenumber;
                linenumber = 0;
            }
        }
    }

    if (cover->debugLevel(3))
    {
        fprintf(stderr, "max number of molecules per timestep: %d\n", maxNumberOfMolecules);
        fprintf(stderr, "number of timesteps: %d\n", numberOfTimesteps);
    }

    viewerDCS = new osg::MatrixTransform;
    parent->addChild(viewerDCS.get());
    m_rootNode = parent;
    DCSList = new osg::ref_ptr<osg::MatrixTransform>[maxNumberOfMolecules];
    for (int i = 0; i < maxNumberOfMolecules; i++)
    {
        DCSList[i] = new osg::MatrixTransform;
        DCSList[i]->ref();
        viewerDCS->addChild(DCSList[i].get());
    }

    coVRAnimationManager::instance()->setNumTimesteps(numberOfTimesteps, plugin);

    // now the number of the parts and the maximum number of molecules are known
    // we can create arrays for the data and read them

    //reset file pointer to beginning
    fseek(molecule_fp, 0, SEEK_SET);

    char holder[16];
    int type;
    float x, y, z;
    float q0, q1, q2, q3;
    osg::Geode *molGeode;
    float cubeSize;

    int numberOfFrames; //quite the same as numberOfTimesteps...

    if (cover->debugLevel(3))
        printf("reading timestep data...\n");
    Frame *newFrame = NULL;
    framelist.reset();
    numberOfFrames = 0;
    int n;
    while (fgets(buf, LINE_SIZE, molecule_fp) != NULL)
    {

        // skip blank lines and comments
        if (*buf == '\0')
            continue;

        // store the data according to the parts...
        if (*buf == '!') // add a molecule to the frame
        {
            n = sscanf(buf, "%s %d %f %f %f %f %f %f %f", holder, &type, &x, &y, &z, &q0, &q1, &q2, &q3);
            if (newFrame != NULL)
            {
                if (n == 5)
                {
                    x = x / 10;
                    y = y / 10;
                    z = z / 10;
                    q0 = 0;
                    q1 = 0;
                    q2 = 0;
                    q3 = 1000;
                    n = 9;
                }
                if (n == 8)
                {
                    q3 = sqrt(1000000.0f - (q0 * q0 + q1 * q1 + q2 * q2));
                    n = 9;
                }
                if (n == 9)
                {
                    molGeode = structure->getMoleculeGeode(type);
                    newFrame->addMolecule(molGeode, x, y, z, q1, q2, q3, q0);
                }
            }
        }

        if (*buf == '#') // add a new frame
        {
            if (sscanf(buf, "%s %f", holder, &cubeSize) == 2)
            {
                newFrame = new Frame(maxNumberOfMolecules, viewerDCS.get(), cubeSize, DCSList);
                if (newFrame == NULL)
                {
                    printf("error creating frame object\n");
                    break;
                }
                framelist.append(newFrame);
                numberOfFrames++;
            }
        }
    }
    fclose(molecule_fp);
    if (cover->debugLevel(3))
        fprintf(stderr, " file operation complete!\n");
    show();

#ifdef USE_OLD_COVISE
    if (!VRCoviseConnection::covconn)
#endif
        m_bDoOnceViewAll = true;
    move();

    return 0;
}

void VRMoleculeViewer::show()
{

    if (framelist.current())
    {
        framelist.current()->hide();
        framelist.reset();
        frameIndex = 1;
        framelist.current()->display();
    }
}

void VRMoleculeViewer::move()
{
#if 0
   if(!VRCoviseConnection::covconn)
   {
   if(numberOfTimesteps>0)
   {
   moving=1;
   cover->setButtonState("Animate", true );
}
                               else
                         {
                               printf("load data first!\n");
                               cover->setButtonState("Animate", false );
}
}
#endif
}

void VRMoleculeViewer::NoMove()
{
    moving = 0;
#if 0
   cover->setButtonState("Animate", false );
#endif
}

void VRMoleculeViewer::stepping()
{
#if 0

   if( (cover->frameTime()-time) < timeBetweenFrames )
   return;

   time = cover->frameTime();

   if(animationSpeed>0)  stepForward();
   if(animationSpeed<0)  stepBackward();
#endif

} // VRMoleculeViewer::stepping()

#if 0
void VRMoleculeViewer::stepForward()
{
    if (numberOfTimesteps > 0)
    {
        framelist.current()->hide();

        if (frameIndex >= numberOfTimesteps)
        {
            framelist.reset();
            frameIndex = 1;
        }
        else
        {
            framelist.next();
            frameIndex++;
        }

        framelist.current()->display();

        float value = float(frameIndex);
        cover->setSliderValue("timesteps", value);
    }
}


void VRMoleculeViewer::stepBackward()
{
    if (numberOfTimesteps > 0)
    {
        framelist.current()->hide();

        if (frameIndex == 1)
        {
            framelist.set(framelist.num() - 1);
            frameIndex = numberOfTimesteps;
        }
        else
        {
            framelist.prev();
            frameIndex--;
        }
        framelist.current()->display();

        float value = float(frameIndex);
        cover->setSliderValue("timesteps", value);
    }
}

void VRMoleculeViewer::slide(float actual)
{

    frameIndex = int(actual);

    framelist.current()->hide();
    framelist.set(frameIndex - 1);
    framelist.current()->display();
}
#endif

void VRMoleculeViewer::clearUp()
// this aims to delete all data and restore the initial state
// before any data was loaded
{
    static bool bReady = false;

    if (!bReady)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "VRMoleculeViewer::clearUp\n");
        bReady = true;
        //stop animation
        NoMove();

#if 0
        //remove frameindex slider from menu
        if (numberOfTimesteps)
            cover->removeButton("timesteps", "Molecules");
#endif

        if (framelist.current())
            framelist.current()->hide();
        framelist.reset();
        while (framelist.current())
            framelist.remove();

        if (DCSList != NULL)
        {
            for (int i = 0; i < maxNumberOfMolecules; i++)
            {
                while (DCSList[i]->getNumChildren())
                    DCSList[i]->removeChild(DCSList[i]->getChild(0));
                if (viewerDCS.valid())
                    viewerDCS->removeChild(DCSList[i].get());
                DCSList[i]->unref();
            }
            delete[] DCSList;
            DCSList = NULL;
        }

        // then we disconnect our top node from cover
        if (viewerDCS.valid())
        {
            osg::Group *p = viewerDCS->getParent(0);
            if (p)
                p->removeChild(viewerDCS.get());
        }
        viewerDCS = NULL;

        maxNumberOfMolecules = 0;
        numberOfTimesteps = 0;
        frameIndex = 0;

        if (structure != NULL)
        {
            delete structure;
            structure = NULL;
        }
        bReady = false;
    }

    coVRAnimationManager::instance()->setNumTimesteps(0, plugin);
}

void VRMoleculeViewer::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin VRMoleculeViewer coVRGuiToRenderMsg %s\n", msg.getString().c_str());

    if (msg.isValid())
    {
        if (msg.getType() == grmsg::coGRMsg::KEYWORD)
        {

            auto &keyWordMsg = msg.as<grmsg::coGRKeyWordMsg>();
            const char *keyword = keyWordMsg.getKeyWord();
            if (cover->debugLevel(3))
                fprintf(stderr, "\tcoGRMsg::KEYWORD keyword=%s\n", keyword);
            if (strcmp(keyword, "presForward") == 0)
            {
                //fprintf(stderr, "presForward\n");
                iterFiles++;

                // only if not end of vsFiles
                if (iterFiles == vsFiles.end())
                    iterFiles = vsFiles.begin();

#ifdef _WIN32
                filename = dirName + "\\";
#else
                filename = dirName + "/";
#endif

                filename += std::string(*iterFiles);

                Init();
                loadData(filename.c_str(), m_rootNode.get());
            }
            else if (strcmp(keyword, "presBackward") == 0)
            {
                //fprintf(stderr, "presBackward\n");
                //fprintf(stderr,"VRMoleculeViewer::message for %s\n", chbuf );

                if (iterFiles != vsFiles.begin())
                {
                    iterFiles--;

#ifdef _WIN32
                    filename = dirName + "\\";
#else
                    filename = dirName + "/";
#endif

                    filename += std::string(*iterFiles);
                    Init();
                    loadData(filename.c_str(), m_rootNode.get());
                }
            }
            else if (strncmp(keyword, "goToStep", 8) == 0)
            {
                stringstream sstream(keyword);
                string sub;
                int step;
                sstream >> sub;
                sstream >> step;
                //fprintf(stderr, "goToStep %d\n", step);
                //fprintf(stderr,"VRMoleculeViewer::message for %s\n", chbuf );
                iterFiles = vsFiles.begin();
                for (int i = 0; i < step; i++)
                    iterFiles++;

#ifdef _WIN32
                filename = dirName + "\\";
#else
                filename = dirName + "/";
#endif

                filename += std::string(*iterFiles);

                Init();
                loadData(filename.c_str(), m_rootNode.get());
            }
        }
    }
}

COVERPLUGIN(VRMoleculeViewer)
