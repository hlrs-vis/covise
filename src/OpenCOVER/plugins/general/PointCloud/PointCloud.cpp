/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description: Rendering binary point data
//
// Author: Philip Weber
//
// Creation Date: 2007-03-07
//
// **************************************************************************

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <functional>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>

#include <OpenVRUI/coButtonInteraction.h>
#include <cover/coVRShader.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <cover/ui/ButtonGroup.h>

// OSG:
#include <osg/Node>
#include <osg/Group>
#include <osg/Switch>
#include <osgDB/ReadFile>

#include <cover/coVRTui.h>
#include <util/unixcompat.h>

// Local:
#include "PointCloud.h"

#ifdef HAVE_E57
#include <e57/E57Foundation.h>
#include <e57/E57Simple.h>
#endif

using namespace osg;
using namespace std;
using covise::coCoviseConfig;
using vrui::coInteraction;

const int MAX_POINTS = 30000000;
PointCloudPlugin *PointCloudPlugin::plugin = NULL;
PointCloudInteractor *PointCloudPlugin::s_pointCloudInteractor = NULL;


COVERPLUGIN(PointCloudPlugin)

// Constructor
PointCloudPlugin::PointCloudPlugin()
: ui::Owner("PointCloud",cover->ui)
, pointSizeValue("pointSizeValue", 4.0)
{
}

static FileHandler handlers[] = {
    { NULL,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::unloadPTS,
      "pts" },
    { NULL,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::unloadPTS,
      "ptx" },
    { NULL,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::unloadPTS,
      "xyz" },
    { NULL,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::unloadPTS,
      "ptsb" },
    { NULL,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::loadPTS,
      PointCloudPlugin::unloadPTS,
      "c2m" },
	  { NULL,
	  PointCloudPlugin::loadPTS,
	  PointCloudPlugin::loadPTS,
	  PointCloudPlugin::unloadPTS,
	  "e57" }
};

bool PointCloudPlugin::init()
{
    if (plugin != NULL)
    {
        return false;
    }
    plugin = this;
    //pointSizeValue = coCoviseConfig::getFloat("COVER.Plugin.PointCloud.PointSize", pointSizeValue);
	std::function<void(void)> update = [this](void) {UpdatePointSizeValue(); };
	pointSizeValue.setUpdateFunction(update);
    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);
    coVRFileManager::instance()->registerFileHandler(&handlers[2]);
    coVRFileManager::instance()->registerFileHandler(&handlers[3]);
	coVRFileManager::instance()->registerFileHandler(&handlers[4]);
	coVRFileManager::instance()->registerFileHandler(&handlers[5]);

    //Create main menu button
    pointCloudMenu = new ui::Menu("PointCloudMenu",this);
    pointCloudMenu->setText("Point cloud");

    // Create menu
#if 0
    char name[100];
    sprintf(name, "PointCloudFiles");
    fileGroup = new ui::Group(pointCloudMenu, name);
    sprintf(name, "Files");
    fileGroup->setText(name);
#endif

    loadMenu = new ui::Menu(pointCloudMenu,"Load");
    //loadGroup = new ui::Group("Load", loadMenu);
    //deleteButton = new ui::Button(fileGroup,"Delete");
    selectionGroup = new ui::Group(pointCloudMenu,"Selection");
    selectionButtonGroup = new ui::ButtonGroup(selectionGroup, "SelectionGroup");
    selectionButtonGroup->enableDeselect(true);
    singleSelectButton = new ui::Button(selectionGroup, "SelectPoints", selectionButtonGroup);
    singleSelectButton->setText("Select Points");
    singleSelectButton->setCallback([this](bool state){
        if (state)
        {
            //enable interaction
            vrui::coInteractionManager::the()->registerInteraction(s_pointCloudInteractor);
            //cover->addPlugin("NurbsSurface");
        }
        else
        {
            vrui::coInteractionManager::the()->unregisterInteraction(s_pointCloudInteractor);
        } 
    });
    deselectButton = new ui::Button(selectionGroup, "DeselectPoints", selectionButtonGroup);
    deselectButton->setText("Deselect Points");
    deselectButton->setCallback([this](bool state){
        if (state)
        {
        //enable interaction
        vrui::coInteractionManager::the()->registerInteraction(s_pointCloudInteractor);
        s_pointCloudInteractor->setDeselection(true);
        }
        else
        {
        vrui::coInteractionManager::the()->unregisterInteraction(s_pointCloudInteractor);
        s_pointCloudInteractor->setDeselection(false);
        }
    });
    createNurbsSurface = new ui::Button(pointCloudMenu,"createNurbsSurface");
    createNurbsSurface->setText("Create nurbs surface from selected points");
    createNurbsSurface->setCallback([this](bool state){
        if (state)
        {
            cover->addPlugin("NurbsSurface");
        }
        else
        {
            cover->removePlugin("NurbsSurface");
        }
    });
/*
    //Create main menu button
    imanPluginInstanceMenuItem = new coSubMenuItem("Point Model Plugin");
    imanPluginInstanceMenuItem->setMenuListener(this);
    cover->getMenu()->add(imanPluginInstanceMenuItem);

    // Create menu
    imanPluginInstanceMenu = new coRowMenu("Point Model Options");
    //enablePointCloudPlugin = new coCheckboxMenuItem("Enable", false);
    //enablePointCloudPlugin->setMenuListener(this);
    //imanPluginInstanceMenu->add(enablePointCloudPlugin);
    imanPluginInstanceMenuItem->setMenu(imanPluginInstanceMenu);

    // add a load sub menu (has files that can be loaded)
    loadMenuItem = new coSubMenuItem("Load");
    loadMenu = new coRowMenu("Files");
    loadMenuItem->setMenu(loadMenu);
    imanPluginInstanceMenu->add(loadMenuItem);

    // need a delete button
    deleteMenuItem = new coButtonMenuItem("Delete");
    imanPluginInstanceMenu->add(deleteMenuItem);
    deleteMenuItem->setMenuListener(this);
*/
    //imanPluginInstanceDrawable = NULL;
    //imanPluginInstanceNode = NULL;
    planetTrans = new MatrixTransform();
    osg::Matrix mat;
    mat.makeIdentity();
    float scale = coCoviseConfig::getFloat("COVER.Plugin.PointCloud.Scale", 1);
    float x = coCoviseConfig::getFloat("x", "COVER.Plugin.PointCloud.Translation", 0);
    float y = coCoviseConfig::getFloat("y", "COVER.Plugin.PointCloud.Translation", 0);
    float z = coCoviseConfig::getFloat("z", "COVER.Plugin.PointCloud.Translation", 0);
    adaptLOD = coCoviseConfig::isOn("COVER.Plugin.PointCloud.AdaptLOD", adaptLOD);
    mat.makeScale(scale, scale, scale);
    mat.setTrans(Vec3(x, y, z));
    planetTrans->setMatrix(mat);
    planetTrans->setName("PointCloudNode");

    // add the base node to the scenegraph
    cover->getObjectsRoot()->addChild(planetTrans);

    pointSet = NULL;

    //read in menu data
    readMenuConfigData("COVER.Plugin.PointCloud.Files", pointVec, loadMenu);


    //PCTab = new coTUITab("PointCloud", coVRTui::instance()->mainFolder->getID());
    //PCTab->setPos(0, 0);

    //viewGroup = new ui::Group(pointCloudMenu,"PCView");
    adaptLODButton = new ui::Button(pointCloudMenu,"adaptLOD");
    adaptLODButton->setState(adaptLOD);
    adaptLODButton->setText("Adapt level of detail");
    adaptLODButton->setCallback([this](bool state){
        adaptLOD = state;
        if (!adaptLOD)
        {
            changeAllLOD(lodScale);
        }
    });

    pointSizeSlider = new ui::Slider(pointCloudMenu, "pointSize");
    pointSizeSlider->setText("Point size");
    pointSizeSlider->setBounds(1.0,10.0);
    pointSizeSlider->setValue(pointSizeValue);
    pointSizeSlider->setCallback([this](double value, bool released){
        pointSizeValue = value;
        changeAllPointSize(pointSizeValue);
    });

    auto lodScaleSlider = new ui::Slider(pointCloudMenu, "lodScale");
    lodScaleSlider->setText("LOD scale");
    lodScaleSlider->setBounds(0.01, 100.);
    lodScaleSlider->setValue(1.);
    lodScaleSlider->setScale(ui::Slider::Logarithmic);
    lodScaleSlider->setCallback([this](double value, bool released){
        lodScale = value;
        if (!adaptLOD)
        {
            changeAllLOD(lodScale);
        }
    });

/*
    adaptLODTui = new coTUIToggleButton("adaptLOD", PCTab->getID());
    adaptLODTui->setEventListener(this);
    adaptLODTui->setState(adaptLOD);
    
    coTUILabel *pointSizeLabel = new coTUILabel("pointSize:");
    pointSizeTui = new coTUIFloatSlider("PointSize", PCTab->getID());
    pointSizeTui->setEventListener(this);
    pointSizeTui->setMin(1.0);
    pointSizeTui->setMax(10.0);
    pointSizeTui->setValue(pointSizeValue);
    

    adaptLODTui->setPos(0, 0);
    pointSizeLabel->setPos(0, 1);
    pointSizeTui->setPos(1, 1);
    */

    assert(!s_pointCloudInteractor);
    s_pointCloudInteractor = new PointCloudInteractor(coInteraction::ButtonA, "PointCloud", coInteraction::High);

    return true;
}

/// Destructor
PointCloudPlugin::~PointCloudPlugin()
{

	coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
	coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);
	coVRFileManager::instance()->unregisterFileHandler(&handlers[2]);
	coVRFileManager::instance()->unregisterFileHandler(&handlers[3]);
	coVRFileManager::instance()->unregisterFileHandler(&handlers[4]);
	coVRFileManager::instance()->unregisterFileHandler(&handlers[5]);

    // clean the scenegraph and free memory
    clearData();

    if (planetTrans)
        cover->getObjectsRoot()->removeChild(planetTrans);
    planetTrans = NULL;

/*    delete imanPluginInstanceMenuItem;
    delete imanPluginInstanceMenu;
    delete loadMenuItem;
    delete loadMenu;
*/
    //clean up TUI
    //delete PCTab;
    //delete adaptLODTui;
    
    delete s_pointCloudInteractor;
    s_pointCloudInteractor = nullptr;
    vector<ImageFileEntry>::iterator itEntry = pointVec.begin();
    for (; itEntry < pointVec.end(); itEntry++)
    {
        delete itEntry->fileMenuItem;
    }
    //delete deleteMenuItem;
}

int PointCloudPlugin::loadPTS(const char *filename, osg::Group *loadParent, const char *)
{
    std::string filen;
    filen = filename;
    osg::Group *g = new osg::Group;
    loadParent->addChild(g);
    if (filename != NULL)
        g->setName(filename);
    plugin->createGeodes(g, filen);
    return 1;
}


// read in and store the menu data from the configuration file
void PointCloudPlugin::readMenuConfigData(const char *menu, vector<ImageFileEntry> &menulist, ui::Group *subMenu)
{
    coCoviseConfig::ScopeEntries e = coCoviseConfig::getScopeEntries(menu);
    const char **entries = e.getValue();
    if (entries)
    {
        while (*entries)
        {
            const char *menuName = *entries;
            entries++;
            const char *fileName = *entries;
            entries++;
            if (fileName && menuName)
            {
                std::string filename= fileName;
                //create button and append it to the submenu
                ui::Button *temp = new ui::Button(subMenu, fileName);
                temp->setCallback([this, filename](bool state){
                    if (state)
                        createGeodes(planetTrans, filename);
                });
                menulist.push_back(ImageFileEntry(menuName, fileName, (ui::Element *)temp));
            }
        }
    }
}


void PointCloudPlugin::changeAllLOD(float lod)
{
    for (std::vector<FileInfo>::iterator fit = files.begin(); fit != files.end(); fit++)
    {
        //TODO calc distance correctly
        for (std::vector<NodeInfo>::iterator nit = fit->nodes.begin(); nit != fit->nodes.end(); nit++)
        {
            auto geo = dynamic_cast<PointCloudGeometry *>(nit->node->getDrawable(0));
            if (geo)
                geo->changeLod(lod);
        }
    }
}

void PointCloudPlugin::changeAllPointSize(float pointSize)
{
    for (std::vector<FileInfo>::iterator fit = files.begin(); fit != files.end(); fit++)
    {
        //TODO calc distance correctly
        for (std::vector<NodeInfo>::iterator nit = fit->nodes.begin(); nit != fit->nodes.end(); nit++)
        {
            auto geo = dynamic_cast<PointCloudGeometry *>(nit->node->getDrawable(0));
            if (geo)
                geo->setPointSize(pointSize);
        }
    }
}
void PointCloudPlugin::UpdatePointSizeValue(void) {
	changeAllPointSize(pointSizeValue);
    pointSizeSlider->setValue(pointSizeValue);
}

// create and add geodes to the scene  //DEFAULT JUST LOADS New_10x10x10.xyz  //UPDATE will be using the menu
void PointCloudPlugin::createGeodes(Group *parent, const string &filename)
{
    opencover::coVRShader *pointShader = opencover::coVRShaderList::instance()->get("Points");
    const char *cfile = filename.c_str();
    if ((strcasecmp(cfile + strlen(cfile) - 3, "pts") == 0) || (strcasecmp(cfile + strlen(cfile) - 3, "ptx") == 0) || (strcasecmp(cfile + strlen(cfile) - 3, "xyz") == 0))
    {
        bool imwfLattice = false;
        intensityOnly = false;
        intColor = false;
        polar = false;
        bool commaSeparated = false;
        FILE *fp = fopen(cfile, "r");
        pointSetSize = 0;
        intensityScale = 10;
        char buf[1000];
        if (!fp)
        {
            cout << "Error opening file" << endl;
            return;
        }

        int psize = 0;
        int numHeaderLines = 0;
        while (!feof(fp))
        {
            if (!fgets(buf, 1000, fp))
            {
                fprintf(stderr, "failed to get line\n");
            }
            if (buf[0] == '#')
            {
                if (strstr(buf, "intensityOnly") != NULL)
                {
                    intensityOnly = true;
                    fprintf(stderr, "intensityOnly\n");
                }
                const char *intensityString;
                if ((intensityString = strstr(buf, "intensityScale")) != NULL)
                {
                    sscanf(intensityString+14,"%f",&intensityScale);
                    fprintf(stderr, "intensityScale %f\n",intensityScale);
                }
                if (strstr(buf, "intColor") != NULL)
                {
                    intColor = true;
                    fprintf(stderr, "intColor\n");
                }
                if (strstr(buf, "polar") != NULL)
                {
                    polar = true;
                    fprintf(stderr, "polar\n");
                }
                if (strstr(buf, "commaSeparated") != NULL)
                {
                    commaSeparated = true;
                    fprintf(stderr, "commaSeparated\n");
                }
                numHeaderLines++;
            }
            else if (strstr(buf, "Lattice=") == buf)
            {
                imwfLattice = true;
                fprintf(stderr, "IMWF lattice - rename file to .indent and use Particles plug-in\n");
                numHeaderLines = 2;
            }
            else
                psize++;
        }
        fseek(fp, 0, SEEK_SET);
        for (int i = 0; i < numHeaderLines; i++)
        {
            if (!fgets(buf, 1000, fp))
            {
                fprintf(stderr, "failed to get header line %d\n", i);
            }
        }

        cerr << "Total num of points is " << psize << endl;
        pointSet = new PointSet[1];

        pointSet[0].colors = new Color[psize];
        pointSet[0].points = new ::Point[psize];
        pointSet[0].size = psize;
        /*int partSize = psize/64;
         int i=0;
         int n=0;
         int s=0;
         while(!feof(fp))
         {
            fgets(buf,1000,fp);
            sscanf(buf,"%f %f %f %f %f %f,",&pointSet[0].points[n].x,&pointSet[0].points[n].y,&pointSet[0].points[n].z,&pointSet[0].colors[n].r,&pointSet[0].colors[n].g,&pointSet[0].colors[n].b);
            i++;
            n+=partSize;
            if(n>psize)
            {
               s++;
               n=s;
            }
         }*/
        int i = 0;
        while (!feof(fp))
        {
            if (!fgets(buf, 1000, fp))
            {
                fprintf(stderr, "failed 2 to get line\n");
            }
            if (imwfLattice)
            {
                int id=0;
                char type[1000];
                float dummy=0.f;
                int numValues = sscanf(buf, "%d %s %f %f %f %f", &id, type, &pointSet[0].points[i].x, &pointSet[0].points[i].y, &pointSet[0].points[i].z, &dummy);
                pointSet[0].colors[i].g = pointSet[0].colors[i].b = pointSet[0].colors[i].r = 1.0;
            }
            else if (commaSeparated)
            {
                int numValues = sscanf(buf, "%f,%f,%f,%f", &pointSet[0].points[i].x, &pointSet[0].points[i].y, &pointSet[0].points[i].z, &pointSet[0].colors[i].r);
                if (numValues == 4)
                {
                    pointSet[0].colors[i].g = pointSet[0].colors[i].b = pointSet[0].colors[i].r;
                }
                else
                {
                    pointSet[0].colors[i].g = pointSet[0].colors[i].b = pointSet[0].colors[i].r = 1.0;
                }
            }
            else if (intensityOnly)
            {
                float intensity;
                int numValues = sscanf(buf, "%f %f %f %f %f %f %f,", &pointSet[0].points[i].x, &pointSet[0].points[i].y, &pointSet[0].points[i].z, &pointSet[0].colors[i].r, &pointSet[0].colors[i].g, &pointSet[0].colors[i].b, &intensity);
                if (numValues == 7)
                {
                    pointSet[0].colors[i].g = pointSet[0].colors[i].b = pointSet[0].colors[i].r = intensity * intensityScale;
                }
                else
                {
                    pointSet[0].colors[i].g = pointSet[0].colors[i].b = pointSet[0].colors[i].r * intensityScale;
                }
            }
            else
            {
                int numValues = sscanf(buf, "%f %f %f %f %f %f,", &pointSet[0].points[i].x, &pointSet[0].points[i].y, &pointSet[0].points[i].z, &pointSet[0].colors[i].r, &pointSet[0].colors[i].g, &pointSet[0].colors[i].b);
                if (numValues < 6)
                {
                    pointSet[0].colors[i].g = pointSet[0].colors[i].b = pointSet[0].colors[i].r;
                }
                if (intColor)
                {
                    pointSet[0].colors[i].g /= 255;
                    pointSet[0].colors[i].b /= 255;
                    pointSet[0].colors[i].r /= 255;
                }

                if (numValues < 3) // invalid coordinate
                    i--;
            }

            i++;
        }
        psize = i;
        pointSet[0].size = psize;
        cerr << "Total num of valid points is " << psize << endl;

        if (polar)
        {
            for (int i = 0; i < psize; i++)
            {
                // convert to cartesian
                float vx = sin(pointSet[0].points[i].x) * cos(pointSet[0].points[i].y);
                float vy = sin(pointSet[0].points[i].x) * sin(pointSet[0].points[i].y);
                float vz = cos(pointSet[0].points[i].x);
                pointSet[0].points[i].x = vx * pointSet[0].points[i].z;
                pointSet[0].points[i].y = vy * pointSet[0].points[i].z;
                pointSet[0].points[i].z = vz * pointSet[0].points[i].z;
            }
        }

        FileInfo fi;
        fi.pointSetSize = pointSetSize;
        fi.pointSet = pointSet;

        //create drawable and geode and add to the scene (make sure the cube is not empty)
        if (pointSet[0].size != 0)
        {
            PointCloudGeometry *drawable = new PointCloudGeometry(&pointSet[0]);
            drawable->changeLod(lodScale);
            drawable->setPointSize(pointSizeValue);
            Geode *currentGeode = new Geode();
            currentGeode->addDrawable(drawable);
            currentGeode->setName(filename);
            parent->addChild(currentGeode);
            NodeInfo ni;
            ni.node = currentGeode;
            fi.nodes.push_back(ni);
            if (pointShader)
                pointShader->apply(currentGeode, drawable);
        }
        files.push_back(fi);
        cerr << "closing the file" << endl;
        fclose(fp);
        return;
    }
    else if (strcasecmp(cfile + strlen(cfile) - 3, "c2m") == 0)
    {
        cout << "iCloud2Max Data: " << filename << endl;

        ifstream file(filename.c_str(), ios::in | ios::binary);

        pointSetSize = 0;
        char *buf = new char[128];
        file.read(buf, 68);
        pointSetSize = ((unsigned int *)(buf+20))[0]; 
        if (file.is_open())
        {
            cerr << "Total num of sets is " << pointSetSize << endl;
            pointSet = new PointSet[pointSetSize];
            FileInfo fi;
            fi.pointSetSize = pointSetSize;
            fi.pointSet = pointSet;
            for (int i = 0; i < pointSetSize; i++)
            {
                int psize;
                file.read(buf, 60);
                psize = ((unsigned int *)(buf))[0]; 
                pointSet[i].colors = new Color[psize];
                pointSet[i].points = new ::Point[psize];
                pointSet[i].size = psize;

                for (int n = 0; n < psize; n++)
                {
                    // read point data
                    file.read(buf, 36);
                    pointSet[i].points[n].x = ((float *)buf)[0];
                    pointSet[i].points[n].y = ((float *)buf)[1];
                    pointSet[i].points[n].z = ((float *)buf)[2];
                    pointSet[i].colors[n].r = (((unsigned char *)buf)[14]) / 255.0;
                    pointSet[i].colors[n].g = (((unsigned char *)buf)[13]) / 255.0;
                    pointSet[i].colors[n].b = (((unsigned char *)buf)[12]) / 255.0;
                }

                //create drawable and geode and add to the scene (make sure the cube is not empty)

                if (pointSet[i].size != 0)
                {
                    PointCloudGeometry *drawable = new PointCloudGeometry(&pointSet[i]);
                    drawable->changeLod(lodScale);
                    drawable->setPointSize(pointSizeValue);
                    Geode *currentGeode = new Geode();
                    currentGeode->addDrawable(drawable);
                    currentGeode->setName(filename);
                    parent->addChild(currentGeode);
                    NodeInfo ni;
                    ni.node = currentGeode;
                    fi.nodes.push_back(ni);
                }
            }
            files.push_back(fi);
            cerr << "closing the file" << endl;
            file.close();
            return;
        }
        cout << "Error opening file" << endl;
        return;
    }
	else if (strcasecmp(cfile + strlen(cfile) - 3, "e57") == 0)
	{
		cout << "e57 Data: " << filename << endl;


#ifdef HAVE_E57

		osg::Matrix m;
		m.makeIdentity();
		try
		{
			e57::Reader	eReader(filename);
			e57::E57Root	rootHeader;
			eReader.GetE57Root(rootHeader);


			//Get the number of scan images available
			int data3DCount = eReader.GetData3DCount();
			e57::Data3D		scanHeader;
			cerr << "Total num of sets is " << data3DCount << endl;
			pointSet = new PointSet[data3DCount];
			FileInfo fi;
			fi.pointSetSize = data3DCount;
			fi.pointSet = pointSet;
			for (int scanIndex = 0; scanIndex < data3DCount; scanIndex++)
			{
				eReader.ReadData3D(scanIndex, scanHeader);
				fprintf(stderr, "reading Name: %s\n", scanHeader.name.c_str());
				osg::Matrix trans;
				trans.makeTranslate(scanHeader.pose.translation.x, scanHeader.pose.translation.y, scanHeader.pose.translation.z);
				osg::Matrix rot;
				rot.makeRotate(osg::Quat(scanHeader.pose.rotation.x, scanHeader.pose.rotation.y, scanHeader.pose.rotation.z, scanHeader.pose.rotation.w));
				m = rot*trans;

				int64_t nColumn = 0;
				int64_t nRow = 0;


				int64_t nPointsSize = 0;	//Number of points


				int64_t nGroupsSize = 0;	//Number of groups
				int64_t nCountSize = 0;		//Number of points per group
				bool	bColumnIndex = false; //indicates that idElementName is "columnIndex"


				eReader.GetData3DSizes(scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, nCountSize, bColumnIndex);


				int64_t nSize = nRow;
				if (nSize == 0) nSize = 1024;	// choose a chunk size

				int8_t * isInvalidData = NULL;
                isInvalidData = new int8_t[nSize];
                if (!scanHeader.pointFields.cartesianInvalidStateField)
                {
                    for (int i = 0; i < nSize; i++)
                        isInvalidData[i] = 0;
                }


				double * xData = NULL;
				if (scanHeader.pointFields.cartesianXField)
					xData = new double[nSize];
				double * yData = NULL;
				if (scanHeader.pointFields.cartesianYField)
					yData = new double[nSize];
				double * zData = NULL;
				if (scanHeader.pointFields.cartesianZField)
					zData = new double[nSize];

				double *	intData = NULL;
				bool		bIntensity = false;
				double		intRange = 0;
				double		intOffset = 0;


				if (scanHeader.pointFields.intensityField)
				{
					bIntensity = true;
					intData = new double[nSize];
					intRange = scanHeader.intensityLimits.intensityMaximum - scanHeader.intensityLimits.intensityMinimum;
					intOffset = scanHeader.intensityLimits.intensityMinimum;
				}


				uint16_t *	redData = NULL;
				uint16_t *	greenData = NULL;
				uint16_t *	blueData = NULL;
				bool		bColor = false;
				int32_t		colorRedRange = 1;
				int32_t		colorRedOffset = 0;
				int32_t		colorGreenRange = 1;
				int32_t		colorGreenOffset = 0;
				int32_t		colorBlueRange = 1;
				int32_t		colorBlueOffset = 0;


				if (scanHeader.pointFields.colorRedField)
				{
					bColor = true;
					redData = new uint16_t[nSize];
					greenData = new uint16_t[nSize];
					blueData = new uint16_t[nSize];
					colorRedRange = scanHeader.colorLimits.colorRedMaximum - scanHeader.colorLimits.colorRedMinimum;
					colorRedOffset = scanHeader.colorLimits.colorRedMinimum;
					colorGreenRange = scanHeader.colorLimits.colorGreenMaximum - scanHeader.colorLimits.colorGreenMinimum;
					colorGreenOffset = scanHeader.colorLimits.colorGreenMinimum;
					colorBlueRange = scanHeader.colorLimits.colorBlueMaximum - scanHeader.colorLimits.colorBlueMinimum;
					colorBlueOffset = scanHeader.colorLimits.colorBlueMinimum;
				}



				int64_t * idElementValue = NULL;
				int64_t * startPointIndex = NULL;
				int64_t * pointCount = NULL;
				if (nGroupsSize > 0)
				{
					idElementValue = new int64_t[nGroupsSize];
					startPointIndex = new int64_t[nGroupsSize];
					pointCount = new int64_t[nGroupsSize];

					if (!eReader.ReadData3DGroupsData(scanIndex, nGroupsSize, idElementValue,
						startPointIndex, pointCount))
						nGroupsSize = 0;
				}

				int8_t * rowIndex = NULL;
				int32_t * columnIndex = NULL;
				if (scanHeader.pointFields.rowIndexField)
					rowIndex = new int8_t[nSize];
				if (scanHeader.pointFields.columnIndexField)
					columnIndex = new int32_t[nRow];




				e57::CompressedVectorReader dataReader = eReader.SetUpData3DPointsData(
					scanIndex,			//!< data block index given by the NewData3D
					nSize,				//!< size of each of the buffers given
					xData,				//!< pointer to a buffer with the x data
					yData,				//!< pointer to a buffer with the y data
					zData,				//!< pointer to a buffer with the z data
					isInvalidData,		//!< pointer to a buffer with the valid indication
					intData,			//!< pointer to a buffer with the lidar return intesity
					NULL,
					redData,			//!< pointer to a buffer with the color red data
					greenData,			//!< pointer to a buffer with the color green data
					blueData/*,*/			//!< pointer to a buffer with the color blue data
											/*NULL,
											NULL,
											NULL,
											NULL,
											rowIndex,			//!< pointer to a buffer with the rowIndex
											columnIndex			//!< pointer to a buffer with the columnIndex*/
				);

				int64_t		count = 0;
				unsigned	size = 0;
				int			col = 0;
				int			row = 0;

				std::vector<Color> colors;
				std::vector<::Point> points;
				colors.reserve(nPointsSize);
				points.reserve(nPointsSize);

				::Point point;
				Color color;
				while (size = dataReader.read())
				{
					for (unsigned int i = 0; i < size; i++)
					{

						if ( isInvalidData[i] == 0 && (xData[i]!=0.0 &&yData[i] != 0.0 &&zData[i] != 0.0))
						{
							osg::Vec3 p(xData[i], yData[i], zData[i]);
							p = p * m;
							point.x = p[0];
							point.y = p[1];
							point.z = p[2];


							if (bIntensity) {		//Normalize intensity to 0 - 1.
								int intensity = ((intData[i] - intOffset) / intRange) * 255;
								color.r = intensity;
								color.g = intensity;
								color.b = intensity;
							}


							if (bColor) {			//Normalize color to 0 - 255
								color.r = (redData[i] - colorRedOffset) / (float)colorRedRange;
								color.g = (greenData[i] - colorGreenOffset) / (float)colorBlueRange;
								color.b = (blueData[i] - colorBlueOffset) / (float)colorBlueRange;

							}
							points.push_back(point);
							colors.push_back(color);

						}
					}

				}

				pointSet[scanIndex].colors = new Color[points.size()];
				pointSet[scanIndex].points = new ::Point[points.size()];
				memcpy(pointSet[scanIndex].points, &points[0], points.size() * sizeof(::Point));
				memcpy(pointSet[scanIndex].colors, &colors[0], points.size() * sizeof(Color));
				pointSet[scanIndex].size = points.size();
				dataReader.close();

				if (isInvalidData) delete isInvalidData;
				if (xData) delete xData;
				if (yData) delete yData;
				if (zData) delete zData;
				if (intData) delete intData;
				if (redData) delete redData;
				if (greenData) delete greenData;
				if (blueData) delete blueData;

                calcMinMax(pointSet[scanIndex]);

				if (pointSet[scanIndex].size != 0)
				{
					PointCloudGeometry *drawable = new PointCloudGeometry(&pointSet[scanIndex]);
                    drawable->changeLod(lodScale);
                    drawable->setPointSize(pointSizeValue);
                    Geode *currentGeode = new Geode();
					currentGeode->addDrawable(drawable);
					currentGeode->setName(filename);
					parent->addChild(currentGeode);
					NodeInfo ni;
					ni.node = currentGeode;
					fi.nodes.push_back(ni);
				}
			}

			files.push_back(fi);
            s_pointCloudInteractor->updatePoints(&files);
			eReader.Close();
			return;
		}
		catch (e57::E57Exception& ex) {
			ex.report(__FILE__, __LINE__, __FUNCTION__);
			return;
		}
		catch (std::exception& ex) {
			cerr << "Got an std::exception, what=" << ex.what() << endl;
			return;
		}
		catch (...) {
			cerr << "Got an unknown exception" << endl;
			return;
		}
#else
		cout << "Missing e75 library " << filename << endl;
#endif
		
	}
    else // ptsb binary randomized blocked
    {
        cout << "Input Data: " << filename << endl;

        ifstream file(filename.c_str(), ios::in | ios::binary);

        pointSetSize = 0;

        if (!file.is_open())
        {
            cerr << "Error opening file" << endl;
            return;
        }

        file.read((char *)&pointSetSize, sizeof(int));
        cerr << "Total num of sets is " << pointSetSize << endl;
        pointSet = new PointSet[pointSetSize];
        FileInfo fi;
        fi.pointSetSize = pointSetSize;
        fi.pointSet = pointSet;
        for (int i = 0; i < pointSetSize; i++)
        {
            int psize;
            file.read((char *)&psize, sizeof(psize));
            pointSet[i].colors = new Color[psize];
            pointSet[i].points = new ::Point[psize];
            pointSet[i].size = psize;

            // read point data
            file.read((char *)(pointSet[i].points), (sizeof(::Point) * psize));
            //read color data
            uint32_t *pc = new uint32_t[psize];
            file.read((char *)(pc), (sizeof(uint32_t) * psize));
            for (int n = 0; n < psize; n++)
            {
                pointSet[i].colors[n].r = (pc[n] & 0xff) / 255.0;
                pointSet[i].colors[n].g = ((pc[n] >> 8) & 0xff) / 255.0;
                pointSet[i].colors[n].b = ((pc[n] >> 16) & 0xff) / 255.0;
            }
            delete[] pc;

            calcMinMax(pointSet[i]);

            //create drawable and geode and add to the scene (make sure the cube is not empty)

            if (pointSet[i].size != 0)
            {
                PointCloudGeometry *drawable = new PointCloudGeometry(&pointSet[i]);
                drawable->changeLod(lodScale);
                drawable->setPointSize(pointSizeValue);
                Geode *currentGeode = new Geode();
                currentGeode->addDrawable(drawable);
                currentGeode->setName(filename);
                parent->addChild(currentGeode);
                NodeInfo ni;
                ni.node = currentGeode;
                fi.nodes.push_back(ni);
            }
        }
        uint32_t version;
        file.read((char *)&version,sizeof(uint32_t));
        bool readScannerPositions = false;
        if (file.good() && !file.eof())
            readScannerPositions= true;
        if (readScannerPositions)
        {
            //read Scanner positions

            cerr << "Version " << (version) << endl;
            uint32_t numPositions;
            file.read((char *)&numPositions, sizeof(uint32_t));
            for (int i=0; i!=numPositions; i++)
            {
                ScannerPosition pos;
                pos.type = 0;
                file.read((char *)&pos.ID, sizeof(uint32_t));
                file.read((char *)&pos.point._v, sizeof(float) * 3);
                positions.push_back(pos);
                //cerr << "Scannerposition " << pos.ID << " x: " << pos.point.x() << " y: " << pos.point.y() << " z: " << pos.point.z() << endl;
            }

            uint32_t size;
            file.read((char *)&size, sizeof(uint32_t));
            cerr << "Total num of sets with scanner position is " << (size) << endl;
            for (uint32_t i = 0; i < size; i++)
            {
                unsigned int psize;
                file.read((char *)&psize, sizeof(psize));
                printf("Size of set %d is %d\n", i, psize);
                // read position ID data
                size_t numP = psize;
                pointSet[i].IDs = new uint32_t[psize];
                file.read((char *)(pointSet[i].IDs), (sizeof(uint32_t) * psize));
            }
        }
        files.push_back(fi);
        cerr << "closing the file" << endl;
        file.close();
        s_pointCloudInteractor->updatePoints(&files);
        return;
    }
}

void PointCloudPlugin::calcMinMax(PointSet& pointSet)
{
    if (pointSet.size >0)
    {
        pointSet.xmax = pointSet.xmin = pointSet.points[0].x;
        pointSet.ymax = pointSet.ymin = pointSet.points[0].y;
        pointSet.zmax = pointSet.zmin = pointSet.points[0].z;

        if (pointSet.size >1)
        {
            for (int k=1; k<pointSet.size; k++)
            {
                if(pointSet.points[k].x<pointSet.xmin)
                    pointSet.xmin= pointSet.points[k].x;
                else if (pointSet.points[k].x>pointSet.xmax)
                    pointSet.xmax= pointSet.points[k].x;

                if(pointSet.points[k].y<pointSet.ymin)
                    pointSet.ymin= pointSet.points[k].y;
                else if (pointSet.points[k].y>pointSet.ymax)
                    pointSet.ymax= pointSet.points[k].y;

                if(pointSet.points[k].z<pointSet.zmin)
                    pointSet.zmin= pointSet.points[k].z;
                else if (pointSet.points[k].z> pointSet.zmax)
                    pointSet.zmax= pointSet.points[k].z;
            }
        }
    }
}

int PointCloudPlugin::unloadFile(std::string filename)
{
    for (std::vector<FileInfo>::iterator fit = files.begin(); fit != files.end(); fit++)
    {
        if (fit->filename == filename)
        {
            for (std::vector<NodeInfo>::iterator nit = fit->nodes.begin(); nit != fit->nodes.end(); nit++)
            {
                if (nit->node->getNumParents() > 0)
                    nit->node->getParent(0)->removeChild(nit->node);
            }
            fit->nodes.clear();
            // remove the poinset data
            if (fit->pointSet)
            {
                for (int i = 0; i < fit->pointSetSize; i++)
                {
                    delete[] fit -> pointSet[i].colors;
                    delete[] fit -> pointSet[i].points;
                }
                delete[] fit -> pointSet;
            }
            pointSet = NULL;
            files.erase(fit);
            return 0;
        }
    }
    files.clear(); // FIXME: really?
    return -1;
}

int PointCloudPlugin::unloadPTS(const char *filename, const char *)
{
    std::string fn = filename;
    return plugin->unloadFile(fn);
}
//remove currently loaded data and free up any memory that has been allocated
void PointCloudPlugin::clearData()
{
    for (std::vector<FileInfo>::iterator fit = files.begin(); fit != files.end(); fit++)
    {
        for (std::vector<NodeInfo>::iterator nit = fit->nodes.begin(); nit != fit->nodes.end(); nit++)
        {
            if (nit->node->getNumParents() > 0)
                nit->node->getParent(0)->removeChild(nit->node);
        }
        fit->nodes.clear();
        // remove the poinset data
        if (fit->pointSet)
        {
            for (int i = 0; i < fit->pointSetSize; i++)
            {
                delete[] fit -> pointSet[i].colors;
                delete[] fit -> pointSet[i].points;
            }
            delete[] fit -> pointSet;
        }
        pointSet = NULL;
    }
    files.clear();
}

//used to handle new menu items in pointset lists
void PointCloudPlugin::selectedMenuButton(ui::Element *menuItem)
{
    string filename;

    // check structures vector for pointer (if found exit)
    vector<ImageFileEntry>::iterator itEntry = pointVec.begin();
    for (; itEntry < pointVec.end(); itEntry++)
    {
        if (itEntry->fileMenuItem == menuItem)
        {
            // call the load method passing in the file name
            filename = itEntry->fileName;
            createGeodes(planetTrans, filename);

            return; //exit
        }
    }
}

/// Called before each frame
void PointCloudPlugin::preFrame()
{
    //resize the speheres of selected and preview points
    s_pointCloudInteractor->resize();

    if (!adaptLOD)
        return;

    //translate viewer position into object space
    //vecBase = (cover->getViewerMat() * Matrix::inverse(CUI::computeLocal2Root(cover->getObjectsRoot()))).getTrans();
    //Matrix ObjectToRoot = CUI::computeLocal2Root(planetTrans);

    // using the center of the cave to determine distance (found world space in object space)
    //vecBase = Matrix::inverse(ObjectToRoot).getTrans();
    vecBase = (cover->getViewerMat() /* * cover->getInvBaseMat()*/).getTrans();

    // level of detail
    float levelOfDetail = 0.4;

    for (std::vector<FileInfo>::iterator fit = files.begin(); fit != files.end(); fit++)
    {
        //TODO calc distance correctly
        for (std::vector<NodeInfo>::iterator nit = fit->nodes.begin(); nit != fit->nodes.end(); nit++)
        {
            osg::Matrix tr;
            tr.makeIdentity();
            osg::Group *parent = nullptr;
            if (nit->node->getNumParents() > 0)
                parent = nit->node->getParent(0);
            while (parent != NULL)
            {
                if (dynamic_cast<osg::MatrixTransform *>(parent))
                {
                    osg::Matrix transformMat = (dynamic_cast<osg::MatrixTransform *>(parent))->getMatrix();
                    tr.postMult(transformMat);
                }
                if (parent->getNumParents())
                    parent = parent->getParent(0);
                else
                    parent = NULL;
            }
            osg::Vec3 nodeCenter = nit->node->getBound().center();
            osg::Vec3 nodeCenterWorld = tr.preMult(nodeCenter);

            double distance = (vecBase - nodeCenterWorld).length2();

            //need to determine values to assign to the distances
            /* if(distance >= 3000000000)
		levelOfDetail = 0.5;
	  else if(distance >= 2400000000 && distance < 3000000000)
		levelOfDetail = 0.6;
	  else if(distance >= 1800000000 && distance < 2400000000)
		levelOfDetail = 0.8;
	  else if(distance >= 1200000000 && distance < 1800000000)
		levelOfDetail = 0.9;
	  else if(distance >= 600000000 && distance < 1200000000)
		levelOfDetail = 0.95; 
	  else if(distance < 600000000)
		levelOfDetail = 1.0;*/
            //10000m = 0.01
            //100m =1.0
            if (distance < 100000000)
                distance = 100000001;
            levelOfDetail = 2000000000.0 / ((distance - 100000000));
            // fprintf(stderr,"%f, %f\n",levelOfDetail, distance);
            if (levelOfDetail > 1.0)
                levelOfDetail = 1.0;
            if (levelOfDetail < 0.01)
                levelOfDetail = 0.01;

            if (adaptLOD)
            {
                ((PointCloudGeometry *)nit->node->getDrawable(0))->changeLod(levelOfDetail * lodScale);
            }
        }
    }
}

/// Called after each frame
void PointCloudPlugin::postFrame()
{
}

void PointCloudPlugin::message(int toWhom, int type, int len, const void *buf)
{
    if (type == PluginMessageTypes::PointCloudSurfaceMsg)
    {

    }
    if (type == PluginMessageTypes::PointCloudSelectionSetMsg)
    {
        int *selectionSet = (int *)buf;
        s_pointCloudInteractor->setSelectionSetIndex(*selectionSet);
    }
    if (type == PluginMessageTypes::PointCloudSelectionIsBoundaryMsg)
    {
        bool *selectionIsBoundary = (bool *)buf;
        s_pointCloudInteractor->setSelectionIsBoundary(*selectionIsBoundary);
    }
}
