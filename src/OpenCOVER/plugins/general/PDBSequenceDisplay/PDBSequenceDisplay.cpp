/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   PDB Sequence Display - Displays Sequence for Protein
//
// Author:        Sendhil Panchadsaram (sendhilp@gmail.com)
//
// Creation Date: April 24th 2006
//
// **************************************************************************
#include <iostream>
#include <ostream>
#include <sstream>
#include <util/unixcompat.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coFrame.h>
#include <config/CoviseConfig.h>
#include <string.h>

//#include <vtk/vtkStructuredGridReader.h>

#include "PDBSequenceDisplay.h"

#include <PluginUtil/PluginMessageTypes.h>

using std::cerr;
using std::endl;
using std::string;
using std::stringstream;
using covise::coCoviseConfig;

PDBSequenceDisplay *plugin = NULL;

static const int NUM_BUTTONS = 8;

/// Constructor
PDBSequenceDisplay::PDBSequenceDisplay()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool PDBSequenceDisplay::init()
{

    if (plugin)
        return false;

    plugin = this;
    string stringChar[NUM_BUTTONS] = { "Arg", "Val", "Leu", "His", "Glu", "Cys", "Ile", "Gly" };
    //	string stringChar[NUM_BUTTONS];
    int x[8] = { 0, 20, 40, 60, 80, 100, 120, 140 };
    int y[7] = { 100, 85, 70, 55, 40, 25, 10 };
    string nums[] = { "  0", "  1", "  2", "  3", "  4", "  5", "  6", "  7", "  8", "  9" };
    float z = 1;
    //int cStatus=0;
    CChain cTemp;
    CSequence csTemp;
    string strChain;
    //string imageLoc;
    string strTitle;
    //string fileLoc = coCoviseConfig::getEntry("COVER.Plugin.PDBSequenceDisplay.Location");
    string strProteinName = coCoviseConfig::getEntry("COVER.Plugin.PDBSequenceDisplay.ProteinName");
    CacheDirectory = coCoviseConfig::getEntry("COVER.Plugin.PDBSequenceDisplay.CacheDirectory");

    //cStatus = 5;
    curChainNum = 0;

    //Set starting positions for the chain
    curChainPos = 0;
    curChainEndPos = 10;

    //cStatus= myProtein.RetrievePositions(overallChain,fileLoc);
    cTemp = overallChain.at(curChainNum);
    curChain = cTemp.name;
    //cStatus = myProtein.RetrieveSubset(oneChain,overallChain,curChain,curChainPos,10);

    for (int i = 0; i < NUM_BUTTONS; i++)
    {
        csTemp = cTemp.chainsequence.at(i);
        stringChar[i] = csTemp.aminoacid;
    }

    //attach a switch node to the root
    mainNode = new osg::Switch();
    cover->getObjectsRoot()->addChild((osg::Node *)mainNode.get());

    //create panel
    panel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    panel->setScale(1);
    panel->resize();
    handle = new coPopupHandle("pdbSequenceDisplay");

    labelChar = new coLabel *[NUM_BUTTONS];
    labelNum = new coLabel *[NUM_BUTTONS];

    //Add Title
    strTitle = "Sequence for:" + strProteinName;
    labelSequence = new coLabel();
    labelSequence->setString(strTitle);
    labelSequence->setPos((x[3]) / 2, y[0], z);
    labelSequence->setFontSize(8);
    panel->addElement(labelSequence);

    //Add Amino Acid List
    for (int i = 0; i < NUM_BUTTONS; i++)
    {
        labelNum[i] = new coLabel();
        labelNum[i]->setString(nums[i + 1]);
        labelNum[i]->setPos(x[i], y[4], z);
        labelNum[i]->setFontSize(8);
        panel->addElement(labelNum[i]);
    }

    //Add number on the chain
    for (int i = 0; i < NUM_BUTTONS; i++)
    {
        labelChar[i] = new coLabel();
        labelChar[i]->setString(stringChar[i]);
        labelChar[i]->setPos(x[i], y[5], z);
        labelChar[i]->setFontSize(8);
        panel->addElement(labelChar[i]);
    }

    //Changes the chain

    upButton = new coPushButton(new coRectButtonGeometry(5, 5, "PDBSequenceDisplay/uparrow"), this);
    upButton->setPos(130, 60, z);
    upButton->setSize(8);
    panel->addElement(upButton);

    downButton = new coPushButton(new coRectButtonGeometry(5, 5, "PDBSequenceDisplay/downarrow"), this);
    downButton->setPos(130, 40, z);
    downButton->setSize(8);
    panel->addElement(downButton);

    showMarkerButton = new coPushButton(new coRectButtonGeometry(5, 5, "PDBSequenceDisplay/showmarker"), this);
    showMarkerButton->setPos(180, 30, z);
    showMarkerButton->setSize(6);
    panel->addElement(showMarkerButton);

    strChain = "Chain :" + cTemp.name;
    labelTitle = new coLabel();
    labelTitle->setString(strChain);
    labelTitle->setPos(120, 90, z);
    labelTitle->setFontSize(8);
    panel->addElement(labelTitle);

    labelChange = new coLabel();
    labelChange->setString("Change Chain");
    labelChange->setPos(175, y[0], z);
    labelChange->setFontSize(8);
    panel->addElement(labelChange);

    //Add buttons (left and right)
    leftButton = new coPushButton(new coRectButtonGeometry(10, 5, "PDBSequenceDisplay/leftarrow"), this);
    leftButton->setPos(90, 110, z);
    leftButton->setSize(5);
    panel->addElement(leftButton);

    rightButton = new coPushButton(new coRectButtonGeometry(10, 5, "PDBSequenceDisplay/rightarrow"), this);
    rightButton->setPos(180, 110, z);
    rightButton->setSize(5);
    panel->addElement(rightButton);

    //Add first and last buttons
    firstButton = new coPushButton(new coRectButtonGeometry(10, 5, "PDBSequenceDisplay/firstelement"), this);
    firstButton->setPos(0, 110, z);
    firstButton->setSize(5);
    panel->addElement(firstButton);

    lastButton = new coPushButton(new coRectButtonGeometry(10, 5, "PDBSequenceDisplay/lastelement"), this);
    lastButton->setPos(270, 110, z);
    lastButton->setSize(5);
    panel->addElement(lastButton);

    labelChange = new coLabel();
    labelChange->setString("Move up/down chain");
    labelChange->setPos(30, y[2], z);
    labelChange->setFontSize(8);
    panel->addElement(labelChange);

    //This sets the panel up inside the popup frame
    frame = new coFrame("UI/Frame"); //Used to be UI/Frame
    panel->setScale(5);
    panel->resize();
    frame->addElement(panel);
    handle->addElement(frame);

    //Setup Menus
    selectMenu = new coSubMenuItem("PDBSequenceDisplay");
    selectRow = new coRowMenu("PDBSequenceDisplay");
    sequenceDisplayButton = new coCheckboxMenuItem("Display Sequence", false);

    selectRow->add(sequenceDisplayButton);
    selectMenu->setMenu(selectRow);

    //Listeners
    sequenceDisplayButton->setMenuListener(this);
    selectMenu->setMenuListener(this);

    // Create main menu button
    PDBSequenceDisplayMenuItem = new coButtonMenuItem("PDB Sequence Display");
    PDBSequenceDisplayMenuItem->setMenuListener(this);
    // add button to main menu (need to adjust)
    cover->getMenu()->add(selectMenu);

    return true;
}

/// Destructor
PDBSequenceDisplay::~PDBSequenceDisplay()
{
    delete panel;
    delete handle;
    delete PDBSequenceDisplayMenuItem;
}

void PDBSequenceDisplay::ChangeChain(int increment)
{
    CChain cTemp;
    int overallSize;
    string strTemp;

    overallSize = overallChain.size();

    if ((curChainNum + increment) < 0)
    {
        curChainNum = 0;
    }
    else if ((curChainNum + increment) == overallSize)
    {
        curChainNum = overallSize - 1;
    }
    else
    {
        curChainNum += increment;
    }

    cTemp = overallChain.at(curChainNum);
    strTemp = "Chain :" + cTemp.name;
    labelTitle->setString(strTemp);
    GoFirst();
}

void PDBSequenceDisplay::ChangeLabel(int increment)
{
    CChain cTemp;
    CSequence csTemp;
    string numTemp;
    string fileLoc = coCoviseConfig::getEntry("COVER.Plugin.PDBSequenceDisplay.Location");
    stringstream out;
    //	int cStatus=0;

    cTemp = overallChain.at(curChainNum);

    if ((curChainPos + increment) <= 0)
    {
        curChainPos = increment = 0;
        curChainEndPos = NUM_BUTTONS;
    }
    else if ((curChainEndPos + increment) >= (int(cTemp.chainsequence.size()) + NUM_BUTTONS))
    {
        curChainPos = cTemp.chainsequence.size() - 1;
        curChainEndPos = curChainPos + NUM_BUTTONS;
    }
    else
    {
        curChainPos += increment;
        curChainEndPos += increment;
    }

    //	cStatus = myProtein.RetrieveSubset(oneChain,overallChain,curChain,curChainPos,curChainEndPos);

    cerr << endl << endl;
    for (int i = 0; i < NUM_BUTTONS; i++)
    {
        if ((i + curChainPos) < int(cTemp.chainsequence.size()))
        {
            csTemp = cTemp.chainsequence.at(i + curChainPos);
            cerr << "Adding " << csTemp.aminoacid << " Num:" << csTemp.num << endl;
            out.str("");
            out << (csTemp.num);
            numTemp = out.str();
            labelChar[i]->setString(csTemp.aminoacid);
            labelNum[i]->setString(numTemp);
        }
        else
        {
            cerr << "Adding a blank"
                 << " Num:" << csTemp.num << endl;
            out.str("");
            out << i + curChainPos;
            numTemp = out.str();
            labelChar[i]->setString("");
            labelNum[i]->setString(numTemp);
        }
    }
}

void PDBSequenceDisplay::GoFirst()
{
    CChain cTemp;
    CSequence csTemp;
    string numTemp;
    string fileLoc = coCoviseConfig::getEntry("COVER.Plugin.PDBSequenceDisplay.Location");
    stringstream out;
    //	int cStatus=0;

    cTemp = overallChain.at(curChainNum);

    curChainPos = 0;
    curChainEndPos = 10;

    //cStatus = myProtein.RetrieveSubset(oneChain,overallChain,curChain,curChainPos,curChainEndPos);

    cerr << endl << endl;
    for (int i = 0; i < NUM_BUTTONS; i++)
    {
        csTemp = cTemp.chainsequence.at(i + curChainPos);
        cerr << "Adding " << csTemp.aminoacid << " Num:" << csTemp.num << endl;
        out.str("");
        out << csTemp.num;
        numTemp = out.str();
        labelChar[i]->setString(csTemp.aminoacid);
        labelNum[i]->setString(numTemp);
    }
}

void PDBSequenceDisplay::GoLast()
{
    CChain cTemp;
    CSequence csTemp;
    string numTemp;
    string fileLoc = coCoviseConfig::getEntry("COVER.Plugin.PDBSequenceDisplay.Location");
    stringstream out;
    //	int cStatus=0;
    cTemp = overallChain.at(curChainNum);

    curChainPos = cTemp.chainsequence.size() - NUM_BUTTONS;
    curChainEndPos = cTemp.chainsequence.size();

    //	cStatus = myProtein.RetrieveSubset(oneChain,overallChain,curChain,curChainPos,curChainEndPos);

    cerr << endl << endl;
    for (int i = 0; i < NUM_BUTTONS; i++)
    {
        csTemp = cTemp.chainsequence.at(i + curChainPos);
        cerr << "Adding " << csTemp.aminoacid << " Num:" << csTemp.num << endl;
        out.str("");
        out << csTemp.num;
        numTemp = out.str();
        labelChar[i]->setString(csTemp.aminoacid);
        labelNum[i]->setString(numTemp);
    }
}
void PDBSequenceDisplay::menuEvent(coMenuItem *menuItem)
{
    // listen for initPDB frame to open close

    if (menuItem == sequenceDisplayButton)
    {
        handle->setVisible(sequenceDisplayButton->getState());
    }
    /*  if(menuItem == PDBSequenceDisplayMenuItem) 
  {
    if(handle->isVisible())
    {
      handle->setVisible(false);
    }
    else
    {
      handle->setVisible(true);
    }
  }
  handle->update();
*/
}

// need to define because abstract
void PDBSequenceDisplay::potiValueChanged(float, float, coValuePoti *, int) {}

// load new structure listener
void PDBSequenceDisplay::buttonEvent(coButton *cobutton)
{
    if (cobutton == leftButton)
    {
        ChangeLabel(-1);
    }
    else if (cobutton == rightButton)
    {
        ChangeLabel(1);
    }
    else if (cobutton == showMarkerButton)
    {
        SendCoordinates();
    }
    else if (cobutton == firstButton)
    {
        GoFirst();
    }
    else if (cobutton == lastButton)
    {
        GoLast();
    }
    else if (cobutton == upButton)
    {
        ChangeChain(-1);
    }
    else if (cobutton == downButton)
    {
        ChangeChain(1);
    }
}

void PDBSequenceDisplay::ChangeProtein(string filename)
{
    // check if the protein has already been loaded into memory
    if (filename == currentProtein)
        return;

    string cmdInput;
    const char *current_path = getcwd(NULL, 0);
    string currentPath = current_path;
    int status;
    string FixedFileName;
    currentProtein = filename;

    // change back to Cache Directory
    if (chdir(CacheDirectory.c_str()) == -1)
        cerr << "failed to change to " << CacheDirectory << endl;
    //chdir(currentPath.c_str());
    //Check to see if the file is already in the cache

    //File does not exist, download it to the cache
    cerr << "Current directory is " << CacheDirectory << endl;
    cerr << "File name is : " << filename << endl;

    FixedFileName = CacheDirectory + "/" + filename + PDB_EXT;

    cerr << "Now the filename is : " << FixedFileName << endl;

    /*	if (FileExists(FixedFileName.c_str()))
		cerr << "The file exists\n";
	else
		cerr << "The file does not exist\n";
*/

    //	if (FileExists(FixedFileName.c_str()))
    //	{
    std::string PDBURL = coCoviseConfig::getEntry("COVER.Plugin.PDB.PDBUrl");
    cmdInput.erase().append(WGET).append(PDBURL).append(filename).append(PDB_EXT);
    status = system(cmdInput.c_str());
    //	}

    if (coVRMSController::instance()->isMaster())
    {
        /* convert pdb to vrml */
        status = system(cmdInput.c_str());

        coVRMSController::instance()->sendSlaves((char *)&status, sizeof(int));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&status, sizeof(int));
    }

    if (status)
    {
        cerr << "Couldn't load file\n";
    }
    else
    {
        //filename = currentProtein + PDB_EXT;
        filename = FixedFileName;
        myProtein.RetrievePositions(overallChain, filename);
        GoFirst();
        labelSequence->setString("Sequence for " + currentProtein);
    }
}

void PDBSequenceDisplay::message(int toWhom, int type, int, const void *buf)
{
    if (type == PluginMessageTypes::PDBMoveMark)
    {

        MessageData *mm = (MessageData *)buf;
        cerr << "The FILENAME:" << mm->filename << endl;
        if (mm->on)
        {

            string filename;
            filename.assign(mm->filename, 4);
            if (filename != currentProtein)
            {
                //cerr << "Numbers recieved are : " << "x:" << mm->x << " y:" << mm->y << " z:" << mm->z << endl;
                cerr << "The filename:" << mm->filename << endl;
                handle->setVisible(true);
                ChangeProtein(filename);
            }
            else
            {
                //Already loaded the file up, so change position
                string smallestChain;
                size_t smallestChainPos;
                myProtein.ClosestAminoAcid(overallChain, mm->x, mm->y, mm->z, smallestChain, smallestChainPos);
                cerr << "Closest Chain is : " << smallestChain << " Closest Postion is:" << smallestChainPos << endl;
                GotoChainAndPos(smallestChain, smallestChainPos);
            }
        }
        else
        {
            cerr << "It's off\n";
        }
    }
}

/// Called before each frame
void PDBSequenceDisplay::preFrame()
{
    handle->update();
}

void PDBSequenceDisplay::SendCoordinates()
{
    MessageData myMessage;
    CSequence csTemp;
    CChain cTemp;

    cTemp = overallChain.at(curChainNum);

    csTemp = cTemp.chainsequence.at(curChainPos);
    myMessage.x = csTemp.x;
    myMessage.y = csTemp.y;
    myMessage.z = csTemp.z;
    myMessage.on = true;

    strcpy(myMessage.filename, currentProtein.c_str());

    cerr << "I'm sending coordinates\n";
    cover->sendMessage(this, "PDB", PluginMessageTypes::PDBMoveMark,
                       sizeof(MessageData), &myMessage);
}

void PDBSequenceDisplay::GotoChainAndPos(string chain, int pos)
{
    CChain cTemp;
    CSequence csTemp;
    stringstream out;
    string numTemp;

    curChainNum = myProtein.ReturnChainNumber(overallChain, chain);

    cTemp = overallChain.at(curChainNum);

    curChainPos = pos;
    curChainEndPos = pos + 10;

    if (curChainPos > (int(cTemp.chainsequence.size()) - NUM_BUTTONS))
        curChainPos = cTemp.chainsequence.size() - NUM_BUTTONS;

    cerr << endl << endl;

    labelTitle->setString("Chain : " + chain);

    for (int i = 0; i < NUM_BUTTONS; i++)
    {
        csTemp = cTemp.chainsequence.at(i + curChainPos);
        out.str("");
        out << csTemp.num;
        numTemp = out.str();
        labelChar[i]->setString(csTemp.aminoacid);
        labelNum[i]->setString(numTemp);
    }
}

int PDBSequenceDisplay::FileExists(const char *filename)
{
    struct stat buffer;

    if (stat(filename, &buffer))
        return 0;
    return 1;
}

COVERPLUGIN(PDBSequenceDisplay)
