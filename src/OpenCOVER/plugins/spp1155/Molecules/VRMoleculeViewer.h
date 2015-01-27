/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//**************************************************************************
//
//			.h File
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
// * History : started 6.7.2001
//
// **************************************************************************

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

class Frame;
class MoleculeStructure;

class VRMoleculeViewer : public coVRPlugin
{

private:
    DLinkList<Frame *> framelist;

    // Performer Objects
    // here is my main DCS
    osg::ref_ptr<osg::MatrixTransform> viewerDCS;
    osg::ref_ptr<osg::Group> scene; // the spitting image
    osg::ref_ptr<osg::Group> m_rootNode; // add to covise to use the menus
    osg::Vec3 *center; // middle of the Molecule
    osg::ref_ptr<osg::MatrixTransform> *DCSList;

    // data variables
    float *radius; // this one *2Pi gives the circumference
    float sceneSize; // the size of the scene from cover
    float sliderValue;
    float sphereRatio;
    double time;
    float animationSpeed;
    double timeBetweenFrames;
    int uniqueMenu;
    int maxNumberOfMolecules;
    int numberOfTimesteps;
    int moving; // indicates if animation of timesteps is on
    int frameIndex; // numer of current timestep
    MoleculeStructure *structure;

    // class functions
    void addMenuEntry(); // Add menu(s) to COVER-pinboard
    void removeMenuEntry(); // And remove it...

    int loadData(const char *moleculepath, osg::Group *parent); // get the Molecule infos from file
    void show(); // put the molecules in shape
    void move();
    void NoMove();
    void stepping(); // step on timestep forward each frame
    void slide(float actual);
    void stepForward();
    void stepBackward();

    static void Init(void *);
    void clearUp();
    void readDirectory(const char *parent);

    std::string m_strFeedbackInfo;
    bool m_bNewObject;
    bool m_bDoOnceViewAll;

    std::list<std::string> vsFiles;
    // hack for cyberclassroom
    std::list<std::string>::iterator iterFiles;

public:
    // Constructor
    VRMoleculeViewer();

    // Destructor
    ~VRMoleculeViewer();
    bool init();

    // Callback function for mouse clicks
    static void menuCallback(void *, buttonSpecCell *);
    static void fileSelection(void *, buttonSpecCell *);
    static void sliderCallback(void *c, buttonSpecCell *spec);
    static void speederCallback(void *c, buttonSpecCell *spec);
    static void sphereCallback(void *c, buttonSpecCell *spec);
    static void updateCallback(void *, buttonSpecCell *);
    void setTimestep(int t);

    void preFrame(); // Update function , called each frame

    std::string dirName;
    std::string filename;

    static int loadFile(const char *name, osg::Group *parent, const char *ck = "");
    static int unloadFile(const char *name, const char *ck = "");

    virtual void guiToRenderMsg(const char *msg);
};
