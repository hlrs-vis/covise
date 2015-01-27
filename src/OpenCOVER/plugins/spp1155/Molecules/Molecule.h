/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
//			.h File
//
// * Description    : Molecule plugin module for the Cover Covise Renderer
//                    Reads Molecule Structures based on the Jorgensen Model
//                    The data is provided from the Itt / University Stuttgart
//
// * Class(es)      :
//
// * inherited from :
//
// * Author  : Thilo Krueger
//
// * History : started 28.5.2001
//
// **************************************************************************

#ifndef _MOLECULE_H
#define _MOLECULE_H
#define LINESIZE 128
#define ATOMS 32 // maximum nuber of sites in all molecules
#define MAGNIFICATION 10 // magnification factor for molecules
// their original size may be too small :-/
// see class Molecule constructor

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

namespace osg
{
class Material;
class ShapeDrawable;
};

class MoleculeStructure
{
private:
    typedef std::vector<osg::ref_ptr<osg::Geode> > GeodeList;
    GeodeList geodeList;

    int molIndex[ATOMS]; //index number of molecule
    float x[ATOMS];
    float y[ATOMS];
    float z[ATOMS];
    float sigma[ATOMS]; //size
    int color[ATOMS];
    float cubeSize; //size of cube of molecules
    int numberOfMolecules;
    int numberOfSites;
    float sphereRatio;

    osg::ref_ptr<osg::StateSet> geoState;

    FILE *fp;

    // member functions
    void readFile();
    void setColor(osg::ShapeDrawable *draw, int colorIndex);
    osg::Geode *createMoleculeGeode(int molnr);

public:
    //constructor
    MoleculeStructure(FILE *fp, float fsphereRatio = 1.0f);

    //destructor
    ~MoleculeStructure();

    osg::Geode *getMoleculeGeode(int molIndex);
    float getBoxSize();
    void setSphereRatio(float fratio);
};

class Frame
{

private:
    int numberOfMolecules;
    float *xCoord, *yCoord, *zCoord;
    float *q1Coord, *q2Coord, *q3Coord, *q0Coord;
    float unit;
    osg::ref_ptr<osg::Geode> *geodeList;
    osg::ref_ptr<osg::MatrixTransform> *pfDCSList;
    osg::ref_ptr<osg::Group> frameGroup;
    int index;

    osg::ref_ptr<osg::MatrixTransform> viewerDCS;

public:
    Frame(int molecules, osg::MatrixTransform *MainNode, float cubeSize,
          osg::ref_ptr<osg::MatrixTransform> *DCSPool);
    ~Frame();
    int addMolecule(osg::Geode *molGeode, float x, float y, float z, float q0, float q1, float q2, float q3);

    void display();
    void hide();
};
#endif
