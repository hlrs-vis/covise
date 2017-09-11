/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
//			Source File
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
// * History : started 6.3.2001
//
// **************************************************************************

#include <iostream>
#include <ostream>
#include <cover/coVRPluginSupport.h>
#include "Molecule.h"
#include "VRMoleculeViewer.h"
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Shape>
#include <osg/Geode>
#include <osg/CullFace>

using std::cerr;
using std::endl;
using namespace osg;

extern VRMoleculeViewer Module;

MoleculeStructure::MoleculeStructure(FILE *datafile, float fSphereRatio)
{

    int state;

    fp = datafile;
    readFile();

    //reset file pointer to beginning of file
    //important to read the rest of the file correctly
    state = fseek(fp, 0, SEEK_SET);
    if (state != 0)
        fprintf(stderr, "File read error!");

    ref_ptr<Material> material = new Material();
    // create GeoState
    // one GeoState for all
    material->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    material->setAmbient(Material::FRONT, Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    material->setDiffuse(Material::FRONT, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    material->setSpecular(Material::FRONT, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    material->setEmission(Material::FRONT, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(Material::FRONT, 16.0f);
    //material->setAlpha(0.8);

    geoState = new osg::StateSet();
    geoState->ref();
    geoState->setGlobalDefaults();
    geoState->setAttributeAndModes(material.get(), StateAttribute::ON);
    geoState->setMode(GL_LIGHTING, StateAttribute::ON);
    CullFace *cullFace = new CullFace();
    cullFace->setMode(CullFace::BACK);
    geoState->setAttributeAndModes(cullFace, StateAttribute::ON);
    //geoState->setMode(PFSTATE_TRANSPARENCY, PFTR_OFF);

    sphereRatio = fSphereRatio;

    for (int i = 1; i <= numberOfMolecules; i++)
    {
        ref_ptr<Geode> g = createMoleculeGeode(i);
        g->ref();
        geodeList.push_back(g);
    }
}

void MoleculeStructure::setSphereRatio(float fratio)
{
    sphereRatio = fratio;
}

MoleculeStructure::~MoleculeStructure()
{
    for (GeodeList::iterator it = geodeList.begin();
         it != geodeList.end();
         it++)
    {
        (*it)->unref();
    }

    geodeList.clear();

    geoState->unref();
}

osg::Geode *MoleculeStructure::getMoleculeGeode(int molIndex)
{
    if (molIndex > (numberOfMolecules))
        return NULL;

    // XXX
    return geodeList.at(molIndex - 1).get();
}

void MoleculeStructure::readFile()
{
    char line[LINESIZE];
    char *first;
    int numScanned;

    int siteCounter = 0;
    int i = 0;

    if (cover->debugLevel(3))
        printf("reading molecule descriptions...\n");
    while (fgets(line, LINESIZE, fp) != NULL)
    {
        first = line;

        // skip blank lines
        while (*first != '\0' && isspace(*first))
            first++;

        if (*first == '\0')
            //read the next line
            continue;

        if (*first == '~')
        {
            i++;
            first++;

            numScanned = sscanf(first, " %d  LJ %f %f %f %f %d",
                                &molIndex[i], &x[i], &y[i], &z[i], &sigma[i], &color[i]);
            if (cover->debugLevel(3))
            {
                printf("scanned #%d:\n", i);
                printf("%d %f %f %f %f %d\n", molIndex[i], x[i], y[i], z[i], sigma[i], color[i]);
            }

            //ignore badly formed lines
            if (numScanned != 6)
            {

                i--;
                continue;
            }

            siteCounter++;
            if (siteCounter == ATOMS)
            {
                printf("too many sites (max. %d)\n", ATOMS);
                break;
            }
        }

        if (*first == '#')
        {
            first++;

            if (sscanf(first, " %f", &cubeSize) != 1)
            {
                cerr << "MoleculeStructure::readFile: sscanf failed" << endl;
            }

            //we have all information, so we break the while-loop here
            break;
        }
    }

    numberOfMolecules = molIndex[i]; //not absolutely correct, will change later...
    numberOfSites = i;

    if (cover->debugLevel(3))
        printf("done reading %d molecules\n", numberOfMolecules);

    return;
}

osg::Geode *MoleculeStructure::createMoleculeGeode(int molnr)
// returns handle to Geode for particular molecule
{
    // Geode creation loop
    // one Geode per molecule
    osg::Geode *molecule = new osg::Geode;

    for (int j = 1; j <= numberOfSites; j++)
    {
        if (molIndex[j] == molnr)
        {
            ref_ptr<Sphere> sphere = new Sphere(Vec3(x[j], y[j], z[j]), sigma[j] / 2.0);
            TessellationHints *hints = new TessellationHints();
            /*       should work, not yet implemented in osg
         hints->setTessellationMode(TessellationHints::USE_TARGET_NUM_FACES);
         hints->setTargetNumFaces(10);
*/
            hints->setDetailRatio(sphereRatio);
            ref_ptr<ShapeDrawable> drawable = new ShapeDrawable(sphere.get(), hints);
            setColor(drawable.get(), color[j]);
            drawable->setStateSet(geoState.get());

            molecule->addDrawable(drawable.get());
        }
    }

    return molecule;
}

void MoleculeStructure::setColor(osg::ShapeDrawable *draw, int colorIndex)
// implements color definition by ITT
{
    osg::Vec4 *color;

    colorIndex = colorIndex % 14;
    switch (colorIndex)
    {
    //grau
    case 0:
        color = new osg::Vec4(0.7, 0.7, 0.7, 1.0);
        break;
    //schwarz
    case 1:
        color = new osg::Vec4(0.1, 0.1, 0.1, 1.0);
        break;
    //rot
    case 2:
        color = new osg::Vec4(1.0, 0.0, 0.0, 1.0);
        break;
    //orange
    case 3:
        color = new osg::Vec4(1.0, 0.5, 0.0, 1.0);
        break;
    //gelb
    case 4:
        color = new osg::Vec4(1.0, 1.0, 0.0, 1.0);
        break;
    //zitron
    case 5:
        color = new osg::Vec4(0.7, 1.0, 0.0, 1.0);
        break;
    //gruen
    case 6:
        color = new osg::Vec4(0.0, 1.0, 0.0, 1.0);
        break;
    //tuerkis
    case 7:
        color = new osg::Vec4(0.0, 1.0, 0.5, 1.0);
        break;
    //cyan
    case 8:
        color = new osg::Vec4(0.0, 1.0, 1.0, 1.0);
        break;
    //wasser
    case 9:
        color = new osg::Vec4(0.0, 0.5, 1.0, 1.0);
        break;
    //blau
    case 10:
        color = new osg::Vec4(0.0, 0.0, 1.0, 1.0);
        break;
    //lila
    case 11:
        color = new osg::Vec4(0.5, 0.0, 1.0, 1.0);
        break;
    //mangenta
    case 12:
        color = new osg::Vec4(1.0, 0.0, 1.0, 1.0);
        break;
    //kirsch
    case 13:
        color = new osg::Vec4(1.0, 0.0, 0.5, 1.0);
        break;
    //blau
    default:
        color = new osg::Vec4(0.0, 0.0, 1.0, 1.0);
        break;
    }

    draw->setColor(*color);

    delete color;

    return;
}

float MoleculeStructure::getBoxSize()
{

    return cubeSize;
}

//*********************************************************************

Frame::Frame(int molecules, MatrixTransform *MainNode, float cubeSize, ref_ptr<MatrixTransform> *DCSPool)
{

    numberOfMolecules = molecules;
    viewerDCS = MainNode;
    unit = cubeSize;
    index = 0;

    pfDCSList = DCSPool;
    geodeList = new osg::ref_ptr<osg::Geode>[molecules];

    //frameGroup = NULL;
    //pfDCSList = NULL;

    //pfSCSList = new pfSCS*[numberOfMolecules];
    //frameGroup = new pfGroup;

    xCoord = new float[molecules];
    yCoord = new float[molecules];
    zCoord = new float[molecules];
    q0Coord = new float[molecules];
    q1Coord = new float[molecules];
    q2Coord = new float[molecules];
    q3Coord = new float[molecules];
}

Frame::~Frame()
{

    /*for(i=0;i<index;i++)
       {
           pfDCSList[i]->removeChild(geodeList[i]);
       }
   */
    for (int i = 0; i < index; i++)
    {
        geodeList[i]->unref();
    }
    delete[] geodeList;
    delete[] xCoord;
    delete[] yCoord;
    delete[] zCoord;
    delete[] q0Coord;
    delete[] q1Coord;
    delete[] q2Coord;
    delete[] q3Coord;
}

int Frame::addMolecule(osg::Geode *molGeode, float x, float y, float z, float q0, float q1, float q2, float q3)
{
    //if(index>=numberOfMolecules) return 0;

    xCoord[index] = x / 1000 * unit;
    yCoord[index] = y / 1000 * unit;
    zCoord[index] = z / 1000 * unit;
    q0Coord[index] = q0 / 1000;
    q1Coord[index] = q1 / 1000;
    q2Coord[index] = q2 / 1000;
    q3Coord[index] = q3 / 1000;

    // store the pointer to corresponding Geode for this molecule
    geodeList[index] = molGeode;
    molGeode->ref();

    index++;
    return index;
}

void Frame::display()
{
    for (int i = 0; i < index; i++)
    {
        osg::Quat rotation(q0Coord[i], q1Coord[i], q2Coord[i], q3Coord[i]);
        osg::Matrix transMatrix;
        rotation.get(transMatrix);

        //transMatrix.makeQuat(rotation);

        // we apply the translation directly to the transformation matrix

        transMatrix.setTrans(Vec3(xCoord[i], yCoord[i], zCoord[i]));
#if 0
      transMatrix.mat[3][0] = xCoord[i];
      transMatrix.mat[3][1] = yCoord[i];
      transMatrix.mat[3][2] = zCoord[i];
#endif

        pfDCSList[i]->setMatrix(transMatrix);
        pfDCSList[i]->addChild(geodeList[i].get());

        //frameGroup->addChild(pfDCSList[i]);
    }

    //viewerDCS->addChild(frameGroup);
}

void Frame::hide()
{
    for (int i = 0; i < index; i++)
    {
        if (pfDCSList[i]->containsNode(geodeList[i].get()))
            pfDCSList[i]->removeChild(geodeList[i].get());
    }

    //viewerDCS->removeChild(frameGroup);
}
