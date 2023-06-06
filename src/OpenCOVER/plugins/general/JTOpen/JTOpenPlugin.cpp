/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: JTOpen Plugin (does nothing usefull)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <PluginUtil/coLOD.h>
#include <config/CoviseConfig.h>
#include <cover/coVRShader.h>

#include "JTOpenPlugin.h"
#include "findNodeVisitor.h"

#include <cover/RenderObject.h>
#include <cover/VRRegisterSceneGraph.h>
#include <osg/LOD>

#include <osg/GL>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/TexMat>
#include <osg/TexEnvCombine>
#include <osg/Texture>
#include <osg/TextureCubeMap>
#include <osg/Texture2D>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/BlendFunc>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osg/Fog>
#include <osg/AlphaFunc>
#include <osg/ColorMask>
#include <osgDB/ReadFile>

#include <osgUtil/SmoothingVisitor>

#ifdef HAVE_OSGNV
#include <osgNVExt/RegisterCombiners>
#include <osgNVExt/CombinerInput>
#include <osgNVExt/CombinerOutput>
#include <osgNVExt/FinalCombinerInput>
#endif

#include <osgText/Font>
#include <osgText/Text>

using namespace osg;

#include <sys/types.h>
#include <string.h>
#include <JtTk/JtkCADImporter.h>
#include <JtTk/JtkCADExporter.h>
#include <JtTk/JtkTraverser.h>
#include <JtTk/JtkEntityFactory.h>
#include <JtTk/JtkAttrib.h>
#include <JtTk/JtkStandard.h>
#include <JtTk/JtkBrep.h>
#include <JtTk/JtkEntity.h>


template<class JT>
std::string getJtName(JT *node)
{
#if JTTK_MAJOR_VERSION >= 8
	JtkString name = node->name();
	JtkUTF8* stringUTF8;
	int length;
	name.getString(stringUTF8, length);
    return std::string(stringUTF8, length);
#else
    return std::string(node->name());
#endif
}

//int my_level = 0;
int want_details = 1;

#define indent(i)                   \
    {                               \
        for (int l = 0; l < i; l++) \
            cout << " ";            \
    }

void printXform(JtkTransform *partXform, int level)
{
    float *elements = NULL;

    indent(level);
    cout << "JtkTRANSFORM\n";

    partXform->getTElements(elements);
    if (elements)
    {
        indent(level + 1);
        cout << elements[0] << ", " << elements[1] << ", "
             << elements[2] << ", " << elements[3] << "\n";
        indent(level + 1);
        cout << elements[4] << ", " << elements[5] << ", "
             << elements[6] << ", " << elements[7] << "\n";
        indent(level + 1);
        cout << elements[8] << ", " << elements[9] << ", "
             << elements[10] << ", " << elements[11] << "\n";
        indent(level + 1);
        cout << elements[12] << ", " << elements[13] << ", "
             << elements[14] << ", " << elements[15] << "\n";
        JtkEntityFactory::deleteMemory(elements);
    }
}

void printMaterial(JtkMaterial *partMaterial, int level)
{
    float *ambient = NULL,
          *diffuse = NULL,
          *specular = NULL,
          *emission = NULL,
          shininess = -999.0;

    indent(level);
    cout << "JtkMATERIAL\n";

    partMaterial->getAmbientColor(ambient);
    if (ambient)
    {
        indent(level + 1);
        cout << "ambient = ( " << ambient[0] << ", " << ambient[1] << ", "
             << ambient[2] << ", " << ambient[3] << " )\n";
        JtkEntityFactory::deleteMemory(ambient);
    }

    partMaterial->getDiffuseColor(diffuse);
    if (diffuse)
    {
        indent(level + 1);
        cout << "diffuse = ( " << diffuse[0] << ", " << diffuse[1] << ", "
             << diffuse[2] << ", " << diffuse[3] << " )\n";
        JtkEntityFactory::deleteMemory(diffuse);
    }

    partMaterial->getSpecularColor(specular);
    if (specular)
    {
        indent(level + 1);
        cout << "specular = ( " << specular[0] << ", " << specular[1] << ", "
             << specular[2] << ", " << specular[3] << " )\n";
        JtkEntityFactory::deleteMemory(specular);
    }

    partMaterial->getEmissionColor(emission);
    if (emission)
    {
        indent(level + 1);
        cout << "emission = ( " << emission[0] << ", " << emission[1] << ", "
             << emission[2] << ", " << emission[3] << " )\n";
        JtkEntityFactory::deleteMemory(emission);
    }

    partMaterial->getShininess(shininess);
    if (shininess != -999.0)
    {
        indent(level + 1);
        cout << "shininess = " << shininess << "\n";
    }
}

void printBrep(JtkBrep * /*partBrep*/, int level)
{
    indent(level);
    cout << "JtkBREP\n";
}

void printWrep(JtkWrep * /*partWrep*/, int level)
{
    indent(level);
    cout << "JtkWREP\n";
}

osg::Node *JTOpenPlugin::createShape(JtkShape *partShape, const char *objName)
{
    //cout << "JtkSHAPE\n";

    Geode *geode = new Geode();
    ref_ptr<Geometry> geom = new Geometry();
    StateSet *geoState = geode->getOrCreateStateSet();
    Vec3Array *vert = new Vec3Array;
    Vec3Array *normalArray = new Vec3Array();
    Vec3Array *colorArray = new Vec3Array();
    Vec2Array *tcArray = new Vec2Array();

    DrawArrayLengths *primitives = NULL;
    if (partShape->typeID() == JtkEntity::JtkPOLYGONSET)
    {
        primitives = new DrawArrayLengths(PrimitiveSet::POLYGON);
    }
    else if (partShape->typeID() == JtkEntity::JtkLINESTRIPSET)
    {
        primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
    }
    else if (partShape->typeID() == JtkEntity::JtkTRISTRIPSET)
    {
        primitives = new DrawArrayLengths(PrimitiveSet::TRIANGLE_STRIP);
    }
    else
    {
        cerr << "unknown partShape->typeID " << partShape->typeID() << endl;
    }
    geode->setName(objName);
    if (primitives)
    {
        for (int set = 0; set < partShape->numOfSets(); set++)
        {
            float *vertex = NULL,
                  *normal = NULL,
                  *color = NULL,
                  *texture = NULL;
            int vertexCount = -1,
                normCount = -1,
                colorCount = -1,
                textCount = -1;

            partShape->getInternal(vertex, vertexCount, normal, normCount,
                                   color, colorCount, texture, textCount, set);

            primitives->push_back(vertexCount);

            // backFaceCulling nur dann, wenn es im CoviseConfig enabled ist
            /*if(backFaceCulling && (mask & Viewer::MASK_SOLID))
         {
         CullFace *cullFace = new CullFace();        // da viele Modelle backface Culling nicht vertragen (nicht richtig modelliert sind)
         cullFace->setMode(CullFace::BACK);
         geoState->setAttributeAndModes(cullFace, StateAttribute::ON);
         }

         // already done in updateMaterial()
         #if 0
         if(Blended)
         {
         BlendFunc *blendFunc = new BlendFunc();
         blendFunc->setFunction(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA);
         geoState->setAttributeAndModes(blendFunc, StateAttribute::ON);
         #if 1
         AlphaFunc *alphaFunc = new AlphaFunc();
         alphaFunc->setFunction(AlphaFunc::ALWAYS,1.0);
         geoState->setAttributeAndModes(alphaFunc, StateAttribute::OFF);
         #endif
         }
         #endif
         #ifdef HAVE_OSGNV
         if((strncmp(d_currentObject->node->name(),"combineTextures",15)==0)||(strncmp(objName,"combineTextures",15)==0))
         {
         geoState->setAttributeAndModes(combineTextures.get(), StateAttribute::ON);
         }
         if((strncmp(d_currentObject->node->name(),"combineEnvTextures",15)==0)||(strncmp(objName,"combineEnvTextures",15)==0))
         {
         geoState->setAttributeAndModes(combineEnvTextures.get(), StateAttribute::ON);
         }
         #endif*/

            if (vertex && (vertexCount > 0))
            {
                for (int elems = 0; elems < vertexCount; elems++)
                {
                    vert->push_back(Vec3(vertex[elems * 3 + 0], vertex[elems * 3 + 1], vertex[elems * 3 + 2]));
                }
                JtkEntityFactory::deleteMemory(vertex);
            }

            if (normal && (normCount > 0))
            {
                for (int elems = 0; elems < normCount; elems++)
                {
                    normalArray->push_back(Vec3(normal[elems * 3 + 0], normal[elems * 3 + 1], normal[elems * 3 + 2]));
                }
                if (normCount == vertexCount)
                {
                }
                else
                {
                    //geom->setNormalBinding(Geometry::BIND_PER_PRIMITIVE);
                    std::cerr << "JTOpen: normals per primitive not supported" << std::endl;
                }
                JtkEntityFactory::deleteMemory(normal);
            }
            else // generate normals
            {
            }

            if (color && (colorCount > 0))
            {

                for (int elems = 0; elems < colorCount; elems++)
                {
                    colorArray->push_back(Vec3(color[elems * 3 + 0], color[elems * 3 + 1], color[elems * 3 + 2]));
                }

                if (colorCount == vertexCount)
                {
                }
                else
                {
                    //geom->setColorBinding(Geometry::BIND_PER_PRIMITIVE);
                    std::cerr << "JTOpen: colors per primitive not supported" << std::endl;
                }
                JtkEntityFactory::deleteMemory(color);
            }

            if (texture && (textCount > 0))
            {

                for (int elems = 0; elems < textCount; elems++)
                {
                    tcArray->push_back(Vec2(texture[elems * 2 + 0], texture[elems * 2 + 1]));
                }
                JtkEntityFactory::deleteMemory(texture);
            }

            /*   if(!(mask & Viewer::MASK_CONVEX))
         {
         osgUtil::Tesselator *tess = new osgUtil::Tesselator;
         tess->retesselatePolygons(*geom);
         //delete[] tess;
         }*/

        }
        geom->setVertexArray(vert);
        geom->addPrimitiveSet(primitives);
        if (normalArray->size() > 0)
        {
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(Geometry::BIND_PER_VERTEX);
        }
        if (colorArray->size() > 0)
        {
            geom->setColorArray(colorArray);
            geom->setColorBinding(Geometry::BIND_PER_VERTEX);
        }
        if (tcArray->size() > 0)
            geom->setTexCoordArray(0, tcArray);
        if (normalArray->size() == 0)
        {
            osgUtil::SmoothingVisitor::smooth(*(geom.get()), 40.0 / 180.0 * M_PI);
        }
        geode->addDrawable(geom.get());
        geode->setStateSet(geoState);
        return geode;
    }
    return NULL;
}

int JTOpenPlugin::myPreactionCB(JtkHierarchy *CurrNode, int level, JtkClientData *cd)
{
    return JTOpenPlugin::plugin->PreAction(CurrNode, level, cd);
}

int JTOpenPlugin::myPostactionCB(JtkHierarchy *CurrNode, int level, JtkClientData *cd)
{
    return JTOpenPlugin::plugin->PostAction(CurrNode, level, cd);
}

int JTOpenPlugin::PostAction(JtkHierarchy * /*CurrNode*/, int /*level*/, JtkClientData *)
{
    Parents.pop_back();
    currentGroup = Parents.back();
    return (Jtk_OK);
}

void JTOpenPlugin::setMaterial(osg::Node *osgNode, JtkHierarchy *CurrNode)
{
	JtkTexImage *partTexture = NULL;
	((JtkPart *)CurrNode)->getTexImage(partTexture);
    JtkMaterial *partMaterial = NULL;
    ((JtkPart *)CurrNode)->getMaterial(partMaterial);
    if (partMaterial)
    {
        float *ambient = NULL,
              *diffuse = NULL,
              *specular = NULL,
              *emission = NULL,
              shininess = -999.0;
        StateSet *stateset = NULL;
        osg::Geode *osgGeode = dynamic_cast<osg::Geode *>(osgNode);
        if (osgGeode)
        {
            osg::Drawable *drawable = osgGeode->getDrawable(0);
            if (drawable)
            {
                stateset = drawable->getOrCreateStateSet();
            }
            else
                stateset = osgGeode->getOrCreateStateSet();
        }
        else
            stateset = osgNode->getOrCreateStateSet();

        osg::Material *mtl = new osg::Material();

        partMaterial->getAmbientColor(ambient);
        if (ambient)
        {
            mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(ambient[0], ambient[1], ambient[2], ambient[3]));
            JtkEntityFactory::deleteMemory(ambient);
        }

        partMaterial->getDiffuseColor(diffuse);
        if (diffuse)
        {
            mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(diffuse[0], diffuse[1], diffuse[2], diffuse[3]));
            JtkEntityFactory::deleteMemory(diffuse);
        }

        partMaterial->getSpecularColor(specular);
        if (specular)
        {
            mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(specular[0], specular[1], specular[2], specular[3]));
            JtkEntityFactory::deleteMemory(specular);
        }

        partMaterial->getEmissionColor(emission);
        if (emission)
        {
            mtl->setEmission(Material::FRONT_AND_BACK, Vec4(emission[0], emission[1], emission[2], emission[3]));
            JtkEntityFactory::deleteMemory(emission);
        }

        partMaterial->getShininess(shininess);
        if (shininess != -999.0)
        {
            mtl->setShininess(Material::FRONT_AND_BACK, shininess);
        }
        stateset->setAttributeAndModes(mtl, StateAttribute::ON);
    }
}

void JTOpenPlugin::setShapeMaterial(osg::Node *osgNode, JtkShape *currShape)
{
    JtkMaterial *partMaterial = NULL;
    currShape->getMaterial(partMaterial);
    if (partMaterial)
    {
        float *ambient = NULL,
              *diffuse = NULL,
              *specular = NULL,
              *emission = NULL,
              shininess = -999.0;
        StateSet *stateset = NULL;
        osg::Geode *osgGeode = dynamic_cast<osg::Geode *>(osgNode);
        if (osgGeode)
        {
            osg::Drawable *drawable = osgGeode->getDrawable(0);
            if (drawable)
            {
                stateset = drawable->getOrCreateStateSet();
            }
            else
                stateset = osgGeode->getOrCreateStateSet();
        }
        else
            stateset = osgNode->getOrCreateStateSet();

        osg::Material *mtl = new osg::Material();

        partMaterial->getAmbientColor(ambient);
        if (ambient)
        {
            mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(ambient[0], ambient[1], ambient[2], ambient[3]));
            JtkEntityFactory::deleteMemory(ambient);
        }

        partMaterial->getDiffuseColor(diffuse);
        if (diffuse)
        {
            mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(diffuse[0], diffuse[1], diffuse[2], diffuse[3]));
            JtkEntityFactory::deleteMemory(diffuse);
        }

        partMaterial->getSpecularColor(specular);
        if (specular)
        {
            mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(specular[0], specular[1], specular[2], specular[3]));
            JtkEntityFactory::deleteMemory(specular);
        }

        partMaterial->getEmissionColor(emission);
        if (emission)
        {
            mtl->setEmission(Material::FRONT_AND_BACK, Vec4(emission[0], emission[1], emission[2], emission[3]));
            JtkEntityFactory::deleteMemory(emission);
        }

        partMaterial->getShininess(shininess);
        if (shininess != -999.0)
        {
            mtl->setShininess(Material::FRONT_AND_BACK, shininess);
        }
        stateset->setAttributeAndModes(mtl, StateAttribute::ON);
    }
}

osg::Group *JTOpenPlugin::createGroup(JtkHierarchy *CurrNode)
{
    osg::Group *newGroup = NULL;
    JtkTransform *partXform = NULL;
    ((JtkPart *)CurrNode)->getTransform(partXform);
    if (partXform)
    {
        float *matElements;
        partXform->getTElements(matElements);
        osg::MatrixTransform *tr = new osg::MatrixTransform;
        osg::Matrix mat(matElements);
        JtkEntityFactory::deleteMemory(matElements);
        tr->setMatrix(mat);
        newGroup = tr;
    }
    else
    {
        newGroup = new osg::Group();
    }

    newGroup->setName(getJtName(CurrNode));
    return newGroup;
}

int JTOpenPlugin::PreAction(JtkHierarchy *CurrNode, int level, JtkClientData *)
{
    indent(level);
    //my_level++;

    /*
         JtkUnitHierarchy* unitRoot = dynamic_cast<JtkUnitHierarchy *>(CurrNode);
         if(unitRoot)
         {
            JtkUnits Units;
            unitRoot->getUnits(Units);
            std::cerr << "Unit: " << Units << std::endl;
         }*/

	for (int i = 0; i < 100; i++)
	{
		JtkAttrib *attr=NULL;
		CurrNode->getAttrib(i, attr);
		if (attr == NULL)
			break;
		fprintf(stderr,"Attrib %s\n", getJtName(attr).c_str());
			
	}
	

    switch (CurrNode->typeID())
    {
    case JtkEntity::JtkNONE:
        cout << "JtkNONE\n";
        break;

    case JtkEntity::JtkBREP:
        cout << "JtkBREP\n";
        break;

    case JtkEntity::JtkREGION:
        cout << "JtkREGION\n";
        break;

    case JtkEntity::JtkSHELL:
        cout << "JtkSHELL\n";
        break;

    case JtkEntity::JtkLOOP:
        cout << "JtkLOOP\n";
        break;

    case JtkEntity::JtkCOEDGE:
        cout << "JtkCOEDGE\n";
        break;

    case JtkEntity::JtkEDGE:
        cout << "JtkEDGE\n";
        break;

    case JtkEntity::JtkVERTEX:
        cout << "JtkVERTEX\n";
        break;

    case JtkEntity::JtkNURBSSURFACE:
        cout << "JtkNURBSSURFACE\n";
        break;

    case JtkEntity::JtkUVCURVE:
        cout << "JtkUVCURVE\n";
        break;

    case JtkEntity::JtkXYZCURVE:
        cout << "JtkXYZCURVE\n";
        break;

    case JtkEntity::JtkTRISTRIPSET:
        cout << "JtkTRISTRIPSET\n";
        break;

    case JtkEntity::JtkPOINTSET:
        cout << "JtkPOINTSET\n";
        break;

    case JtkEntity::JtkLINESTRIPSET:
        cout << "JtkLINESTRIPSET\n";
        break;

    case JtkEntity::JtkPOLYGONSET:
        cout << "JtkPOLYGONSET\n";
        break;

    case JtkEntity::JtkPOINT:
        cout << "JtkPOINT\n";
        break;

    case JtkEntity::JtkMATERIAL:
        cout << "JtkMATERIAL\n";
        break;

    case JtkEntity::JtkTRANSFORM:
        cout << "JtkTRANSFORM\n";
        break;

    case JtkEntity::JtkPROPERTY:
        cout << "JtkPROPERTY\n";
        break;

    case JtkEntity::JtkPART:
    {
        osg::Group *newGroup;
        newGroup = createGroup(CurrNode);
        currentGroup.get()->addChild(newGroup);
        currentGroup = newGroup;
        setMaterial(currentGroup.get(), CurrNode);

        JtkWrep *partWrep = NULL;
        ((JtkPart *)CurrNode)->getWrep(partWrep);
        if (partWrep)
        {
            printWrep(partWrep, level + 1);
        }

        int partNumShapeLODs = -1;
        partNumShapeLODs = ((JtkPart *)CurrNode)->numPolyLODs();
        int numPrimLODs = ((JtkPart *)CurrNode)->numPrimLODs();
        if (partNumShapeLODs > 1)
        {
            newGroup = new coLOD();
            currentGroup->addChild(newGroup);
        }
        if (partNumShapeLODs == 0) // no LODs, tessellate breps
        {
            /*    ((JtkPart*) CurrNode)->autoNormalsON();
            JtkBrep* brep
            ((JtkPart*) CurrNode)->getBrep(brep);*/

            JtkXTBody *xtBody;
            int error;

            ((JtkPart *)CurrNode)->getBody(0, xtBody, error);
            JtkBrep *partBrep = NULL;
            ((JtkPart *)CurrNode)->getBrep(partBrep);
            if (partBrep)
            {
                int numFaces = partBrep->numChildren(JtkEntity::JtkFACE);
                int numRegions = partBrep->numChildren(JtkEntity::JtkREGION);
                int numShells = partBrep->numChildren(JtkEntity::JtkSHELL);
                cout << endl << "JtkBRep: ";
                cout << " faces: " << numFaces << endl;
                cout << " region: " << numRegions << endl;
                cout << " shell: " << numShells << endl;
                if (xtBody)
                    cout << endl << "has xtBody: " << endl;
            }
        }
        cout << "JtkPART: ";
        cout << CurrNode->name() << " lods: " << partNumShapeLODs << " Primlods: " << numPrimLODs << endl;
        for (int lod = 0; lod < partNumShapeLODs; lod++)
        {
            //indent(level+1);
            //cout << "LOD#" << lod << ":\n";

            int partNumShapes = -1;
            partNumShapes = ((JtkPart *)CurrNode)->numPolyShapes(lod);
            osg::Group *newLODGroup = newGroup;
            if (partNumShapes > 1)
            {
                newLODGroup = new osg::Group();
                newGroup->addChild(newLODGroup);
            }
            if (partNumShapes > 1)
            {
                cout << "partNumShapes#" << partNumShapes << ":" << endl;
            }
            for (int shNum = 0; shNum < partNumShapes; shNum++)
            {
                //indent(level+2);
                //cout << "Shape#" << shNum << ":" << endl;

                JtkShape *partShape = NULL;
                ((JtkPart *)CurrNode)->getPolyShape(partShape, lod, shNum);
                if (partShape)
                {
                    std::string stringUTF8 = getJtName(CurrNode);
                    char *shapeName = new char[stringUTF8.length() + 30];
                    sprintf(shapeName, "%s_%d_%d", stringUTF8.c_str(), lod, shNum);
                    osg::Node *n = createShape(partShape, shapeName);
		    
                    setShapeMaterial(n, partShape);
                    newLODGroup->addChild(n);
                }
            }
            if (partNumShapeLODs > 1)
            {
                if (lod < partNumShapeLODs - 1)
                    ((coLOD *)newGroup)->setRange(lod, plugin->lodScale * lod, plugin->lodScale * (lod + 1));
                else
                    ((coLOD *)newGroup)->setRange(lod, plugin->lodScale * lod, 100000000);
            }
        }

        if (partNumShapeLODs > 1)
        {
        }
    }
    break;

    case JtkEntity::JtkASSEMBLY:
    {
        cout << "JtkASSEMBLY: ";
        cout << CurrNode->name() << "("
             << ((JtkAssembly *)CurrNode)->numChildren()
             << " children)\n";

        osg::Group *newGroup = createGroup(CurrNode);
        currentGroup.get()->addChild(newGroup);
        currentGroup = newGroup;
        setMaterial(currentGroup.get(), CurrNode);
    }
    break;

    case JtkEntity::JtkINSTANCE:
    {
        cout << "JtkINSTANCE: ";
        cout << CurrNode->name() << "\n";

        // Declare an instance of 'findNodeVisitor' class and set its
        // searchForName string equal to "sw1"
        findNodeVisitor findNode(getJtName(CurrNode));

        // Initiate traversal of this findNodeVisitor instance starting
        // from tankTwoGroup, searching all its children. Build a list
        // of nodes whose names matched the searchForName string above.
        firstGroup->accept(findNode);

        osg::Group *newGroup = createGroup(CurrNode);
        currentGroup.get()->addChild(newGroup);
        currentGroup = newGroup;
        setMaterial(currentGroup.get(), CurrNode);
        osg::Group *gr = NULL;
        osg::Node *g = NULL;
        if (findNode.getNodeList().size() > 0)
        {
            g = *(findNode.getNodeList().begin());
            gr = dynamic_cast<osg::Group *>(g);
            if (gr) // if this is a group, add all its children to the new group
            {
                for (unsigned int i = 0; i < gr->getNumChildren(); i++)
                {
                    newGroup->addChild(gr->getChild(i));
                }
            }
            if (!gr)
            {
                newGroup->addChild(g);
            }
        }
        else
        {
            fprintf(stderr, "Instance not found %s\n", getJtName(CurrNode).c_str());
        }

        /*
         if( want_details )
         {
         JtkTransform   *partXform= NULL;
         ((JtkPart*) CurrNode)->getTransform(partXform);
         if( partXform )
         {
         printXform(partXform, level+1);
         }

         JtkMaterial *partMaterial= NULL;
         ((JtkPart*) CurrNode)->getMaterial(partMaterial);
         if( partMaterial )
         {
         printMaterial(partMaterial, level+1);
         }
         }*/
    }
    break;

    case JtkEntity::JtkCLIENTDATA:
        cout << "JtkCLIENTDATA\n";
        break;

    case JtkEntity::JtkWIRE:
        cout << "JtkWIRE\n";
        break;
    default:
        cout << "unknown JtkEntry\n";
        break;
    }

    Parents.push_back(currentGroup);
    return (Jtk_OK);
}

JTOpenPlugin *JTOpenPlugin::plugin = NULL;

static FileHandler handlers[] = {
    { NULL,
      JTOpenPlugin::loadJT,
      JTOpenPlugin::unloadJT,
      "jt" }
};

int JTOpenPlugin::loadJT(const char *filename, osg::Group *loadParent, const char *)
{

    // Try to create an JtkCADImporter to test for JT read licensing
    JtkCADImporter *jtreader = NULL;
    jtreader = JtkEntityFactory::createCADImporter();
    if (!jtreader)
    {
        cout << "No import license found.\n";
        return (-1);
    }
    else
    {
        jtreader->ref();
        jtreader->unref();
        jtreader = NULL;
    }

    JtkCADImporter *importer = NULL;
    importer = JtkEntityFactory::createCADImporter();
    if (importer)
    {
        importer->ref();
        importer->setShapeLoadOption(JtkCADImporter::JtkALL_LODS);
        importer->setBrepLoadOption(JtkCADImporter::JtkTESS_ONLY);
        //importer->setBrepLoadOption(JtkCADImporter::JtkBREP_ONLY);
        JtkHierarchy *root = NULL;

        root = importer->import(filename);

        if (root)
        {
            root->ref();
            JtkTraverser *trav = JtkEntityFactory::createTraverser();
            trav->setupPreActionCallback(myPreactionCB);
            trav->setupPostActionCallback(myPostactionCB);
            while (JTOpenPlugin::plugin->Parents.size())
                JTOpenPlugin::plugin->Parents.pop_back();
            JtkUnitHierarchy *unitRoot = dynamic_cast<JtkUnitHierarchy *>(root);

            // We need a single, dedicated root node for registration in vr-prepare.
            // If no registration is done, we can skip this part.
            if (!VRRegisterSceneGraph::instance()->isBlocked())
            {
                osg::Group *rootGroup = new osg::Group;
                loadParent->addChild(rootGroup);
                loadParent->setName("JT root");
                loadParent = rootGroup;
            }

            // loadParent is a new node if loading JT directly or a PLMXML-node if loading via PLMXML
            // in both cases we can disable intersection safely
            loadParent->setNodeMask(loadParent->getNodeMask() & (~Isect::Intersection));

            if (unitRoot)
            {
                JtkUnits Units;
                unitRoot->getUnits(Units);

                float scale = 1;
                if (Units == JtkCENTIMETERS)
                {
                    scale = 10;
                }
                if (Units == JtkDECIMETERS)
                {
                    scale = 100;
                }
                if (Units == JtkMETERS)
                {
                    scale = 1000;
                }
                if (Units == JtkKILOMETERS)
                {
                    scale = 1000000;
                }
                if (Units == JtkINCHES)
                {
                    scale = 25.4;
                }
                if (Units == JtkFEET)
                {
                    scale = 304.8;
                }
                if (Units == JtkYARDS)
                {
                    scale = 914.4;
                }
                if (Units == JtkMILES)
                {
                    scale = 1609000.0;
                }
                if (Units == JtkMILS)
                {
                    scale = 25.4 / 1000.0;
                }
                if ((scale != 1) && plugin->scaleGeometry)
                {
                    osg::MatrixTransform *mt;
                    mt = new osg::MatrixTransform;
                    mt->setMatrix(osg::Matrix::scale(scale, scale, scale));
                    mt->setName("scaleToJtUnits");
                    loadParent->addChild(mt);
                    loadParent = mt;
                }
                std::cerr << "Unit: " << Units << std::endl;
            }
            JTOpenPlugin::plugin->currentGroup = loadParent;
            JTOpenPlugin::plugin->firstGroup = loadParent;
            JTOpenPlugin::plugin->Parents.push_back(JTOpenPlugin::plugin->currentGroup);
            if (trav)
            {
                trav->ref();
                trav->traverseGraph(root);
                trav->unref();
                trav = NULL;
            }
            else
            {
                JtkError << "Unable to create JtkTraverser.\n";
                return (-1);
            }
            JTOpenPlugin::plugin->Parents.clear();
            root->unref();
            root = NULL;
	    
    coVRShaderList::instance()->get("SolidClippingObject")->apply(loadParent);
            VRRegisterSceneGraph::instance()->registerNode(loadParent, filename);
        }
        else
        {
            cerr << "Unable in find root node.  Check file...\n";
            return (-1);
        }

        importer->unref();
        importer = NULL;
    }
    else
    {
        cerr << "Unable to create JtkCADImporter.  Check license...\n";
    }

    return 0;
}

int JTOpenPlugin::unloadJT(const char *filename, const char *)
{
    (void)filename;

    return 0;
}

JTOpenPlugin::JTOpenPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    // Initialize JtTk
    JtkEntityFactory::init();

    // Note, non-evaluation JT Open Toolkit licensees must uncomment the
    // following line, inserting their "Sold_To_ID". Each licensee has a
    // unique Sold_To_ID issued by UGS Corp.
    //
    long soldToId = covise::coCoviseConfig::getInt("COVER.Plugin.JTOpen.SoldToId", -1);
    if (soldToId == -1)
    {
        cerr << "Did not find JTOpen license in config file" << endl;
    }
    else
    {
        JtkEntityFactory::registerCustomer(soldToId);
    }

    scaleGeometry = coCoviseConfig::isOn("COVER.Plugin.JTOpen.ScaleGeometry", true);
    lodScale = coCoviseConfig::getFloat("COVER.Plugin.JTOpen.LodScale", 2000.0f);

    plugin = this;
}

bool JTOpenPlugin::init()
{
    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
JTOpenPlugin::~JTOpenPlugin()
{
    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    // Uninitialize JtTk
    JtkEntityFactory::fini();
}

void
JTOpenPlugin::preFrame()
{
}

COVERPLUGIN(JTOpenPlugin)
