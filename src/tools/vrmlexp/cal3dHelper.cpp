/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
*<
FILE: cal3dHelper.cpp

DESCRIPTION:  A Cal3D helper implementation

CREATED BY: Uwe Woessner

HISTORY: created 25 Apr. 2011

*> Copyright (c) 1996, All Rights Reserved.
**********************************************************************/
#ifndef NO_CAL3D
#include "vrml.h"
#include "bookmark.h"
#include "cal3dHelper.h"

// Parameter block indices
#define PB_LENGTH 0

//----------------------------------------------------------------------------//
// Static member variables initialization                                     //
//----------------------------------------------------------------------------//

const int Cal3DCoreHelper::STATE_IDLE = 0;
const int Cal3DCoreHelper::STATE_FANCY = 1;
const int Cal3DCoreHelper::STATE_MOTION = 2;

//----------------------------------------------------------------------------//
// Constructors                                                               //
//----------------------------------------------------------------------------//

Cal3DCoreHelper::Cal3DCoreHelper()
{
    m_calCoreModel = new CalCoreModel("dummy");

    m_state = 0;
    m_motionBlend[0] = 0.6f;
    m_motionBlend[1] = 0.1f;
    m_motionBlend[2] = 0.3f;
    m_animationCount = 0;
    m_meshCount = 0;
    m_renderScale = 1.0f;
    m_lodLevel = 1.0f;
    written = false;
}

//----------------------------------------------------------------------------//
// Destructor                                                                 //
//----------------------------------------------------------------------------//

Cal3DCoreHelper::~Cal3DCoreHelper()
{
}

//----------------------------------------------------------------------------//
// Execute an action of the model                                             //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::executeAction(int action)
{
    m_calModel->getMixer()->executeAction(m_animationId[0], 0.3f, 0.3f);
}

//----------------------------------------------------------------------------//
// Get the lod level of the model                                             //
//----------------------------------------------------------------------------//

float Cal3DCoreHelper::getLodLevel()
{
    return m_lodLevel;
}

//----------------------------------------------------------------------------//
// Get the motion blend factors state of the model                            //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::getMotionBlend(float *pMotionBlend)
{
    pMotionBlend[0] = m_motionBlend[0];
    pMotionBlend[1] = m_motionBlend[1];
    pMotionBlend[2] = m_motionBlend[2];
}

//----------------------------------------------------------------------------//
// Get the render scale of the model                                          //
//----------------------------------------------------------------------------//

float Cal3DCoreHelper::getRenderScale()
{
    return m_renderScale;
}

//----------------------------------------------------------------------------//
// Get the animation state of the model                                       //
//----------------------------------------------------------------------------//

int Cal3DCoreHelper::getState()
{
    return m_state;
}

//----------------------------------------------------------------------------//
// Read a int from file stream (to avoid Little/Big endian issue)
//----------------------------------------------------------------------------//

int readInt(std::ifstream *file)
{
    int x = 0;
    for (int i = 0; i < 32; i += 8)
    {
        char c;
        file->read(&c, 1);
        x += c << i;
    }
    return x;
}

//----------------------------------------------------------------------------//
// Initialize the model                                                       //
//----------------------------------------------------------------------------//

#define CTL_CHARS 31
#define SINGLE_QUOTE 39
static TCHAR *myVRMLName(const TCHAR *name)
{
    static TCHAR buffer[256];
    static int seqnum = 0;
    TCHAR *cPtr;
    int firstCharacter = 1;

    _tcscpy(buffer, name);
    cPtr = buffer;
    while (*cPtr)
    {
        if (*cPtr <= CTL_CHARS || *cPtr == ' ' || *cPtr == SINGLE_QUOTE || *cPtr == '"' || *cPtr == '\\' || *cPtr == '/' || *cPtr == '{' || *cPtr == '}' || *cPtr == ',' || *cPtr == '.' || *cPtr == '[' || *cPtr == ']' || *cPtr == '-' || *cPtr == '#' || *cPtr >= 127 || (firstCharacter && (*cPtr >= '0' && *cPtr <= '9')))
            *cPtr = '_';
        firstCharacter = 0;
        cPtr++;
    }
    if (firstCharacter)
    { // if empty name, quick, make one up!
        *cPtr++ = '_';
        *cPtr++ = '_';
        _stprintf(cPtr, _T("%d"), seqnum++);
    }

    return buffer;
}
bool Cal3DCoreHelper::loadCfg(const std::string &strFilename)
{
    // open the model configuration file
    m_name = strFilename;
#if MAX_PRODUCT_VERSION_MAJOR > 14
    TSTR tmpN;
    tmpN.FromUTF8(m_name.c_str());
    TSTR tmpName = myVRMLName(tmpN);
    m_VrmlName = tmpName.ToCStr().data();
#else
    m_VrmlName = myVRMLName(m_name.c_str());
#endif

    std::ifstream file;
    file.open(strFilename.c_str(), std::ios::in | std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open model configuration file '" << strFilename << "'." << std::endl;
        return false;
    }
    size_t pathend = strFilename.find_last_of("/\\");
    if (pathend != std::string::npos)
    {
        m_path = strFilename.substr(0, pathend + 1);
    }

    // initialize the data path
    std::string strPath = m_path;

    // initialize the animation count
    m_animationCount = 0;

    // parse all lines from the model configuration file
    int line;
    for (line = 1;; line++)
    {
        // read the next model configuration line
        std::string strBuffer;
        std::getline(file, strBuffer);

        // stop if we reached the end of file
        if (file.eof())
            break;

        // check if an error happened while reading from the file
        if (!file)
        {
            std::cerr << "Error while reading from the model configuration file '" << strFilename << "'." << std::endl;
            return false;
        }

        // find the first non-whitespace character
        std::string::size_type pos;
        pos = strBuffer.find_first_not_of(" \t");

        // check for empty lines
        if ((pos == std::string::npos) || (strBuffer[pos] == '\n') || (strBuffer[pos] == '\r') || (strBuffer[pos] == 0))
            continue;

        // check for comment lines
        if (strBuffer[pos] == '#')
            continue;

        // get the key
        std::string strKey;
        strKey = strBuffer.substr(pos, strBuffer.find_first_of(" =\t\n\r", pos) - pos);
        pos += strKey.size();

        // get the '=' character
        pos = strBuffer.find_first_not_of(" \t", pos);
        if ((pos == std::string::npos) || (strBuffer[pos] != '='))
        {
            std::cerr << strFilename << "(" << line << "): Invalid syntax." << std::endl;
            return false;
        }

        // find the first non-whitespace character after the '=' character
        pos = strBuffer.find_first_not_of(" \t", pos + 1);

        // get the data
        std::string strData;
        strData = strBuffer.substr(pos, strBuffer.find_first_of("\n\r", pos) - pos);

        // handle the model creation
        if (strKey == "scale")
        {
            // set rendering scale factor
            m_renderScale = (float)atof(strData.c_str());
        }
        else if (strKey == "path")
        {
            // set the new path for the data files if one hasn't been set already
            if (m_path == "")
                strPath = strData;
        }
        else if (strKey == "skeleton")
        {
            // load core skeleton
            std::cout << "Loading skeleton '" << strData << "'..." << std::endl;
            if (!m_calCoreModel->loadCoreSkeleton(strPath + strData))
            {
                CalError::printLastError();
                return false;
            }
        }
        else if (strKey == "animation")
        {
            // load core animation
            std::cout << "Loading animation '" << strData << "'..." << std::endl;
            m_animationId[m_animationCount] = m_calCoreModel->loadCoreAnimation(strPath + strData);
            if (m_animationId[m_animationCount] == -1)
            {
                CalError::printLastError();
                return false;
            }

            m_animationCount++;
        }
        else if (strKey == "mesh")
        {
            // load core mesh
            std::cout << "Loading mesh '" << strData << "'..." << std::endl;
            if (m_calCoreModel->loadCoreMesh(strPath + strData) == -1)
            {
                CalError::printLastError();
                return false;
            }
        }
        else if (strKey == "material")
        {
            // load core material
            std::cout << "Loading material '" << strData << "'..." << std::endl;
            if (m_calCoreModel->loadCoreMaterial(strPath + strData) == -1)
            {
                CalError::printLastError();
                return false;
            }
        }
        else
        {
            std::cerr << strFilename << "(" << line << "): Invalid syntax." << std::endl;
            return false;
        }
    }

    // explicitely close the file
    file.close();

    // load all textures and store the opengl texture id in the corresponding map in the material
    int materialId;
    for (materialId = 0; materialId < m_calCoreModel->getCoreMaterialCount(); materialId++)
    {
        // get the core material
        CalCoreMaterial *pCoreMaterial;
        pCoreMaterial = m_calCoreModel->getCoreMaterial(materialId);

        // loop through all maps of the core material
        int mapId;
        for (mapId = 0; mapId < pCoreMaterial->getMapCount(); mapId++)
        {
            // get the filename of the texture
            std::string strFilename;
            strFilename = pCoreMaterial->getMapFilename(mapId);

            // load the texture from the file
            // GLuint textureId;
            // textureId = loadTexture(strPath + strFilename);

            // store the opengl texture id in the user data of the map
            // pCoreMaterial->setMapUserData(mapId, (Cal::UserData)textureId);
        }
    }

    // make one material thread for each material
    // NOTE: this is not the right way to do it, but this viewer can't do the right
    // mapping without further information on the model etc.
    for (materialId = 0; materialId < m_calCoreModel->getCoreMaterialCount(); materialId++)
    {
        // create the a material thread
        m_calCoreModel->createCoreMaterialThread(materialId);

        // initialize the material thread
        m_calCoreModel->setCoreMaterialId(materialId, 0, materialId);
    }

    // Calculate Bounding Boxes

    m_calCoreModel->getCoreSkeleton()->calculateBoundingBoxes(m_calCoreModel);

    m_calModel = new CalModel(m_calCoreModel);

    // attach all meshes to the model
    int meshId;
    for (meshId = 0; meshId < m_calCoreModel->getCoreMeshCount(); meshId++)
    {
        m_calModel->attachMesh(meshId);
    }

    // set the material set of the whole model
    m_calModel->setMaterialSet(0);

    // set initial animation state
    m_state = 0;
    m_calModel->getMixer()->blendCycle(m_animationId[0], 1.0, 0.0f);

    return true;
}

//----------------------------------------------------------------------------//
// Render the skeleton of the model                                           //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::renderSkeleton()
{
    // draw the bone lines
    /*  float lines[1024][2][3];
   int nrLines;
   nrLines =  m_calModel->getSkeleton()->getBoneLines(&lines[0][0][0]);
   //  nrLines = m_calModel->getSkeleton()->getBoneLinesStatic(&lines[0][0][0]);

   glLineWidth(3.0f);
   glColor3f(1.0f, 1.0f, 1.0f);
   glBegin(GL_LINES);
   int currLine;
   for(currLine = 0; currLine < nrLines; currLine++)
   {
   glVertex3f(lines[currLine][0][0], lines[currLine][0][1], lines[currLine][0][2]);
   glVertex3f(lines[currLine][1][0], lines[currLine][1][1], lines[currLine][1][2]);
   }
   glEnd();
   glLineWidth(1.0f);

   // draw the bone points
   float points[1024][3];
   int nrPoints;
   nrPoints =  m_calModel->getSkeleton()->getBonePoints(&points[0][0]);
   //  nrPoints = m_calModel->getSkeleton()->getBonePointsStatic(&points[0][0]);

   glPointSize(4.0f);
   glBegin(GL_POINTS);
   glColor3f(0.0f, 0.0f, 1.0f);
   int currPoint;
   for(currPoint = 0; currPoint < nrPoints; currPoint++)
   {
   glVertex3f(points[currPoint][0], points[currPoint][1], points[currPoint][2]);
   }
   glEnd();
   glPointSize(1.0f);*/
}

//----------------------------------------------------------------------------//
// Render the bounding boxes of a model                                       //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::renderBoundingBox()
{

    /*  CalSkeleton *pCalSkeleton = m_calModel->getSkeleton();

   std::vector<CalBone*> &vectorCoreBone = pCalSkeleton->getVectorBone();

   glColor3f(1.0f, 1.0f, 1.0f);
   glBegin(GL_LINES);      

   for(size_t boneId=0;boneId<vectorCoreBone.size();++boneId)
   {
   CalBoundingBox & calBoundingBox  = vectorCoreBone[boneId]->getBoundingBox();

   CalVector p[8];
   calBoundingBox.computePoints(p);


   glVertex3f(p[0].x,p[0].y,p[0].z);
   glVertex3f(p[1].x,p[1].y,p[1].z);

   glVertex3f(p[0].x,p[0].y,p[0].z);
   glVertex3f(p[2].x,p[2].y,p[2].z);

   glVertex3f(p[1].x,p[1].y,p[1].z);
   glVertex3f(p[3].x,p[3].y,p[3].z);

   glVertex3f(p[2].x,p[2].y,p[2].z);
   glVertex3f(p[3].x,p[3].y,p[3].z);

   glVertex3f(p[4].x,p[4].y,p[4].z);
   glVertex3f(p[5].x,p[5].y,p[5].z);

   glVertex3f(p[4].x,p[4].y,p[4].z);
   glVertex3f(p[6].x,p[6].y,p[6].z);

   glVertex3f(p[5].x,p[5].y,p[5].z);
   glVertex3f(p[7].x,p[7].y,p[7].z);

   glVertex3f(p[6].x,p[6].y,p[6].z);
   glVertex3f(p[7].x,p[7].y,p[7].z);

   glVertex3f(p[0].x,p[0].y,p[0].z);
   glVertex3f(p[4].x,p[4].y,p[4].z);

   glVertex3f(p[1].x,p[1].y,p[1].z);
   glVertex3f(p[5].x,p[5].y,p[5].z);

   glVertex3f(p[2].x,p[2].y,p[2].z);
   glVertex3f(p[6].x,p[6].y,p[6].z);

   glVertex3f(p[3].x,p[3].y,p[3].z);
   glVertex3f(p[7].x,p[7].y,p[7].z);  

   }

   glEnd();*/
}

//----------------------------------------------------------------------------//
// Render the mesh of the model                                               //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::buildMesh(Mesh &mesh, float scale, TimeValue t)
{

    int timeDiff = t - oldT;
    oldT = t;
    m_calModel->update(((float)timeDiff) / 1000.0f);
    // get the number of meshes
    int meshCount;

    int meshId;
    int submeshId;
    int numVerts = 0;
    int numFaces = 0;
    //static CalIndex meshFaces[50000][3];
    std::vector<CalMesh *> &vectorMesh = m_calModel->getVectorMesh();
    //meshCount = m_calCoreModel->getCoreMeshCount();
    meshCount = (int)vectorMesh.size();
    for (meshId = 0; meshId < meshCount; meshId++)
    {
        /*const CalCoreMesh *cm = m_calCoreModel->getCoreMesh(meshId);
      // get the number of submeshes
      submeshCount = cm->getCoreSubmeshCount();*/
        int submeshCount;
        submeshCount = vectorMesh[meshId]->getSubmeshCount();

        // render all submeshes of the mesh
        for (submeshId = 0; submeshId < submeshCount; submeshId++)
        {
            CalSubmesh *m_pSelectedSubmesh = vectorMesh[meshId]->getSubmesh(submeshId);
            /*const CalCoreSubmesh *csm = cm->getCoreSubmesh(submeshId);
         int faceCount;
         int vertexCount;
         faceCount = (int)csm->getVectorFace().size();
         vertexCount = (int)csm->getVectorVertex().size();*/
            numVerts += m_pSelectedSubmesh->getVertexCount();
            numFaces += m_pSelectedSubmesh->getFaceCount();
        }
    }

    mesh.setNumVerts(numVerts);
    mesh.setNumFaces(numFaces);

    int v = 0;
    int f = 0;
    numVerts = 0;

    // render all meshes of the model
    for (meshId = 0; meshId < meshCount; meshId++)
    {
        //const CalCoreMesh *cm = m_calCoreModel->getCoreMesh(meshId);
        // get the number of submeshes
        int submeshCount;
        submeshCount = vectorMesh[meshId]->getSubmeshCount();
        //submeshCount = cm->getCoreSubmeshCount();

        // render all submeshes of the mesh
        for (submeshId = 0; submeshId < submeshCount; submeshId++)
        {
            CalSubmesh *m_pSelectedSubmesh = vectorMesh[meshId]->getSubmesh(submeshId);
            int faceCount = m_pSelectedSubmesh->getFaceCount();
            int vertexCount = m_pSelectedSubmesh->getVertexCount();
            for (int i = 0; i < vertexCount; i++)
            {
                //CalCoreSubmesh::Vertex vert= m_calModel->getPhysique()->calculateVertex(m_pSelectedSubmesh, i);
                CalVector vert = m_calModel->getPhysique()->calculateVertex(m_pSelectedSubmesh, i);
                mesh.setVert(v, Point3(vert[0] * scale, vert[1] * scale, vert[2] * scale));
                //mesh.setNormal(v, Point3(  vert.normal[0],vert.normal[1],vert.normal[2]));
                v++;
            }
            const CalCoreSubmesh *csm = m_pSelectedSubmesh->getCoreSubmesh();
            const std::vector<CalCoreSubmesh::Face> &vectorFace = csm->getVectorFace();
            for (int faceId = 0; faceId < faceCount; faceId++)
            {
                const CalCoreSubmesh::Face &face = vectorFace[faceId];
                mesh.faces[f].setVerts(face.vertexId[0] + numVerts, face.vertexId[1] + numVerts, face.vertexId[2] + numVerts);
                mesh.faces[f].setEdgeVisFlags(1, 1, 1);
                mesh.faces[f].setSmGroup(0);
                f++;
            }

            numVerts += m_pSelectedSubmesh->getVertexCount();

            /*  const CalCoreSubmesh *csm = cm->getCoreSubmesh(submeshId);
         const std::vector<CalCoreSubmesh::Face>& vectorFace = csm->getVectorFace();
         const std::vector<CalCoreSubmesh::Vertex>& vectorVertex = csm->getVectorVertex();
         int faceCount = (int)vectorFace.size();
         int vertexCount = (int)vectorVertex.size();
         for(int i=0;i<vertexCount;i++)
         {
            //const CalCoreSubmesh::Vertex& vert = vectorVertex[i];
            CalCoreSubmesh::Vertex vert= m_calModel->getPhysique()->calculateVertex(csm, i);
            mesh.setVert(v, Point3(  vert.position[0]*scale,vert.position[1]*scale,vert.position[2]*scale)); 
            //mesh.setNormal(v, Point3(  vert.normal[0],vert.normal[1],vert.normal[2])); 
            v++;
         }
         for(int faceId=0;faceId<faceCount;faceId++)
         {
            const CalCoreSubmesh::Face& face = vectorFace[faceId];
            mesh.faces[f].setVerts(face.vertexId[0],face.vertexId[1],face.vertexId[2]);
            mesh.faces[f].setEdgeVisFlags(1, 1, 1);
            mesh.faces[f].setSmGroup(0);
            f++;
         }*/
        }
    }

    //mesh.InvalidateTopologyCache();
}

//----------------------------------------------------------------------------//
// Render the model                                                           //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::onRender()
{
    // set global OpenGL states
    /*  glEnable(GL_DEPTH_TEST);
   glShadeModel(GL_SMOOTH);


   //CalSkeleton *pCalSkeleton = m_calModel->getSkeleton();

   // Note :
   // You have to call coreSkeleton.calculateBoundingBoxes(calCoreModel)
   // during the initialisation (before calModel.create(calCoreModel))
   // if you want to use bounding boxes.

   m_calModel->getSkeleton()->calculateBoundingBoxes();

   // Note:
   // Uncomment the next lines if you want to test the experimental collision system.
   // Also remember that you need to call calculateBoundingBoxes()

   //m_calModel->getSpringSystem()->setForceVector(CalVector(0.0f,0.0f,0.0f));
   //m_calModel->getSpringSystem()->setCollisionDetection(true);  

   // check if we need to render the skeleton
   if(theMenu.isSkeleton()==1)
   {
   renderSkeleton();
   }
   else if(theMenu.isSkeleton()==2)
   {
   renderBoundingBox();
   }

   // check if we need to render the mesh
   if(theMenu.isSkeleton()==0 || theMenu.isWireframe())
   {
   renderMesh(theMenu.isWireframe(), theMenu.isLight());
   }

   // clear global OpenGL states
   glDisable(GL_DEPTH_TEST);*/
}

//----------------------------------------------------------------------------//
// Update the model                                                           //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::onUpdate(float elapsedSeconds)
{
    // update the model
    m_calModel->update(elapsedSeconds);
}

//----------------------------------------------------------------------------//
// Shut the model down                                                        //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::onShutdown()
{
    delete m_calModel;
    delete m_calCoreModel;
}

//----------------------------------------------------------------------------//
// Set the lod level of the model                                             //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::setLodLevel(float lodLevel)
{
    m_lodLevel = lodLevel;

    // set the new lod level in the cal model renderer
    m_calModel->setLodLevel(m_lodLevel);
}

//----------------------------------------------------------------------------//
// Set the motion blend factors state of the model                            //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::setMotionBlend(float *pMotionBlend, float delay)
{
    m_motionBlend[0] = pMotionBlend[0];

    m_calModel->getMixer()->clearCycle(m_animationId[0], delay);
    m_calModel->getMixer()->blendCycle(m_animationId[0], m_motionBlend[0], delay);

    m_state = 0;
}

//----------------------------------------------------------------------------//
// Set a new animation state within a given delay                             //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::setState(int state, float delay)
{
    // check if this is really a new state
    if (state != m_state && state < m_animationCount)
    {
        m_calModel->getMixer()->blendCycle(m_animationId[state], 1.0f, delay);
        m_calModel->getMixer()->clearCycle(m_animationId[m_state], delay);
        m_state = state;
    }
}

//----------------------------------------------------------------------------//
// Set a path to override any config file path                                //
//----------------------------------------------------------------------------//

void Cal3DCoreHelper::setPath(const std::string &strPath)
{
    m_path = strPath;
}

//----------------------------------------------------------------------------//

Cal3DCoreHelpers::~Cal3DCoreHelpers()
{
    std::list<Cal3DCoreHelper *>::iterator it;
    for (it = cores.begin(); it != cores.end(); it++)
    {
        delete (*it);
    }
}

void Cal3DCoreHelpers::clearWritten()
{
    std::list<Cal3DCoreHelper *>::iterator it;
    for (it = cores.begin(); it != cores.end(); it++)
    {
        (*it)->clearWritten();
    }
}
Cal3DCoreHelper *Cal3DCoreHelpers::getCoreHelper(const std::string &name)
{
    std::list<Cal3DCoreHelper *>::iterator it;
    for (it = cores.begin(); it != cores.end(); it++)
    {
        if ((*it)->getName() == name)
        {
            return (*it);
        }
    }
    return NULL;
}
Cal3DCoreHelper *Cal3DCoreHelpers::addHelper(const std::string &name)
{
    Cal3DCoreHelper *core;
    core = getCoreHelper(name);
    if (core != NULL)
        return core;
    core = new Cal3DCoreHelper();
    if (core->loadCfg(name) == false)
    {
        delete core;
        return NULL;
    }
    cores.push_back(core);
    return core;
}

#include "audio.h"

#define SEGMENTS 32

ISpinnerControl *Cal3DObject::sizeSpin;
ISpinnerControl *Cal3DObject::animationIDSpin;
ISpinnerControl *Cal3DObject::actionIDSpin;

//------------------------------------------------------

class Cal3DClassDesc : public ClassDesc
{
public:
    int IsPublic() { return 1; }
    void *Create(BOOL loading = FALSE)
    {
        return new Cal3DObject;
    }
    const TCHAR *ClassName() { return GetString(IDS_CAL3D_CLASS); }
    SClass_ID SuperClassID() { return HELPER_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(CAL3D_CLASS_ID1,
                                         CAL3D_CLASS_ID2); }
    const TCHAR *Category() { return _T("COVER"); }
};

static Cal3DClassDesc Cal3DDesc;

ClassDesc *GetCal3DDesc() { return &Cal3DDesc; }

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

ICustButton *Cal3DObject::Cal3DPickButton = NULL;

HWND Cal3DObject::hRollup = NULL;
int Cal3DObject::dlgPrevSel = -1;

class Cal3DObjPick : public PickModeCallback
{
    Cal3DObject *sound;

public:
    BOOL HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m, int flags);
    BOOL Pick(IObjParam *ip, ViewExp *vpt);

    void EnterMode(IObjParam *ip);
    void ExitMode(IObjParam *ip);

    HCURSOR GetHitCursor(IObjParam *ip);
    void SetCal3D(Cal3DObject *l) { sound = l; }
};

//static Cal3DObjPick thePick;
static Cal3DObjPick thePick;
static BOOL pickMode = FALSE;
static CommandMode *lastMode = NULL;
Cal3DCoreHelpers *Cal3DObject::cores = NULL;

static void
SetPickMode(Cal3DObject *o)
{
    if (pickMode)
    {
        pickMode = FALSE;
        GetCOREInterface()->PushCommandMode(lastMode);
        lastMode = NULL;
        GetCOREInterface()->ClearPickMode();
    }
    else
    {
        pickMode = TRUE;
        lastMode = GetCOREInterface()->GetCommandMode();
        thePick.SetCal3D(o);
        GetCOREInterface()->SetPickMode(&thePick);
    }
}

BOOL
Cal3DObjPick::HitTest(IObjParam *ip, HWND hWnd, ViewExp *vpt, IPoint2 m,
                      int flags)
{
    INode *node = ip->PickNode(hWnd, m);
    if (node == NULL)
        return FALSE;
    Object *obj = node->EvalWorldState(0).obj;
    return obj->ClassID() == AudioClipClassID;
}

void
Cal3DObjPick::EnterMode(IObjParam *ip)
{
    ip->PushPrompt(GetString(IDS_PICK_AUDIOCLIP));
}

void
Cal3DObjPick::ExitMode(IObjParam *ip)
{
    ip->PopPrompt();
}

BOOL
Cal3DObjPick::Pick(IObjParam *ip, ViewExp *vpt)
{
    if (vpt->HitCount() == 0)
        return FALSE;

    INode *node;
    if ((node = vpt->GetClosestHit()) != NULL)
    {
#if MAX_PRODUCT_VERSION_MAJOR > 8
        RefResult ret = sound->ReplaceReference(1, node);
#else
        RefResult ret = sound->MakeRefByID(FOREVER, 1, node);
#endif

        SetPickMode(NULL);
        // sound->iObjParams->SetCommandMode(sound->previousMode);
        // sound->previousMode = NULL;
        sound->Cal3DPickButton->SetCheck(FALSE);
        HWND hw = sound->hRollup;
        Static_SetText(GetDlgItem(hw, IDC_NAME), sound->audioClip->GetName());
        return FALSE;
    }
    return FALSE;
}

HCURSOR
Cal3DObjPick::GetHitCursor(IObjParam *ip)
{
    return LoadCursor(hInstance, MAKEINTRESOURCE(IDC_LOD_CURSOR));
}

#define RELEASE_SPIN(x)         \
    if (th->x)                  \
    {                           \
        ReleaseISpinner(th->x); \
        th->x = NULL;           \
    }
#define RELEASE_BUT(x)             \
    if (th->x)                     \
    {                              \
        ReleaseICustButton(th->x); \
        th->x = NULL;              \
    }

BOOL CALLBACK
    RollupDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam,
                     Cal3DObject *th)
{
    if (!th && message != WM_INITDIALOG)
        return FALSE;

    switch (message)
    {
    case WM_INITDIALOG:
    {
        TimeValue t = 0;

        SendMessage(GetDlgItem(hDlg, IDC_CAL_CFG_URL), WM_SETTEXT, 0, (LPARAM)th->cal3d_cfg.data());
        EnableWindow(GetDlgItem(hDlg, IDC_CAL_CFG_URL), TRUE);

        th->hRollup = hDlg;

        if (pickMode)
            SetPickMode(th);

        return TRUE;
    }
    case WM_DESTROY:
        if (pickMode)
            SetPickMode(th);
        // th->iObjParams->ClearPickMode();
        // th->previousMode = NULL;
        RELEASE_SPIN(animationIDSpin);
        RELEASE_SPIN(actionIDSpin);

        RELEASE_BUT(Cal3DPickButton);
        return FALSE;

    case CC_SPINNER_BUTTONDOWN:
    {
        int id = LOWORD(wParam);
        switch (id)
        {
        case IDC_ANIMATION_SPIN:
        case IDC_ACTION_SPIN2:
            theHold.Begin();
            return TRUE;
        default:
            return FALSE;
        }
    }
    break;

    case CC_SPINNER_BUTTONUP:
    {
        if (!HIWORD(wParam))
        {
            theHold.Cancel();
            break;
        }
        int id = LOWORD(wParam);
        switch (id)
        {
        case IDC_ANIMATION_SPIN:
        case IDC_ACTION_SPIN2:

            if (th->getCoreHelper())
                th->getCoreHelper()->setState(th->animationIDSpin->GetIVal(), 0.0);
            theHold.Accept(GetString(IDS_DS_PARAMCHG));
            return TRUE;
        default:
            return FALSE;
        }
    }

    case CC_SPINNER_CHANGE:
    {
        int animID;
        int id = LOWORD(wParam);
        TimeValue t = 0; // not really needed...yet

        switch (id)
        {
        case IDC_ANIMATION_SPIN:
        case IDC_ACTION_SPIN2:
            if (!HIWORD(wParam))
                theHold.Begin();

            //actionID = th->actionIDSpin->GetIVal();

            //th->animationIDSpin->SetValue(animID, FALSE);
            //th->actionIDSpin->SetValue(actionID, FALSE);
            animID = th->animationIDSpin->GetIVal();
            if (th->getCoreHelper())
                th->getCoreHelper()->setState(animID, 0.0);

            if (!HIWORD(wParam))
                theHold.Accept(GetString(IDS_DS_PARAMCHG));
            return TRUE;
        default:
            return FALSE;
        }
    }

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_CAL_CFG_URL:
            switch (HIWORD(wParam))
            {
            case EN_SETFOCUS:
                DisableAccelerators();
                break;
            case EN_KILLFOCUS:
                EnableAccelerators();
                break;
            case EN_CHANGE:
                int len = (int)SendDlgItemMessage(hDlg, IDC_CAL_CFG_URL, WM_GETTEXTLENGTH, 0, 0);
                TSTR temp;
                temp.Resize(len + 1);
                SendDlgItemMessage(hDlg, IDC_CAL_CFG_URL, WM_GETTEXT, len + 1, (LPARAM)temp.data());

#if MAX_PRODUCT_VERSION_MAJOR > 14
                th->setURL(temp.ToUTF8().data());
#else
                th->setURL((char *)temp);
#endif
                break;
            }
            break;
        }
        return FALSE;
    default:
        return FALSE;
    }

    return FALSE;
}

static ParamUIDesc descParam[] = {
    // Size
    ParamUIDesc(
        PB_CAL_SIZE,
        EDITTYPE_UNIVERSE,
        IDC_CAL_ICON_SIZE, IDC_CAL_ICON_SIZE_SPINNER,
        0.0f, 10.0f,
        0.1f),

    // anim
    ParamUIDesc(
        PB_CAL_ANIM,
        EDITTYPE_INT,
        IDC_ANIMATION_EDIT, IDC_ANIMATION_SPIN,
        0.0f, 100.0f,
        1),

    // Min action
    ParamUIDesc(
        PB_CAL_ACTION,
        EDITTYPE_INT,
        IDC_ACTION_EDIT2, IDC_ACTION_SPIN2,
        0.0f, 100.0f,
        1),

};

#define PARAMDESC_LENGTH 3

static ParamBlockDescID descVer0[] = {
    { TYPE_FLOAT, NULL, FALSE, 0 },
    { TYPE_INT, NULL, FALSE, 1 },
    { TYPE_INT, NULL, FALSE, 2 },
};

#define CURRENT_VERSION 0

class Cal3DParamDlgProc : public ParamMapUserDlgProc
{
public:
    Cal3DObject *ob;

    Cal3DParamDlgProc(Cal3DObject *o) { ob = o; }
    INT_PTR DlgProc(TimeValue t, IParamMap *map, HWND hWnd, UINT msg,
                    WPARAM wParam, LPARAM lParam);
    void DeleteThis() { delete this; }
};

INT_PTR Cal3DParamDlgProc::DlgProc(TimeValue t, IParamMap *map, HWND hWnd,
                                   UINT msg, WPARAM wParam, LPARAM lParam)
{
    return RollupDialogProc(hWnd, msg, wParam, lParam, ob);
}

IParamMap *Cal3DObject::pmapParam = NULL;
void Cal3DObject::setURL(const std::string &url)
{

#if MAX_PRODUCT_VERSION_MAJOR > 14
    cal3d_cfg.FromUTF8(url.c_str());
#else

    cal3d_cfg = url.c_str();
#endif
    coreHelper = cores->addHelper(url);
}

void
Cal3DObject::BeginEditParams(IObjParam *ip, ULONG flags,
                             Animatable *prev)
{
    iObjParams = ip;
    TimeValue t = ip->GetTime(); // not really needed...yet

    if (pmapParam)
    {

        // Left over from last Cal3D created
        pmapParam->SetParamBlock(pblock);
    }
    else
    {

        // Gotta make a new one.
        pmapParam = CreateCPParamMap(descParam, PARAMDESC_LENGTH,
                                     pblock,
                                     ip,
                                     hInstance,
                                     MAKEINTRESOURCE(IDD_CAL3D),
                                     _T("Cal3D" /*JP_LOC*/),
                                     0);
    }

    if (pmapParam)
    {
        // A callback for dialog
        pmapParam->SetUserDlgProc(new Cal3DParamDlgProc(this));
    }

    animationIDSpin = GetISpinner(GetDlgItem(hRollup, IDC_ANIMATION_SPIN));
    actionIDSpin = GetISpinner(GetDlgItem(hRollup, IDC_ACTION_SPIN2));

    sizeSpin = GetISpinner(GetDlgItem(hRollup, IDC_CAL_ICON_SIZE_SPINNER));
}

void
Cal3DObject::EndEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
    if (flags & END_EDIT_REMOVEUI)
    {
        if (pmapParam)
            DestroyCPParamMap(pmapParam);
        pmapParam = NULL;
    }
}

Cal3DObject::Cal3DObject()
    : HelperObject()
{
    scale = 1.0f;
    coreHelper = NULL;
    cal3d_cfg = _T("");
    if (cores == NULL)
        cores = new Cal3DCoreHelpers();
    pblock = NULL;
    IParamBlock *pb = CreateParameterBlock(descVer0, PB_CAL_LENGTH,
                                           CURRENT_VERSION);
    pb->SetValue(PB_CAL_SIZE, 0, 1.0f);
    pb->SetValue(PB_CAL_ANIM, 0, 0);
    pb->SetValue(PB_CAL_ACTION, 0, 1);
#if MAX_PRODUCT_VERSION_MAJOR > 8
    ReplaceReference(0, pb);
#else
    MakeRefByID(FOREVER, 0, pb);
#endif
    assert(pblock);
    previousMode = NULL;
}

Cal3DObject::~Cal3DObject()
{
    DeleteAllRefsFromMe();
}

IObjParam *Cal3DObject::iObjParams;

// This is only called if the object MAKES references to other things.
#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult Cal3DObject::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                        PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult Cal3DObject::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                        PartID &partID, RefMessage message)
#endif
{
    // FIXME handle these messages
    switch (message)
    {
    case REFMSG_TARGET_DELETED:
        break;
    case REFMSG_NODE_NAMECHANGE:
        break;
    }
    return REF_SUCCEED;
}

RefTargetHandle
Cal3DObject::GetReference(int ind)
{
    if (ind == 0)
        return pblock;

    return NULL;
}

void
Cal3DObject::SetReference(int ind, RefTargetHandle rtarg)
{
    if (ind == 0)
    {
        pblock = (IParamBlock *)rtarg;
    }
}

ObjectState
Cal3DObject::Eval(TimeValue time)
{
    return ObjectState(this);
}

Interval
Cal3DObject::ObjectValidity(TimeValue time)
{
    Interval ivalid;
    ivalid.SetInfinite();
    return ivalid;
}

void
Cal3DObject::GetMat(TimeValue t, INode *inode, ViewExp *vpt, Matrix3 &tm)
{
    tm = inode->GetObjectTM(t);
}

void
Cal3DObject::GetLocalBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                              Box3 &box)
{
    BuildMesh(t); // 000829  --prs.
    box = mesh.getBoundingBox();
}

void
Cal3DObject::GetWorldBoundBox(TimeValue t, INode *inode, ViewExp *vpt,
                              Box3 &box)
{
    Matrix3 tm;
    GetMat(t, inode, vpt, tm);

    GetLocalBoundBox(t, inode, vpt, box);
    int nv = mesh.getNumVerts();
    box = box * tm;
}

void
Cal3DObject::MakeQuad(int *f, int a, int b, int c, int d, int vab, int vbc, int vcd, int vda)
{
    mesh.faces[*f].setVerts(a, b, c); // back Face
    mesh.faces[*f].setEdgeVisFlags(vab, vbc, 0);
    mesh.faces[(*f)++].setSmGroup(0);

    mesh.faces[*f].setVerts(c, d, a);
    mesh.faces[*f].setEdgeVisFlags(vcd, vda, 0);
    mesh.faces[(*f)++].setSmGroup(0);
}

void
Cal3DObject::BuildMesh(TimeValue t)
{

    pblock->GetValue(PB_CAL_SIZE, t, scale, FOREVER);
    float r = scale,
          r2 = r / 3.0f;
    if (coreHelper)
    {
        int animI, actionI;
        pblock->GetValue(PB_CAL_ANIM, t, animI, FOREVER);
        pblock->GetValue(PB_CAL_ACTION, t, actionI, FOREVER);
        coreHelper->setState(animI, 0.0);
        coreHelper->buildMesh(mesh, scale, t);

        // mesh.InvalidateStrips();
        //mesh.BuildStripsAndEdges();
    }
    else
    {

        mesh.setNumVerts(28);
        mesh.setNumFaces(52);

        int v = 0;
        mesh.setVert(v++, Point3(-r2, r2, r2)); //0 -- back of center cube of the plus
        mesh.setVert(v++, Point3(-r2, r2, -r2));
        mesh.setVert(v++, Point3(r2, r2, -r2));
        mesh.setVert(v++, Point3(r2, r2, r2));

        mesh.setVert(v++, Point3(-r2, -r2, r2)); //4 -- front of center cube of the plus
        mesh.setVert(v++, Point3(-r2, -r2, -r2));
        mesh.setVert(v++, Point3(r2, -r2, -r2));
        mesh.setVert(v++, Point3(r2, -r2, r2));

        mesh.setVert(v++, Point3(-r2, -r, r2)); //8 -- front of the plus
        mesh.setVert(v++, Point3(-r2, -r, -r2));
        mesh.setVert(v++, Point3(r2, -r, -r2));
        mesh.setVert(v++, Point3(r2, -r, r2));

        mesh.setVert(v++, Point3(-r, r2, r2)); //12 -- left end
        mesh.setVert(v++, Point3(-r, r2, -r2));
        mesh.setVert(v++, Point3(-r, -r2, -r2));
        mesh.setVert(v++, Point3(-r, -r2, r2));

        mesh.setVert(v++, Point3(r, r2, r2)); //16 -- right end
        mesh.setVert(v++, Point3(r, r2, -r2));
        mesh.setVert(v++, Point3(r, -r2, -r2));
        mesh.setVert(v++, Point3(r, -r2, r2));

        mesh.setVert(v++, Point3(-r2, r2, r)); //20 -- top end
        mesh.setVert(v++, Point3(r2, r2, r));
        mesh.setVert(v++, Point3(r2, -r2, r));
        mesh.setVert(v++, Point3(-r2, -r2, r));

        mesh.setVert(v++, Point3(-r, r, -r)); //24 -- bottom end
        mesh.setVert(v++, Point3(r, r, -r));
        mesh.setVert(v++, Point3(r, -r, -r));
        mesh.setVert(v++, Point3(-r, -r, -r));

        /* Now the Faces */
        int f = 0;
        // TOP
        MakeQuad(&f, 23, 22, 21, 20, 1, 1, 1, 1); // Top
        MakeQuad(&f, 7, 22, 23, 4, 1, 0, 0, 1); // Front
        MakeQuad(&f, 3, 21, 22, 7, 1, 0, 0, 1); // Right
        MakeQuad(&f, 0, 20, 21, 3, 1, 0, 0, 0); // Back
        MakeQuad(&f, 4, 23, 20, 0, 1, 0, 0, 1); // Left

        // FRONT
        MakeQuad(&f, 8, 9, 10, 11, 1, 1, 1, 1); // End
        MakeQuad(&f, 4, 8, 11, 7, 1, 0, 0, 0); // Top
        MakeQuad(&f, 7, 11, 10, 6, 1, 0, 0, 1); // Right
        MakeQuad(&f, 6, 10, 9, 5, 1, 0, 1, 1); // Bottom
        MakeQuad(&f, 5, 9, 8, 4, 1, 0, 0, 1); // Left

        // LEFT
        MakeQuad(&f, 12, 13, 14, 15, 1, 1, 1, 1); // End
        MakeQuad(&f, 0, 12, 15, 4, 1, 0, 0, 0); // Top
        MakeQuad(&f, 4, 15, 14, 5, 1, 0, 0, 0); // Right
        MakeQuad(&f, 5, 14, 13, 1, 1, 0, 0, 1); // Bottom
        MakeQuad(&f, 1, 13, 12, 0, 1, 0, 0, 0); // Left

        // BACK
        MakeQuad(&f, 3, 2, 1, 0, 0, 1, 0, 0); // Left

        // RIGHT
        MakeQuad(&f, 19, 18, 17, 16, 1, 1, 1, 1); // End
        MakeQuad(&f, 7, 19, 16, 3, 1, 0, 0, 0); // Top
        MakeQuad(&f, 3, 16, 17, 2, 1, 0, 0, 0); // Right
        MakeQuad(&f, 2, 17, 18, 6, 1, 0, 0, 1); // Bottom
        MakeQuad(&f, 6, 18, 19, 7, 1, 0, 0, 0); // Left

        // BASE
        MakeQuad(&f, 24, 25, 26, 27, 1, 1, 1, 1); // Bottom
        MakeQuad(&f, 5, 27, 26, 6, 1, 0, 0, 0); // Front
        MakeQuad(&f, 6, 26, 25, 2, 1, 0, 0, 0); // Right
        MakeQuad(&f, 2, 25, 24, 1, 1, 0, 0, 0); // Back
        MakeQuad(&f, 1, 24, 27, 5, 1, 0, 0, 0); // Left
    }
    mesh.InvalidateGeomCache();
    mesh.EnableEdgeList(1);
    mesh.buildBoundingBox();
}

int
Cal3DObject::Display(TimeValue t, INode *inode, ViewExp *vpt, int flags)
{
    pblock->GetValue(PB_CAL_SIZE, t, scale, FOREVER);
    if (scale <= 0.0)
        return 0;
    BuildMesh(t);
    Matrix3 m;
    GraphicsWindow *gw = vpt->getGW();
    Material *mtl = gw->getMaterial();

    DWORD rlim = gw->getRndLimits();
    gw->setRndLimits(/*GW_WIREFRAME*/ GW_ILLUM | GW_FLAT | GW_Z_BUFFER | GW_BACKCULL);
    GetMat(t, inode, vpt, m);
    gw->setTransform(m);
    if (inode->Selected())
        gw->setColor(LINE_COLOR, 1.0f, 1.0f, 1.0f);
    else if (!inode->IsFrozen())
        gw->setColor(LINE_COLOR, 0.0f, 1.0f, 0.0f);
    mesh.render(gw, mtl, NULL, COMP_ALL);
    gw->setRndLimits(rlim);
    return (0);
}

int
Cal3DObject::HitTest(TimeValue t, INode *inode, int type, int crossing,
                     int flags, IPoint2 *p, ViewExp *vpt)
{
    HitRegion hitRegion;
    DWORD savedLimits;
    int res = FALSE;
    Matrix3 m;
    GraphicsWindow *gw = vpt->getGW();
    Material *mtl = gw->getMaterial();
    MakeHitRegion(hitRegion, type, crossing, 4, p);
    gw->setRndLimits(((savedLimits = gw->getRndLimits()) | GW_PICK) & ~GW_ILLUM);
    GetMat(t, inode, vpt, m);
    gw->setTransform(m);
    gw->clearHitCode();
    if (mesh.select(gw, mtl, &hitRegion, flags & HIT_ABORTONHIT))
        return TRUE;
    gw->setRndLimits(savedLimits);
    return res;
}

class Cal3DCreateCallBack : public CreateMouseCallBack
{
private:
    IPoint2 sp0;
    Point3 p0;
    Cal3DObject *Cal3D_Object;

public:
    int proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m,
             Matrix3 &mat);
    void SetObj(Cal3DObject *obj) { Cal3D_Object = obj; }
};

int
Cal3DCreateCallBack::proc(ViewExp *vpt, int msg, int point, int flags, IPoint2 m, Matrix3 &mat)
{
    Point3 p1, center;

    switch (msg)
    {
    case MOUSE_POINT:
    case MOUSE_MOVE:
        switch (point)
        {
        case 0: // only happens with MOUSE_POINT msg
            sp0 = m;
            p0 = vpt->SnapPoint(m, m, NULL, SNAP_IN_PLANE);
            mat.SetTrans(p0);

            if (msg == MOUSE_POINT)
            {
                return CREATE_STOP;
            }
            break;
        }
        break;
    case MOUSE_ABORT:
        return CREATE_ABORT;
    }

    return TRUE;
}
// A single instance of the callback object.
static Cal3DCreateCallBack Cal3DCreateCB;

// This method allows MAX to access and call our proc method to
// handle the user input.
CreateMouseCallBack *
Cal3DObject::GetCreateMouseCallBack()
{
    Cal3DCreateCB.SetObj(this);
    return (&Cal3DCreateCB);
}

RefTargetHandle
Cal3DObject::Clone(RemapDir &remap)
{

    Cal3DObject *co = new Cal3DObject();
    co->scale = scale;
    co->cal3d_cfg = cal3d_cfg;
    co->coreHelper = coreHelper;

    co->ReplaceReference(0, pblock->Clone(remap));
    BaseClone(this, co, remap);
    return co;
}

// IO
#define CAL3D_URL_CHUNK 0xacb1

IOResult
Cal3DObject::Save(ISave *isave)
{

    isave->BeginChunk(CAL3D_URL_CHUNK);
#ifdef _UNICODE
    isave->WriteWString(cal3d_cfg.data());
#else
    isave->WriteCString(cal3d_cfg.data());
#endif
    isave->EndChunk();

    return IO_OK;
}

IOResult
Cal3DObject::Load(ILoad *iload)
{
    IOResult res;

    while (IO_OK == (res = iload->OpenChunk()))
    {
        switch (iload->CurChunkID())
        {
        case CAL3D_URL_CHUNK:
        {
            char *n;
#ifdef _UNICODE
            iload->ReadWStringChunk(&n);
#else
            iload->ReadCStringChunk(&n);
#endif
            setURL(n);
            break;
        }
        }
        iload->CloseChunk();
        if (res != IO_OK)
            return res;
    }
    return IO_OK;
}

#endif