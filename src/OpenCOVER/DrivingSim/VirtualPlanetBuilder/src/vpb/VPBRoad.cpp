/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <vpb/VPBRoad>
#include <cover/coVRShader.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <direct.h>
#endif
#include <xercesc/util/PlatformUtils.hpp>

VPBRoad::VPBRoad(std::string xodrName)
{
    system = NULL;
    rootElement = NULL;
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &/*toCatch*/)
    {
        // Do your failure processing here
    }
    loadRoadSystem(xodrName);
}
VPBRoad::~VPBRoad()
{
    xercesc::XMLPlatformUtils::Terminate();
}
bool VPBRoad::loadRoadSystem(std::string filename)
{
    std::cerr << "Loading road system!" << std::endl;
    if (system == NULL)
    {
        //Building directory string to xodr file
        xodrDirectory.clear();
        if (filename[0] != '/' && filename[0] != '\\' && (!(filename[1] == ':' && (filename[2] == '/' || filename[2] == '\\'))))
        { // / or backslash or c:/
            char *workingDir = getcwd(NULL, 0);
            xodrDirectory.assign(workingDir);
            free(workingDir);
        }
        size_t lastSlashPos = filename.find_last_of('/');
        size_t lastSlashPos2 = filename.find_last_of('\\');
        if (lastSlashPos != filename.npos && (lastSlashPos2 == filename.npos || lastSlashPos2 < lastSlashPos))
        {
            if (!xodrDirectory.empty())
                xodrDirectory += "/";
            xodrDirectory.append(filename, 0, lastSlashPos);
        }
        if (lastSlashPos2 != filename.npos && (lastSlashPos == filename.npos || lastSlashPos < lastSlashPos2))
        {
            if (!xodrDirectory.empty())
                xodrDirectory += "\\";
            xodrDirectory.append(filename, 0, lastSlashPos2);
        }

        opencover::coVRShaderList::instance();
        system = RoadSystem::Instance();

        xercesc::DOMElement *openDriveElement = getOpenDriveRootElement(filename);
        if (!openDriveElement)
        {
            std::cerr << "No regular xodr file " << filename << " at: " + xodrDirectory << std::endl;
            return false;
        }

        system->parseOpenDrive(openDriveElement);
        if (rootElement)
            this->parseOpenDrive(rootElement);
    }
    return true;
}
void VPBRoad::parseOpenDrive(xercesc::DOMElement *rootElement)
{
	XMLCh *t1 = NULL, *t2 = NULL;
    xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();

    for (int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
    {
        xercesc::DOMElement *sceneryElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
        if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), t1 = xercesc::XMLString::transcode("scenery")) == 0)
        {
            /*

   std::string fileString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("file")));
   std::string vpbString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("vpb")));

   std::vector<BoundingArea> voidBoundingAreaVector;
   std::vector<std::string> shapeFileNameVector;

   xercesc::DOMNodeList* sceneryChildrenList = sceneryElement->getChildNodes();
   xercesc::DOMElement* sceneryChildElement;
   for(unsigned int childIndex=0; childIndex<sceneryChildrenList->getLength(); ++childIndex) {
sceneryChildElement = dynamic_cast<xercesc::DOMElement*>(sceneryChildrenList->item(childIndex));
if(!sceneryChildElement) continue;

if(xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), xercesc::XMLString::transcode("void"))==0) {
double xMin = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("xMin"))));
double yMin = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("yMin"))));
double xMax = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("xMax"))));
double yMax = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("yMax"))));

voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(xMin, yMin),osg::Vec2(xMax, yMax)));
//voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(506426.839,5398055.357),osg::Vec2(508461.865,5399852.0)));
}
else if(xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), xercesc::XMLString::transcode("shape"))==0) {
std::string fileString = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("file")));
shapeFileNameVector.push_back(fileString);
}

}
*/

            /*
         if(!fileString.empty())
         {
            if(!coVRFileManager::instance()->fileExist((xodrDirectory+"/"+fileString).c_str()))
            {
               std::cerr << "\n#\n# file not found: this may lead to a crash! \n#" << endl;
            }
            coVRFileManager::instance()->loadFile((xodrDirectory+"/"+fileString).c_str());
         }

         if(!vpbString.empty())
{
coVRPlugin* roadTerrainPlugin = cover->addPlugin("RoadTerrain");
fprintf(stderr,"loading %s\n",vpbString.c_str());
if(RoadTerrainPlugin::plugin)
{
osg::Vec3d offset(0,0,0);
const RoadSystemHeader& header = RoadSystem::Instance()->getHeader();
offset.set(header.xoffset, header.yoffset, 0.0);
fprintf(stderr,"loading %s offset: %f %f\n",(xodrDirectory+"/"+vpbString).c_str(),offset[0],offset[1]);
RoadTerrainPlugin::plugin->loadTerrain(xodrDirectory+"/"+vpbString,offset, voidBoundingAreaVector, shapeFileNameVector);
}
}*/
        }
        else if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), t2 = xercesc::XMLString::transcode("environment")) == 0)
        {
            /*std::string tessellateRoadsString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("tessellateRoads")));
if(tessellateRoadsString=="false" || tessellateRoadsString=="0") {
   tessellateRoads = false;
}
else {
   tessellateRoads = true;
}

std::string tessellatePathsString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("tessellatePaths")));
if(tessellatePathsString=="false" || tessellatePathsString=="0") {
   tessellatePaths = false;
}
else {
tessellatePaths = true;
}

std::string tessellateBattersString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("tessellateBatters")));
if(tessellateBattersString=="true" ) {
tessellateBatters = true;
}
else {
tessellateBatters = false;
}

std::string tessellateObjectsString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("tessellateObjects")));
if(tessellateObjectsString=="true") {
tessellateObjects = true;
}
else {
tessellateObjects = false;
}*/
        }
		xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
    }
}
xercesc::DOMElement *VPBRoad::getOpenDriveRootElement(std::string filename)
{
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cout << "Error during initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
        return NULL;
    }

    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(filename.c_str());
    }
    catch (...)
    {
        std::cerr << "Couldn't parse OpenDRIVE XML-file " << filename << "!" << std::endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    return rootElement;
}
