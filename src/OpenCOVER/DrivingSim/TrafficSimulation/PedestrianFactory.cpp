/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PedestrianFactory.h"

/**
 * Create PedestrianFactory as a singleton
 */
PedestrianFactory *PedestrianFactory::__instance = NULL;
PedestrianFactory *PedestrianFactory::Instance()
{
    if (__instance == NULL)
    {
        __instance = new PedestrianFactory();
    }
    return __instance;
}
void PedestrianFactory::Destroy()
{
    delete __instance;
    __instance = NULL;
}
PedestrianFactory::PedestrianFactory()
    : maximumPeds(-1)
{
    // Set default values (hardcoded, can be overridden in .xodr)
    pedDefaults = PedestrianSettings("", "", "300", "0", // id, name, rangeLOD, debugLvl,
                                     "cally", "0.01", "0.0", // modelFile, scale, heading,
                                     "road", "1", "1", "0.0", "0.0", "1.2", "0.6", // road, lane, dir, sOff, vOff, vel, acc,
                                     "0", "0.0", "1", "0.6", "2", "1.5", "3", "3.0", "-1", "-1"); // animation mapping

    // Create OSG group for pedestrians
    pedestrianGroup = new osg::Group();
    pedestrianGroup->setName("PedestrianSystem");
    opencover::cover->getObjectsRoot()->addChild(pedestrianGroup);
}

PedestrianFactory::~PedestrianFactory()
{
	opencover::cover->getObjectsRoot()->removeChild(pedestrianGroup);
}

/**
 * Delete a given pedestrian from the set
 */
void PedestrianFactory::deletePedestrian(Pedestrian *p)
{
    p->getPedestrianGeometry()->removeFromSceneGraph();
    delete p;
}

/**
 * Load the Cal3D cfg file and return the core model
 */
osgCal::CoreModel *PedestrianFactory::getCoreModel(const std::string &modelFile)
{
    // Determine actual location of Cal3D .cfg file
    std::string tmpModelFile = modelFile;
    // Check that the given file exists and is a .cfg file before opening it (if it doesn't exist, use default location)
    if (tmpModelFile.rfind(".cfg") == std::string::npos || fopen(tmpModelFile.c_str(), "r") == NULL)
    {
        // Doesn't exist, look for the default model in COVISEDIR/data/cal3d
        // Prune at the last slash and drop extension (e.g., ..../skeleton/skeleton.cfg --> skeleton)

        // Erase all characters up to final '/'
        size_t pos = tmpModelFile.find_last_of('/');
        if (pos != std::string::npos)
            tmpModelFile.erase(0, pos);

        // Erase .cfg from end of file (if it's there)
        pos = tmpModelFile.rfind(".cfg");
        if (pos != std::string::npos)
            tmpModelFile.erase(pos, tmpModelFile.length());

        tmpModelFile = std::string(getenv("COVISEDIR")) + std::string("/data/cal3d/") + tmpModelFile + std::string("/") + tmpModelFile + std::string(".cfg");
    }

    // If some core models have already been loaded, we need to perform a search
    if (coreModelMap.size() > 0)
    {
        // Check if this core model is already loaded
        std::map<std::string, osg::ref_ptr<osgCal::CoreModel> >::iterator cmIt = coreModelMap.find(tmpModelFile);

        // It it is found in the map
        if (cmIt != coreModelMap.end())
        {
            // Return it
            return cmIt->second.get();
        }
    }

    // Load core model
    osg::ref_ptr<osgCal::CoreModel> coreModel = new osgCal::CoreModel();
    osg::ref_ptr<osgCal::MeshParameters> meshParams = new osgCal::MeshParameters;
    meshParams->useDepthFirstMesh = false;
    meshParams->software = false;
    coreModel->load(tmpModelFile, meshParams.get());

    // Add it to the map
    coreModelMap.insert(pair<std::string, osg::ref_ptr<osgCal::CoreModel> >(tmpModelFile, coreModel));

    // Report that this core model has been loaded
    fprintf(stderr, "PedestrianFactory::getCoreModel(%s) A new core model has been loaded\n", tmpModelFile.c_str());

    // Return it
    return coreModel.get();
}

/**
 * Create a new pedestrian at a fiddleyard source (use default/template settings and the given start-values)
 */
Pedestrian *PedestrianFactory::createPedestrian(const std::string &name, const std::string &tmpl, const std::string &r, const int l, const int dir, const double sOff, const double vOff, const double vel, const double acc)
{
    // Get the base (default+template) settings
    PedestrianSettings pedSettings(pedDefaults);
    applyTemplateToSettings(tmpl, &pedSettings);

    // Set name and id
    pedSettings.id = name;
    pedSettings.name = name;

    // Override starting position values
    pedSettings.startRoadId = r;
    char laneBuf[8];
    sprintf(laneBuf, "%d", l);
    pedSettings.startLane = std::string(laneBuf);
    char dirBuf[8];
    sprintf(dirBuf, "%d", dir);
    pedSettings.startDir = std::string(dirBuf);
    char sOffBuf[16];
    sprintf(sOffBuf, "%.2f", sOff);
    pedSettings.startSOff = std::string(sOffBuf);
    char vOffBuf[16];
    sprintf(vOffBuf, "%.2f", vOff);
    pedSettings.startVOff = std::string(vOffBuf);
    char velBuf[16];
    sprintf(velBuf, "%.2f", vel);
    pedSettings.startVel = std::string(velBuf);
    char accBuf[16];
    sprintf(accBuf, "%.2f", acc);
    pedSettings.startAcc = std::string(accBuf);

    // Create a new pedestrian with these settings
    return createPedestrian(pedSettings);
}

/**
 * Create a new pedestrian with the given settings
 */
Pedestrian *PedestrianFactory::createPedestrian(PedestrianSettings ps)
{
    // Get starting road
    Road *startRoad = RoadSystem::Instance()->getRoad(ps.startRoadId);

    if (startRoad != NULL && (maxPeds() < 0 || PedestrianManager::Instance()->numPeds() < maxPeds()))
    {
        // Make sure starting position is on the road
        if (atof(ps.startSOff.c_str()) < 0.0 || atof(ps.startSOff.c_str()) > startRoad->getLength())
        {
            fprintf(stderr, " Pedestrian '%s': WARNING position(%s) is not on the road\n", ps.name.c_str(), ps.startSOff.c_str());
        }

        // Get starting lane according to position on the road; check that it's a sidewalk
        LaneSection *laneSec = startRoad->getLaneSection(atof(ps.startSOff.c_str()));
        Vector2D laneCenter = laneSec->getLaneCenter(atoi(ps.startLane.c_str()), atof(ps.startSOff.c_str()));
        int laneNumCheck = laneSec->searchLane(atof(ps.startSOff.c_str()), laneCenter[0]);
        if (laneNumCheck != Lane::NOLANE)
        {
            Lane::LaneType laneType = laneSec->getLane(laneNumCheck)->getLaneType();
            if (laneType == Lane::SIDEWALK)
            {
                PedestrianAnimations anim(atoi(ps.idleIdx.c_str()), atof(ps.idleVel.c_str()),
                                          atoi(ps.slowIdx.c_str()), atof(ps.slowVel.c_str()),
                                          atoi(ps.walkIdx.c_str()), atof(ps.walkVel.c_str()),
                                          atoi(ps.jogIdx.c_str()), atof(ps.jogVel.c_str()),
                                          atoi(ps.lookIdx.c_str()), atoi(ps.waveIdx.c_str()));

                // Create a pedestrian
                Pedestrian *p = new Pedestrian(ps.name, startRoad, atoi(ps.startLane.c_str()), atoi(ps.startDir.c_str()), atof(ps.startSOff.c_str()), atof(ps.startVOff.c_str()), atof(ps.startVel.c_str()), atof(ps.startAcc.c_str()), atof(ps.heading.c_str()), atoi(ps.debugLvl.c_str()), new PedestrianGeometry(ps.name, ps.modelFile, atof(ps.scale.c_str()), atof(ps.rangeLOD.c_str()), anim, pedestrianGroup));
                PedestrianManager::Instance()->addPedestrian(p);

                // Only print debug level if it's greater than 0
                char dbg[15];
                dbg[0] = '\0';
                if (atoi(ps.debugLvl.c_str()) > 0)
                {
                    strcpy(dbg, ")/dbgLvl(");
                    strcat(dbg, ps.debugLvl.c_str());
                }

                // Print pedestrian settings
                if (atoi(ps.debugLvl.c_str()) > 0)
                    fprintf(stderr, " Pedestrian '%s': created on roadId(%s)/lane(%s)/dir(%s)/pos(%s)/vel(%s%s)\n", ps.name.c_str(), ps.startRoadId.c_str(), ps.startLane.c_str(), ps.startDir.c_str(), ps.startSOff.c_str(), ps.startVel.c_str(), dbg);
                return p;
            }
            else
            {
                fprintf(stderr, " Pedestrian '%s': invalid lane type, must start on sidewalk\n", ps.name.c_str());
            }
        }
        else
        {
            fprintf(stderr, " Pedestrian '%s': invalid lane(%s) or position(%s)\n", ps.name.c_str(), ps.startLane.c_str(), ps.startSOff.c_str());
        }
    }
    else if (startRoad == NULL)
    {
        fprintf(stderr, " Pedestrian '%s': invalid road(%s)\n", ps.name.c_str(), ps.startRoadId.c_str());
    }
    else
    {
        fprintf(stderr, "PedestrianFactory::createPedestrian(%s) Not created, maximum pedestrians already active\n", ps.name.c_str());
    }
    return NULL;
}

/**
 * Parse the OpenDRIVE XML file for settings
 */
void PedestrianFactory::parseOpenDrive(xercesc::DOMElement *rootElement, const std::string &xodr)
{
    xodrDir = xodr;

    // Get all child entries of the root element
    xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();

    // Loop over children of root element
    for (int childIndex = 0; childIndex < documentChildrenList->getLength(); childIndex++)
    {
        // Get "<pedestrians>...</pedestrians>" element
        xercesc::DOMElement *pedestriansElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
        if (tagsMatch(pedestriansElement, "pedestrians"))
        {
            // Get spawn range (if set)
            if (getValOfAttr(pedestriansElement, "spawnRange").length() > 0)
                PedestrianManager::Instance()->setSpawnRange(atof(getValOfAttr(pedestriansElement, "spawnRange").c_str()));

            // Get maximum number of pedestrians (if set)
            if (getValOfAttr(pedestriansElement, "maxPeds").length() > 0)
                maximumPeds = atoi(getValOfAttr(pedestriansElement, "maxPeds").c_str());

            // Get reportInterval (if set)
            if (getValOfAttr(pedestriansElement, "reportInterval").length() > 0)
                PedestrianManager::Instance()->setReportInterval(atof(getValOfAttr(pedestriansElement, "reportInterval").c_str()));

            // Get avoidCount (if set)
            if (getValOfAttr(pedestriansElement, "avoidCount").length() > 0)
                PedestrianManager::Instance()->setAvoidCount(atoi(getValOfAttr(pedestriansElement, "avoidCount").c_str()));

            // Get avoidTime (if set)
            if (getValOfAttr(pedestriansElement, "avoidTime").length() > 0)
                PedestrianManager::Instance()->setAvoidTime(atof(getValOfAttr(pedestriansElement, "avoidTime").c_str()));

            // Get movingFiddle (if set)
            if (getValOfAttr(pedestriansElement, "movingFiddle").compare("true") == 0)
                PedestrianManager::Instance()->enableMovingFiddleyards();
            // Get autoFiddle (if set, and movingFiddle is not set)
            else if (getValOfAttr(pedestriansElement, "autoFiddle").compare("true") == 0)
                PedestrianManager::Instance()->generateAutoFiddleyards();

            // Loop over children of "<pedestrians>"
            xercesc::DOMNodeList *pedestriansChildrenList;
            xercesc::DOMElement *pedestriansChildElement;
            pedestriansChildrenList = pedestriansElement->getChildNodes();

            for (int childIndex = 0; childIndex < pedestriansChildrenList->getLength(); childIndex++)
            {
                // Get "<default>...</default>" element
                pedestriansChildElement = dynamic_cast<xercesc::DOMElement *>(pedestriansChildrenList->item(childIndex));

                if (tagsMatch(pedestriansChildElement, "default"))
                {
                    // Get settings
                    parseElementForSettings(pedestriansChildElement, &pedDefaults);
                }
            }

            for (int childIndex = 0; childIndex < pedestriansChildrenList->getLength(); childIndex++)
            {
                // Get "<template>...</template>" elements
                pedestriansChildElement = dynamic_cast<xercesc::DOMElement *>(pedestriansChildrenList->item(childIndex));

                if (tagsMatch(pedestriansChildElement, "template"))
                {
                    // Create empty settings for this template before populating
                    PedestrianSettings pedTemplate;

                    // Get id for this template
                    pedTemplate.id = getValOfAttr(pedestriansChildElement, "id"); // required

                    // Get other settings
                    parseElementForSettings(pedestriansChildElement, &pedTemplate);

                    // Have the template, so store it in the list
                    pedTemplatesList.push_back(pedTemplate);
                }
            }

            for (int childIndex = 0; childIndex < pedestriansChildrenList->getLength(); childIndex++)
            {
                // Get "<ped>...</ped>" elements
                pedestriansChildElement = dynamic_cast<xercesc::DOMElement *>(pedestriansChildrenList->item(childIndex));

                if (tagsMatch(pedestriansChildElement, "ped"))
                {
                    // Get the base (default+template) settings
                    PedestrianSettings pedSettings(pedDefaults);
                    applyTemplateToSettings(getValOfAttr(pedestriansChildElement, "templateId"), &pedSettings);

                    // Get id/name for this pedestrian
                    pedSettings.id = getValOfAttr(pedestriansChildElement, "id"); // required
                    pedSettings.name = getValOfAttr(pedestriansChildElement, "name"); // required

                    // Get other settings
                    parseElementForSettings(pedestriansChildElement, &pedSettings);

                    // Store the settings of this instance in the list
                    pedInstancesList.push_back(pedSettings);

                    // Create a new pedestrian
                    Pedestrian *p = NULL;
                    p = createPedestrian(pedSettings);
                }
            }
        }
    }
}

/**
 * Given a templateId string, apply the associated template's settings to the given settings
 * If the template is not found, do not alter the settings
 */
void PedestrianFactory::applyTemplateToSettings(std::string tmpl, PedestrianSettings *pedSettings)
{
    // Search for any apply the requested template's values
    if (tmpl.length() > 0)
    {
        for (list<PedestrianSettings>::iterator tmpIt = pedTemplatesList.begin(); tmpIt != pedTemplatesList.end(); tmpIt++)
        {
            PedestrianSettings ps = (*tmpIt);
            if (ps.id.compare(tmpl) == 0)
            {
                // This is the requested template
                // Override default settings with template settings, if they were provided
                if (ps.debugLvl.length() > 0)
                    pedSettings->debugLvl = ps.debugLvl;

                if (ps.modelFile.length() > 0)
                    pedSettings->modelFile = ps.modelFile;
                if (ps.scale.length() > 0)
                    pedSettings->scale = ps.scale;
                if (ps.heading.length() > 0)
                    pedSettings->heading = ps.heading;

                if (ps.startRoadId.length() > 0)
                    pedSettings->startRoadId = ps.startRoadId;
                if (ps.startLane.length() > 0)
                    pedSettings->startLane = ps.startLane;
                if (ps.startDir.length() > 0)
                    pedSettings->startDir = ps.startDir;
                if (ps.startSOff.length() > 0)
                    pedSettings->startSOff = ps.startSOff;
                if (ps.startVOff.length() > 0)
                    pedSettings->startVOff = ps.startVOff;
                if (ps.startVel.length() > 0)
                    pedSettings->startVel = ps.startVel;
                if (ps.startAcc.length() > 0)
                    pedSettings->startAcc = ps.startAcc;

                if (ps.idleIdx.length() > 0)
                    pedSettings->idleIdx = ps.idleIdx;
                if (ps.idleVel.length() > 0)
                    pedSettings->idleVel = ps.idleVel;
                if (ps.slowIdx.length() > 0)
                    pedSettings->slowIdx = ps.slowIdx;
                if (ps.slowVel.length() > 0)
                    pedSettings->slowVel = ps.slowVel;
                if (ps.walkIdx.length() > 0)
                    pedSettings->walkIdx = ps.walkIdx;
                if (ps.walkVel.length() > 0)
                    pedSettings->walkVel = ps.walkVel;
                if (ps.jogIdx.length() > 0)
                    pedSettings->jogIdx = ps.jogIdx;
                if (ps.jogVel.length() > 0)
                    pedSettings->jogVel = ps.jogVel;
                if (ps.lookIdx.length() > 0)
                    pedSettings->lookIdx = ps.lookIdx;
                if (ps.waveIdx.length() > 0)
                    pedSettings->waveIdx = ps.waveIdx;

                // And stop looking
                break;
            }
        }
    }
}

/**
 * Given a DOMElement from the OpenDRIVE file, parse the settings contained in its attributes and child elements,
 * and store these values in the given settings
 */
void PedestrianFactory::parseElementForSettings(xercesc::DOMElement *element, PedestrianSettings *pedSettings)
{
    // Get LOD value
    if (hasAttr(element, "rangeLOD"))
        pedSettings->rangeLOD = getValOfAttr(element, "rangeLOD");

    // Get debug value
    if (hasAttr(element, "debugLvl"))
        pedSettings->debugLvl = getValOfAttr(element, "debugLvl");

    // Loop over element's children
    xercesc::DOMNodeList *childrenList = element->getChildNodes();
    xercesc::DOMElement *childElement;
    for (int childIndex = 0; childIndex < childrenList->getLength(); childIndex++)
    {
        // 1. Get "<geometry ... />" element
        childElement = dynamic_cast<xercesc::DOMElement *>(childrenList->item(childIndex));
        if (tagsMatch(childElement, "geometry"))
        {
            if (hasAttr(childElement, "modelFile"))
            {
                std::string tmp = getValOfAttr(childElement, "modelFile");
                if (tmp[0] == '/' || tmp[0] == '\\' || tmp[0] == ':')
                {
                    // Have an absolute path
                    pedSettings->modelFile = tmp;
                }
                else
                {
                    // Have a relative path, check both the .xodr directory and COVISEDIR
                    pedSettings->modelFile = xodrDir + "/" + tmp;
                }

                // Load the core model now, to prevent delays later
                getCoreModel(pedSettings->modelFile);
            }

            // Get model scale
            if (hasAttr(childElement, "scale"))
                pedSettings->scale = getValOfAttr(childElement, "scale");

            // Get heading adjustment
            if (hasAttr(childElement, "heading"))
                pedSettings->heading = getValOfAttr(childElement, "heading");
        }

        // 2. Get "<start ... />" element
        else if (tagsMatch(childElement, "start"))
        {
            // Get starting position settings
            if (hasAttr(childElement, "roadId"))
                pedSettings->startRoadId = getValOfAttr(childElement, "roadId");
            if (hasAttr(childElement, "lane"))
                pedSettings->startLane = getValOfAttr(childElement, "lane");
            if (hasAttr(childElement, "direction"))
                pedSettings->startDir = getValOfAttr(childElement, "direction");
            if (hasAttr(childElement, "sOffset"))
                pedSettings->startSOff = getValOfAttr(childElement, "sOffset");
            if (hasAttr(childElement, "vOffset"))
                pedSettings->startVOff = getValOfAttr(childElement, "vOffset");
            if (hasAttr(childElement, "velocity"))
                pedSettings->startVel = getValOfAttr(childElement, "velocity");
            if (hasAttr(childElement, "acceleration"))
                pedSettings->startAcc = getValOfAttr(childElement, "acceleration");
        }

        // 3. Get "<animations>...</animations>" element
        else if (tagsMatch(childElement, "animations"))
        {
            xercesc::DOMNodeList *animChildrenList = childElement->getChildNodes();
            xercesc::DOMElement *animChildElement;
            for (int childIndex = 0; childIndex < animChildrenList->getLength(); childIndex++)
            {
                animChildElement = dynamic_cast<xercesc::DOMElement *>(animChildrenList->item(childIndex));
                if (tagsMatch(animChildElement, "idle"))
                {
                    if (hasAttr(animChildElement, "index"))
                        pedSettings->idleIdx = getValOfAttr(animChildElement, "index");
                    if (hasAttr(animChildElement, "velocity"))
                        pedSettings->idleVel = getValOfAttr(animChildElement, "velocity");
                }
                else if (tagsMatch(animChildElement, "slow"))
                {
                    if (hasAttr(animChildElement, "index"))
                        pedSettings->slowIdx = getValOfAttr(animChildElement, "index");
                    if (hasAttr(animChildElement, "velocity"))
                        pedSettings->slowVel = getValOfAttr(animChildElement, "velocity");
                }
                else if (tagsMatch(animChildElement, "walk"))
                {
                    if (hasAttr(animChildElement, "index"))
                        pedSettings->walkIdx = getValOfAttr(animChildElement, "index");
                    if (hasAttr(animChildElement, "velocity"))
                        pedSettings->walkVel = getValOfAttr(animChildElement, "velocity");
                }
                else if (tagsMatch(animChildElement, "jog"))
                {
                    if (hasAttr(animChildElement, "index"))
                        pedSettings->jogIdx = getValOfAttr(animChildElement, "index");
                    if (hasAttr(animChildElement, "velocity"))
                        pedSettings->jogVel = getValOfAttr(animChildElement, "velocity");
                }
                else if (tagsMatch(animChildElement, "look"))
                {
                    if (hasAttr(animChildElement, "index"))
                        pedSettings->lookIdx = getValOfAttr(animChildElement, "index");
                }
                else if (tagsMatch(animChildElement, "wave"))
                {
                    if (hasAttr(animChildElement, "index"))
                        pedSettings->waveIdx = getValOfAttr(animChildElement, "index");
                }
            }
        }
    }
}

/**
 * Check that the given element has a particular attribute
 */
bool PedestrianFactory::hasAttr(xercesc::DOMElement *element, const char *attr)
{
    if (element->hasAttribute(xercesc::XMLString::transcode(attr)))
        return true;
    else
        return false;
}

/**
 * Check that the given element exists and whether its tag matches the given string
 */
bool PedestrianFactory::tagsMatch(xercesc::DOMElement *element, const char *tag)
{
    if (element && xercesc::XMLString::compareIString(element->getTagName(), xercesc::XMLString::transcode(tag)) == 0)
        return true;
    else
        return false;
}

/**
 * Get the value of the given element's attribute
 */
std::string PedestrianFactory::getValOfAttr(xercesc::DOMElement *element, const char *attr)
{
    return std::string(xercesc::XMLString::transcode(element->getAttribute(xercesc::XMLString::transcode(attr))));
}
