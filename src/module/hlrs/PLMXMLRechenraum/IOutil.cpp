/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "include/IOutil.h"
#include <Parser/PLMXMLParser.h>
//#include <boost/filesystem.hpp>

#define MAX_CUBES 64
//using namespace std;

/**
 *
 * @param xml_file
 *
 * @param positionen
 * @param dimensionen
 *
 */
bool readPLM(const char *xml_file, double (*positionen)[MAX_CUBES][3],
             double (*dimensionen)[MAX_CUBES][3], double (*bounds)[3],
             double (*power)[MAX_CUBES], string *dir)
{
    //  osg::ref_ptr<osg::Group> wurzel(new osg::Group());

    DOMDocument *document;

    PLMXMLParser myParser;

    // relevant keywords of plmxml - file
    XMLCh *TAG_transform = XMLString::transcode("Transform");
    XMLCh *TAG_UserData = XMLString::transcode("UserData");
    XMLCh *TAG_UserValue = XMLString::transcode("UserValue");
    XMLCh *TAG_value = XMLString::transcode("value");
    XMLCh *TAG_values = XMLString::transcode("values");
    XMLCh *TAG_title = XMLString::transcode("title");
    XMLCh *TAG_ID = XMLString::transcode("id");
    XMLCh *TAG_Dimension = XMLString::transcode("Dimensions");
    XMLCh *TAG_RevisionView = XMLString::transcode("ProductRevisionView");
    XMLCh *TAG_Bound = XMLString::transcode("Bound");
    XMLCh *TAG_Name = XMLString::transcode("name");
    XMLCh *TAG_Room = XMLString::transcode("Room");
    XMLCh *TAG_room = XMLString::transcode("room");
    XMLCh *TAG_Power = XMLString::transcode("Power");
    XMLCh *TAG_CurrentPowerUsage = XMLString::transcode("CurrentPowerUsage");
    XMLCh *TAG_Rack = XMLString::transcode("Rack");
    XMLCh *TAG_FlowDirection = XMLString::transcode("FlowDirection");
    XMLCh *TAG_ProductInstance = XMLString::transcode("ProductInstance");

    //   myParser->parse(xml_file, wurzel);

    document = myParser.parse(xml_file, document);
    if (document == NULL)
    {
        cerr << "PLMXML-File not found!" << endl;
        return false;
    }

    //  cout << XMLString::transcode(document->getTextContent()) << endl;

    //all categories
    DOMNodeList *listTransform = document->getElementsByTagName(TAG_transform);
    //all instances
    DOMNodeList *listRacks = document->getElementsByTagName(TAG_Bound);
    // all power values
    DOMNodeList *listUserValue = document->getElementsByTagName(TAG_UserValue);

    double positionen2[MAX_CUBES][3] = { 0 };
    double dimensionen2[MAX_CUBES][3] = { 0 };

    double power2[MAX_CUBES] = { 0 };

    double bounds2[3] = { 0 };

    vector<string> dir2(MAX_CUBES, "0");
    //	char *dir2[MAX_CUBES] = { };

    int angle[MAX_CUBES] = {};

    //  vector<double[3]>* posi;
    //  vector<double[3]>* dime;

    /**
	 * new way of processing
	 */

    //	DOMElement *startNode = document->getDocumentElement();
    ////find Node to Begin
    //	while ((XMLString::compareString(startNode->getNodeName(),
    //			TAG_ProductInstance)) != 0
    //			&& (XMLString::compareString(startNode->getNodeName(),
    //					TAG_RevisionView))) {
    //		DOMNode *childNode = startNode->getFirstChild();
    //		while (childNode->getNodeType() != DOMNode::ELEMENT_NODE) {
    //			childNode = childNode->getNextSibling();
    //		}
    //		startNode = dynamic_cast<DOMElement*>(childNode);
    //	}
    //	cout << "Node " << XMLString::transcode(startNode->getNodeName())
    //			<< " has parent: "
    //			<< XMLString::transcode(startNode->getParentNode()->getNodeName())
    //			<< endl;
    //
    //	DOMNodeList *startNodeList = startNode->getChildNodes();
    int iR = 0; //iterator for racks

    //------------------------old way below
    for (int i = 0; i < listTransform->getLength(); i++)
    {
        DOMElement *element = dynamic_cast<DOMElement *>(listTransform->item(i));
        DOMElement *parent = dynamic_cast<DOMElement *>(element->getParentNode());
        string transformID = XMLString::transcode(
            element->getAttribute(TAG_ID));
        //		cerr << transformID << endl;
        transformID.erase(0, 2);
        //      cout << "---------------------------------------" << endl;

        string parentstr = XMLString::transcode(parent->getAttribute(TAG_Name));
        //		cout << transformID << " parent:" << parentstr << endl;
        // bei Abfrage auch nach id schauen?
        if (parent->hasAttribute(TAG_Name)
            && (strstr(parentstr.c_str(), "Rack") != NULL))
        {

            for (int ii = 0; ii < listRacks->getLength(); ii++)
            {
                DOMElement *elementRev = dynamic_cast<DOMElement *>(listRacks->item(ii));
                string racksID = XMLString::transcode(
                    elementRev->getAttribute(TAG_ID));
                //				transformID.erase(0, 2);

                DOMElement *revParent = dynamic_cast<DOMElement *>(elementRev->getParentNode());
                string parentName = XMLString::transcode(
                    revParent->getAttribute(TAG_Name));

                if ((strstr(racksID.c_str(), transformID.c_str()) != NULL)
                    && (strstr(parentName.c_str(), "Rack")))
                {
                    //					cout << racksID << " " << transformID << endl;
                    //
                    //					cout << "---------------------------------------" << endl;

                    char *tempstring = XMLString::transcode(
                        elementRev->getAttribute(TAG_values));
                    char *posstring = XMLString::transcode(
                        element->getTextContent());
                    cout << ".";
                    //					cout << tempstring << " @ " << posstring << endl;
                    double mat[16];
                    if (sscanf(posstring,
                               "%lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",
                               &mat[0], &mat[1], &mat[2], &mat[3], &mat[4],
                               &mat[5], &mat[6], &mat[7], &mat[8], &mat[9],
                               &mat[10], &mat[11], &mat[12], &mat[13], &mat[14],
                               &mat[15]) == 16)
                    {
                        positionen2[iR][0] = mat[12];
                        positionen2[iR][1] = mat[13];
                        positionen2[iR][2] = mat[14];

                        if (mat[0] > 0)
                            angle[iR] = 0;
                        if (mat[1] < 0)
                            angle[iR] = 90;
                        if (mat[0] < 0)
                            angle[iR] = 180;
                        if (mat[1] > 0)
                            angle[iR] = 270;
                    }
                    double n[6];
                    if (sscanf(tempstring, "%lg %lg %lg %lg %lg %lg", &n[0],
                               &n[1], &n[2], &n[3], &n[4], &n[5]) == 6)
                    {
                        dimensionen2[iR][0] = 2.0 * n[3];
                        dimensionen2[iR][1] = 2.0 * n[4];
                        dimensionen2[iR][2] = 2.0 * n[5];
                    }
                    break;
                }
            }
            for (int ii = 0; ii < listUserValue->getLength(); ii++)
            {
                DOMElement *eleUserValue = dynamic_cast<DOMElement *>(listUserValue->item(ii));
                DOMElement *userValueParent = dynamic_cast<DOMElement *>(eleUserValue->getParentNode());
                string parentID = XMLString::transcode(
                    userValueParent->getAttribute(TAG_ID));
                //				cerr
                //						<< XMLString::transcode(
                //								eleUserValue->getAttribute(TAG_title)) << endl;
                //				char *cstring = new char;

                if (strstr(parentID.c_str(), transformID.c_str()) != NULL)
                {
                    if ((XMLString::compareString(
                             eleUserValue->getAttribute(TAG_title), TAG_Power)
                         == 0)
                        || (XMLString::compareString(
                               eleUserValue->getAttribute(TAG_title),
                               TAG_CurrentPowerUsage)) == 0)
                    {

                        char *powerValue = XMLString::transcode(
                            eleUserValue->getAttribute(TAG_value));
                        //						cout << parentID << "--> " << powerValue << "[W]"
                        //								<< endl;

                        double n;
                        if (sscanf(powerValue, "%lg", &n) == 1)
                            power2[iR] = n;
                        //						break;
                    }
                    if (strstr(
                            XMLString::transcode(
                                eleUserValue->getAttribute(TAG_title)),
                            "FlowDirection") != NULL)
                    {

                        char *dirValue = XMLString::transcode(
                            eleUserValue->getAttribute(TAG_value));
                        //						cout << parentID << " " << dirValue << endl;
                        dir[iR] = dirValue;
                        //						if (dir->size() != iR)
                        //							fprintf(stderr, "dir Vector mishap\n");
                        //						break;
                    }
                }
            }
            if (power2[iR] != 0 && (dir[iR].empty() || dir[iR] == "0"))
            {
                printf("power: %lg, index: %d, direction: %s\n", power2[iR], iR,
                       dir[iR].c_str());
                printf("Power Value, but no direction!");
                return false;
            }
            iR++;
        }
        else
        {
            //          cout << transformID << " is not a rack" << endl;
        }
    }
    //	cout << endl;

    for (int i = 0; i < MAX_CUBES; i++)
    {
        if (dimensionen2[i][0] > 0 && dimensionen2[i][1] > 0
            && dimensionen2[i][2] > 0)
        {
            //			printf("No: %d size: %lgx%lgx%lg\n", i, dimensionen2[i][0],
            //					dimensionen2[i][1], dimensionen2[i][2]);
            //			printf("location: x: %lg y: %lg z: %lg \n", positionen2[i][0],
            //					positionen2[i][1], positionen2[i][2]);
            //			printf("orientation: %s, angle: %d\n", dir[i].c_str(), angle[i]);

            switch (angle[i])
            {
            case 0: // Not rotated -> Do nothing
                break;
            case 90:
                positionen2[i][1] -= dimensionen2[i][0];
                std::swap(dimensionen2[i][0], dimensionen2[i][1]);
                if (strcmp(dir[i].c_str(), "+x") == 0)
                    dir[i] = "-y";
                else if (strcmp(dir[i].c_str(), "+y") == 0)
                    dir[i] = "+x";
                else if (strcmp(dir[i].c_str(), "-x") == 0)
                    dir[i] = "+y";
                else if (strcmp(dir[i].c_str(), "-y") == 0)
                    dir[i] = "-x";
                else
                    printf(
                        "Invalid direction identifier, valid values are \"+x\",\"-y\", etc\n");
                break;
            case 180:
                positionen2[i][0] -= dimensionen2[i][0];
                positionen2[i][1] -= dimensionen2[i][1];
                if (strncmp(dir[i].c_str(), "+", 1) == 0)
                    dir[i].replace(0, 1, "-");
                else if (strncmp(dir[i].c_str(), "-", 1) == 0)
                    dir[i].replace(0, 1, "+");
                else
                    printf(
                        "Invalid direction identifier, valid values are \"+x\",\"-y\", etc\n");
                break;
            case 270:
                positionen2[i][0] -= dimensionen2[i][1];
                std::swap(dimensionen2[i][0], dimensionen2[i][1]);
                if (strcmp(dir[i].c_str(), "+x") == 0)
                    dir[i] = "+y";
                else if (strcmp(dir[i].c_str(), "+y") == 0)
                    dir[i] = "-x";
                else if (strcmp(dir[i].c_str(), "-x") == 0)
                    dir[i] = "-y";
                else if (strcmp(dir[i].c_str(), "-y") == 0)
                    dir[i] = "+x";
                else
                    printf(
                        "Invalid direction identifier, valid values are \"+x\",\"-y\", etcs\n");
                break;
            default:
                printf("Faulty angle provided!");
            }
            //			printf("No: %d size: %lgx%lgx%lg\n", i, dimensionen2[i][0],
            //					dimensionen2[i][1], dimensionen2[i][2]);
            //			printf("location: x: %lg y: %lg z: %lg \n", positionen2[i][0],
            //					positionen2[i][1], positionen2[i][2]);
            //			printf("orientation: %s\n---------------------\n", dir[i].c_str());
        }
    }
    //	printf("");
    //----------------------------------------------------------
    memcpy(positionen, &positionen2, sizeof(positionen2));
    memcpy(dimensionen, &dimensionen2, sizeof(dimensionen2));
    memcpy(power, &power2, sizeof(power2));

    //-----------------------------------------------------------------------------
    DOMNodeList *room = document->getElementsByTagName(TAG_Bound);
    for (int i = 0; i < room->getLength(); i++)
    {

        DOMElement *element = dynamic_cast<DOMElement *>(room->item(i));
        DOMElement *eleParent = dynamic_cast<DOMElement *>(element->getParentNode());
        if (eleParent->hasAttribute(TAG_Name))
        {

            string name = XMLString::transcode(
                eleParent->getAttribute(TAG_Name));
            if ((strstr(name.c_str(), XMLString::transcode(TAG_Room)) != NULL)
                || (strstr(name.c_str(), XMLString::transcode(TAG_room))))
            {
                char *bounds = XMLString::transcode(
                    element->getAttribute(TAG_values));

                double b[6];
                if (sscanf(bounds, "%lg %lg %lg %lg %lg %lg ", &b[0], &b[1],
                           &b[2], &b[3], &b[4], &b[5]))
                {
                    bounds2[0] = b[3];
                    bounds2[1] = b[4];
                    bounds2[2] = b[5];
                }
                break;
            }
        }
    }
    memcpy(bounds, &bounds2, sizeof(bounds2));
    printf("file-parsing done!\n");
    return true;
}
bool writePLM(char *xml_file)
{
    return true;
}
void addNode(DOMNode *node)
{
}

void delNode(DOMNode *node)
{
}
//double getValue(std::string path, std::string var, covise::Host *dbHost) {
//	double value = -1.;
//	if (dbHost->hasValidAddress() && dbHost->hasValidName()) {
//		covise::SimpleClientConnection *conn =
//				new covise::SimpleClientConnection(dbHost, 9451);
//		if (!conn->is_connected()) {
//			fprintf(stderr, "Could not connect to: %s port: %d\n",
//					dbHost->getName(), 9451);
//			return value;
//		}
//		string command;
//		command = "getLastMetricByMetricName " + path + " " + var + "\n";
//		fprintf(stderr, "%s\n", command.c_str());
//		conn->send(command.c_str(), command.length());
//		char line[1000];
//		line[0] = '\n';
//		line[999] = '\n';
//		int numRead = conn->getSocket()->Read(line, 1000);
//		if (numRead >= 0)
//			line[numRead] = '\0';
//		fprintf(stderr, "%s\n", line);
//		char *valstr = strstr(line, "value=");
//
//		if (valstr)
//			sscanf(valstr + 7, "%lf", &value);
//		fprintf(stderr, "%f\n", (float) value);
//	}
//	return value;
//}
