/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "CSV.h"

#include <util/unixcompat.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>
#include <cover/coVRTui.h>
#include <iostream>
#include <string>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace boost;

CSVPlugin *CSVPlugin::plugin = NULL;

void VrmlNodeCSV::initFields(VrmlNodeCSV *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("enabled", node->d_enabled),
                     exposedField("row", node->d_row, [node](auto f){
                        double timeStamp = System::the->time();
                        if (node->d_row.get() < node->rows.size())
                        {
                            node->d_floats.set(node->d_numColumns.get(), node->rows[node->d_row.get()]);
                            node->eventOut(timeStamp, "floats_changed", node->d_floats);
                        }
                     }),
                     exposedField("fileName", node->d_fileName, [node](auto f){
                        node->loadFile(node->d_fileName.get());
                     }),
                     exposedField("labelFileName", node->d_labelFileName, [node](auto f){
                        node->loadLabelFile(node->d_labelFileName.get());
                     }),
                     exposedField("gpsFileName", node->d_gpsFileName, [node](auto f){
                        node->loadGPSFile(node->d_gpsFileName.get());
                     }));

    if (t)
    {
        t->addEventOut("floats_changed", VrmlField::MFFLOAT);
        t->addEventOut("numRows", VrmlField::SFINT32);
        t->addEventOut("numColumns", VrmlField::SFINT32);
    }
}

const char *VrmlNodeCSV::name()
{
    return "CSV";
}

VrmlNodeCSV::VrmlNodeCSV(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_enabled(true)
{
    changedFile = false;
    setModified();
    CSVPlugin::plugin->CSVNode = this;
}

VrmlNodeCSV::VrmlNodeCSV(const VrmlNodeCSV &n)
    : VrmlNodeChild(n)
    , d_enabled(n.d_enabled)
{
    changedFile = false;
    setModified();
    CSVPlugin::plugin->CSVNode = this;
}

VrmlNodeCSV *VrmlNodeCSV::toCSV() const
{
    return (VrmlNodeCSV *)this;
}

void VrmlNodeCSV::eventIn(double timeStamp,
                              const char *eventName,
                              const VrmlField *fieldValue)
{

    // Check exposedFields
    //else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    if (strcmp(eventName, "row") == 0)
    {
        double timeStamp = System::the->time();
        if (d_row.get() < rows.size())
        {
            d_floats.set(d_numColumns.get(), rows[d_row.get()]);
            eventOut(timeStamp, "floats_changed", d_floats);
        }
    }

}

void VrmlNodeCSV::render(Viewer *)
{
    if (!d_enabled.get())
        return;
    if (changedFile)
    {
        double timeStamp = System::the->time();
        d_numColumns.set(numColumns);
        eventOut(timeStamp, "numColumns", d_numColumns);
        d_numRows.set(rows.size());
        eventOut(timeStamp, "numRows", d_numRows);
        if (rows.size() > 0)
        {
            d_floats.set(numColumns, rows[0]);
            eventOut(timeStamp, "floats_changed", d_floats);
        }
        changedFile = false;
    }
    clearModified();
    //double timeStamp = System::the->time();
}

bool VrmlNodeCSV::loadFile(const std::string &fileName)
{
    FILE *fp = fopen(fileName.c_str(), "r");
    if (fp == NULL)
        return false;
    const int lineSize = 1000;
    const int maxFloats = 100;
    char buf[lineSize];
    char buf2[lineSize];
    float tmpData[maxFloats];
    numColumns = -1;
    RowCount = 0;
    if (!fgets(buf, lineSize, fp))
        return false;
    char *c = buf;
    while (*c != '\0')
    {
        if (*c == ',')
            *c = '.';
        c++;
    }
    strcpy(buf2, buf);
    char *cbuf;
    if ((cbuf = strtok(buf, "\t ;")) != NULL)
    {
        sscanf(cbuf, "%f", &tmpData[0]);
    }
    else
    {
        fprintf(stderr, "Error parsing line %d", RowCount + 1);
    }

    int ii = 1;
    while ((cbuf = strtok(NULL, "\t ;")) != NULL)
    {
        sscanf(cbuf, "%f", &tmpData[ii]);
        ii++;
    }
    numColumns = ii;
    for (auto it = rows.begin(); it != rows.end(); it++)
        delete[] * it;
    rows.clear();
    rows.reserve(10000);

    strcpy(buf, buf2);
    do
    {
        float *fdata = new float[numColumns];
        char *cbuf;
        if ((cbuf = strtok(buf, "\t ;")) != NULL)
        {
            sscanf(cbuf, "%f", &fdata[0]);
        }
        else
        {
            fprintf(stderr, "Error parsing line %d", RowCount + 1);
        }

        int i = 1;
        while ((cbuf = strtok(NULL, "\t ;")) != NULL)
        {
            sscanf(cbuf, "%f", &fdata[i]);
            i++;
        }
        if (i != numColumns)
        {
            fprintf(stderr, "Error parsing line %d", RowCount + 1);
            break;
        }
        rows.push_back(fdata);
        if (!fgets(buf, lineSize, fp))
            break;

        char *c = buf;
        while (*c != '\0')
        {
            if (*c == ',')
                *c = '.';
            c++;
        }
    } while (!feof(fp));
    changedFile = true;
    return true;
}


bool VrmlNodeCSV::loadLabelFile(const std::string &fileName)
{
    FILE *fp = fopen(fileName.c_str(), "r");
    if (fp == NULL)
        return false;
    const int lineSize = 1000;
    char buf[lineSize];
    if (!fgets(buf, lineSize, fp))
        return false;
    MenuLabel = buf;
    MenuLabel.erase(MenuLabel.find_last_not_of(" \n\r\t") + 1);
    int RowCount = 1;
    while(!feof(fp))
    {
        char *cbuf=NULL;
        if (!fgets(buf, lineSize, fp))
            break;
        int64_t startTime;
        int64_t stopTime;
        std::string label;
        if ((cbuf = strtok(buf, "\t;")) != NULL)
        {
            label = cbuf;
        }
        else
        {
            fprintf(stderr, "Error parsing line %d", RowCount);
        }
        if ((cbuf = strtok(NULL, "\t;")) != NULL)
        {
            startTime = strtoll(cbuf, NULL, 10);
        }
        else
        {
            fprintf(stderr, "Error parsing line %d", RowCount);
        }
        if ((cbuf = strtok(NULL, "\t;")) != NULL)
        {
            stopTime = strtoll(cbuf, NULL, 10);
        }
        else
        {
            fprintf(stderr, "Error parsing line %d", RowCount);
        }
        CSVPlugin::plugin->labels.push_back(LabelInfo(label, startTime, stopTime));
        RowCount++;
    }
    changedLabelFile = true;
    CSVPlugin::plugin->createMenu(MenuLabel);
    return true;
}

bool VrmlNodeCSV::loadGPSFile(const std::string &fileName)
{
    FILE *fp = fopen(fileName.c_str(), "r");
    if (fp == NULL)
        return false;
    const int lineSize = 1000;
    char buf[lineSize];
    int RowCount = 1;
    while (!feof(fp))
    {
        char *cbuf = NULL;
        if (!fgets(buf, lineSize, fp))
            break;
        std::string line(buf);
        std::replace(line.begin(), line.end(), ',', '.'); // replace all 'x' to 'y'
        gpsData gpsd;
        char_separator<char> sep(";");
        tokenizer<char_separator<char>> tokens(line, sep);
        auto tok = tokens.begin();
        gpsd.timestamp = std::strtoull(tok->c_str(),NULL,10);
        tok++;
        gpsd.lat = std::strtof(tok->c_str(), NULL);
        tok++;
        gpsd.lon = std::strtof(tok->c_str(), NULL);
        tok++;
        gpsd.alt = std::strtof(tok->c_str(), NULL);
        tok++;
        gpsd.velocity = std::strtof(tok->c_str(), NULL);
        path.push_back(gpsd);
    }
    float minHeight = 10000000;
    float maxHeight = -10000000;
    for (auto gpsd = path.begin(); gpsd != path.end();gpsd++)
    {
        CSVPlugin::plugin->tuiEarthMap->addPathNode(gpsd->lat, gpsd->lon);
	if(gpsd->alt > maxHeight)
	    maxHeight = gpsd->alt;
	if(gpsd->alt < minHeight)
	    minHeight = gpsd->alt;
    }
    
    CSVPlugin::plugin->tuiEarthMap->setMinMax(minHeight,maxHeight);
    CSVPlugin::plugin->tuiEarthMap->updatePath();
    return true;
}

void CSVPlugin::createMenu(const std::string &label)
{
    labelMenu = new ui::Menu(label, CSVPlugin::plugin);
    if (labels.size() > 0)
        currentLabel = new ui::Label(labelMenu, labels[0].label);
    currentLabelNumber = 0;
}
void CSVPlugin::updateLabel(int64_t ts)
{
    double seconds = ts / 60.0;
    if (CSVNode)
    {
        for (int i = 0; i < CSVNode->path.size(); i++)
        {
            if (CSVNode->path[i].timestamp - CSVNode->path[0].timestamp > seconds*1000.0)
            {
                tuiEarthMap->setPosition(CSVNode->path[i].lat, CSVNode->path[i].lon, CSVNode->path[i].alt);
                break;
            }
        }
    }
    if (labels.size()>currentLabelNumber && labels[currentLabelNumber].start < seconds && labels[currentLabelNumber].stop > seconds)
        return;
    for (uint i = 0; i < labels.size(); i++)
    {
        if (labels[i].start < seconds && labels[i].stop > seconds)
        {
            currentLabelNumber = i;
            currentLabel->setText(labels[i].label);
            break;
        }
    }
}

CSVPlugin::CSVPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("CSVPlugin", cover->ui)
{
    fprintf(stderr, "CSVPlugin::CSVPlugin\n");
    plugin = this;
    currentLabelNumber=0;
    CSVNode = NULL;


    CSVTab = new coTUITab("CSV", coVRTui::instance()->mainFolder->getID());
    CSVTab->setPos(0, 0);
    tuiEarthMap = new coTUIEarthMap("earthtest", CSVTab->getID());
    tuiEarthMap->setPos(4, 0);

}

// this is called if the plugin is removed at runtime
CSVPlugin::~CSVPlugin()
{
    fprintf(stderr, "CSVPlugin::~CSVPlugin\n");

}

bool CSVPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeCSV>());

    return true;
}

bool
CSVPlugin::update()
{
	return false;
}

void CSVPlugin::setTimestep(int t)
{
    updateLabel(t);
}

COVERPLUGIN(CSVPlugin)
