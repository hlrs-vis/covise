/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "CSV.h"

#include <util/unixcompat.h>

CSVPlugin *CSVPlugin::plugin = NULL;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeCSV(scene);
}

// Define the built in VrmlNodeType:: "CSV" fields

VrmlNodeType *VrmlNodeCSV::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("CSV", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addEventOut("floats_changed", VrmlField::MFFLOAT);
    t->addEventOut("numRows", VrmlField::SFINT32);
    t->addEventOut("numColumns", VrmlField::SFINT32);
    t->addExposedField("row", VrmlField::SFINT32);
    t->addExposedField("fileName", VrmlField::SFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeCSV::nodeType() const
{
    return defineType(0);
}

VrmlNodeCSV::VrmlNodeCSV(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_enabled(true)
{
    changedFile = false;
    setModified();
}

VrmlNodeCSV::VrmlNodeCSV(const VrmlNodeCSV &n)
    : VrmlNodeChild(n.d_scene)
    , d_enabled(n.d_enabled)
{
    changedFile = false;
    setModified();
}

VrmlNodeCSV::~VrmlNodeCSV()
{
}

VrmlNode *VrmlNodeCSV::cloneMe() const
{
    return new VrmlNodeCSV(*this);
}

VrmlNodeCSV *VrmlNodeCSV::toCSV() const
{
    return (VrmlNodeCSV *)this;
}

ostream &VrmlNodeCSV::printFields(ostream &os, int indent)
{
    if (!d_enabled.get())
        PRINT_FIELD(enabled);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeCSV::setField(const char *fieldName,
                               const VrmlField &fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(row, SFInt)
    else if
        TRY_FIELD(fileName, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
    if (strcmp(fieldName, "fileName") == 0)
    {
        loadFile(fieldValue.toSFString()->get());
    }
    else if (strcmp(fieldName, "row") == 0)
    {
        double timeStamp = System::the->time();
        if (d_row.get() < rows.size())
        {
            d_floats.set(d_numColumns.get(), rows[d_row.get()]);
            eventOut(timeStamp, "floats_changed", d_floats);
        }
    }
}

const VrmlField *VrmlNodeCSV::getField(const char *fieldName)
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "floats_changed") == 0)
        return &d_floats;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
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
    fgets(buf, lineSize, fp);
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
        fgets(buf, lineSize, fp);

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


CSVPlugin::CSVPlugin()
{
    fprintf(stderr, "CSVPlugin::CSVPlugin\n");
    plugin = this;

}

// this is called if the plugin is removed at runtime
CSVPlugin::~CSVPlugin()
{
    fprintf(stderr, "CSVPlugin::~CSVPlugin\n");

}

bool CSVPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeCSV::defineType());

    return true;
}

bool
CSVPlugin::update()
{
	return false;
}

COVERPLUGIN(CSVPlugin)
