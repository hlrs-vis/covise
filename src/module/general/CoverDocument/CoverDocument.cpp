/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Module CoverDocument
//
// This class sends a dummy geometry to COVER and adds a
// attribute:
//
//     DOCUMENT <name> <numFiles> fullpath1 fullpath2 ..
//
// Initial version: 2004-04-29 [dr]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//    we 2004-04-30  Translated to new API
//    mf 2004-04-30  Reads consecutive filenames now

#include <do/coDoPoints.h>
#include "CoverDocument.h"
#include <stdlib.h>
#include <api/coFeedback.h>

CoverDocument::CoverDocument(int argc, char *argv[])
    : coModule(argc, argv, "Cover Document Viewer access module")
{
    // create a valid startup location for the file
    char filebase[1024];

    const char *basePath = getenv("COVISEDIR");
    if (!basePath)
        basePath = ".";
    sprintf(filebase, "%s/nofile.png", basePath);

    // Create parameters and ports
    filenameParam = addFileBrowserParam("Filename", "Name of first file");
    filenameParam->setValue(filebase, "*.png;*.PNG;*.rgb;*.RGB;*.tif;*.tiff");

    docnameParam = addStringParam("Title", "Title of Document in Menu");
    docnameParam->setValue("Document");

    outPort = addOutputPort("Document", "Points", "Dummy Object to carry DOCUMENT attriute");
}

int CoverDocument::compute(const char *)
{
    // File name
    const char *docPath = filenameParam->getValue();
    sendInfo("Document path is [%s]", docPath);

    // Document name
    const char *docName = docnameParam->getValue();
    sendInfo("Document name is [%s]", docName);

    // Document name 2, needed for replacing blanks.
    char *docName2 = new char[strlen(docName) + 1];
    strcpy(docName2, docName);

    // replacing blanks with underscores...
    for (int i = 0; i < strlen(docName2); i++)
    {

        if (docName2[i] == ' ')
            docName2[i] = '_';
    }
    //   sendInfo("Document Name2 is [%s]",docName2);

    // create 0-sized Points and check it
    coDoPoints *point = new coDoPoints(outPort->getObjName(), 0);
    if (!point->objectOk())
    {
        Covise::sendError("ERROR: Could not create Point Object");
        return STOP_PIPELINE;
    }

    // get document name
    //
    //
    // if there are several similar documents for example
    // "/data/Kunden/NorshHydro/GraneC23/epicurus1.png"
    // "/data/Kunden/NorshHydro/GraneC23/epicurus2.png"
    // "/data/Kunden/NorshHydro/GraneC23/epicurus3.png"
    // "/data/Kunden/NorshHydro/GraneC23/epicurus4.png"

    // Syntax:
    // DOCUMENT <documentName> <numpages> <documentPath1> <documentpath2> ...
    //
    // Example:
    // DOCUMENT epicurus 1 /data/kunden/NorskHydro/GraneC23/epicurus.png
    //

    char *nextFilename = NULL;
    string result = "";
    int noFiles = 0;

    char pathBuf[500];
    getname(pathBuf, docPath, NULL);

    coStepFile myStepFile(pathBuf);
    myStepFile.get_nextpath(&nextFilename);

    while (nextFilename != NULL)
    {
        noFiles++;
        result += nextFilename;
        result += " ";
        // debug: send filename to info message
        //sendInfo("nextFilename is [%s]", nextFilename);
        myStepFile.get_nextpath(&nextFilename);
    }

    char *str = new char[strlen(docName2) + 1 + 32 + 1 + result.length() + 1];
    sprintf(str, "%s %d %s", docName2, noFiles, result.c_str());

    // original code (only one filename):
    //char *str = new char[strlen(docName) + 1 + 32 + 1 + strlen(docPath) + 1];
    //sprintf(str,"%s 1 %s", docName, docPath);

    // debug: send to info message
    //sendInfo("Setting DOCUMENT to \"%s\"\n", str);

    // attach Attribute to Object
    point->addAttribute("DOCUMENT", str);

    // attach MODULE attribute which starts plugin DocumentViewer if it is not already running
    coFeedback feedback("DocumentViewer");
    feedback.apply(point);

    delete[] str;
    delete[] nextFilename;
    delete[] docName2;

    // hand over Object to port
    outPort->setCurrentObject(point);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, CoverDocument)
