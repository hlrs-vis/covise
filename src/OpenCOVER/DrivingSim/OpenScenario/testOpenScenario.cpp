/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <OpenScenarioBase.h>
#include <oscHeader.h>

#include <iostream>

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>


using namespace OpenScenario;

int main(int argc, char **argv)
{
    OpenScenarioBase *osdb = new OpenScenarioBase();
    //default file to read in
    std::string fileName = "testScenario.xosc";

    //command line parameter:
    //-c: write xosc file imported with parser to console
    //-f: write xosc file imported with parser to console to another file with name writeComleteXmlToFile
    //first appearance of non empty argv[i] is filename to read
    //second appearance of non empty argv[i] is filename to write
    bool writeToConsole = false;
    bool writeComleteXML = false;
    std::string writeComleteXmlToFile = "complete_osc-document.xml";
    std::string readFileToUse;
    std::string writeFileToUse;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-c"))
        {
            writeToConsole = true;
        }
        else if (!strcmp(argv[i], "-f"))
        {
            writeComleteXML = true;
        }
        else if (readFileToUse == "" && argv[i] != "") //first appearance
        {
            readFileToUse = argv[i];
        }
        else if (writeFileToUse == "" && argv[i] != "") //second appearance
        {
            writeFileToUse = argv[i];
        }
    }

    if (readFileToUse == "")
    {
        readFileToUse = fileName;
    }

    //read file
    std::cerr << "trying to load " << readFileToUse << std::endl;
    std::cerr << std::endl;
    if(osdb->loadFile(readFileToUse) == false)
    {
        std::cerr << std::endl;
        std::cerr << "failed to load OpenScenarioBase from file " << readFileToUse << std::endl;
        std::cerr << std::endl;
        delete osdb;
        return -1;
    }

    //////
    //for writing to console or complete xml document
    //
    xercesc::DOMDocument *parsedXmlDoc = osdb->getDocument();

    xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();
    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    // set the format-pretty-print feature
    if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
    {
        writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    }

    xercesc::DOMLSOutput *output = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();

    // print to std::out
    if (writeToConsole)
    {
        xercesc::XMLFormatTarget *consoleTarget = new xercesc::StdOutFormatTarget();
        std::cerr << std::endl;
        output->setByteStream(consoleTarget);
        writer->write(parsedXmlDoc, output);
        std::cerr << std::endl;
    }
    //

    //write into a file
    if (writeComleteXML)
    {
        std::cerr << std::endl;
        std::cerr << "save xml DOM to file: complete_osc-document.xml" << std::endl;
        xercesc::XMLFormatTarget *fileTarget = new xercesc::LocalFileFormatTarget(writeComleteXmlToFile.c_str());
        output->setByteStream(fileTarget);
        writer->write(parsedXmlDoc, output);
        std::cerr << std::endl;
    }
    //
    //////

    //print some values to console
    if (osdb->header.getObject() != NULL)
    {
        std::cerr << "revMajor:" << osdb->header->revMajor.getValue() << std::endl;
        std::cerr << "revMinor:" << osdb->header->revMinor.getValue() << std::endl;
        std::cerr << "Author:" << osdb->header->author.getValue() << std::endl;
    }

    //write object structure to xml document
    if(writeFileToUse != "")
    {
        std::cerr << std::endl;
        std::cerr << "trying to save to " << writeFileToUse << std::endl;
        if(osdb->saveFile(writeFileToUse,true) == false)
        {
            std::cerr << std::endl;
            std::cerr << "failed to save OpenScenarioBase to file " << writeFileToUse << std::endl;
            std::cerr << std::endl;
            delete osdb;
            return -1;
        }
    }

    delete osdb;

    return 0;
}
