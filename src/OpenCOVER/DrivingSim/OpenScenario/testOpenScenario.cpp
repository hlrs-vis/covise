/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <OpenScenarioBase.h>
#include <oscHeader.h>

#include <iostream>

// Mandatory for using any feature of Xerces.
#include <xercesc/util/PlatformUtils.hpp>
// Required for outputting a Xerces DOMDocument
// to a standard output stream (Also see: XMLFormatTarget)
#include <xercesc/framework/StdOutFormatTarget.hpp>
// Required for outputting a Xerces DOMDocument
// to the file system (Also see: XMLFormatTarget)
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
    std::string fileName = "testScenario.xosc";
    if(argc > 1)
    {
        fileName = argv[1];
    }
    std::cerr << "trying to load " << fileName << std::endl;
    if(osdb->loadFile(fileName)== false)
    {
        std::cerr << "failed to load OpenScenarioBase from file " << fileName << std::endl;
        return -1;
    }


    //////
    // print to std::out
    //
    std::cerr << std::endl;

    xercesc::DOMDocument *parsedXmlDoc = osdb->getDocument();

    xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();
    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    // set the format-pretty-print feature
    if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
    {
        writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    }

    xercesc::DOMLSOutput *output = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();

    /*
    Choose a location for the serialized output. The 3 options are:
        1) StdOutFormatTarget     (std output stream -  good for debugging)
        2) MemBufFormatTarget     (to Memory)
        3) LocalFileFormatTarget  (save to file)
        (Note: You'll need a different header file for each one)
    */
//    xercesc::XMLFormatTarget *consoleTarget = new xercesc::StdOutFormatTarget();

//    output->setByteStream(consoleTarget);
//    writer->write(parsedXmlDoc, output);

    std::cerr << std::endl;
    //
    //////

    //////
    //write into a file
    //
    std::cerr << "save xml DOM to file: complete_osc-document.xml" << std::endl;
    xercesc::XMLFormatTarget *fileTarget = new xercesc::LocalFileFormatTarget(std::string("complete_osc-document.xml").c_str());

    output->setByteStream(fileTarget);
    writer->write(parsedXmlDoc, output);

    std::cerr << std::endl;
    //
    //////


    if (osdb->header.getObject() != NULL)
    {
        std::cerr << "revMajor:" << osdb->header->revMajor.getValue() << std::endl;
        std::cerr << "revMinor:" << osdb->header->revMinor.getValue() << std::endl;
        std::cerr << "Author:" << osdb->header->author.getValue() << std::endl;
    }
    if(argc > 2)
    {
        fileName = argv[2];
        std::cerr << "trying to save to " << fileName << std::endl;
        if(osdb->saveFile(fileName,true)== false)
        {
            std::cerr << "failed to save OpenScenarioBase to file " << fileName << std::endl;
            delete osdb;
            return -1;
        }
    }
    delete osdb;
    return 0;
}
