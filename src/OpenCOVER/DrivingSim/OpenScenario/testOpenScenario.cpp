/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "OpenScenarioBase.h"
#include "oscFileHeader.h"

#include <iostream>
#ifdef WIN32
#include <time.h>
#endif

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>


using namespace OpenScenario;


void displayHelp(const char *progName, const std::string &writeCompleteXmlToFile)
{
    std::cout << progName << " [options] input [output]\n"
            << "\n"
            << " input:       OpenSCENARIO main input file - must be specified -\n"
            << " output:      write object structure to DOM and than to output\n"
            << "\n"
            << "Options:\n"
            << "\n"
            << " -c           write xosc file with all included files to console,\n"
            << "               if import with parser (XInclude enabled) was successful\n"
            << "\n"
            << " -f           write xosc file with all included files to to file\n"
            << "               with name '" << writeCompleteXmlToFile << "',\n"
            << "               if import with parser (XInclude enabled) was successful\n"
            << "\n"
            << " -frc         full read of all available catalog objects, store them in\n"
            << "               the object structure and write them back to files.\n"
            << "\n"
            << " -h, --help   print this help\n"
            << "\n"
            << " -nv          disable validation\n"
            << "\n"
            << std::endl;
}

bool getopt(const char *argument, const std::string &option)
{
    if(argument[0] == '-')
    {
        std::string arg = argument;
        arg.erase(0, 1);

        if (arg == option)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}


int main(int argc, char **argv)
{
    //initialize random seed
    srand(time(NULL));

    OpenScenarioBase *osdb = new OpenScenarioBase();

    //command line parameter:
    //-c:      write xosc file imported with parser to console
    //-f:      write xosc file imported with parser to another file with name writeComleteXmlToFile
    //-frc:    full read of catalog objects in oscObjectBase::parseFromXML()
    //-h:      print help message
    //--help:  print help message
    //-nv:     disable validation
    //first appearance of non empty argv[i] is filename to read
    //second appearance of non empty argv[i] is filename to write
    //
    bool writeToConsole = false;
    bool writeCompleteXML = false;
    std::string writeCompleteXmlToFile = "complete_osc-document.xml";
    std::string readFileToUse;
    std::string writeFileToUse;

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            if (getopt(argv[i],"c") == true)
            {
                writeToConsole = true;
            }
            else if (getopt(argv[i],"f") == true)
            {
                writeCompleteXML = true;
            }
            else if (getopt(argv[i],"frc") == true)
            {
                osdb->setFullReadCatalogs(true);
            }
            else if (getopt(argv[i],"h") == true || getopt(argv[i],"-help") == true)
            {
                displayHelp(argv[0], writeCompleteXmlToFile);
                return -1;
            }
            else if (getopt(argv[i],"nv") == true)
            {
                osdb->setValidation(false);
            }
            else
            {
                std::cerr << "Option '" << argv[i] << "' not recognized.\n" << std::endl;
                displayHelp(argv[0], writeCompleteXmlToFile);
                return -1;
            }
        }
        else if (readFileToUse == "" && argv[i][0] != '\0') //first appearance
        {
            readFileToUse = argv[i];
        }
        else if (writeFileToUse == "" && argv[i][0] != '\0') //second appearance
        {
            writeFileToUse = argv[i];
        }
    }

    if (readFileToUse == "")
    {
        displayHelp(argv[0], writeCompleteXmlToFile);
        return -1;
    }


    //read file
    std::cerr << "trying to load " << readFileToUse << std::endl;
    std::cerr << std::endl;
    if(osdb->loadFile(readFileToUse, "OpenSCENARIO", "OpenSCENARIO") == false)
    {
        std::cerr << std::endl;
        std::cerr << "failed to load OpenSCENARIO from file " << readFileToUse << std::endl;
        std::cerr << std::endl;
        delete osdb;

        return -1;
    }


    //////
    //write complete xml document to console or into a file
    //
    if (writeToConsole || writeCompleteXML)
    {
        xercesc::DOMDocument *parsedXmlDoc = osdb->getDocument();

        xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();
        xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
        // set the format-pretty-print feature
        if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
        {
            writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
        }

        xercesc::DOMLSOutput *output = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();

        //print to console
        if (writeToConsole)
        {
            xercesc::XMLFormatTarget *consoleTarget = new xercesc::StdOutFormatTarget();
            std::cerr << std::endl;
            output->setByteStream(consoleTarget);
            writer->write(parsedXmlDoc, output);
            std::cerr << std::endl;

            delete consoleTarget;
        }

        //write into a file
        if (writeCompleteXML)
        {
            std::cerr << std::endl;
            std::cerr << "save xml DOM to file: complete_osc-document.xml" << std::endl;
            xercesc::XMLFormatTarget *fileTarget = new xercesc::LocalFileFormatTarget(writeCompleteXmlToFile.c_str());
            output->setByteStream(fileTarget);
            writer->write(parsedXmlDoc, output);
            std::cerr << std::endl;

            delete fileTarget;
        }

        //delete used objects
        delete writer;
        delete output;
    }
    //
    //////


    //print some values to console
    if (osdb->FileHeader.getObject() != NULL)
    {
        std::cerr << std::endl;
        std::cerr << "revMajor:" << osdb->FileHeader->revMajor.getValue() << std::endl;
        std::cerr << "revMinor:" << osdb->FileHeader->revMinor.getValue() << std::endl;
        std::cerr << "Author:" << osdb->FileHeader->author.getValue() << std::endl;
    }


    //write object structure to xml document
    if(writeFileToUse != "")
    {
        std::cerr << std::endl;
        std::cerr << "trying to save to " << writeFileToUse << std::endl;
        if(osdb->saveFile(writeFileToUse,true) == false)
        {
            std::cerr << std::endl;
            std::cerr << "failed to save OpenSCENARIO to file " << writeFileToUse << std::endl;
            std::cerr << std::endl;
            delete osdb;

            return -1;
        }
    }


    delete osdb;

    return 0;
}
