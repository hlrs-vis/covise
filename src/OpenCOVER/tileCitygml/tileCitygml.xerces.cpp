#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace xercesc;

void writeNodeToFile(const std::vector<DOMNode*> &nodes, const std::string& filename) {
    // Create a new document to hold the node without namespace context
    DOMImplementation* impl = DOMImplementationRegistry::getDOMImplementation(XMLString::transcode("LS"));
    DOMDocument* newDoc = impl->createDocument(nullptr, nullptr, nullptr);
    //DOMNode* newNode = new DOM("CityModel");
    // 
    DOMElement* groupNode = newDoc->createElementNS(XMLString::transcode("http://www.opengis.net/citygml/2.0"),XMLString::transcode("core:CityModel"));
    newDoc->appendChild(groupNode);
    XMLCh* nsURI = XMLString::transcode("http://www.opengis.net");
    XMLCh* nsURI2 = XMLString::transcode("http://www.opengis.net/gml");
    XMLCh* xmlnsPrefix2 = XMLString::transcode("xmlns:gml");
    // Set the additional namespace as an attribute on the root element
    groupNode->setAttributeNS(XMLString::transcode("http://www.w3.org/2000/xmlns/"), xmlnsPrefix2, nsURI2);
    groupNode->setAttributeNS(XMLString::transcode("http://www.w3.org/2000/xmlns/"), XMLString::transcode("xmlns:bldg"), XMLString::transcode("http://www.opengis.net/citygml/building/2.0"));
    groupNode->setAttributeNS(XMLString::transcode("http://www.w3.org/2000/xmlns/"), XMLString::transcode("xmlns:gen"), XMLString::transcode("http://www.opengis.net/citygml/generics/2.0"));
    // Convert DOMNode to XML string
    DOMLSSerializer* serializer = impl->createLSSerializer();
    DOMLSOutput* output = impl->createLSOutput();

    // Make sure the output is nicely formatted
    if (serializer->getDomConfig()->canSetParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true)) {
        serializer->getDomConfig()->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true);
    }

    // Set encoding to UTF-8
    output->setEncoding(XMLString::transcode("UTF-8"));

    // Set the target for writing output to a file
    XMLFormatTarget* formatTarget = new LocalFileFormatTarget(filename.c_str());
    output->setByteStream(formatTarget);


        //file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<core:CityModel xmlns:brid=\"http://www.opengis.net/citygml/bridge/2.0\" xmlns:tran=\"http://www.opengis.net/citygml/transportation/2.0\" xmlns:frn=\"http://www.opengis.net/citygml/cityfurniture/2.0\" xmlns:wtr=\"http://www.opengis.net/citygml/waterbody/2.0\" xmlns:sch=\"http://www.ascc.net/xml/schematron\" xmlns:veg=\"http://www.opengis.net/citygml/vegetation/2.0\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" xmlns:tun=\"http://www.opengis.net/citygml/tunnel/2.0\" xmlns:tex=\"http://www.opengis.net/citygml/texturedsurface/2.0\" xmlns:gml=\"http://www.opengis.net/gml\" xmlns:gen=\"http://www.opengis.net/citygml/generics/2.0\" xmlns:dem=\"http://www.opengis.net/citygml/relief/2.0\" xmlns:app=\"http://www.opengis.net/citygml/appearance/2.0\" xmlns:luse=\"http://www.opengis.net/citygml/landuse/2.0\" xmlns:xAL=\"urn:oasis:names:tc:ciq:xsdschema:xAL:2.0\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:smil20lang=\"http://www.w3.org/2001/SMIL20/Language\" xmlns:pbase=\"http://www.opengis.net/citygml/profiles/base/2.0\" xmlns:smil20=\"http://www.w3.org/2001/SMIL20/\" xmlns:bldg=\"http://www.opengis.net/citygml/building/2.0\" xmlns:core=\"http://www.opengis.net/citygml/2.0\" xmlns:grp=\"http://www.opengis.net/citygml/cityobjectgroup/2.0\">" << std::endl;
        for (const auto& node : nodes)
        {
            // Import the node into the new document (detaches it from namespace context)
            DOMNode* importedNode = newDoc->importNode(node, true);

            groupNode->appendChild(importedNode);

        }
        // Make sure the output is nicely formatted
        if (serializer->getDomConfig()->canSetParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true)) {
            serializer->getDomConfig()->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true);
        }

        // Serialize the node to an XML string without xmlns attributes
        serializer->write(newDoc, output);

        std::cout << "Written to " << filename << std::endl;

    // Clean up
    delete formatTarget;
    output->release();
    serializer->release();
    newDoc->release();
}

int main(int argc, char* argv[]) {
    try {
        // Initialize Xerces
        XMLPlatformUtils::Initialize();
    } catch (const XMLException& e) {
        char* message = XMLString::transcode(e.getMessage());
        std::cerr << "Xerces initialization error: " << message << std::endl;
        XMLString::release(&message);
        return 1;
    }

    // Create parser
    XercesDOMParser* parser = new XercesDOMParser();
    parser->setValidationScheme(XercesDOMParser::Val_Always);
    parser->setDoNamespaces(true);  // Enable namespaces

    // Error handler
    ErrorHandler* errorHandler = (ErrorHandler*) new HandlerBase();
    parser->setErrorHandler(errorHandler);

    // Parse the XML file
    try {
        parser->parse(argv[1]);  // Input GML file
    } catch (const XMLException& e) {
        char* message = XMLString::transcode(e.getMessage());
        std::cerr << "XML Exception: " << message << std::endl;
        XMLString::release(&message);
        return 1;
    } catch (const DOMException& e) {
        char* message = XMLString::transcode(e.getMessage());
        std::cerr << "DOM Exception: " << message << std::endl;
        XMLString::release(&message);
        return 1;
    } catch (...) {
        std::cerr << "Unexpected exception during parsing" << std::endl;
        return 1;
    }

    // Get the document root element
    DOMDocument* doc = parser->getDocument();
    DOMElement* rootElement = doc->getDocumentElement();

    // Find all "core:cityObjectMember" elements
    DOMNodeList* cityObjectMembers = rootElement->getElementsByTagName(XMLString::transcode("core:cityObjectMember"));

    //6377179 6633514
    //6377-6633 = 256
    //nx = 256
    //738969 370083
    //738-370
    //ny = 368
    size_t nx = 258;
    size_t ny = 370;
    std::vector<std::vector<DOMNode*>> cityObjectMembersSorted;
    cityObjectMembersSorted.resize(nx * ny);

    static int oldp=1000;
    // Iterate over each "core:cityObjectMember" node and save to individual files
    XMLSize_t length = cityObjectMembers->getLength();
#pragma omp parallel for
    for (XMLSize_t n = 0; n < length; ++n) {
        int perc = (int)((double)n/(double)length) *1000.0;
        if(oldp!=perc)
        {
            #pragma omp critical
            {
                oldp=perc;
                std::cerr << "PercentDone: " << (float)perc/10.0 << std::endl;
            }
        }
        DOMNode* cityObjectMember = cityObjectMembers->item(n);

        DOMElement* currentElement = dynamic_cast<DOMElement*>(cityObjectMember);
        DOMNodeList* posLists = currentElement->getElementsByTagName(XMLString::transcode("gml:posList"));

        if (posLists->getLength() > 0)
        {
            DOMNode* posList = posLists->item(0);
            //std::cerr << "nodeName" << XMLString::transcode(posList->getNodeName()) << std::endl;
            //std::cerr << "textContent" << XMLString::transcode(posList->getTextContent()) << std::endl;
            const XMLCh* val = posList->getTextContent();
            char* strVal = XMLString::transcode(val);
            float x = 0, y = 0;
            int num = sscanf(strVal, "%f %f", &x, &y);
            if (num == 2)
            {
                size_t xi = (size_t)x;
                size_t yi = (size_t)y;
                size_t i, j;
                i = (xi / 1000) - 6377;
                j = (yi / 1000) - 370;
                if (i < nx && j < ny)
                {
                    #pragma omp critical
                    {
                        cityObjectMembersSorted[i * ny + j].push_back(cityObjectMember);
                    }
                }
            }
        }
        DOMNodeList* surfaceList = currentElement->getElementsByTagName(XMLString::transcode("gml:MultiSurface"));
        XMLSize_t numSurf=surfaceList->getLength();
        for (XMLSize_t m = 0; m < numSurf; ++m)
        {
            DOMNode* multiSurface = surfaceList->item(m);
            DOMElement* currentElement = dynamic_cast<DOMElement*>(multiSurface);
            if (currentElement)
            {
                #pragma omp critical
                {
                    currentElement->setAttribute(XMLString::transcode("orientation"), XMLString::transcode("-"));
                }
            }
        }
    }
    for (size_t i = 0; i < nx; i++)
    {
#pragma omp parallel for
        for (size_t j = 0; j < ny; j++)
        {
            // Write the nodes to files
            // Construct filename
            std::ostringstream filename;
            filename << "lod2_" << i*1000 << "_" << j*1000 << ".gml";
            if (cityObjectMembersSorted[i * ny + j].size() > 0)
            {
                writeNodeToFile(cityObjectMembersSorted[i * ny + j], filename.str());
            }
        }
    }

    // Clean up
    delete parser;
    delete errorHandler;
    XMLPlatformUtils::Terminate();

    return 0;
}
