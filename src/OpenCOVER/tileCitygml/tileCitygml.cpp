#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace xercesc;

void writeNodeToFile(const std::vector<DOMNode*> &nodes, const std::string& filename) {
    // Convert DOMNode to XML string
    DOMImplementation* impl = DOMImplementationRegistry::getDOMImplementation(XMLString::transcode("LS"));
    DOMLSSerializer* serializer = impl->createLSSerializer();

    // Make sure the output is nicely formatted
    if (serializer->getDomConfig()->canSetParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true))
        serializer->getDomConfig()->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true);


    // Write the XML string to a file
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto& node : nodes)
        {
            // Serialize the node to an XML string
            XMLCh* xmlStr = serializer->writeToString(node);

            // Convert XMLCh* to std::string
            char* cStr = XMLString::transcode(xmlStr);
            std::string xmlContent(cStr);
            XMLString::release(&cStr);
            file << xmlContent;
            XMLString::release(&xmlStr);
        }
        file.close();
        std::cout << "Written to " << filename << std::endl;
    } else {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
    }

    // Clean up
    serializer->release();
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


    // Iterate over each "core:cityObjectMember" node and save to individual files
    for (XMLSize_t n = 0; n < cityObjectMembers->getLength(); ++n) {
        DOMNode* cityObjectMember = cityObjectMembers->item(n);

        DOMElement* currentElement = dynamic_cast<DOMElement*>(cityObjectMember);
        DOMNodeList* posLists = currentElement->getElementsByTagName(XMLString::transcode("gml:posList"));

        if (posLists->getLength() > 0)
        {
            DOMNode* posList = posLists->item(0);
            //char* value = XMLString::transcode(posList->getNodeValue());
            DOMText* data = dynamic_cast<DOMText*>(posList);
            //if(DOMNode::TEXT_NODE == posList->getNodeType())
            if(data != nullptr)
            {
                const XMLCh* val = data->getWholeText();
                char *strVal = XMLString::transcode(val);
                float x, y;
                sscanf(strVal, "%f %f",&x,&y);
                size_t xi = (size_t)x;
                size_t yi = (size_t)y;
                size_t i, j;
                i = (xi / 1000) - 6633;
                j = (yi / 1000) - 370;
                cityObjectMembersSorted[i * ny + j].push_back(cityObjectMember);
            }
        }
    }
    for (size_t i = 0; i < nx; i++)
    {
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
