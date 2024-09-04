#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

struct DataEntry {
    std::streampos cityObjectMemberPos;  // Store the position of "cityObjectMember" in the file
    size_t i;
    size_t j;
};

int main(int argc, char **argv) {
    std::ifstream file(argv[1], std::ios::in);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    std::string line;
    //6377179 6633514
    //6377-6633 = 256
    //nx = 256
    //738969 370083
    //738-370
    //ny = 368
    size_t nx = 258;
    size_t ny = 370;
    std::vector<std::vector<DataEntry>> cityObjectMembersSorted;
    cityObjectMembersSorted.resize(nx * ny);


    // Step 1: Read and store positions of "cityObjectMember" and "posList" data
    // Store the position of "cityObjectMember"
    std::streampos cityObjectMemberPos = file.tellg();
    while (std::getline(file, line)) {

        // Search for "cityObjectMember"
        if (line.find("<core:cityObjectMember>") != std::string::npos) {
            std::string posListLine;
            
            // Continue searching for "posList" after finding "cityObjectMember"
            while (std::getline(file, posListLine)) {
                size_t pospos = posListLine.find("posList");
                if (pospos != std::string::npos) {
                    // Extract the next two floating-point numbers
                    std::string coordinates = posListLine.substr(pospos+8);
                    //std::cerr << coordinates << std::endl;
                    std::istringstream iss(coordinates);
                    double x, y;
                    iss >> x >> y;
                    size_t xi = (size_t)x;
                    size_t yi = (size_t)y;
                    size_t i, j;
                    i = (xi / 1000) - 6377;
                    j = (yi / 1000) - 370;
                    if (i>0 && j >0 && i < nx && j < ny)
                    {
                        {
                            cityObjectMembersSorted[i * ny + j].push_back({cityObjectMemberPos, i+6377, j+370});
                            break; // Stop looking for "posList" after the first occurrence
                        }
                    }
                    
                }
            }
        }
        // Store the position of the next "cityObjectMember"
        cityObjectMemberPos = file.tellg();
    }

    file.close();
        std::ifstream inputFile(argv[1], std::ios::in);
        if (!inputFile.is_open()) {
            std::cerr << "Error opening file for processing!" << std::endl;
            return 1;
        }

    for (size_t i = 0; i < nx; i++)
    {
        std::cerr << i << std::endl;
        for (size_t j = 0; j < ny; j++)
        {
            if (cityObjectMembersSorted[i * ny + j].size() > 0)
            {
            auto dataEntry = cityObjectMembersSorted[i * ny + j][0];
            // Write the nodes to files
            // Construct filename
            std::ostringstream filename;
            filename << "lod2_" << dataEntry.i*1000 << "_" << dataEntry.j*1000 << ".gml";
                //writeNodeToFile(cityObjectMembersSorted[i * ny + j], filename.str());
 

        // Open the output file (e.g., output1.txt, output2.txt, etc.)
        std::ofstream outputFile(filename.str(), std::ios::out);

        if (!outputFile.is_open()) {
            std::cerr << "Error opening output file!" << std::endl;
            return 1;
        }
        outputFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<core:CityModel xmlns:brid=\"http://www.opengis.net/citygml/bridge/2.0\" xmlns:tran=\"http://www.opengis.net/citygml/transportation/2.0\" xmlns:frn=\"http://www.opengis.net/citygml/cityfurniture/2.0\" xmlns:wtr=\"http://www.opengis.net/citygml/waterbody/2.0\" xmlns:sch=\"http://www.ascc.net/xml/schematron\" xmlns:veg=\"http://www.opengis.net/citygml/vegetation/2.0\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" xmlns:tun=\"http://www.opengis.net/citygml/tunnel/2.0\" xmlns:tex=\"http://www.opengis.net/citygml/texturedsurface/2.0\" xmlns:gml=\"http://www.opengis.net/gml\" xmlns:gen=\"http://www.opengis.net/citygml/generics/2.0\" xmlns:dem=\"http://www.opengis.net/citygml/relief/2.0\" xmlns:app=\"http://www.opengis.net/citygml/appearance/2.0\" xmlns:luse=\"http://www.opengis.net/citygml/landuse/2.0\" xmlns:xAL=\"urn:oasis:names:tc:ciq:xsdschema:xAL:2.0\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:smil20lang=\"http://www.w3.org/2001/SMIL20/Language\" xmlns:pbase=\"http://www.opengis.net/citygml/profiles/base/2.0\" xmlns:smil20=\"http://www.w3.org/2001/SMIL20/\" xmlns:bldg=\"http://www.opengis.net/citygml/building/2.0\" xmlns:core=\"http://www.opengis.net/citygml/2.0\" xmlns:grp=\"http://www.opengis.net/citygml/cityobjectgroup/2.0\">" << std::endl;

        bool foundEnd = false;
        for (const auto& dataEntry : cityObjectMembersSorted[i * ny + j])
        {
        // Seek to the stored position in the input file
        inputFile.seekg(dataEntry.cityObjectMemberPos);

        // Process lines until we hit "/core:cityObjectMember"
        while (std::getline(inputFile, line)) {
            // Stop if we find "/core:cityObjectMember"
            if (line.find("/core:cityObjectMember") != std::string::npos) {
                foundEnd = true;
                outputFile << line << std::endl;
                break;
            }

            // Search and replace in lines containing "gml:MultiSurface"
            if (line.find("gml:MultiSurface") != std::string::npos && line.find("/gml:MultiSurface") == std::string::npos) {
                size_t pos = line.find(">");
                if (pos != std::string::npos) {
                    line.insert(pos, " orientation=\"-\"");
                }
            }

            // Write the modified or original line to the output file
            outputFile << line << std::endl;
        }
            }
        outputFile << "</core:CityModel>" << std::endl;
        // If we didn't find "/core:cityObjectMember", issue a warning
        if (!foundEnd) {
            std::cerr << "Warning: Did not find '/core:cityObjectMember' for entry " << i + 1 << std::endl;
        }

        // Close files
        outputFile.close();
        }

        }
    }
    inputFile.close();

    return 0;
}
