#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <gdal.h>
#include <ogrsf_frmts.h>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/ProxyNode>
#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>
enum geotype
{
    windturbine = 0,
    powerline
};
struct DataEntry {
    std::streampos cityObjectMemberPos;  // Store the position of "cityObjectMember" in the file
    size_t i;
    size_t j;
};
struct FeatureEntry {
    enum geotype type;
    OGRFeature* feature;  // 
    size_t i;
    size_t j;
    double x;
    double y;
    double z;
    double angle;
};

int tileCitygml(const std::string& filename)
{

    std::ifstream file(filename, std::ios::in);
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
                    std::string coordinates = posListLine.substr(pospos + 8);
                    //std::cerr << coordinates << std::endl;
                    std::istringstream iss(coordinates);
                    double x, y;
                    iss >> x >> y;
                    size_t xi = (size_t)x;
                    size_t yi = (size_t)y;
                    size_t i, j;
                    i = (xi / 1000) - 6377;
                    j = (yi / 1000) - 370;
                    if (i > 0 && j > 0 && i < nx && j < ny)
                    {
                        {
                            cityObjectMembersSorted[i * ny + j].push_back({ cityObjectMemberPos, i + 6377, j + 370 });
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
    std::ifstream inputFile(filename, std::ios::in);
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
                filename << "lod2_" << dataEntry.i * 1000 << "_" << dataEntry.j * 1000 << ".gml";
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
                        /*if (line.find("gml:MultiSurface") != std::string::npos && line.find("/gml:MultiSurface") == std::string::npos) {
                            size_t pos = line.find(">");
                            if (pos != std::string::npos) {
                                line.insert(pos, " orientation=\"-\"");
                            }
                        }*/
                        size_t pospos = line.find("posList");
                        if (pospos != std::string::npos) {
                            std::ostringstream oss;
                            oss << line.substr(0, pospos + 8);
                            std::string coordinatesStr = line.substr(pospos + 8);
                            std::istringstream iss(coordinatesStr);
                            double x = 0, y = 0, z = 0;
                            std::vector<double> coordinates;
                            oss << std::setprecision(12);

                            // Read triplets of coordinates (x, y, z)
                            while (iss >> x >> y >> z) {
                                auto nextChar = iss.peek();
                                if (nextChar == '<') {
                                    break; // Stop parsing coordinates if "<" is found
                                }
                                if (x != 0 && y != 0)
                                {
                                    // Switch x and y for each coordinate triple
                                    coordinates.push_back(y);
                                    coordinates.push_back(x);
                                    coordinates.push_back(z);
                                }
                                x = 0; y = 0; z = 0;
                            }
                            for (const auto& c : coordinates)
                            {
                                oss << c << " ";
                            }
                            oss << "</gml:posList>";
                            // Write the modified or original line to the output file
                            outputFile << oss.str() << std::endl;
                        }
                        else
                        {


                            // Write the modified or original line to the output file
                            outputFile << line << std::endl;
                        }
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

int tileShp(const std::string& filename)
{

    // Initialize GDAL/OGR library
    GDALAllRegister();


    // Open the shapefile
    GDALDataset* dataset = static_cast<GDALDataset*>(GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr));
    if (dataset == nullptr) {
        std::cerr << "Failed to open shapefile." << std::endl;
        return 1;
    }

    // Get the layer (assuming the shapefile has a single layer)
    OGRLayer* layer = dataset->GetLayer(0);
    if (layer == nullptr) {
        std::cerr << "Failed to get layer." << std::endl;
        GDALClose(dataset);
        return 1;
    }

    // Loop through features
    OGRFeature* feature;
    layer->ResetReading(); // Reset reading to the beginning of the layer

    //6377179 6633514
    //6377-6633 = 256
    //nx = 256
    //738969 370083
    //738-370
    //ny = 368
    size_t nx = 258;
    size_t ny = 370;
    std::vector<std::vector<FeatureEntry>> shapeObjects;
    shapeObjects.resize(nx * ny);
    size_t featureNumber = 0;
    while ((feature = layer->GetNextFeature()) != nullptr) {
        bool used = false;
        // Get the feature's class/type
        const char* Type = feature->GetFieldAsString("tyyp");  // Adjust field name

        // Check if the feature's class is "e_602_tehnopaigaldis_p"
        if (Type && std::string(Type) == "10")
        {
        }
        else if (Type && std::string(Type) == "20")
        {
            // Print geometry information or any other relevant info
            OGRGeometry* geometry = feature->GetGeometryRef();
            if (geometry != nullptr) {
                if (wkbFlatten(geometry->getGeometryType()) == wkbPoint) {
                    OGRPoint* point = static_cast<OGRPoint*>(geometry);

                    // Output X, Y, and Z coordinates
                    double x = point->getY();
                    double y = point->getX();
                    double z = point->getZ();  // Z coordinate (may be 0 if 2D data)

                    size_t xi = (size_t)x;
                    size_t yi = (size_t)y;
                    size_t i, j;
                    i = (xi / 1000) - 6377;
                    j = (yi / 1000) - 370;
                    if (i > 0 && j > 0 && i < nx && j < ny)
                    {
                        used = true;
                        shapeObjects[i * ny + j].push_back({ windturbine, feature,i,j, x,y,z ,0.0});
                    }
                }
            }
        }
        else
        {
            // Output the feature's fields
           /* for (int i = 0; i < feature->GetFieldCount(); i++) {
                const char* fieldName = feature->GetFieldDefnRef(i)->GetNameRef();
                const char* fieldValue = feature->GetFieldAsString(i);
                std::cout << fieldName << ": " << fieldValue << std::endl;
            }*/
        }
        const char* Code = feature->GetFieldAsString("kood");  // Adjust field name
        if (Code && std::string(Code) == "601") // power lines
        {
            // Print geometry information or any other relevant info
            OGRGeometry* geometry = feature->GetGeometryRef();
            if (geometry != nullptr)
            {
                if (wkbFlatten(geometry->getGeometryType()) == wkbLineString) {
                    OGRLineString* lineString = static_cast<OGRLineString*>(geometry);

                    // Output the number of points in the linestring
                    int numPoints = lineString->getNumPoints();
                    std::cout << "LineString with " << numPoints << " vertices:" << feature->GetFieldAsString("nimipinge") << " volt"<< std::endl;
                    // Loop through the vertices of the linestring
                    for (int n = 0; n < numPoints; n++) {
                        double x = lineString->getY(n);
                        double y = lineString->getX(n);
                        double z = lineString->getZ(n);  // Z coordinate (may be 0 if 2D data)
                        double x2, y2;
                        if (n == 0)
                        {
                            x2 = lineString->getY(n + 1);
                            y2 = lineString->getX(n + 1);
                        }
                        else
                        {
                            x2 = lineString->getY(n - 1);
                            y2 = lineString->getX(n - 1);
                        }
                        size_t xi = (size_t)x;
                        size_t yi = (size_t)y;
                        size_t i, j;
                        i = (xi / 1000) - 6377;
                        j = (yi / 1000) - 370;
                        double angle = atan2(x2 - x, y2 - y) + M_PI_2;
                        if (i > 0 && j > 0 && i < nx && j < ny)
                        {
                            used = true;
                            shapeObjects[i * ny + j].push_back({ powerline, feature,i,j, x,y,z ,angle});
                        }
                        //std::cout << "Vertex " << i + 1 << ": X = " << x << ", Y = " << y << ", Z = " << z << std::endl;
                    }

                }
            }
        }

        // Destroy the feature to avoid memory leaks
        if (!used)
        {
            OGRFeature::DestroyFeature(feature);
        }
    }

    // Clean up
    GDALClose(dataset);


    for (size_t i = 0; i < nx; i++)
    {
        std::cerr << i << std::endl;
        for (size_t j = 0; j < ny; j++)
        {
            if (shapeObjects[i * ny + j].size() > 0)
            {
                auto featureEntry = shapeObjects[i * ny + j][0];
                // Write the nodes to files
                // Construct filename
                std::ostringstream filename;
                filename << "objects_" << featureEntry.i * 1000 << "_" << featureEntry.j * 1000 << ".ive";
                //writeNodeToFile(cityObjectMembersSorted[i * ny + j], filename.str());
                osg::ref_ptr<osg::Group> root = new osg::Group;
                for (const auto& featureEntry : shapeObjects[i * ny + j])
                {
                    osg::ProxyNode* p = new osg::ProxyNode();
                    osg::MatrixTransform* m = new osg::MatrixTransform();
                    float scale=1;
                    float angle = featureEntry.angle;
                    if(featureEntry.type==windturbine)
                    {
                        const char* Height = featureEntry.feature->GetFieldAsString("korgus");  // Adjust field name
                        float height = std::stof(Height);
                        scale = height / 1.053; // our model is 1.053 m high
                        p->setFileName(0, "Windrad.ive");
                    }
                    else if (featureEntry.type == powerline)
                    {
                        const char* Voltage = featureEntry.feature->GetFieldAsString("nimipinge");  // Adjust field name

                        float v = 0;
                        if(Voltage != "")
                        {
                            try {
                                v = std::stof(Voltage);
                            }
                            catch(...){ }
                            
                        }
                        if (v <= 10)
                        {
                            p->setFileName(0, "FreileitungSmall.ive");
                        }
                        else if (v >= 110)
                        {
                            p->setFileName(0, "Freileitung.ive");
                        }
                        else
                        {
                            p->setFileName(0, "Freileitung20.ive");
                        }
                    }
                    osg::Vec3d position(featureEntry.y,featureEntry.x,featureEntry.z);
                   // osg::Vec3 direction;
                    //float angle = atan2(direction.x(), direction.y());
                    m->setMatrix(osg::Matrix::scale(osg::Vec3(scale, scale, scale)) * osg::Matrix::rotate(angle, osg::Vec3(0, 0, 1)) * osg::Matrix::translate(position));

                    m->addChild(p);
                    root->addChild(m);
                }
                osgDB::writeNodeFile(*root.get(), filename.str());
            }

        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2)
    {
        std::string filename = argv[1];
        if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".shp")
            return tileShp(filename);
        else if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".gml")
            return tileCitygml(filename);
        else
        {

            std::cerr << "Error: unrecognized file type" << std::endl;
            std::cerr << "Usage: tileCitygml [file.gml] [file.shp]" << std::endl;
        }
    }
    else
    {
        std::cerr << "Usage: tileCitygml [file.gml] [file.shp]" << std::endl;
    }

    return 0;
}
