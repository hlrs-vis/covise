#include "nozzlemanager.h"

nozzle* nozzleManager::createNozzle(std::string nozzleName)
{
    //Will never be called
    std::string newNozzleName = nozzleName;

    printf("Start creating new nozzle!\n");
    if(nozzleName.empty())
    {
        char ptr[33];
        sprintf(ptr, "coNozzle%i", nextNozzleID);
        newNozzleName.clear();
        newNozzleName.append(ptr);
    }

    osg::Matrix initialMat;
    initialMat.makeIdentity();
//    initialMat.setTrans(cover->getPointerMat().getTrans());
//    float baseTransform[] = {1,0,0,0,
//                             0,1,0,0,
//                             0,0,1,0,
//                             0,0,0,1
//                            };
//    initialMat.set(baseTransform);
    initialMat = cover->getPointerMat()*initialMat;

    class nozzle* newNozzle = new class nozzle(initialMat, 1 , newNozzleName);
    newNozzle->setID(nextNozzleID);

    newNozzle->enableIntersection();

    newNozzle->resizeBoundingBox(currentBoundingBox);

    nozzleList.push_back(newNozzle);

    nozzleCount++;
    nextNozzleID++;

    printf("New nozzle created!\n");

    return newNozzle;

}

nozzle* nozzleManager::createImageNozzle(std::string nozzleName, std::string pathName, std::string fileName)
{

    std::string newNozzleName = nozzleName;

    printf("Start creating new nozzle!\n");
    if(nozzleName.empty())
    {
        char ptr[33];
        sprintf(ptr, "coNozzle%i", nextNozzleID);
        newNozzleName.clear();
        newNozzleName.append(ptr);
    }

    osg::Matrix initialMat;
    initialMat.makeIdentity();
//    initialMat.setTrans(cover->getPointerMat().getTrans());
//    float baseTransform[] = {1,0,0,0,
//                             0,1,0,0,
//                             0,0,1,0,
//                             0,0,0,1
//                            };
//    initialMat.set(baseTransform);
    initialMat = cover->getPointerMat()*initialMat;
    class imageNozzle* newNozzle = new class imageNozzle(pathName, fileName, initialMat, 1, newNozzleName);
    if(newNozzle->isFailed())
    {
        delete newNozzle;
        return NULL;
    }

    newNozzle->setID(nextNozzleID);
    newNozzle->enableIntersection();

    nozzleList.push_back(newNozzle);

    nozzleCount++;
    nextNozzleID++;

    printf("New nozzle created!\n");

    return newNozzle;
}

nozzle* nozzleManager::createStandardNozzle(std::string nozzleName, float sprayAngle, std::string decoy)
{
    std::string newNozzleName = nozzleName;

    printf("Start creating new nozzle!\n");
    if(nozzleName.empty())
    {
        char ptr[33];
        sprintf(ptr, "coNozzle%i", nextNozzleID);
        newNozzleName.clear();
        newNozzleName.append(ptr);
    }

    osg::Matrix initialMat;
    initialMat.makeIdentity();
//    initialMat.setTrans(cover->getPointerMat().getTrans());
//    float baseTransform[] = {1,0,0,0,
//                             0,1,0,0,
//                             0,0,1,0,
//                             0,0,0,1
//                            };
//    initialMat.set(baseTransform);
    initialMat = cover->getPointerMat()*initialMat;

    class standardNozzle* newNozzle = new class standardNozzle(sprayAngle, decoy, initialMat, 1, newNozzleName);

    newNozzle->setID(nextNozzleID);

    newNozzle->enableIntersection();

    nozzleList.push_back(newNozzle);

    nozzleCount++;
    nextNozzleID++;

    printf("New nozzle created!\n");

    return newNozzle;

}

void nozzleManager::init()
{
    nozzleList.clear();
    nextNozzleID = 0;
}

int nozzleManager::removeNozzle(int index)
{
    if(!nozzleList.empty())
    {
        for(auto i = nozzleList.begin();i != nozzleList.end(); i++){
            class nozzle* current = *i;
            if(current->getID() == index)
            {
                delete current;
                nozzleList.erase(i);
                nozzleCount--;
                printf("Remove completed!\n");
                return 0;
            };
        }

    }
    else
        printf("Remove failed!\n");
    return 1;
}

void nozzleManager::saveNozzle(std::string pathName, std::string fileName)
{
    printf("Starting to save nozzle configuration.\n");

    std::string newFileName = fileName;

    if(fileName.empty())
    {
        printf("Filename Field is empty!\n");
        newFileName = "untitled.txt";
    }

    FILE* saving = new FILE;
    char ptr[10];
    saving = fopen((pathName +newFileName).c_str(), "w+");
    if(saving != NULL)
    {
        fputs("Nozzlecount: ", saving);
        sprintf(ptr, "%i", nozzleCount);
        fputs(ptr, saving);
        fputs("\n\n", saving);
        fclose(saving);

        if(!nozzleList.empty())
        {
            for(auto i = nozzleList.begin();i != nozzleList.end(); i++)
            {
                class nozzle* current = *i;
                current->save(pathName, newFileName);
            }
        }
        saving = fopen((pathName +newFileName).c_str(), "a");
        if(saving != NULL)
        {
            fputs("\n", saving);
            fputs("#", saving);
        }
        fclose(saving);

        printf("Saving complete!\n");
    }
    else printf("Saving is currently impossible!\n");
}

void nozzleManager::loadNozzle(std::string pathName, std::string fileName)
{
    printf("Starting to load nozzle configuration.\n");

    std::string line;
    int nozzleCount = 0;
    std::ifstream mystream(pathName+fileName);

    //Currently, the parser is written for a file with name, position, spray angle, decoy, filename, pathname
    //minimum and deviation

    if(mystream.is_open())
    {
        std::getline(mystream,line);
        std::stringstream ss(line);
        std::cout << line << std::endl;
        std::getline(ss,line,':');
        if(line.compare("Nozzlecount") == 0)
            {                
                std::getline(ss,line,'\n');
            }
        nozzleCount = std::stoi(line);

        nozzleParam param;

        while(std::getline (mystream,line))
        {
            if(line.empty())
                {                    
                    continue;
                }
            std::stringstream nextss(line);
            getline(nextss,line, '=');

            if(line.compare("name ") == 0 || line.compare("name") == 0)
            {                
                std::getline(nextss,line,'\n');
                if(line[0] == '0') line.erase(0,1);
                param.name = line;
            }

            if(line.compare("position ") == 0 || line.compare("position") == 0)
            {
                std::getline (mystream,line);
                for(int i = 0; i<4; i++)
                {
                    std::stringstream posStream(line);

                    for(int j = 0; j<4;j++){
                        std::getline(posStream,line,' ');
                        param.position[j+i*4] = stof(line);
                    }
                    std::getline(mystream, line);

                }

            }

            if(line.compare("minimum ") == 0 || line.compare("minimum") == 0)
            {
                std::getline(nextss,line,'\n');
                param.minimum = stof(line);
            }

            if(line.compare("deviation ") == 0 || line.compare("deviation") == 0)
            {
                std::getline(nextss,line,'\n');
                param.deviation = stof(line);
            }

            if(line.compare("type ") == 0 || line.compare("type") == 0)
            {                
                std::getline(nextss,line,'\n');
                if(line[0] == '0') line.erase(0,1);
                param.type = line;

                if(line.compare(" standard") == 0 || line.compare("standard") == 0){
                    std::cout << "found standard type" << std::endl;
                }

                if(line.compare(" image") == 0 || line.compare("image") == 0){
                    std::cout << "found image type" << std::endl;
                }

                for(int i = 0; i<2;i++){
                    std::getline (mystream,line);
                    std::stringstream typess(line);
                    std::getline(typess,line, '=');

                    if(line.compare("pathname ") == 0 || line.compare("pathname") == 0)
                    {
                        std::getline(typess,line,'\n');
                        int x = 0;
                        while(line[x] == ' ')
                            x++;
                        line.erase(0,x);
                        param.pathname = line;
                    }

                    if(line.compare("filename ") == 0 || line.compare("filename") == 0)
                    {
                        std::getline(typess,line,'\n');
                        int x = 0;
                        while(line[x] == ' ')
                            x++;
                        line.erase(0,x);
                        param.filename = line;
                    }

                    if(line.compare("angle ") == 0 || line.compare("angle") == 0)
                    {
                        std::getline(typess,line,'\n');
                        param.angle = stof(line);
                    }

                    if(line.compare("decoy ") == 0 || line.compare("decoy") == 0)
                    {
                        std::getline(typess,line,'\n');
                        if(line[0] == '0') line.erase(0,1);
                        param.decoy = line;
                    }
                }

            }
            if(line.compare("-") == 0)
            {
                std::cout << "Creating a " << param.type << "nozzle" << std::endl;
                if(param.type.compare("standard") == 0 || param.type.compare(" standard") == 0)
                {
                    osg::Matrix initialMat;
                    initialMat.set(param.position);

                    class standardNozzle* newNozzle = new class standardNozzle(param.angle, param.decoy, initialMat,1 ,param.name);

                    newNozzle->setID(nextNozzleID);

                    newNozzle->enableIntersection();

                    newNozzle->setMinimum(param.minimum);

                    newNozzle->setDeviation(param.deviation);

                    nozzleList.push_back(newNozzle);

                    nozzleCount++;
                    nextNozzleID++;
                }

                if(param.type.compare("image") == 0 || param.type.compare(" image") == 0)
                {
                    osg::Matrix initialMat;
                    initialMat.set(param.position);

                    class imageNozzle* newNozzle = new class imageNozzle(param.pathname, param.filename, initialMat,1 ,param.name);

                    newNozzle->setID(nextNozzleID);

                    newNozzle->enableIntersection();

                    newNozzle->setMinimum(param.minimum);

                    newNozzle->setDeviation(param.deviation);

                    nozzleList.push_back(newNozzle);

                    nozzleCount++;
                    nextNozzleID++;

                }
                std::cout << "end of current nozzle definition" << std::endl;
            }

            if(line.compare("#") == 0)
            {
                std::cout << "end of nozzle definitions" << std::endl;
                break;
            }
        }
        mystream.close();
        std::cout << "Loading has finished!" << std::endl;
    }

    else cout << "Couldn't open the desired file" << std::endl;


}

void nozzleManager::update()
{
        if(!nozzleList.empty())
        {
            for(auto i = nozzleList.begin();i != nozzleList.end(); i++)
            {
                class nozzle* current = *i;
                current->updateGen();
            }
        }
}

nozzle* nozzleManager::getNozzle(int index)
{
    if(!nozzleList.empty())
    {
        for(auto i = nozzleList.begin();i != nozzleList.end(); i++)
        {
            class nozzle* current = *i;
            if(current->getID() == index)
                return current;
        }
    }
    std::cout << "Nozzle doesn't exist" <<std::endl;
    return 0;
}

void nozzleManager::remove_all()
{
    if(!nozzleList.empty())
    {
        for(auto i = nozzleList.begin();i != nozzleList.end(); i++)
        {
            class nozzle* current = *i;

            delete current;
            std::cout << "Element cleared" << std::endl;
            nozzleCount--;
        }
    }
    else
        std::cout << "List is already empty" << std::endl;

}

nozzle* nozzleManager::checkAll()
{
    if(!nozzleList.empty())
    {
        for(auto i = nozzleList.begin();i != nozzleList.end(); i++)
        {
            class nozzle* current = *i;
            if(!current->isRegistered())
                return current;
        }
    }
    return 0;
}

void nozzleManager::setBoundingBox(osg::Vec3 newBoundingBox)
{    
    currentBoundingBox = newBoundingBox;
    if(!nozzleList.empty()){
        for(auto i = nozzleList.begin();i != nozzleList.end(); i++)
        {
            class nozzle* current = *i;
            current->resizeBoundingBox(newBoundingBox);
        }
    }
    else
        printf("List is empty\n");
}

void nozzleManager::autoremove(bool state)
{
    if(!nozzleList.empty())
    {
        for(auto i = nozzleList.begin();i != nozzleList.end(); i++)
        {
            class nozzle* current = *i;
            current->autoremove(state);
        }
    }
    else
        printf("List is empty\n");
}

void nozzleManager::removeAllParticles()
{
    if(!nozzleList.empty())
    {
        for(auto i = nozzleList.begin();i != nozzleList.end(); i++)
        {
            class nozzle* current = *i;
            current->deleteAllGen();
        }
    }
    else
        printf("List is empty\n");
}

