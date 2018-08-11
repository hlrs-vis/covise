#ifndef NOZZLEMANAGER_H
#define NOZZLEMANAGER_H

#include <list>
#include <string>
#include "nozzle.h"
#include <osg/Matrix>
#include <sstream>
#include <fstream>
#include <iostream>
#include "types.h"

#include "parser.h"


typedef struct{
    int count;
    std::string name;
    float position[15];
    std::string type;

    std::string filename;
    std::string pathname;

    float angle;
    std::string decoy;

}nozzleParam;

class nozzleManager
{
private:
    static nozzleManager* _instance;
    nozzleManager(){}
    nozzleManager(const nozzleManager&);
    ~nozzleManager(){}
    std::list<nozzle*> nozzleList;

    class nozzleManagerGuard
    {
    public:
        ~nozzleManagerGuard(){
            if(NULL != nozzleManager::_instance){
                delete nozzleManager::_instance;
                nozzleManager::_instance = NULL;
            }
        }

    };

    int nozzleCount = 0;
    int nextNozzleID = 0;


public:
    static nozzleManager* instance()
    {
        static nozzleManagerGuard g;
        if(!_instance)
        {
            _instance = new nozzleManager;            
        }
        return _instance;

    }

    void init();

    nozzle *createNozzle(std::string nozzleName);
    nozzle *createImageNozzle(std::string nozzleName, std::string pathName, std::string fileName);
    nozzle *createStandardNozzle(std::string nozzleName, float sprayAngle, std::string decoy);

    int removeNozzle(int index);
    void saveNozzle(std::string pathName, std::string fileName);
    void loadNozzle(std::string pathName, std::string fileName);

    void update();
    void remove_all();
    nozzle* checkAll();

    nozzle* getNozzle(int index);
};


#endif // NOZZLEMANAGER_H
