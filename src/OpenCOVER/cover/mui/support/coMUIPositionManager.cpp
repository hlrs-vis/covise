#include "coMUIPositionManager.h"
#include "coMUIElementManager.h"
#include <boost/lexical_cast.hpp>


// constructor:
coMUIPositionManager::coMUIPositionManager(){
    MaxPosX=5;
}

// destructor:
coMUIPositionManager::~coMUIPositionManager(){
}

// create stuct:
struct coMUIPositionManager::Pos{
    std::string UniqueIdentifier;
    std::string UniqueIdentifierParent;
    bool autoassigned;
    int posx;
    int posy;
};


// adds an element/position to the list
void coMUIPositionManager::addPos(std::string UniqueIdentifier, int posx, int posy, std::string UniqueIdentifierParent, bool autoassigned){
    if (posx > MaxPosX){
        std::cerr << "WARNING: coMUIPositionManager::addPos(): X-Position of " << UniqueIdentifier << " exceeds allowed MaxPosX= " << MaxPosX << std::endl;
    }
    PosList.push_back(Pos());
    PosList.back().UniqueIdentifier= UniqueIdentifier;
    PosList.back().UniqueIdentifierParent = UniqueIdentifierParent;
    PosList.back().autoassigned = autoassigned;
    PosList.back().posx = posx;
    PosList.back().posy = posy;
}


// find a free position and returns the coordinates
std::vector <int> coMUIPositionManager::getFreePos(std::string UniqueIdentifierParent){
    std::vector <int> returnCoordinates(2);
    returnCoordinates[0]=0;
    returnCoordinates[1]=0;
    int y = 0;
    while (true){                                   // loop infinite y-positions
        ++y;                                        // increase y-position in every loop
        returnCoordinates[1]=y;
        for (int x=1; x<MaxPosX; ++x){              // loop over all allowed x-positions
            returnCoordinates[0]=x;
            if (!PosInPosList(returnCoordinates, UniqueIdentifierParent)){
                return returnCoordinates;
            }
        }
    }
}


// determine, if the coordinates already are in PosList
bool coMUIPositionManager::PosInPosList(std::vector <int> Coords, std::string UniqueIdentifierParent){
    for (int i=0; i<PosList.size(); ++i){
        if ((Coords[0]==PosList[i].posx) && (Coords[1]==PosList[i].posy) && (UniqueIdentifierParent==PosList[i].UniqueIdentifierParent)){
            return true;
        }
    }
    return false;
}

// delete entrys from PosList
void coMUIPositionManager::deletePos(std::string UniqueIdentifier){
    for (int i=0; i<PosList.size(); ++i){
        if (PosList[i].UniqueIdentifier == UniqueIdentifier){
            PosList.erase(PosList.begin()+i);
            return;
        }
    }
    std::cout << "WARNING: coMUIPositionManager::deletePos(): " << UniqueIdentifier << " can't be removed. Reason: not found in PosList." << std::endl;
}

// returns the position of element with identifier "UniqueIdentifier"
std::vector <int> coMUIPositionManager::getPos(std::string UniqueIdentifier){
    std::vector <int> returnCoordinates(2);
    for (int i=0; i<PosList.size(); ++i){
        if (PosList[i].UniqueIdentifier == UniqueIdentifier){
            returnCoordinates[0] = PosList[i].posx;
            returnCoordinates[1] = PosList[i].posy;
            return returnCoordinates;
        }
    }
    std::cerr << "WARNING: coMUIPositionManager::getPos(): Position of " << UniqueIdentifier << " can't be returned. Reason: not found in PosList." << std::endl;
    returnCoordinates[0] = returnCoordinates[1] = 0;
    return returnCoordinates;
}

// changes position of element with identifier "UniqueIdentifier"
void coMUIPositionManager::changePos(std::string UniqueIdentifier, int xPos, int yPos){
    for (int i=0; i<PosList.size(); ++i){
        if (PosList[i].UniqueIdentifier == UniqueIdentifier){
            PosList[i].posx = xPos;
            PosList[i].posy = yPos;
        }
    }
}

// returns true, if position is occupied, else returns false
bool coMUIPositionManager::isOccupied(int posx, int posy, std::string UniqueIdentifierParent){
    for (int i=0; i<PosList.size(); ++i){
        if (PosList[i].UniqueIdentifierParent == UniqueIdentifierParent){
            if ((PosList[i].posx == posx) && (PosList[i].posy == posy)){
                return true;
            }
        }
    }
    return false;
}

// returns true, if position is occupied by an autoassigned element; else returns false
bool coMUIPositionManager::isAutoassigned(int posx, int posy, std::string UniqueIdentifierParent){
    for (int i=0; i<PosList.size(); ++i){
        if (PosList[i].UniqueIdentifierParent == UniqueIdentifierParent){
            if ((PosList[i].posx == posx) && (PosList[i].posy == posy)){
                return PosList[i].autoassigned;
            }
        }
    }
    return false;
}

// returns the identifier of the element at position posx/posy and parent UniqueIdentifierParent
std::string coMUIPositionManager::getIdentifier(int posx, int posy, std::string UniqueIdentifierParent){
    for (int i=0; i<PosList.size(); ++i){
        if (PosList[i].UniqueIdentifierParent == UniqueIdentifierParent){
            if ((PosList[i].posx == posx) && (PosList[i].posy == posy)){
                return PosList[i].UniqueIdentifier;
            }
        }
    }
    return "";
}

// returns all positions of PosList in one string
std::string coMUIPositionManager::printPos(){
    std::string Positions="";
    for (int i=0; i<PosList.size(); ++i){
        Positions.append(PosList[i].UniqueIdentifier);
        Positions.append(": (");
        Positions.append(boost::lexical_cast<std::string>(PosList[i].posx));
        Positions.append(",");
        Positions.append(boost::lexical_cast<std::string>(PosList[i].posy));
        Positions.append("); ");
    }
    return Positions;
}
