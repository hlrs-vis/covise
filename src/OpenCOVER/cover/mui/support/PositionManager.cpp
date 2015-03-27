#include "PositionManager.h"
#include "ElementManager.h"
#include <boost/lexical_cast.hpp>

using namespace mui;

// constructor:
PositionManager::PositionManager()
{
    MaxPosX=5;
}

// destructor:
PositionManager::~PositionManager()
{
}

// create stuct:
struct PositionManager::Pos
{
    std::string UniqueIdentifier;
    std::string UniqueIdentifierParent;
    bool autoassigned;
    std::pair <int,int> pos;
};


// adds an element/position to the list
void PositionManager::addPosToPosList(std::string UniqueIdentifier, std::pair<int,int> pos, std::string UniqueIdentifierParent, bool autoassigned)
{
    if (pos.first > MaxPosX)
    {
        std::cerr << "WARNING: PositionManager::addPos(): X-Position of " << UniqueIdentifier << " exceeds allowed MaxPosX= " << MaxPosX << std::endl;
    }
    PosList.push_back(Pos());
    PosList.back().UniqueIdentifier= UniqueIdentifier;
    PosList.back().UniqueIdentifierParent = UniqueIdentifierParent;
    PosList.back().autoassigned = autoassigned;
    PosList.back().pos = pos;
}


// find a free position and returns the coordinates
std::pair <int,int> PositionManager::getFreePos(std::string UniqueIdentifierParent)
{
    std::pair <int,int> returnCoordinates;
    int y = 0;
    while (true)                                   // loop infinite y-positions
    {
        returnCoordinates.second=y;
        for (int x=0; x<MaxPosX; ++x)              // loop over all allowed x-positions
        {
            returnCoordinates.first=x;
            if (!PosInPosList(returnCoordinates, UniqueIdentifierParent))
            {
                return returnCoordinates;
            }
        }
        ++y;                                                // increase y-position in every loop
    }
}

std::pair <int,int> PositionManager::getFreePosExeptOfPos(std::vector<std::pair <int,int> > exceptPos, std::string UniqueIdentifierParent)
{
    std::pair <int,int> returnCoordinates;
    int y = 0;
    while (true)                                   // loop infinite y-positions
    {
        returnCoordinates.second=y;
        for (int x=0; x<MaxPosX; ++x)              // loop over all allowed x-positions
        {
            returnCoordinates.first=x;
            if (!PosInPosList(returnCoordinates, UniqueIdentifierParent))
            {
                bool existflag = false;
                for (size_t i=0; i<exceptPos.size(); ++i)
                {
                    std::pair <int,int> tempPos = exceptPos[i];
                    if (returnCoordinates.first == tempPos.first)
                    {
                        existflag = true;
                    }
                }
                if (!existflag)
                {
                    return returnCoordinates;
                }
            }
        }
        ++y;                                                // increase y-position in every loop
    }
}


// determine, if the coordinates already are in PosList
bool PositionManager::PosInPosList(std::pair <int,int> Coords, std::string UniqueIdentifierParent)
{
    for (size_t i=0; i<PosList.size(); ++i)
    {
        if ((Coords == PosList[i].pos) && (UniqueIdentifierParent==PosList[i].UniqueIdentifierParent))
        {
            return true;
        }
    }
    return false;
}

// delete entrys from PosList
void PositionManager::deletePosFromPosList(std::string UniqueIdentifier)
{
    bool deleteFlag = false;
    for (ssize_t i=PosList.size()-1; i >= 0; --i)
    {
        if (PosList[i].UniqueIdentifier == UniqueIdentifier)
        {
            PosList.erase(PosList.begin()+i);
            deleteFlag = true;
        }
    }
    if (deleteFlag)
    {
        return;
    }
    else
    {
        std::cout << "WARNING: PositionManager::deletePos(): " << UniqueIdentifier << " can't be removed. Reason: not found in PosList." << std::endl;
    }
}

// returns the position of element with identifier "UniqueIdentifier"
std::pair <int,int> PositionManager::getPosOfElement(std::string UniqueIdentifier)
{
    std::pair <int,int> returnCoordinates;
    for (size_t i=0; i<PosList.size(); ++i)
    {
        if (PosList[i].UniqueIdentifier == UniqueIdentifier)
        {
            returnCoordinates = PosList[i].pos;
            return returnCoordinates;
        }
    }
    std::cerr << "ERROR: PositionManager::getPos(): Position of " << UniqueIdentifier << " can't be returned. Reason: not found in PosList." << std::endl;
    returnCoordinates.first = returnCoordinates.second = 0;
    return returnCoordinates;
}

// changes position of element with identifier "UniqueIdentifier"
void PositionManager::changePosInPosList(std::string UniqueIdentifier, std::pair<int,int> pos)
{
    for (size_t i=0; i<PosList.size(); ++i)
    {
        if (PosList[i].UniqueIdentifier == UniqueIdentifier)
        {
            PosList[i].pos = pos;
        }
    }
}

// returns true, if position is occupied, else returns false
bool PositionManager::isPosOccupied(std::pair<int,int> pos, std::string UniqueIdentifierParent)
{
    for (size_t i=0; i<PosList.size(); ++i)
    {
        if (PosList[i].UniqueIdentifierParent == UniqueIdentifierParent)
        {
            if (PosList[i].pos == pos)
            {
                return true;
            }
        }
    }
    return false;
}

// returns true, if position is occupied by an autoassigned element; else returns false
bool PositionManager::isAutoassigned(std::pair<int,int> pos, std::string UniqueIdentifierParent)
{
    for (size_t i=0; i<PosList.size(); ++i)
    {
        if (PosList[i].UniqueIdentifierParent == UniqueIdentifierParent)
        {
            if (PosList[i].pos == pos)
            {
                return PosList[i].autoassigned;
            }
        }
    }
    return false;
}

// returns the identifier of the element at position posx/posy and parent UniqueIdentifierParent
std::string PositionManager::getIdentifierByPos(std::pair<int,int> pos, std::string UniqueIdentifierParent)
{
    for (size_t i=0; i<PosList.size(); ++i)
    {
        if (PosList[i].UniqueIdentifierParent == UniqueIdentifierParent)
        {
            if (PosList[i].pos == pos)
            {
                return PosList[i].UniqueIdentifier;
            }
        }
    }
    return "";
}

// returns all positions of PosList in one string
std::string PositionManager::printPos()
{
    std::string Positions="";
    for (size_t i=0; i<PosList.size(); ++i)
    {
        Positions.append(PosList[i].UniqueIdentifier);
        Positions.append(":(");
        Positions.append(boost::lexical_cast<std::string>(PosList[i].pos.first));
        Positions.append(",");
        Positions.append(boost::lexical_cast<std::string>(PosList[i].pos.second));
        Positions.append("); \n");
    }
    return Positions;
}
