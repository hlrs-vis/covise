// assigning positions for TUI-Elements

#ifndef coMUIPositionManager_H
#define coMUIPositionManager_H

#include <iostream>
#include <vector>
#include <boost/smart_ptr.hpp>


// forward declaration
class coMUIElementManager;
class coMUIElement;
class coMUIWidget;
class coMUIContainer;


class coMUIPositionManager
{
public:
    coMUIPositionManager();
    ~coMUIPositionManager();

    std::pair <int,int> getFreePos(std::string UniqueIdentifierParent);         // returns Coordinates of free Position
    std::pair <int,int> getFreePosExeptOfPos(std::vector<std::pair <int,int> > exceptPos, std::string UniqueIdentifierParent);
    void addPosToPosList(std::string UniqueIdentifier, std::pair <int,int> Pos, std::string UniqueIdentifierParent, bool autoassigned);
    void deletePosFromPosList(std::string UniqueIdentifier);
    std::pair <int,int> getPosOfElement(std::string UniqueIdentifier);
    void changePosInPosList(std::string UniqueIdentifier, std::pair<int,int> Pos);
    bool isPosOccupied(std::pair<int,int> Pos, std::string UniqueIdentifierParent);
    bool isAutoassigned(std::pair<int,int> Pos, std::string UniqueIdentifierParent);
    std::string getIdentifierByPos(std::pair<int,int> Pos, std::string UniqueIdentifierParent);
    std::string printPos();


private:
    // variables:
    struct Pos;                         // struct for the List of all occupied positions
    std::vector <Pos> PosList;


    boost::shared_ptr<coMUIWidget> setNewPosWidget;
    boost::shared_ptr<coMUIContainer> setNewPosContainer;
    boost::shared_ptr<coMUIElementManager> ElementManager;

    int MaxPosX;                      // max allowed positions in x-direction (shall depend on size of display)

    // methods:
    bool PosInPosList(std::pair <int,int> Coordinates, std::string UniqueIdentifierParent);
};


#endif
