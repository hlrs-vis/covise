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


class coMUIPositionManager{
public:
    coMUIPositionManager();
    ~coMUIPositionManager();

    std::vector <int> getFreePos(std::string UniqueIdentifierParent);         // returns Coordinates of free Position
    void addPos(std::string UniqueIdentifier, int posx, int posy, std::string UniqueIdentifierParent, bool autoassigned);
    void deletePos(std::string UniqueIdentifier);
    std::vector <int> getPos(std::string UniqueIdentifier);
    void changePos(std::string UniqueIdentifier, int xPos, int yPos);
    bool isOccupied(int posx, int posy, std::string UniqueIdentifierParent);
    bool isAutoassigned(int posx, int posy, std::string UniqueIdentifierParent);
    std::string getIdentifier(int posx, int posy, std::string UniqueIdentifierParent);
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
    bool PosInPosList(std::vector <int> Coordinates, std::string UniqueIdentifierParent);
};


#endif
