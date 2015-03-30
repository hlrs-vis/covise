#ifndef MUIDEFAULTVALUES_H
#define MUIDEFAULTVALUES_H

#include <iostream>


namespace mui
{
// Class DefaultValues handles the default-values and returns them to the other functions
// Default-values can easily be changed here
class DefaultValues
{
private:
    struct device
    {
        std::string UI;
        std::string Device;
    };
public:
    DefaultValues();
    ~DefaultValues();

    // accerss to the default-values:
    int *getDefaultPosition();

    std::string getKeywordCAVE();       // returns keyword for CAVE in configuration file
    std::string getKeywordTablet();     // returns keyword for Tablet in configuration file
    std::string getKeywordTUI();        // returns keyword for TUI in configuration file
    std::string getKeywordVRUI();       // returns keyword for VRUI in configuration file
    std::string getKeywordPowerwall();  // returns keyword for Powerwall in configuration file
    std::string getKeywordPhone();      // returns keyword for Phone in configuration file
    std::string getKeywordMainWindow(); // returns keyword for MainWindow in configuration file

    std::string getKeywordVisible();
    std::string getKeywordParent();
    std::string getKeywordXPosition();
    std::string getKeywordYPosition();
    std::string getKeywordLabel();
    std::string getKeywordClass();

    std::string getCorrectLabel();
    std::string getCorrectLabel(std::string label);

private:

    // default-values:
    std::string keywordTUI;                           // keyword for TUI in configuration file
    std::string keywordVRUI;                          // keyword for VRUI in configuration file
    std::string keywordTablet;                        // keyword for Tablet in configuration file
    std::string keywordCAVE;                          // keyword for CAVE in configuration file
    std::string keywordPhone;                         // keyword for Phone in configuration file
    std::string keywordPowerwall;                     // keyword for Powerwall in configuration file
    std::string keywordMainWindow;                    // keyword for MainWindow in configuration file
    std::string keywordVisible;
    std::string keywordParent;
    std::string keywordXPosition;
    std::string keywordYPosition;
    std::string keywordLabel;
    std::string keywordClass;


    bool visible;                 // should be visible all the time
    std::pair<int,int> position;                    // return an integer-Array of size 2
    bool PositionFirstCall;

};
} // end namespace
#endif
