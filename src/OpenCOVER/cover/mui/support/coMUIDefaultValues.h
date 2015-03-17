#ifndef COMUIDEFAULTVALUES_H
#define COMUIDEFAULTVALUES_H

#include <iostream>

/*const std::string keywordTUI = "TUI";
const std::string keywordVRUI = "VRUI";
const std::string keywordTablet = "Tablet";
const std::string keywordCAVE = "CAVE";
const std::string keywordPhone = "Phone";
const std::string keywordPowerwall = "Powerwall";
*/
// Class coMUIDefaultValues handles the default-values and returns them to the other functions
// Default-values can easily be changed here
class coMUIDefaultValues{
private:
    struct device{
        std::string UI;
        std::string Device;
    };
public:
    coMUIDefaultValues();
    ~coMUIDefaultValues();

    // accerss to the default-values:
    int *getDefaultPosition();

    std::string getKeywordCAVE();       // returns keyword for CAVE in configuration file
    std::string getKeywordTablet();     // returns keyword for Tablet in configuration file
    std::string getKeywordTUI();        // returns keyword for TUI in configuration file
    std::string getKeywordVRUI();       // returns keyword for VRUI in configuration file
    std::string getKeywordPowerwall();  // returns keyword for Powerwall in configuration file
    std::string getKeywordPhone();      // returns keyword for Phone in configuration file
    std::string getKeywordMainWindow(); // returns keyword for MainWindow in configuration file

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

    bool visible;                 // should be visible all the time
    int position[2];                    // return an integer-Array of size 2
    bool PositionFirstCall;

};

#endif
