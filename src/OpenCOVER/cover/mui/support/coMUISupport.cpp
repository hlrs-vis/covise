// no class; just some support-functions

#include "coMUISupport.h"
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>

using namespace std;

int coMUISupport::readIntFromString(const string String, int pos){
    string NumberString= "0123456789";
    string actualNumber_str;
    string zwischenstring;
    int actualNumber_int;
    int counter = 0;

    int returnArray[readIntFromStringGetArraySize(String)];

    for (int i=0; i<=String.size(); i++){                       // pass each character of the string
        if (NumberString.find(String[i])!=string::npos){        // character is a number
            zwischenstring = String[i];
            actualNumber_str.append(zwischenstring);            // append the actual numeral to the actual number

        }
        else{                                                   // character is not a numeral
            if (!actualNumber_str.empty()){                     // there is already a numeral in actualNumber
                actualNumber_int = boost::lexical_cast<int>(actualNumber_str);      // actualNumber_str is casted into int and saved in actaulNumber_int
                counter ++;
                returnArray[counter]=actualNumber_int;
                actualNumber_str.clear();
            }
            else{                                               // no number in actualNumber

            }
        }
    }
    return returnArray[pos];
}


int coMUISupport::readIntFromStringGetArraySize(const std::string String){
    string NumberString = "0123456789";
    int sizeReturnArray = 0;
    bool lastCharZahl = 0;

    if (NumberString.find(String[0])!=string::npos){            // check, if first element is a number
        lastCharZahl=1;
        sizeReturnArray++;
    }
    for (int i=0; i<String.size(); ++i){                        // determine the size of returnArray
        if (NumberString.find(String[i])!=string::npos){        // character is a number
            if (lastCharZahl==0){                               // previous character was not a number
                sizeReturnArray++;
                lastCharZahl=1;
            }
        }
        else if (!(NumberString.find(String[i])!=string::npos)){  // character is not a number
            lastCharZahl=0;
        }
    }
    return sizeReturnArray;
}
