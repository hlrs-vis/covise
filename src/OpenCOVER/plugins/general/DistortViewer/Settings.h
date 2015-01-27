/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include <iostream>

//Singleton -> http://oette.wordpress.com/2009/09/11/singletons-richtig-verwenden/
class Settings
{
public:
    Settings(void);
    ~Settings(void);

    static Settings *getInstance(void);
    void loadFromXML();
    void saveToXML();

    int visResolutionW;
    int visResolutionH;
    std::string imagePath;
    std::string fragShaderFile;
    std::string vertShaderFile;

private:
    std::string section;
    std::string plugPath;
    std::string path;
};
