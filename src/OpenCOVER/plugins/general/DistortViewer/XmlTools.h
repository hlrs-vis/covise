/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cmath>
#include <iostream>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include <sstream>

#include <cover/coVRPluginSupport.h>
#include <config/coConfig.h>
#include <config/CoviseConfig.h>
using namespace covise;
using namespace opencover;

//Singleton Klasse
class XmlTools
{
private:
    XmlTools(void);
    ~XmlTools(void);

    coConfigGroup *config;
    std::string plugPath;

public:
    //Singletoninstanz nicht als Member der Klasse, sondern in getInstance als static definiert
    //(vgl. http://oette.wordpress.com/2009/09/11/singletons-richtig-verwenden/)
    //-> Singletoninstanz wird erst erzeugt, wenn Funktion getInstance() erstmals aufgerufen(Lazy Creation)
    //-> wird als static erzeugt, d.h. sie exitiert nur einmal!
    //-> kein release() nötig, statische Variable wird durch einen dynamischen AtExit Destruktor am Programmende zerstört
    static XmlTools *getInstance();

    void setNewConfigFile(std::string new_file);

    std::string getPlugPath(void)
    {
        return plugPath;
    };

    //XML-Einträge zum laden/schreiben setzen
    bool chkEntry(std::string section, std::string variable);
    void saveStrValue(std::string str_value, std::string section, std::string variable);
    std::string loadStrValue(std::string section, std::string variable, std::string defValue = "");
    void saveIntValue(int int_value, std::string section, std::string variable);
    int loadIntValue(std::string section, std::string variable, int defValue = 0);
    void saveFloatValue(float float_value, std::string section, std::string variable);
    float loadFloatValue(std::string section, std::string variable, float defValue = 0.0f);
    void saveBoolValue(bool bool_value, std::string section, std::string variable);
    bool loadBoolValue(std::string section, std::string variable, bool defBool = true);
    void saveMatrix(osg::Matrix mat, std::string variable);
    osg::Matrix loadMatrix(std::string variable, osg::Matrix defMat = osg::Matrix::identity());
    void saveVec3(osg::Vec3 vec, std::string variable);
    osg::Vec3 loadVec3(std::string variable, osg::Vec3 defVec = osg::Vec3(1.0f, 0.0f, 0.0f));
    void saveVec4(osg::Vec4 vec, std::string variable);
    osg::Vec4 loadVec4(std::string variable, osg::Vec4 defVec = osg::Vec4(1.0f, 0.0f, 0.0f, 0.0f));

    //Gesetzte Einträge in XML schreiben
    void saveToXml();
};
