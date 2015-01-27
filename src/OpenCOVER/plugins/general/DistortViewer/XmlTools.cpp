/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "XmlTools.h"
#include "HelpFuncs.h"

XmlTools::XmlTools()
{
    config = new coConfigGroup("Global");

    //Konfig-File zum speichern festlegen
    QString xmlFile = coConfigDefaultPaths::getDefaultGlobalConfigFileName();
    config->addConfig(xmlFile, "distortion", true);
    coConfig::getInstance()->addConfig(config);
    plugPath = "COVER.Plugin.DistortViewer";
}

XmlTools::~XmlTools()
{
    delete config;
}

XmlTools *XmlTools::getInstance()
{
    static XmlTools theInstance;
    return &theInstance;
}

void XmlTools::setNewConfigFile(std::string new_file)
{
    //alte Datei entfernen
    config->removeConfig("distortion");
    //neue Datei hinzufügen
    config->addConfig(QString(new_file.c_str()), "distortion", true);
}

bool XmlTools::chkEntry(std::string section, std::string variable)
{
    coConfigString strValue(config, QString(variable.c_str()), QString(section.c_str()));
    if (strValue.hasValidValue())
        return true;
    else
        return false;
}

void XmlTools::saveToXml()
{
    config->save();
}

void XmlTools::saveStrValue(std::string str_value, std::string section, std::string variable)
{
    coConfigString strValue(config, QString(variable.c_str()), QString(section.c_str())); //Angabe der Config -> setSaveToGroup
    strValue = str_value.c_str();
    saveToXml();
}

std::string XmlTools::loadStrValue(std::string section, std::string variable, std::string defValue)
{
    QString xmlValue;
    xmlValue = config->getValue(QString(variable.c_str()), QString(section.c_str()), QString(defValue.c_str()));
    return xmlValue.toStdString();
}

void XmlTools::saveIntValue(int int_value, std::string section, std::string variable)
{
    coConfigInt intValue(config, QString(variable.c_str()), QString(section.c_str()));
    intValue = int_value;
    saveToXml();
}

int XmlTools::loadIntValue(std::string section, std::string variable, int defValue)
{
    coConfigInt intValue(config, QString(variable.c_str()), QString(section.c_str()));
    if (intValue.hasValidValue())
        return intValue;
    else
        return defValue;
}

void XmlTools::saveFloatValue(float float_value, std::string section, std::string variable)
{
    coConfigFloat floatValue(config, QString(variable.c_str()), QString(section.c_str()));
    floatValue = float_value;
    saveToXml();
}

float XmlTools::loadFloatValue(std::string section, std::string variable, float defValue)
{
    coConfigFloat floatValue(config, QString(variable.c_str()), QString(section.c_str()));
    if (floatValue.hasValidValue())
        return floatValue;
    else
        return defValue;
}

void XmlTools::saveBoolValue(bool bool_value, std::string section, std::string variable)
{
    coConfigBool boolValue(config, QString(variable.c_str()), QString(section.c_str()));
    boolValue = bool_value;
    saveToXml();
}

bool XmlTools::loadBoolValue(std::string section, std::string variable, bool defValue)
{
    coConfigBool boolValue(config, QString(variable.c_str()), QString(section.c_str()));
    if (boolValue.hasValidValue())
        return boolValue;
    else
        return defValue;
}

void XmlTools::saveMatrix(osg::Matrix mat, std::string section)
{
    std::string ssi, ssj;
    std::string variable;
    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned int j = 0; j < 4; j++)
        {
            HelpFuncs::IntToString(i, ssi);
            HelpFuncs::IntToString(j, ssj);
            variable = "elem" + ssi + ssj;
            saveFloatValue(mat(i, j), section.c_str(), variable.c_str());
        }
    }
    saveToXml();
}

osg::Matrix XmlTools::loadMatrix(std::string section, osg::Matrix defMat)
{
    float mat[4][4];
    std::string ssi, ssj;
    std::string variable;
    float xmlValue;
    osg::Matrix matrix;

    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned int j = 0; j < 4; j++)
        {
            HelpFuncs::IntToString(i, ssi);
            HelpFuncs::IntToString(j, ssj);
            variable = "elem" + ssi + ssj;

            //Falls noch kein Eintrag existiert
            if (!chkEntry(section.c_str(), variable.c_str()))
                return defMat;

            xmlValue = loadFloatValue(section.c_str(), variable.c_str());
            mat[i][j] = xmlValue;
        }
    }
    matrix = osg::Matrix(mat[0][0], mat[0][1], mat[0][2], mat[0][3],
                         mat[1][0], mat[1][1], mat[1][2], mat[1][3],
                         mat[2][0], mat[2][1], mat[2][2], mat[2][3],
                         mat[3][0], mat[3][1], mat[3][2], mat[3][3]);

    return matrix;
}

void XmlTools::saveVec3(osg::Vec3 vec, std::string section)
{
    std::string ssi, ssj;
    std::string variable;
    float vecValue;
    for (unsigned int i = 0; i < 3; i++)
    {
        HelpFuncs::IntToString(i, ssi);
        variable = "elem" + ssi;
        vecValue = vec[i];
        saveFloatValue(vecValue, section.c_str(), variable.c_str());
    }
}

osg::Vec3 XmlTools::loadVec3(std::string section, osg::Vec3 defVec)
{
    float vec[3];
    std::string ssi;
    std::string variable;
    float xmlValue;
    osg::Vec3 vector;
    for (unsigned int i = 0; i < 3; i++)
    {
        HelpFuncs::IntToString(i, ssi);
        variable = "elem" + ssi;

        //Falls noch kein Eintrag existiert
        if (!chkEntry(section.c_str(), variable.c_str()))
            return defVec;

        xmlValue = loadFloatValue(section.c_str(), variable.c_str());
        vec[i] = xmlValue;
    }
    vector = osg::Vec3(vec[0], vec[1], vec[2]);

    return vector;
}

void XmlTools::saveVec4(osg::Vec4 vec, std::string section)
{
    std::string ssi;
    std::string variable;
    float vecValue;
    for (unsigned int i = 0; i < 4; i++)
    {
        HelpFuncs::IntToString(i, ssi);
        variable = "elem" + ssi;
        vecValue = vec[i];
        saveFloatValue(vecValue, section.c_str(), variable.c_str());
    }
}

osg::Vec4 XmlTools::loadVec4(std::string section, osg::Vec4 defVec)
{
    float vec[4];
    std::string ssi;
    std::string variable;
    float xmlValue;
    osg::Vec4 vector;

    for (unsigned int i = 0; i < 4; i++)
    {
        HelpFuncs::IntToString(i, ssi);
        variable = "elem" + ssi;

        //Falls noch kein Eintrag existiert
        if (!chkEntry(section.c_str(), variable.c_str()))
            return defVec;

        xmlValue = loadFloatValue(section.c_str(), variable.c_str());
        vec[i] = xmlValue;
    }
    vector = osg::Vec4(vec[0], vec[1], vec[2], vec[3]);

    return vector;
}
