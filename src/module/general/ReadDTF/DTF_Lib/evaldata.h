/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/evaldata.h
 * @brief contains definition of class DTF_Lib::EvalData
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope.
 */

/** @class DTF_Lib::ClassInfo_DTFLibEvalData
 * @brief used to register class DTF_Lib::EvalData at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::EvalData and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibEvalData to create new objects of type DTF_Lib::EvalData.
 */

/** @class DTF_Lib::EvalData
 * @brief contains evaluation data contained in VC and BC records
 *
 * \b Description:
 *
 * VC records and BC records contain evaluation data. This data is divided into
 * - int names and values
 * - real names and values
 * - string names and values
 *
 * To shorten certain function calls, that data is saved and returned in objects
 * of class EvalData.
 */

/** @fn DTF_Lib::EvalData::EvalData();
 * @brief default constructor
 *
 * \b Description:
 *
 * Erases all vector contents.
 */

/** @fn DTF_Lib::EvalData::~EvalData();
 * @brief destructor
 *
 * \b Description:
 *
 * Erases the contents in all vectors of the object.
 */

/** @fn bool DTF_Lib::EvalData::addValue(string name,
 int value);
 * @brief Adds name and value of a int parameter
 *
 * @param name - name of the int parameter
 * @param value - value of the int parameter
 *
 * @return \c false on error, \c true on success.
 */

/** @fn bool DTF_Lib::EvalData::addValue(string name,
            double value);
 * @brief Adds name and value of a real parameter
 *
 * @param name - name of the real parameter
 * @param value - value of the real parameter
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 */

/** @fn bool DTF_Lib::EvalData::addValue(string name,
            string value);
 * @brief Adds name and value of a string parameter
 *
 * @param name - name of the string parameter
 * @param value - value of the string parameter
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 */

/** @fn bool DTF_Lib::EvalData::getInts(vector<string>& names,
           vector<int>& values);
 * @brief gets names and values of all stored int parameters
 *
 * @param names - vector containing the names of the int parameters (output)
 * @param values - vector containing the values of the int parameters (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
*
 * The contents of names and values are completely overwritten.
 */

/** @fn bool DTF_Lib::EvalData::getReals(vector<string>& names,
            vector<double>& values);
 * @brief gets names and values of all stored real values
 *
 * @param names - vector with names of the real parameters
 * @param values - vector with values of the real parameters
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
*
 * The contents of names and values are completely overwritten.
 */

/** @fn bool DTF_Lib::EvalData::getStrings(vector<string>& names,
         vector<string>& values);
 * @brief gets names and values of all stored string values
 *
 * @param names - vector with names of the string parameters
 * @param values - vector with values of the string parameters
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
*
 * The contents of names and values are completely overwritten.
 */

/** @var vector<string> DTF_Lib::EvalData::intNames;
 * @brief vector containing the names of the stored int parameters.
 */

/** @var vector<int> DTF_Lib::EvalData::intValues;
 * @brief vector containing the values of the stored int parameters.
 */

/** @var vector<string> DTF_Lib::EvalData::realNames;
 * @brief vector containing the names of the stored real parameters.
 */

/** @var vector<double> DTF_Lib::EvalData::realValues;
 * @brief vector containing the values of the stored real parameters.
 */

/** @var vector<string> DTF_Lib::EvalData::stringNames;
 * @brief vector containing the names of the stored string parameters.
 */

/** @var vector<string> DTF_Lib::EvalData::stringValues;
 * @brief vector containing the values of the stored string parameters.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_EVALDATA_H_
#define __DTF_LIB_EVALDATA_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibEvalData;

class EvalData : public Tools::BaseObject
{
    friend class ClassInfo_DTFLibEvalData;

private:
    vector<string> intNames;
    vector<int> intValues;
    vector<string> realNames;
    vector<double> realValues;
    vector<string> stringNames;
    vector<string> stringValues;

    EvalData();
    EvalData(string className, int objectID);

    static ClassInfo_DTFLibEvalData classInfo;

public:
    virtual ~EvalData();

    virtual void clear();

    bool addValue(string name,
                  int value);

    bool addValue(string name,
                  double value);

    bool addValue(string name,
                  string value);

    bool getInts(vector<string> &names,
                 vector<int> &values);

    bool getReals(vector<string> &names,
                  vector<double> &values);

    bool getStrings(vector<string> &names,
                    vector<string> &values);
};

CLASSINFO(ClassInfo_DTFLibEvalData, EvalData);
};
#endif

/** EOC */
