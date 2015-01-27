/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "main.h"

#include <config/coConfigGroup.h>
#include <config/coConfigFloat.h>
#include <config/coConfig.h>

//#include <covise/covise_config.h>

#include <sys/timeb.h>

#include <iostream>
using namespace std;

int main(int, char **)
{

    coConfig::getInstance()->save("testout.xml");

    //   coConfigGroup * group = new coConfigGroup("config");

    //   group->addConfig("./testconfig-1.xml", "testconfig");

    //   if (group->getValue("TestValue") == group->getValue("value", "TestValue")) {
    //     cout << "TestValue.value          = " << group->getValue("TestValue") << endl;
    //   } else {
    //     cout << "Error: Comparison of simple and complex query failed ["
    //          << group->getValue("TestValue") << "|" << group->getValue("value", "TestValue")
    //          << "]" << endl;
    //   }

    //   if (group->getValue("attribute", "TestValue") == QString("testattribute")) {
    //     cout << "TestValue.attribute      = " << group->getValue("attribute", "TestValue") << endl;
    //   } else {
    //     cout << "Error: Retrieval of named variable failed" << endl;
    //   }

    //   QString tf = "TestFloat";

    //   coConfigFloat floatVarSimple(group, tf, false);
    //   coConfigFloat floatVarComplex(group, "value", "TestFloat", false);
    //   coConfigFloat floatVarSimple2(group, "TestFloat2", false);

    //   float floatValue = floatVarSimple;

    //   cout << "TestFloat.value (" << group->getValue("TestFloat") << ")    = " << floatValue << endl;

    //   if (floatVarSimple == floatVarComplex) {
    //     cout << "FloatVarSimple (" << floatVarSimple << ")      == FloatVarComplex (" << floatVarComplex << ")" << endl;
    //   } else {
    //     cout << "Error: FloatVarSimple (" << floatVarSimple << ") != FloatVarComplex (" << floatVarComplex << " )" << endl;
    //   }

    //   if (floatVarSimple == floatVarSimple2) {
    //     cout << "Error: FloatVarSimple (" << floatVarSimple << ")      == FloatVarSimple2 (" << floatVarSimple2 << ")" << endl;
    //   } else {
    //     cout << "FloatVarSimple (" << floatVarSimple << ")      != FloatVarSimple2 (" << floatVarSimple2 << ")" << endl;
    //   }

    //   floatValue = -1.0f;
    //   floatVarSimple = floatValue;
    //   cout << "FloatVarSimple (-1.0f)   = " << group->getValue("TestFloat") << endl;

    //   floatVarComplex = -2.0f;
    //   cout << "FloatVarComplex (-2.0f)  = " << group->getValue("TestFloat") << endl;

    //  group->setValue("value", "3.0f", "TestFloat.TestFloat.TestFloat");
    //  cout << "AddValue (3.0f)  = " << group->getValue("TestFloat.TestFloat.TestFloat") << endl;

    //   timeb start;
    //   timeb end;

    //   QString search1("Tracker");
    //   QString search2("OpenSGRenderer.Cluster");
    //   ftime(&start);
    //   for (int ctr = 0; ctr < 100000; ++ctr) {
    //     group->getValue(search1);
    //     group->getValue(search2);
    //   }
    //   ftime(&end);
    //   cerr << "coConfig:    200.000 getValue: " << (end.time * 1000 + end.millitm) - (start.time * 1000 + start.millitm) << "ms" << endl;

    //   ftime(&start);
    //   for (int ctr = 0; ctr < 100000; ++ctr) {
    //     group->getEntry("Tracker.value");
    //     group->getEntry("OpenSGRenderer.Cluster.value");
    //   }
    //   ftime(&end);
    //   cerr << "coConfig:    200.000 getEntry: " << (end.time * 1000 + end.millitm) - (start.time * 1000 + start.millitm) << "ms" << endl;

    //   ftime(&start);
    //   if (CoviseConfig::getEntry("TabletPC.Server")) {
    //     for (int ctr = 0; ctr <  99999; ++ctr) {
    //       CoviseConfig::getEntry("TabletPC.Server");
    //       CoviseConfig::getEntry("OpenSGRenderer.Cluster");
    //     }
    //     ftime(&end);
    //     cerr << "CoviseConfig: 200.000 getEntry: " << (end.time * 1000 + end.millitm) - (start.time * 1000 + start.millitm) << "ms" << endl;
    //   }

    //  delete group;
}
