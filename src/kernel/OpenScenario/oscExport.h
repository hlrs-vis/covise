/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_EXPORT_H
#define OSC_EXPORT_H

#if defined(__APPLE__) || defined(CO_rhel3) || (defined(CO_ia64icc) && (__GNUC__ >= 4))
#define EXPORT_TEMPLATE(x)
#define EXPORT_TEMPLATE2(x, y)
#define EXPORT_TEMPLATE3(x, y, z)
#define INST_TEMPLATE1(x)
#define INST_TEMPLATE2(x, y)
#define INST_TEMPLATE3(x, y, z)
#else
#define EXPORT_TEMPLATE(x) x;
#define EXPORT_TEMPLATE2(x, y) x, y;
#define EXPORT_TEMPLATE3(x, y, z) x, y, z;
#define INST_TEMPLATE1(x) x;
#define INST_TEMPLATE2(x, y) x, y;
#define INST_TEMPLATE3(x, y, z) x, y, z;
#endif

#if defined(_WIN32) && !defined(NODLL)
#define COIMPORT __declspec(dllimport)
#define COEXPORT __declspec(dllexport)

#elif(defined(__GNUC__) && __GNUC__ >= 4 && !defined(CO_ia64icc)) || defined(__clang__)
#define COEXPORT __attribute__((visibility("default")))
#define COIMPORT COEXPORT

#else
#define COIMPORT
#define COEXPORT
#endif

#if defined(coOpenScenario_EXPORTS)
#define OPENSCENARIOEXPORT COEXPORT
#else
#define OPENSCENARIOEXPORT COIMPORT
#endif

#endif //OSC_EXPORT_H
