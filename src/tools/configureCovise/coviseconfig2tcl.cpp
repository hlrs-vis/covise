/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "parser.h"

#include "generatedAction.h"

int main(int argc, char **argv)
{
    if (argc == 2)
    {
        cout.precision(100);
        generatedAction *a = new generatedAction;
        Parser parser(a, argv[1]);
        parser.yyparse();
        //    delete a;
    }
    else
    {
        cerr << "usage: covisevonfig2tcl <filename>" << endl;
    }
}
