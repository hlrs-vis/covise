/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CChain.h"

using namespace std;

int main()
{
    CProtein myProtein;
    std::string filename = "1A01.pdb";
    int i = 0;
    vector<CChain> myChain;
    CChain subChain, temp;

    i = myProtein.RetrievePositions(myChain, filename);

    temp = myChain.at(0);

    i = myProtein.RetrieveSubset(subChain, filename, temp.name, 0, 10);
    myProtein.PrintChain(subChain);
    return 0;
}
