/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <fstream>
#include <math.h>
#include "CChain.h"

using namespace std;

void CChain::SetData(string strName, int iNum)
{
    name = strName;
    num = iNum;
}

void CSequence::SetData(string aa, int iNum)
{
    num = iNum;
    aminoacid = aa;
}

void CSequence::SetData(string aa, int iNum, float X, float Y, float Z)
{
    num = iNum;
    aminoacid = aa;
    x = X;
    y = Y;
    z = Z;
}

void CProtein::PrintChain(vector<CChain> userChain)
{
    CChain temporarychain;
    CSequence temporarysequence;
    for (int i = 0; i < userChain.size(); i++)
    {
        temporarychain = userChain.at(i);
        cout << "\n***Chain " << temporarychain.name << "(" << temporarychain.num << "):";
        if (!temporarychain.chainsequence.empty())
        {
            for (int j = 0; j < temporarychain.chainsequence.size(); j++)
            {
                temporarysequence = temporarychain.chainsequence.at(j);
                cout << endl << temporarysequence.aminoacid << "(" << temporarysequence.num << ") " << temporarysequence.x << " " << temporarysequence.y << " " << temporarysequence.z;
            }
        }
    }
}

int CProtein::ReturnChainNumber(std::vector<CChain> &userChain, std::string strChainLabel)
{
    CChain chainTemp;

    for (int i = 0; i < userChain.size(); i++)
    {
        chainTemp = userChain.at(i);
        if (chainTemp.name == strChainLabel)
            return i;
    }

    return 0;
}

void CProtein::PrintChain(CChain userChain)
{
    CSequence temporarysequence;

    cout << "\n***Chain " << userChain.name << "(" << userChain.num << "):";
    if (!userChain.chainsequence.empty())
    {
        for (int j = 0; j < userChain.chainsequence.size(); j++)
        {
            temporarysequence = userChain.chainsequence.at(j);
            cout << endl << temporarysequence.aminoacid << "(" << temporarysequence.num << ") " << temporarysequence.x << " " << temporarysequence.y << " " << temporarysequence.z;
        }
    }
}

int CProtein::RetrievePositions(std::vector<CChain> &userChain, std::string strFileName)
{
    CChain tempChain, eraserChain;
    CSequence tempSequence;
    vector<CSequence> eraserSequence;
    vector<CChain> masterVector;
    ifstream inputfile;
    string isOld = " ";
    int iAverageCount = 0, iCurPos = 0, iOldPos = 0, iChainCount = 0, iMasterChainCount = 0; //Used to go char to char in the seqres line
    float x = 0.0, y = 0.0, z = 0.0, xavg = 0.0, yavg = 0.0, zavg = 0.0;
    string strLine, strAminoAcid, strx, stry, strz, strAANumber;
    string strOldAA = " ";
    string strChain = " ";
    string strSearchParameter = "ATOM      1"; //Used to isolate sequence
    string::size_type pos = 0;
    inputfile.open(strFileName.c_str());

    while (!inputfile.eof())
    {
        pos = 0;
        getline(inputfile, strLine);
        pos = strLine.find(strSearchParameter, 0);

        if (pos != string::npos) //Making sure were on the right line
        {
            if (strSearchParameter == "ATOM      1")
                strSearchParameter = "ATOM";

            strAminoAcid = strLine.substr(17, 3);
            strChain = strLine.substr(21, 1);
            strx = strLine.substr(31, 8);
            stry = strLine.substr(39, 8);
            strz = strLine.substr(47, 8);
            strAANumber = strLine.substr(23, 3);
            iCurPos = atoi(strAANumber.c_str());

            //			cout << endl << strChain << " " << strAminoAcid << " " << strAANumber << " " << strx << " " << stry << " " << strz << " avg count = " << iAverageCount;

            if (isOld == " ") //First Chain
            {
                tempChain.SetData(strChain, ++iMasterChainCount);
                isOld = strChain;
                iCurPos = iOldPos = 1;
                strOldAA = strAminoAcid;
                //				cout << "\nFIRST CHAIN"; //DEBUG
            }

            if (isOld != strChain) //New Chain
            {
                //Take care of finishing off last amino acid
                xavg = x / iAverageCount;
                yavg = y / iAverageCount;
                zavg = z / iAverageCount;
                x = y = z = iAverageCount = 0;
                tempSequence.SetData(strOldAA, ++iChainCount, xavg, yavg, zavg);
                tempChain.chainsequence.push_back(tempSequence);
                strOldAA = strAminoAcid;
                isOld = strChain;
                iOldPos = iCurPos;
                iChainCount = 0;

                //Now push the vector onto the master list
                masterVector.push_back(tempChain);
                tempChain.chainsequence = eraserSequence;
                tempChain.SetData(strChain, ++iMasterChainCount);
                //				cout << "\nNEW CHAIN"; //Debug
                //				cout << endl << "isold = " << isOld << " strchain = " << strChain;  //DEBUG
            }

            if (iCurPos != iOldPos) //Next AA
            {
                //Average Everything and Put it Into Sequence
                xavg = x / iAverageCount;
                yavg = y / iAverageCount;
                zavg = z / iAverageCount;
                x = y = z = iAverageCount = 0;
                tempSequence.SetData(strOldAA, ++iChainCount, xavg, yavg, zavg);
                tempChain.chainsequence.push_back(tempSequence);
                strOldAA = strAminoAcid;
                iOldPos = iCurPos;
                //				cout << "\nNEW AMINO ACID"; // DEBUG
                //				cout << endl << "iCurPos = " << iCurPos << " iOldPos = " << iOldPos; //DEBUG
            }

            x += atof(strx.c_str());
            y += atof(stry.c_str());
            z += atof(strz.c_str());
            iAverageCount++;
        }
    }

    //Finish off Last Amino Acid Sequence and Append it to the master vector
    xavg = x / iAverageCount;
    yavg = y / iAverageCount;
    zavg = z / iAverageCount;
    x = y = z = iAverageCount = 0;
    tempSequence.SetData(strOldAA, ++iChainCount, xavg, yavg, zavg);
    tempChain.chainsequence.push_back(tempSequence);
    masterVector.push_back(tempChain);
    userChain = masterVector;
    inputfile.close();
    return 1;
}

int CProtein::RetrieveSubset(CChain &userChain, std::string strFileName, std::string strChain, int startPos, int endPos)
{
    std::vector<CChain> tempChain;
    CChain holderChain, returnChain;
    CSequence tempSequence;
    RetrievePositions(tempChain, strFileName);
    int i = 0, chainExists = 0, chainNumber = 0;

    //Check if the chain is there
    for (i = 0; i < tempChain.size(); i++)
    {
        holderChain = tempChain.at(i);
        if (holderChain.name == strChain)
        {
            chainExists = 1;
            chainNumber = i;
            break;
        }
    }

    if (!chainExists)
        return 0; //Chain's not there

    //Check to see if the numbers are valid (we'll accept larger than the chain size as endpos, but startpos can't be higher
    holderChain = tempChain.at(chainNumber);

    if (startPos < 0)
        startPos = 0; //Oops, no negatives

    if (startPos >= (holderChain.chainsequence.size()))
        return 0; //Sequence desired is above the chain size

    if (endPos < 0)
        return 0; // Can't finish in the negatives

    if (endPos >= (holderChain.chainsequence.size()))
        endPos = (holderChain.chainsequence.size()) - 1; //We don't want their range exceeding the bound, so we'll correct it for them

    //Now that error checking is done, let's pull out our sequence and spit it back out
    for (i = startPos; i <= endPos; i++)
    {
        tempSequence = holderChain.chainsequence.at(i);
        returnChain.chainsequence.push_back(tempSequence);
    }

    returnChain.name = holderChain.name;
    returnChain.num = holderChain.num;
    userChain = returnChain;
    return 1;
}

int CProtein::ClosestAminoAcid(std::vector<CChain> &userChain, float
                                                                   xpos,
                               float ypos, float zpos, std::string &smallestChain, int &smallestChainPos)

{
    std::vector<CChainMatrix> distanceMatrix;
    std::string currentChain;
    std::vector<float> distances;
    CChainMatrix tempMatrix, tempMatrix2;
    CChain temporarychain;
    CSequence temporarysequence;
    int smallestIndex = 0;
    float smallestDistance = 10000000;
    int absolutej = 0;

    for (int i = 0; i < userChain.size(); i++)
    {
        temporarychain = userChain.at(i);
        currentChain = temporarychain.name;
        if (!temporarychain.chainsequence.empty())
        {
            for (int j = 0; j < temporarychain.chainsequence.size(); j++)
            {
                temporarysequence = temporarychain.chainsequence.at(j);
                //Insert Sequence Subtraction Code
                tempMatrix.x = temporarysequence.x - xpos;
                tempMatrix.y = temporarysequence.y - ypos;
                tempMatrix.z = temporarysequence.z - zpos;
                tempMatrix.distance = sqrt(pow(tempMatrix.x, 2) + pow(tempMatrix.y, 2) + pow(tempMatrix.z, 2));
                tempMatrix.chain = currentChain;
                tempMatrix.seqno = j;
                distanceMatrix.push_back(tempMatrix);

                if (smallestDistance > tempMatrix.distance)
                {
                    smallestIndex = absolutej;
                    smallestDistance = tempMatrix.distance;
                }
                absolutej++;
            }
        }
    }

    //	PrintMatrix(distanceMatrix);
    tempMatrix = distanceMatrix.at(smallestIndex);
    smallestChain = tempMatrix.chain;
    smallestChainPos = tempMatrix.seqno;
    return 1;
}

void CProtein::PrintMatrix(std::vector<CChainMatrix> userMatrix)
{
    CChainMatrix tempMatrix;

    for (int j = 0; j < userMatrix.size(); j++)
    {
        tempMatrix = userMatrix.at(j);
        cout << "Chain " << tempMatrix.chain << " # " << tempMatrix.seqno << " x:" << tempMatrix.x << " y:" << tempMatrix.y << " z:" << tempMatrix.z << " distance: " << tempMatrix.distance << "\n";
    }
}
