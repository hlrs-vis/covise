/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <util/coviseCompat.h>

#include "CaseFile.h"

CaseFile::CaseFile()
    : there_(false)
    , numTimesteps_(0)
    , enVersion_(V5)

{
}

CaseFile::CaseFile(const string &fileNm)
    : there_(false)
    , numTimesteps_(0)
    , enVersion_(V5)

{
    read(fileNm);

    parse();
}

void
CaseFile::read(const string &fileNm)
{
    ifstream inp(fileNm.c_str(), ios::in);

    if (inp)
    {
        char buf[bufLen];
        while (inp.get(buf, bufLen))
        {
            inp.seekg(1L, ios::cur);
            string line(buf);
            content_.push_back(line);
        }
        inp.close();
        there_ = true;
    }
    else
    {
        cerr << "CaseFile::read(..): Could not open file " << fileNm << endl;
        there_ = false;
    }
}

void
CaseFile::parse()
{
    int i;
    string model;
    for (i = 0; i < content_.size(); ++i)
    {
        string line = content_[i];
        // find FORMAT line -> enight version
        if (line.find("FORMAT") != string::npos)
        {
            string nextL = content_[i + 1]; // the format field is usually the first one
            string::size_type begin = nextL.find_first_not_of("type:");
            string verStr = nextL.substr(begin);

            if (verStr.find("ensight") == string::npos)
            {
                enVersion_ = V6;
            }
            if (verStr.find("ensight gold") == string::npos)
            {
                enVersion_ = GOLD;
            }
        }

        // find FORMAT line -> geometry file
        if (line.find("GEOMETRY") != string::npos)
        {
            string nextL = content_[i + 1]; // the format field is usually the first one
            string::size_type begin = nextL.find_first_not_of("model:");
            model = nextL.substr(begin);
        }

        // Time section
        if (line.find("TIME") != string::npos)
        {
            int j = 1;
            string nextL = content_[i + j];
            while (nextL.find_first_not_of(" ") != string::npos)
            {
                // get number of timesteps
                if (nextL.find("number of steps:") != string::npos)
                {
                    string::size_type begin = nextL.find_first_not_of("number of steps:");
                    string numStr = nextL.substr(begin);
                    numTimesteps_ = atoi(numStr.c_str());
                    //cerr << "CaseFile::parse(): got number of timesteps " << numTimesteps_ << endl;
                }

                j++;
                nextL = content_[i + j];
            }
        }

        // VARIABLE section
        if (line.find("VARIABLE") != string::npos)
        {
            int j = 1;
            string nextL = content_[i + j];
            while (nextL.find_first_not_of(" ") != string::npos)
            {
                // extract the last entry
                string::size_type a = 0;
                string::size_type b = 0;
                string file;

                while (a != string::npos)
                {
                    file = nextL.substr(b);
                    a = nextL.find_first_of(" ", b);
                    b = nextL.find_first_not_of(" ", a);
                }

                file = filterWildcard(file);
                //		cerr << "CaseFile::parse() got filtered file name : <" << file << ">" << endl;
                if (nextL.find("per node:"))
                {
                    dataModes_[file] = PER_NODE;
                }

                if (nextL.find("per element:"))
                {
                    dataModes_[file] = PER_CELL;
                }

                j++;
                nextL = content_[i + j];
            }
        }

        // 	if (line.find_first_not_of(" ") == string::npos) {
        // 	    cerr << "CaseFile::parse(): empty line # " << i << endl;
        // 	}
    }

    // check if we have a * notation
    if (!model.empty())
    {
        string::size_type end = model.find_first_of("*");
        if (end != string::npos)
        {
            string::size_type begin = model.find_first_not_of(" ");
            string basic = model.substr(begin, (end - begin));

            int j;
            for (j = 0; j < numTimesteps_; ++j)
            {
                char buf[6];
                sprintf(buf, "%4.4d", j);
                string post(buf);
                //cerr << "CaseFile::parse(): got geo-file name " << basic+post << endl;
                geoFiles_.push_back(basic + post);
            }
        }
        else
        {
            geoFiles_.push_back(model);
        }
    }
}

DataMode
CaseFile::getDataMode(const string &file) const
{

    string fl = filterWildcard(file);
    DataModeMap::const_iterator elem = dataModes_.find(fl);

    if (elem != dataModes_.end())
    {
        return (*elem).second;
    }
    else
    {
        return MYERROR;
    }
}

string
CaseFile::filterWildcard(const string &str) const
{
    if (str.find("*") != string::npos)
    {
        string::size_type a = str.find_first_of(".");
        return str.substr(0, a);
    }
    else
    {
        return str;
    }
}

CaseFile::~CaseFile(void)
{
}
