/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "functions.h"

using namespace std;

bool
transformSTL(std::string stl_file, std::string stl_file_trans, osg::Matrix M)
{
    std::string dateiname;
    dateiname = stl_file_trans;

    int l = dateiname.length();
#ifdef _WIN32
    int pos = dateiname.find_last_of("/\\");
#else
    int pos = dateiname.find_last_of("/");
#endif

    pos = pos + 1;

    dateiname = dateiname.substr(pos, l);

    /*
   std::ofstream datei1;
   datei1.open("dateinamen.txt", std::ios::app);


   if (datei1.is_open())
   {
   datei1 << dateiname << std::endl;
   datei1.close();
   }
   else
   {
   std::cout << "Could not open dateinamen.txt" << std::endl;
   }
   */

    std::cout << "stl_file_BLA: " << stl_file << std::endl;

    std::cout << "stl_file_trans_B1: " << stl_file_trans << std::endl;

    std::cout << dateiname << std::endl;

    std::ifstream file1;
    file1.open(stl_file.c_str());
    if (file1.is_open())
    {
        std::ofstream file2;
        file2.open(stl_file_trans.c_str());
        if (file2.is_open())
        {
            std::string s1, s2, s3;

            file1 >> s1 >> s2;

            file2 << s1 << " " << s2 << std::endl;

            while (file1.good() && !file1.eof())
            {
                s1 = "";
                s2 = "";
                file1 >> s1 >> s2;

                if (s1 == std::string("endsolid"))
                {
                    file2 << s1 << " ";
                    file2 << s2;
                }
                else
                {
                    file2 << s1 << " " << s2 << " ";

                    if (s1 == "facet" && s2 == "normal")
                    {
                        double nx = 0, ny = 0, nz = 0;
                        file1 >> nx >> ny >> nz;
                        osg::Vec3d N(nx, ny, nz);

                        osg::Matrix IM = M;
                        bool test1 = IM.invert(M);
                        if (test1 == true)
                        {
                            //N=IM.preMult(N); 	//geaendert am 7.3.2013 8:30
                            N = osg::Matrix::transform3x3(IM, N);
                            N.normalize();
                        }
                        file2 << scientific << setprecision(6) << N << std::endl;
                    }

                    file1 >> s1 >> s2;
                    file2 << s1 << " " << s2 << std::endl;
                    if (s1 == "outer" && s2 == "loop")
                    {
                        file1 >> s1;
                        file2 << s1 << " ";
                        double P1x = 0, P1y = 0, P1z = 0;
                        file1 >> P1x >> P1y >> P1z;
                        osg::Vec3d P1(P1x, P1y, P1z);
                        P1 = M.preMult(P1);
                        file2 << scientific << setprecision(6) << P1 << std::endl;

                        file1 >> s1;
                        file2 << s1 << " ";
                        double P2x = 0, P2y = 0, P2z = 0;
                        file1 >> P2x >> P2y >> P2z;
                        osg::Vec3d P2(P2x, P2y, P2z);
                        P2 = M.preMult(P2);
                        file2 << scientific << setprecision(6) << P2 << std::endl;

                        file1 >> s1;
                        file2 << s1 << " ";
                        double P3x = 0, P3y = 0, P3z = 0;
                        file1 >> P3x >> P3y >> P3z;
                        osg::Vec3d P3(P3x, P3y, P3z);
                        P3 = M.preMult(P3);
                        file2 << scientific << setprecision(6) << P3 << std::endl;

                        file1 >> s1;
                        file2 << s1 << std::endl;
                        ;

                        file1 >> s1;
                        file2 << s1 << std::endl;
                    }
                }
            }
        }
        else
        {
            std::cerr << "could not open transform1-file" << std::endl;
            return false;
        }
        file2.close();
    }
    else
    {
        std::cerr << "could not open transform2file" << std::endl;
        return false;
    }
    file1.close();

    return true;
}
