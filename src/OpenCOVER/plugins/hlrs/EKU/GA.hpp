/* This file is part of COVISE.
 *

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once
#include <openGA.hpp>
#include <Cam.h>
#define NUMBER_OF_CAMS 432
//8:384 , 9:423
class GA
{
public:
    GA(std::vector<Cam*>& cam,const size_t nbrpoints);
    ~GA();
    std::array<int,NUMBER_OF_CAMS> getfinalCamPos() const;

private:
    std::ofstream output_file;              //store result of GA
    std::vector<Cam*>& camlist;
    const size_t nbrpoints;                 //number of points to observe
    const size_t nbrcams=camlist.size();    //number of cameras
   // std::vector<int> cam=std::vector<int>(nbrcams);

    struct MySolution{   // FIXME: Use template to generate std::array with another size
        // std::vector<int> cam(100);
         std::array<int,NUMBER_OF_CAMS> cam; //NOTE: why this type of declaration?


         std::string to_string() const
         {

             std::string myString;
             int cnt =1;
             for(auto i : cam)
             {
                myString += "cam"+std::to_string(cnt)+":"+std::to_string(i)+" ";
                cnt++;
             }
             return
                 std::string("{") + myString + "}";

         }

     };
    struct MyMiddleCost{int objective;};
    struct MyTest;
    typedef EA::Genetic<MySolution,MyMiddleCost> GA_Type;
    typedef EA::GenerationType<MySolution,MyMiddleCost> Generation_Type;
    GA_Type ga_obj;
    int myrandom();
    int myrandom2();
    void init_genes(MySolution& p,const std::function<double(void)> &rnd01);
    MySolution mutate(const MySolution& X_base,const std::function<double(void)> &rnd01,double shrink_scale);
    MySolution crossover(const MySolution& X1, const MySolution& X2,const std::function<double(void)> &rnd01);
    bool eval_solution(const MySolution& p,MyMiddleCost &c);
    double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X);
    void SO_report_generation(int generation_number,const EA::GenerationType<MySolution,MyMiddleCost> &last_generation,const MySolution& best_genes);

};
