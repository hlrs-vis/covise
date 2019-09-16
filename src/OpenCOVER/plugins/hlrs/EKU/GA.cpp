 /* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include<cstdlib>
#include<ctime>

#include "GA.hpp"
#define CAMS_PER_POINT 4
#define NUMBER_OF_CAMS 432
//384
//returns a float between 0 & 1
#define RANDOM_NUM ((float)rand()/(RAND_MAX+1))
struct GA::MySolution
{   // FIXME: Use template to generate std::array with another size
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

struct GA:: MyMiddleCost
{

    // This is where the results of simulation
    // is stored but not yet finalized.

    int objective;
};



int GA:: myrandom() {
    if (rand() % 2 == 0)
        return 1;
    else return 0;
}

/*int GA::myrandom2()
{
    std::srand(std::time(0));
    int uniform_random_variable = std::rand() % CAMS_PER_POINT; //produces values between 0 an 15 (16 is nbr of possible cams per cam location)
    return uniform_random_variable;
}
*/
void GA::init_genes(MySolution& p,const std::function<double(void)> &rnd01)
{
/*    p.cam.fill(0); // fill up with 0
    int a = p.cam.size();
    for(size_t i=0;i<=p.cam.size()/CAMS_PER_POINT;i++)
    {
        const size_t count2 = i*CAMS_PER_POINT+myrandom2();
        p.cam[count2]=1;
    }
*/  for(auto& i : p.cam)
       i = myrandom();

}

bool GA::eval_solution(const MySolution& p,MyMiddleCost &c)
{
    EA::Chronometer timer;
    timer.tic();
    /*
    The heavy process of evaluation of solutions are assumed to be perfomed in eval function
    The result of this function is callde middle cost as it is not finalized.
    Constraint checking is also done here
    */

    /*
     meine Lösung. aber wozu brauche ich neue variable überhaupt?
     kann doch direkt Mysolution aufaddieren?
    std::array<int,4> cam;
    for(auto i : p.cam)
    {
        i += p.cam;
    }
 */

 /* generierte Lösung
    const int& cam1=p.cam1;
    const int& cam2=p.cam2;
    const int& cam3=p.cam3;
    const int& cam4=p.cam4;

    c.objective=cam1+cam2+cam3+cam4;
*/

    /*for(auto x:EKU::cameras){
        for(auto i:x)
            i*p.cam[0]
    }
        x->visMat
    EKU::cameras[0]->visMat
    */

 /*   int visMatrix [3][4] = {{1,0,0,1},
                            {0,1,0,0},
                            {0,0,0,0}};
   */    // TODO: implement constraint as function not as if statement!
   c.objective = std::accumulate(p.cam.begin(), p.cam.end(),0);
/*      if(
         (p.cam[0]*visMatrix[0][0]+p.cam[1]*visMatrix[0][1]+p.cam[2]*visMatrix[0][2]+p.cam[3]*visMatrix[0][3])>= 2 &&
         (p.cam[0]*visMatrix[1][0]+p.cam[1]*visMatrix[1][1]+p.cam[2]*visMatrix[1][2]+p.cam[3]*visMatrix[1][3])>= 1 &&
         (p.cam[0]*visMatrix[2][0]+p.cam[1]*visMatrix[2][1]+p.cam[2]*visMatrix[2][2]+p.cam[3]*visMatrix[2][3])>= 0)
         return true; // solution is accepted
      else
         return false;
*/
    // 1. constraint: each observation Point must be observed with at least 1 camera
    //Loop over all Points in Visibility Matrix
    for(size_t it =0; it<nbrpoints; ++it )
    {
        int result =0;
        //For each Point go over each camera
        for(size_t it2 =0; it2<nbrcams; ++it2 )
           {
            result += p.cam[it2]*camlist[it2]->visMat[it];
           }
        if(result<1)
            goto exit;

    }
    // at this point constraint 1 is fullfilled
 //   std::cout<<" eval_solution "<<timer.toc()<<" seconds."<<std::endl;
    return true;
    exit:
      //  std::cout<<"false"<<std::endl;
         return false;

}

GA::MySolution GA:: mutate(const MySolution& X_base,const std::function<double(void)> &rnd01,double shrink_scale)
{
    EA::Chronometer timer;
    timer.tic();
    MySolution X_new;
    auto test=shrink_scale;
    bool in_range;
  /*  do{
        in_range=true;
      */  X_new=X_base;

    /*   for(auto &i : X_new.cam)
        {
            i+=0.2*(rnd01()-rnd01())*shrink_scale;
            in_range=in_range&&(i>=0 && i<1);
        }
    */ /*   X_new.cam1+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam1>=0 && X_new.cam1<1);
        X_new.cam2+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam2>=0 && X_new.cam2<1);
        X_new.cam3+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam3>=0 && X_new.cam3<1);
        X_new.cam4+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam4>=0 && X_new.cam4<1);
    */
  // } while(!in_range);

 //   std::cout<<"mutate: "<<timer.toc()<<" seconds."<<std::endl;
 /*   if(RANDOM_NUM < shrink_scale)
    {
        for(auto &i : X_new.cam)
            i = 1-i;
    }
   */ return X_new;
}

GA::MySolution GA::crossover(const MySolution& X1, const MySolution& X2,const std::function<double(void)> &rnd01)
{
    EA::Chronometer timer;
    timer.tic();

    MySolution X_new;
    double r;
   for(size_t it =0; it<X1.cam.size(); ++it )
   {
       auto test = X1.cam.size();
        X_new.cam[it]=r*X1.cam[it]+(1.0-r)*X2.cam[it];
        auto test1 = r*X1.cam[it]+(1.0-r)*X2.cam[it];
        r=rnd01();
   }

  /*  r=rnd01();
    X_new.cam1=r*X1.cam1+(1.0-r)*X2.cam1;
    r=rnd01();
    X_new.cam2=r*X1.cam2+(1.0-r)*X2.cam2;
    r=rnd01();
    X_new.cam3=r*X1.cam3+(1.0-r)*X2.cam3;
    r=rnd01();
    X_new.cam4=r*X1.cam4+(1.0-r)*X2.cam4;
   */
//    std::cout<<"crossover: "<<timer.toc()<<" seconds."<<std::endl;
    return X_new;
}

double GA::calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
    // finalize the cost
    //obtain the final cost from the middle cost
    double final_cost=0;
    final_cost+=X.middle_costs.objective;
    return final_cost;
}



void GA::SO_report_generation(int generation_number,const EA::GenerationType<MySolution,MyMiddleCost> &last_generation,const MySolution& best_genes)
{
    std::cout
        <<"Generation ["<<generation_number<<"], "
        <<"Best="<<last_generation.best_total_cost<<", "
        <<"Average="<<last_generation.average_cost<<", "
        <<"Best genes=("<<best_genes.to_string()<<")"<<", "
        <<"Exe_time="<<last_generation.exe_time
        <<std::endl;

    output_file
        <<generation_number<<"\t"
        <<last_generation.average_cost<<"\t"
        <<last_generation.best_total_cost<<"\t"
        <<best_genes.to_string()<<"\n";
}

std::array<int,NUMBER_OF_CAMS> GA::getfinalCamPos() const
{
   std::array<int,NUMBER_OF_CAMS> result= ga_obj.last_generation.chromosomes.at(ga_obj.last_generation.best_chromosome_index).genes.cam;
   return result ;
}
GA::GA(std::vector<Cam *> &cam, const size_t nbrpoints):camlist(cam),nbrpoints(nbrpoints)
{
    output_file.open("results.txt");
    output_file<<"step"<<"\t"<<"cost_avg"<<"\t"<<"cost_best"<<"\t"<<"solution_best"<<"\n";

    EA::Chronometer timer;
    timer.tic();

    using namespace std::placeholders;
    ga_obj.problem_mode=EA::GA_MODE::SOGA;
    ga_obj.multi_threading=true;
    ga_obj.verbose=true;
    ga_obj.population=3000;
    ga_obj.generation_max=1000;
    ga_obj.calculate_SO_total_fitness=std::bind( &GA::calculate_SO_total_fitness, this, _1);
    ga_obj.init_genes=std::bind( &GA::init_genes, this, _1,_2);
    ga_obj.eval_solution=std::bind( &GA::eval_solution, this, _1,_2 );
    ga_obj.mutate=std::bind( &GA::mutate, this, _1,_2,_3 );
    ga_obj.crossover=std::bind( &GA::crossover, this, _1,_2,_3 );
    ga_obj.SO_report_generation=std::bind( &GA::SO_report_generation, this, _1,_2,_3 );
    ga_obj.best_stall_max=10;
    ga_obj.elite_count=10;
    ga_obj.crossover_fraction=0.7;
    ga_obj.mutation_rate=0.2;
    ga_obj.best_stall_max=10;
    ga_obj.elite_count=10;
    ga_obj.solve();

    std::cout<<"The problem is optimized in "<<timer.toc()<<" seconds.###########################################"<<std::endl;

    output_file.close();


}
