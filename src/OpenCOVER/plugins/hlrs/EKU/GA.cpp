 /* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <fstream>

#include "GA.hpp"

struct GA::MySolution
{   // FIXME: Use template to generate std::array with another size
   // std::vector<int> cam(100);
    std::vector<int> cam=std::vector<int>(16); //NOTE: why this type of declaration?


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

    double objective;
};



int GA:: myrandom() {
    if (rand() % 2 == 0)
        return 1;
    else return 0;
}

void GA::init_genes(MySolution& p,const std::function<double(void)> &rnd01)
{
    // rnd01() gives a random number in 0~1
    for(auto& i : p.cam)
        i = 0+1*myrandom();
}

bool GA::eval_solution(const MySolution& p,MyMiddleCost &c)
{
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

    return true;
    exit:
         return false;

}

GA::MySolution GA:: mutate(const MySolution& X_base,const std::function<double(void)> &rnd01,double shrink_scale)
{
    MySolution X_new;
    bool in_range;
    do{
        in_range=true;
        X_new=X_base;

        for(auto &i : X_new.cam)
        {
            i+=0.2*(rnd01()-rnd01())*shrink_scale;
            in_range=in_range&&(i>=0 && i<1);
        }
      /*  X_new.cam1+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam1>=0 && X_new.cam1<1);
        X_new.cam2+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam2>=0 && X_new.cam2<1);
        X_new.cam3+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam3>=0 && X_new.cam3<1);
        X_new.cam4+=0.2*(rnd01()-rnd01())*shrink_scale;
        in_range=in_range&&(X_new.cam4>=0 && X_new.cam4<1);
      */
    } while(!in_range);
    return X_new;
}

GA::MySolution GA::crossover(const MySolution& X1, const MySolution& X2,const std::function<double(void)> &rnd01)
{
    MySolution X_new;
    double r;
   for(size_t it =0; it<X1.cam.size(); ++it )
   {
        X_new.cam[it]=r*X1.cam[it]+(1.0-r)*X2.cam[it];
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
    return X_new;
}

double GA::calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
    // finalize the cost
    //obtain the final cost from the middle cost
    double final_cost=0.0;
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

std::vector<int>GA::getfinalCamPos() const
{
   std::vector<int> result= ga_obj.last_generation.chromosomes.at(ga_obj.last_generation.best_chromosome_index).genes.cam;
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
    ga_obj.multi_threading=false;
    ga_obj.verbose=false;
    ga_obj.population=1000;
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

    std::cout<<"The problem is optimized in "<<timer.toc()<<" seconds."<<std::endl;

    output_file.close();


}
