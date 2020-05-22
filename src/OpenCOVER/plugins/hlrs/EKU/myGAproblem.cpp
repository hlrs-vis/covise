#include <myGAproblem.hpp>.hpp>



MyGAproblem::MyGAproblem()
{
    output_file.open("results.txt");
    output_file<<"step"<<"\t"<<"cost_avg"<<"\t"<<"cost_best"<<"\t"<<"solution_best"<<"\n";

    EA::Chronometer timer;
    timer.tic();
    GA_Type ga_obj;

    ga_obj.problem_mode=EA::GA_MODE::SOGA;
    ga_obj.multi_threading=false;
    ga_obj.verbose=false;
    ga_obj.population=1000;
    ga_obj.generation_max=1000;
    ga_obj.calculate_SO_total_fitness=calculate_SO_total_fitness;
    ga_obj.init_genes=init_genes;
    ga_obj.eval_solution=eval_solution;
    ga_obj.mutate=mutate;
    ga_obj.crossover=crossover;
    ga_obj.SO_report_generation=SO_report_generation;
    ga_obj.best_stall_max=10;
    ga_obj.elite_count=10;
    ga_obj.crossover_fraction=0.7;
    ga_obj.mutation_rate=0.2;
    ga_obj.best_stall_max=10;
    ga_obj.elite_count=10;
    ga_obj.solve();
    std::cout<<"The problem is optimized in "<<timer.toc()<<" seconds."<<std::endl;
}



MyGAproblem::~MyGAproblem()
{
    output_file.close();
}

struct MyGAproblem::MySolution
{
    std::string to_string() const
    {
        std::string myString;
        int cnt=1;
        for(auto i : cam)
        {
           myString += "cam"+std::to_string(cnt)+":"+std::to_string(i)+" ";
           cnt++;
        }
        return
            std::string("{") + myString + "}";

    }
}

int MyGAproblem::myrandom()
{
    if (rand() % 2 == 0)
        return 1;
    else
        return 0;
}

void MyGAproblem::init_genes(MySolution& p,const std::function<double(void)> &rnd01)
{
    // rnd01() gives a random number in 0~1
    for(auto& i : p.cam)
        i = 0+1*myrandom();
}

bool MyGAproblem::eval_solution(const MySolution& p,MyMiddleCost &c)
{
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
    c.objective = std::accumulate(p.cam.begin(), p.cam.end(),0);

    int visMatrix [3][4] = {{1,0,0,0},
                            {0,1,0,0},
                            {0,0,1,0}};
    // TODO: implement constraint as function not as if statement!
    if((p.cam[0]*visMatrix[0][0]+p.cam[1]*visMatrix[0][1]+p.cam[2]*visMatrix[0][2]+p.cam[3]*visMatrix[0][3])>= 1 &&
       (p.cam[0]*visMatrix[1][0]+p.cam[1]*visMatrix[1][1]+p.cam[2]*visMatrix[1][2]+p.cam[3]*visMatrix[1][3])>= 1 &&
       (p.cam[0]*visMatrix[2][0]+p.cam[1]*visMatrix[2][1]+p.cam[2]*visMatrix[2][2]+p.cam[3]*visMatrix[2][3])>= 1)
       return true; // solution is accepted
    else
       return false;
}

MySolution MyGAproblem:: mutate(const MySolution& X_base,
                                const std::function<double(void)> &rnd01,
                                 double shrink_scale)
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

MySolution MyGAproblem::crossover(const MySolution& X1, const MySolution& X2,const std::function<double(void)> &rnd01)
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

double MyGAproblem::calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
    // finalize the cost
    double final_cost=0.0;
    final_cost+=X.middle_costs.objective;
    return final_cost;
}

void MyGAproblem::SO_report_generation(int generation_number,
                                       const EA::GenerationType<MySolution,MyMiddleCost> &last_generation,
                                       const MySolution& best_genes)
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



