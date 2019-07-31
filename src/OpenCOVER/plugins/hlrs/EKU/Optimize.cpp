#include <string>
#include <iostream>
#include <fstream>
#include "openga.hpp"

using std::string;
using std::cout;
using std::endl;

struct MySolution
{
	int cam1;
	int cam2;
	int cam3;
	int cam4;

	string to_string() const
	{
		return 
			string("{")
			+  "cam1:"+std::to_string(cam1)
			+", cam2:"+std::to_string(cam2)
			+", cam3:"+std::to_string(cam3)
			+", cam4:"+std::to_string(cam4)
			+"}";
	}
};

struct MyMiddleCost
{
	// This is where the results of simulation
	// is stored but not yet finalized.
	double objective;
};

typedef EA::Genetic<MySolution,MyMiddleCost> GA_Type;
typedef EA::GenerationType<MySolution,MyMiddleCost> Generation_Type;

void init_genes(MySolution& p,const std::function<double(void)> &rnd01)
{
	// rnd01() gives a random number in 0~1
	p.cam1=0+1*rnd01();
	p.cam2=0+1*rnd01();
	p.cam3=0+1*rnd01();
	p.cam4=0+1*rnd01();
}

bool eval_solution(
	const MySolution& p,
	MyMiddleCost &c)
{
	const int& cam1=p.cam1;
	const int& cam2=p.cam2;
	const int& cam3=p.cam3;
	const int& cam4=p.cam4;

	c.objective=var1+var2+var3+var4;
	return true; // solution is accepted
}

MySolution mutate(
	const MySolution& X_base,
	const std::function<double(void)> &rnd01,
	double shrink_scale)
{
	MySolution X_new;
	bool in_range;
	do{
		in_range=true;
		X_new=X_base;
		X_new.cam1+=0.2*(rnd01()-rnd01())*shrink_scale;
		in_range=in_range&&(X_new.cam1>=0 && X_new.cam1<1);
		X_new.cam2+=0.2*(rnd01()-rnd01())*shrink_scale;
		in_range=in_range&&(X_new.cam2>=0 && X_new.cam2<1);
		X_new.cam3+=0.2*(rnd01()-rnd01())*shrink_scale;
		in_range=in_range&&(X_new.cam3>=0 && X_new.cam3<1);
		X_new.cam4+=0.2*(rnd01()-rnd01())*shrink_scale;
		in_range=in_range&&(X_new.cam4>=0 && X_new.cam4<1);
	} while(!in_range);
	return X_new;
}

MySolution crossover(
	const MySolution& X1,
	const MySolution& X2,
	const std::function<double(void)> &rnd01)
{
	MySolution X_new;
	double r;
	r=rnd01();
	X_new.cam1=r*X1.cam1+(1.0-r)*X2.cam1;
	r=rnd01();
	X_new.cam2=r*X1.cam2+(1.0-r)*X2.cam2;
	r=rnd01();
	X_new.cam3=r*X1.cam3+(1.0-r)*X2.cam3;
	r=rnd01();
	X_new.cam4=r*X1.cam4+(1.0-r)*X2.cam4;
	return X_new;
}

double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
	// finalize the cost
	double final_cost=0.0;
	final_cost+=X.middle_costs.objective;
	return final_cost;
}

std::ofstream output_file;

void SO_report_generation(
	int generation_number,
	const EA::GenerationType<MySolution,MyMiddleCost> &last_generation,
	const MySolution& best_genes)
{
	cout
		<<"Generation ["<<generation_number<<"], "
		<<"Best="<<last_generation.best_total_cost<<", "
		<<"Average="<<last_generation.average_cost<<", "
		<<"Best genes=("<<best_genes.to_string()<<")"<<", "
		<<"Exe_time="<<last_generation.exe_time
		<<endl;

	output_file
		<<generation_number<<"\t"
		<<last_generation.average_cost<<"\t"
		<<last_generation.best_total_cost<<"\t"
		<<best_genes.to_string()<<"\n";
}

int main()
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

	cout<<"The problem is optimized in "<<timer.toc()<<" seconds."<<endl;

	output_file.close();
	return 0;
}