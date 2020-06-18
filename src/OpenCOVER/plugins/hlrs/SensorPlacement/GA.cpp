#include "GA.h"
#include "DataManager.h"
#include "Sensor.h"

bool maxCoverage1(const Solution& p, MiddleCost &c)
{
    std::cout<<"Fitness: max Coverage 1" <<std::endl;
    return false; //FIXME
}

bool maxCoverage2(const Solution& p, MiddleCost &c)
{
    std::cout<<"Fitness: max Coverage 2" <<std::endl;
    return false; //FIXME
}

GA::GA(FitnessFunction fitness) : m_FitnessFunction(fitness)
{
	EA::Chronometer timer;
	timer.tic();	
	using std::bind;
	using std::placeholders::_1;
	using std::placeholders::_2;
	using std::placeholders::_3;

	ga_obj.problem_mode=EA::GA_MODE::SOGA;
    ga_obj.multi_threading=true;
    ga_obj.idle_delay_us=1; // switch between threads quickly
    ga_obj.dynamic_threading=m_DynamicThreading;
    ga_obj.verbose=false;
    ga_obj.population = m_PopulationSize;
    ga_obj.generation_max = m_MaxGeneration;
    ga_obj.calculate_SO_total_fitness = std::bind( &GA::calculate_SO_total_fitness, this, _1);
    ga_obj.init_genes = std::bind( &GA::init_genes, this, _1,_2);
    ga_obj.eval_solution = std::bind( &GA::optimizationStrategy, this, _1,_2 );
    ga_obj.mutate = std::bind( &GA::mutate, this, _1,_2,_3 );
    ga_obj.crossover = std::bind( &GA::crossover, this, _1,_2,_3 );
    ga_obj.SO_report_generation = std::bind( &GA::SO_report_generation, this, _1,_2,_3 );
    ga_obj.crossover_fraction = m_CrossoverRate;
    ga_obj.mutation_rate = m_MutationRate;
    ga_obj.best_stall_max = 10;
    ga_obj.elite_count = ga_obj.population / 100 * 6; //6% of population size;
    ga_obj.solve();
	
	std::cout<<"The problem is optimized in "<<timer.toc()<<" seconds."<<std::endl;
}

bool GA::optimizationStrategy(const Solution& p, MiddleCost &c)
{
	return m_FitnessFunction(p,c);
}

// SensorPosition* GA::getRandomSensor(int sensorPosition ,const std::function<double(void)> &rnd01)
// {

// }

void GA::init_genes(Solution& p,const std::function<double(void)> &rnd01)
{

}

Solution GA::mutate(const Solution& X_base,const std::function<double(void)> &rnd01,double shrink_scale)
{
    return X_base; //FIXME
}

Solution GA::crossover(const Solution& X1, const Solution& X2,const std::function<double(void)> &rnd01)
{
    return X1; //FIXME
}

double GA::calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
    return 0.; //FIXME
}

void GA::SO_report_generation(int generation_number,const EA::GenerationType<Solution,MiddleCost> &last_generation,const Solution& best_genes)
{

}
