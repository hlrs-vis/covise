#pragma once
#include <openGA.h>
#include <Sensor.h>

#include <fstream>

struct Solution{

    std::vector<SensorPosition*> sensorNetwork;

};

struct MiddleCost{


};

// available Fitness Functions
bool maxCoverage1(const Solution& p, MiddleCost &c);
bool maxCoverage2(const Solution& p, MiddleCost &c);

class GA
{
public:
    using FitnessFunction = std::function<bool(const Solution& p, MiddleCost &c)>;
    GA(FitnessFunction fitness);

    float m_CrossoverRate{0.7};
    float m_MutationRate{0.3};
    int m_PopulationSize{3000};
    int m_MaxGeneration{1000};
    bool m_DynamicThreading{false};

private:
    FitnessFunction m_FitnessFunction; 
    std::ofstream m_Results;

    typedef EA::Genetic<Solution,MiddleCost> GA_Type;
    typedef EA::GenerationType<Solution,MiddleCost> Generation_Type;
    GA_Type ga_obj;

    void init_genes(Solution& p,const std::function<double(void)> &rnd01);
    Solution mutate(const Solution& X_base,const std::function<double(void)> &rnd01,double shrink_scale);
    Solution crossover(const Solution& X1, const Solution& X2,const std::function<double(void)> &rnd01);
    double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X);
    void SO_report_generation(int generation_number,const EA::GenerationType<Solution,MiddleCost> &last_generation,const Solution& best_genes);
    bool optimizationStrategy(const Solution& p, MiddleCost &c);

    const Orientation* getRandomSensor(int sensorPosition ,const std::function<double(void)> &rnd01)const;
};
