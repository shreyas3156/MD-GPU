#include "main.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    initSimulation();
    //timeKernel(500);

    mainLoop();
    runSimulation();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the duration
    std::cout << "Kernel execution took " << milliseconds << " milliseconds." << std::endl;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// Runtime methods //
void initSimulation(){
    N = 2000 ;
    T = 0.5 ;
    rcut = 2.5 ;
    vol = 2700.0 ;
    step_max = 10000 ;
    delt = 0.002;
    print_freq = 50 ;
    neigh_freq = 10 ;
  
    USE_CELLS = 1 ; // Set this value to 1 to use cell lists, 0 to use N^2 loop
  
  
  
  
    // Parameters derived from defined quantities //
    rcut2 = rcut * rcut ;
    double rc6 = rcut2*rcut2*rcut2 ;
    Ucut = 4.0/(rc6*rc6) - 4.0/rc6 ;
  
    for ( i=0; i<3 ; i++ ) {
      // Sets up a cubic simulation box
      box[i] = pow( vol , 1.0/3.0 );  
      
      // Pre-calculating 0.5* box length is useful
      boxh[i] = 0.5 * box[i] ;
    }
    
    initCUDA(N);
}

void mainLoop(){
    
}