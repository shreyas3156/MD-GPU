#pragma once
#include "kernel.h"
int main(int argc, char* argv[]);

double ran2( void );
double dgauss( double , double );
double pbc_vdr( double*, double*, double* );
double U_system( void );
// double U_system_cells( void ) ;
void read_input_config( void );
void write_xyz( void );

/////////////////////////////
// Define global variables //
// Number of particles, current MC step number, maximum number of MC steps
// print_freq = number of steps between writing to console and log.dat
int N , step, step_max , print_freq, neigh_freq ;

// random number seed
long int idum ;

// particle positions, box dimensions, 0.5*box dimensions, box volume, time step size
double **v, **f, **x, box[3], boxh[3], vol, delt;
// cut-off distance, cut-off distance^2, system potential energy, kinetic energy
double rcut, rcut2, PE, KE, Ucut ;

// System temperature, beta = 1.0 / T, and maximum displacement 
double T, beta , dr_max ;
// End global variables //
//////////////////////////

// Simulation methods //

// Initialize params
void initSimulation();

// Main MC Loop
void mainLoop();

// CUDA call for each simulation step
void runCUDA();
