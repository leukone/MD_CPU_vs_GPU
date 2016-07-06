#include <cuda.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdio.h>

using namespace std;






/*####################################################       ERROR HANDLING       ##################################################################*/

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}





/*##################################################################################################################################################*/










__global__ void temperatura_kernel(int Nk, float4* polozenia, float4* predkosci, float* temp);
__global__ void licz_krok_kernel(int Nk, int pk, float dtk, float lD, float4 *pozycje, float4* predkosci);
__device__ float3 licz_kafelek(float4 mojaPozycja, float4 *pozycje, float3 przyspieszenie, float Nk, float pk)
__device__ float3 przyspieszenie_update(float4 bi, float4 bj, float3 ai);
void initPositions();     // places particles on an fcc lattice
void initVelocities(); 
void rescaleVelocities();
double gasdev();
         // Gaussian distributed random numbers



#define sigma (1)
#define epsilon (1e-30)
#define kb 1

int M = 16;
const int N = 4*8; // number of particles
const int p = 4;
const float dt =0.0000000001;    
double rho = 1.0;         // density (number per unit volume)
double T = 30.0;           // temperature
double weight = 1.0;
double **r;               // positions
double **v;               // velocities
float4 *polozeniaH;
float4 *predkosciH;
double L;
int main()
{
	cublasHandle_t cublashandle;
	cublasCreate(&cublashandle);

    	srand( time( NULL ) );

	L = (float) pow(N*1.0/rho, 1.0/3);
	printf("%f", L);
	polozeniaH = new float4[N];
	predkosciH = new float4[N];

	const float lD = L;
	float4* polozeniaD;
	CudaSafeCall(cudaMalloc( (void**)&polozeniaD, N*sizeof(float4)) );
	float4* predkosciD;
	CudaSafeCall(cudaMalloc( (void**)&predkosciD, N*sizeof(float4)) );
	float* tempD;
	CudaSafeCall(cudaMalloc( (void**)&tempD, N*sizeof(float)) );

        initPositions();
	initVelocities();



	fstream plikout;
	plikout.open("dupa_in.txt", fstream::out | fstream::trunc );
	for(int i = 0; i < N; i++) plikout << predkosciH[i].x << ", " << predkosciH[i].y << ", " << predkosciH[i].z << endl;
	plikout.close();

	fstream plik_out;
	plikout.open("poloz_in.txt", fstream::out | fstream::trunc );
	for(int i = 0; i < N; i++) plikout << polozeniaH[i].x << ", " << polozeniaH[i].y << ", " << polozeniaH[i].z << endl;
	plik_out.close();

	CudaSafeCall(cudaMemcpy( polozeniaD, polozeniaH, N*sizeof(float4), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy( predkosciD, predkosciH, N*sizeof(float4), cudaMemcpyHostToDevice));



	
	for(int i = 0; i < 2; i++)
	{	
		//printf("%s", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
		licz_krok_kernel<<<N/p, p, p*sizeof(float4)>>>(N, p, dt, lD, polozeniaD, predkosciD);
		CudaCheckError();

		if( !(i%10) )	
		{
			float temperatura=0;
			temperatura_kernel<<<N/p,p>>>(N, polozeniaD, predkosciD, tempD);
			CudaCheckError();
			cublasSasum(cublashandle, N, tempD, 1, &temperatura);
			cout << setprecision(30) << temperatura << endl;

		}
	}
	CudaSafeCall( cudaMemcpy( predkosciH, predkosciD, N*sizeof(float4), cudaMemcpyDeviceToHost));
	plikout.open("dupa_out.txt", fstream::out | fstream::trunc );
	for(int i = 0; i < N; i++) plikout << predkosciH[i].x << ", " << predkosciH[i].y << ", " << predkosciH[i].z << endl;
	plikout.close();


	cudaFree(polozeniaD);
	cudaFree(predkosciD);
	cudaFree(tempD);
	
	return 0;
}


/*############################Funkcje na hoscie (CPU) - inicjalizujace predkosci i polozenia##########*/
void initPositions() {

    // compute side of cube from number of particles and number density

    // find M large enough to fit N atoms on an fcc lattice.
   // The minimum energy con guration of this Lennard-Jones system is an fcc lattice.  This has // 4 lattice sites in each conventional cubic unit cell.  If the number of atoms N= 4M^3, 
//where M = 1;2,3... then the atoms can  fill a cubical volume. 
    r = new double* [N]; //a dynamic 2D array is basically an array of pointers to arrays
    v = new double* [N];
    for (int i = 0; i < N; i++) {
        r[i] = new double [4];
        v[i] = new double [4];
    }

    int M = 1;
    while (4 * M * M * M < N)
        ++M;
    double a = L / M;           // lattice constant of conventional cell

    // 4 atomic positions in fcc unit cell
    double xCell[4] = {0.25, 0.75, 0.75, 0.25};
    double yCell[4] = {0.25, 0.75, 0.25, 0.75};
    double zCell[4] = {0.25, 0.25, 0.75, 0.75};

    int n = 0;                  // atoms placed so far
    for (int x = 0; x < M; x++)
        for (int y = 0; y < M; y++)
            for (int z = 0; z < M; z++)
                for (int k = 0; k < 4; k++)
                    if (n < N) {
                        r[n][0] = (x + xCell[k]) * a;
                        r[n][1] = (y + yCell[k]) * a;
                        r[n][2] = (z + zCell[k]) * a;
			polozeniaH[n] = make_float4((float) r[n][0], (float) r[n][1], (float) r[n][2], (float) weight);
                        ++n;
                    }
}


double gasdev () {
     static bool available = false;
     static double gset;
     double fac, rsq, v1, v2;
     if (!available) {
          do {
               v1 = 2.0 * rand() / double(RAND_MAX) - 1.0;
               v2 = 2.0 * rand() / double(RAND_MAX) - 1.0;
               rsq = v1 * v1 + v2 * v2;
          } while (rsq >= 1.0 || rsq == 0.0);
          fac = sqrt(-2.0 * log(rsq) / rsq);
          gset = v1 * fac;
          available = true;
          return v2*fac;
     } else {
          available = false;
          return gset;
     }
}

void initVelocities() {

    // Gaussian with unit variance
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            v[n][i] = gasdev();

    // Adjust velocities so center-of-mass velocity is zero
    double vCM[3] = {0, 0, 0};
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vCM[i] += v[n][i];
    for (int i = 0; i < 3; i++)
        vCM[i] /= N;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            v[n][i] -= vCM[i];

    // Rescale velocities to get the desired instantaneous temperature
    rescaleVelocities();
    for (int i = 0; i < N; i++){
    	predkosciH[i] = make_float4((float) v[i][0], (float) v[i][1], (float) v[i][2], 0.0);
	//printf("%d")
    }

}

void rescaleVelocities() {
    double vSqdSum = 0;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vSqdSum += v[n][i] * v[n][i];
    double lambda = sqrt( 3 * (N-1) * T / vSqdSum );
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            v[n][i] *= lambda;
}


/*#########################################################################################*/


__global__ void temperatura_kernel(int Nk, float4* polozenia, float4* predkosci, float* temp)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	if(tid < Nk)
	{
		float4 predkosc = predkosci[tid];
		float masa = polozenia[tid].w;
		float kwadrat_predkosci = predkosc.x*predkosc.x + predkosc.y*predkosc.y + predkosc.z*predkosc.z;
		float energia = masa * kwadrat_predkosci;	
		temp[tid] = energia;
	}
}






__global__ void licz_krok_kernel(int Nk, int pk, float dtk, float lD, float4 *pozycje, float4* predkosci)
{
	extern __shared__ float4 shPozycja[];

	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	if( tid < Nk )
	{
		float4 pozycja = pozycje[tid];
		float4 predkosc = predkosci[tid];
	        //printf("%s", "++++++++++++++++++++++++++++++++");
		//printf(" ppppp %f %d \n ", pozycja.x, tid);;
			
	// == liczenie przyspieszen ==
		float3 przyspieszenie = {0.0f, 0.0f, 0.0f};

		int kafelek,i;
	#pragma unroll
		przyspieszenie = licz_kafelek(pozycja, pozycje, przyspieszenie, Nk, pk);
		
	// ===========================
		printf(" aaaa %f %d \n ", przyspieszenie.x, tid);;
		pozycja.x += predkosc.x * dtk + 0.5 *przyspieszenie.x * dtk *dtk;
		pozycja.y += predkosc.y * dtk + 0.5 *przyspieszenie.y * dtk *dtk;
		pozycja.z += predkosc.z * dtk + 0.5 *przyspieszenie.z * dtk *dtk;
		if (pozycja.x < 0.0)
			pozycja.x += lD;
		if (pozycja.x >= lD)
			pozycja.x -= lD;
		if (pozycja.y < 0.0)
			pozycja.y += lD;
		if (pozycja.y >= lD)
			pozycja.y -= lD;
		if (pozycja.z < 0.0)
			pozycja.z += lD;
		if (pozycja.z >= lD)
			pozycja.z -= lD;

		__syncthreads();
		//printf("%s", "=========");
		printf(" ppppp %f %d \n ", pozycja.x, tid);
		predkosc.x += 0.5 * przyspieszenie.x * dtk;
		predkosc.y += 0.5 * przyspieszenie.y * dtk;
		predkosc.z += 0.5 * przyspieszenie.z * dtk;

		przyspieszenie = make_float3(0.0f, 0.0f, 0.0f);
		
		przyspieszenie = licz_kafelek(pozycja, pozycje, przyspieszenie, Nk, pk);
		predkosc.x += 0.5 * przyspieszenie.x * dtk;
		predkosc.y += 0.5 * przyspieszenie.y * dtk;
		predkosc.z += 0.5 * przyspieszenie.z * dtk;

		pozycje[tid] = pozycja;
		predkosci[tid] = predkosc;
	}
}




// liczy kafelek przyspieszen
__device__ float3 licz_kafelek(float4 mojaPozycja, float4 *pozycje, float3 przyspieszenie, float Nk, float pk)
{
	extern __shared__ float4 shPozycja[];
	float number_tiles = Nk/pk
	
	for(kafelek = 0; i < number_tiles, kafelek++)
	{			
		int idx = threadIdx.x + kafelek*blockDim.x;
		shPozycja[threadIdx.x] = pozycje[idx];
		__syncthreads();
	#pragma unroll
		for (unsigned int counter = 0; counter < blockDim.x; counter++)
       		{
	       		 przyspieszenie = przyspieszenie_update(sharedPos[counter], bodyPos, przyspieszenie);
       		}
       		__syncthreads();
	}
	return przyspieszenie;
	
}




// funkcja oblicza nowe przyspieszenie na podstawie starego oraz pozycji dwóch cząstek
__device__ float3 przyspieszenie_update(float4 bi, float4 bj, float3 ai)
{

// obliczanie odleglosci
	float3 r;

	//r_ij [3 FLOPS]
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;
	
	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z ;	

// obliczanie siły

	float factor = 1.0/distSqr;

	float eight = factor * factor * factor * factor;
	float sixteenth = eight * eight;
	float fourteenth = sixteenth / factor; 
	float force = 24.0 * 10.0*(2 * fourteenth - eight);					
	force /= bi.w;
// obliczanie a(t+dt) = a(t) + f(t+dt)/m
	ai.x += r.x * force;
	ai.y += r.y * force;
	ai.z += r.z * force;
	//printf("force %f dist2 %f 14pow %f ai.x %f r.x %f \n", force, distSqr, fourteenth, ai.x, r.x);
	return ai;
}


