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
__global__ void licz_krok_kernel(int Nk, int pk, float dtk, float4 *pozycje, float4* predkosci);
__device__ float3 licz_kafelek(float4 mojaPozycja, float3 przyspieszenie);
__device__ float3 przyspieszenie_update(float4 bi, float4 bj, float3 ai);


#define sigma (1)
#define epsilon (1e-30)
#define kb 1


const int N = 256*256;
const int p = 256;
const float dt = 1e-12;

int main()
{
	cublasHandle_t cublashandle;
	cublasCreate(&cublashandle);

    	srand( time( NULL ) );


	float4* polozeniaH = new float4[N];
	float4* predkosciH = new float4[N];
	
	float4* polozeniaD;
	CudaSafeCall(cudaMalloc( (void**)&polozeniaD, N*sizeof(float4)) );
	float4* predkosciD;
	CudaSafeCall(cudaMalloc( (void**)&predkosciD, N*sizeof(float4)) );
	float* tempD;
	CudaSafeCall(cudaMalloc( (void**)&tempD, N*sizeof(float)) );

 	for(int i = 0; i < N; i++)
	{
		float a = 1000000;
		polozeniaH[i] = make_float4( a*rand()/RAND_MAX , a*rand()/RAND_MAX, a*rand()/RAND_MAX , 1);
		predkosciH[i] = make_float4( 1, 2, 3, 4);
	}


	fstream plikout;
	plikout.open("dupa_in.txt", fstream::out | fstream::trunc );
	for(int i = 0; i < N; i++) plikout << predkosciH[i].x << ", " << predkosciH[i].y << ", " << predkosciH[i].z << endl;
	plikout.close();

	CudaSafeCall(cudaMemcpy( polozeniaD, polozeniaH, N*sizeof(float4), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy( predkosciD, predkosciH, N*sizeof(float4), cudaMemcpyHostToDevice));


	
	for(int i = 0; i < 100; i++)
	{

		licz_krok_kernel<<<N/p, p, p*sizeof(float4)>>>(N, p, dt, polozeniaD, predkosciD);
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






__global__ void licz_krok_kernel(int Nk, int pk, float dtk, float4 *pozycje, float4* predkosci)
{
	extern __shared__ float4 shPozycja[];

	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	if( tid < Nk )
	{
		float4 pozycja = pozycje[tid];
		float4 predkosc = predkosci[tid];
			
	// == liczenie przyspieszen ==
		float3 przyspieszenie = {0.0f, 0.0f, 0.0f};

		int kafelek,i;
	#pragma unroll
		for(i = 0, kafelek = 0; i < Nk; i += pk, kafelek++)
		{
			int idx = threadIdx.x + kafelek*blockDim.x;
			shPozycja[threadIdx.x] = pozycje[idx];
			__syncthreads();
			przyspieszenie = licz_kafelek(pozycja, przyspieszenie);
			__syncthreads();
		}
	// ===========================

		predkosc.x += przyspieszenie.x * dtk;
		predkosc.y += przyspieszenie.y * dtk;
		predkosc.z += przyspieszenie.z * dtk;

		pozycja.x += predkosc.x * dtk;
		pozycja.y += predkosc.y * dtk;
		pozycja.z += predkosc.z * dtk;

		pozycje[tid] = pozycja;
		predkosci[tid] = predkosc;
	}
}




// liczy kafelek przyspieszen
__device__ float3 licz_kafelek(float4 mojaPozycja, float3 przyspieszenie)
{
	extern __shared__ float4 shPozycja[];
	int i;

	for (i = 0; i < blockDim.x; i++) 
	{
		przyspieszenie = przyspieszenie_update(mojaPozycja, shPozycja[i], przyspieszenie);
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
	float sigmaSqr = sigma*sigma;

	float factor = sigmaSqr/distSqr;

	float eight = factor * factor * factor * factor;
	float sixteenth = eight * eight;
	float fourteenth = sixteenth / factor; 
	float force = 24.0f * epsilon*(2 * fourteenth - eight);					
	force /= bi.w;
// obliczanie a(t+dt) = a(t) + f(t+dt)/m
	ai.x += r.x * force;
	ai.y += r.y * force;
	ai.z += r.z * force;
	
	return ai;
}


