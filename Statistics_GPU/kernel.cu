
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <cassert>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <ctime>
//#include "cuda.h"
//
//
//
//#include "cuda_runtime_api.h"


#define MIN 2
#define MAX 7
#define ITER 10000000

#define THREADS 512 // 2^9
#define BLOCKS 32768 // 2^15
#define NUM_VALS THREADS*BLOCKS

using namespace std;

typedef struct Data
{
	float *numbers_ptr;
	int numbers_length;
} Data;

__global__ void setup_kernel(curandState *state)
{

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(7 + idx, idx, 0, &state[idx]);
}


//__global__ void generateData(Data data, int data_length)
//{
//	int idx = threadIdx.x + blockDim.x * blockIdx.x;
//	
//	for(int i = 0; i < data_length; i++)
//	{
//		float myrandf = curand_uniform(curandstate + idx);
//		myrandf *= (max_rand_int[idx] - min_rand_int[idx] + 0.999999);
//		myrandf += min_rand_int[idx];
//		int myrand = (int)truncf(myrandf);
//
////		assert(myrand <= max_rand_int[idx]);
////		assert(myrand >= min_rand_int[idx]);
//		result[myrand - min_rand_int[idx]]++;
//		
//	}
//}

void print_elapsed(clock_t start, clock_t stop)
{
	double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.3fs\n", elapsed);
}

int random_float(int range)
{
	return rand() % range;
}

void array_print(float *arr, int length)
{
	int i;
	for (i = 0; i < length; ++i) {
		printf("%1.3f ", arr[i]);
	}
	printf("\n");
}



__global__ void generate_kernel(curandState *my_curandstate, const unsigned int n, const unsigned *max_rand_int, const unsigned *min_rand_int, unsigned int *result)
{

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	for(int i = 0; i < n; i++)
	{
		float myrandf = curand_uniform(my_curandstate + idx);
		myrandf *= (max_rand_int[idx] - min_rand_int[idx] + 0.999999);
		myrandf += min_rand_int[idx];
		int myrand = (int)truncf(myrandf);

//		assert(myrand <= max_rand_int[idx]);
//		assert(myrand >= min_rand_int[idx]);
		result[myrand - min_rand_int[idx]]++;
		
	}
}

struct varianceshifteop
	: std::unary_function<float, float>
{
	varianceshifteop(float m)
		: mean(m)
	{ /* no-op */
	}

	const float mean;

	__device__ float operator()(float data) const
	{
		return ::pow(data - mean, 2.0f);
	}
};
__device__ float calculateMean(long long sum, int length)
{
	return (float)sum / (float)length;
}

__device__ float calculateVariance(float mean)
{


//	return variance;
}

__device__ float calculateStandardDerivative(float variance)
{
	// standard dev is just a sqrt away
	float stdv = std::sqrtf(variance);

	return stdv;
}





__device__ float calculateMedian(int *data, int length)
{
	
	if(length % 2)
	{
		return (float)(data[length / 2] + data[length / 2 + 1]) / (float)2;
	}
	
	return (float)data[length / 2 + 1];
}

__device__ int calculateMode(int *data, int length, int *count_values, int range)
{
	for(int i = 0; i < length; i++)
		count_values[data[i]]++;
	
	int max = 0, value = 0;
	for(int i = 0; i < range; i++)
		if(max < count_values[i])
		{
			max = count_values[i];
			value = i;
		}

	return value;
}

__global__ void calculateStatistics(int *data, int length, int* count_values, int range, long long sum, float variance)
{
	float mean = calculateMean(sum, length);
	float std_dv = calculateStandardDerivative(variance);
	float median = calculateMedian(data, length);
	float mode = calculateMode(data, length, count_values, range);

//	printf("Median = %f Mode = %f", median, mode);
	printf("Mean = %f Variance = %f Standard Deviation = %f Median = %f Mode = %f", mean, variance, std_dv, median, mode);
}

//__host__ void generateData(int *data, int length)
//{
//	srand(0);
//	for (int i = 0; i < length; i++)
//	{
//		data[i] = rand();
//	}
//}

__host__ void generateData(int *arr, int length, int range)
{
	srand(time(NULL));
	int i;
	for (i = 0; i < length; ++i)
		arr[i] = random_float(range);
}

int main()
{
	int length = 1000000;
	int *h_data = new int[length], *d_data;
	int range = 1000;
	generateData(h_data, length, range);
	thrust::sort(h_data, h_data + length);
	cudaMalloc(&d_data, length * sizeof(int));
	cudaMemcpy(d_data, h_data, length * sizeof(int), cudaMemcpyHostToDevice);

	int *d_count_values;
	cudaMalloc(&d_count_values, range * sizeof(int));
	cudaMemset(d_count_values, 0, range * sizeof(int));
	thrust::device_vector< int > iVec(h_data, h_data + length);
	long long sum = thrust::reduce(iVec.begin(), iVec.end(), 0, thrust::plus<int>());
	float variance = thrust::transform_reduce(
		iVec.cbegin(),
		iVec.cend(),
		varianceshifteop((float)sum / (float)length),
		0.0f,
		thrust::plus<float>()) / (iVec.size() - 1);
	calculateStatistics<<<1, 1>>>(d_data, length, d_count_values, range, sum, variance);
	cudaDeviceSynchronize();

	return 0;
}