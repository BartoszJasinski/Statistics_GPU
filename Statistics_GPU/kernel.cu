#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <ctime>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>


#define MIN 2
#define MAX 7
#define ITER 10000000

#define THREADS 512 // 2^9
#define BLOCKS 32768 // 2^15
#define NUM_VALS THREADS*BLOCKS

using namespace std;
using namespace thrust;

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


void printElapsed(clock_t start, clock_t stop)
{
	double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.3fs\n", elapsed);
}

int randomFloat(int range)
{
	return rand() % range;
}

void printArray(float *arr, int length)
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

		result[myrand - min_rand_int[idx]]++;
		
	}
}

struct varianceshifteop: std::unary_function<float, float>
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


__host__ float calculateMean(device_vector<int> data)
{
	long long sum = thrust::reduce(data.begin(), data.end(), 0, thrust::plus<int>());
	return (float)sum / (float)data.size();
}


 float calculateVariance(device_vector<int> data, float mean)
{

	float variance = thrust::transform_reduce(
		data.cbegin(),
		data.cend(),
		varianceshifteop(mean),
		0.0f,
		thrust::plus<float>()) / (data.size() - 1);

	return variance;
}

__host__ float calculateStandardDerivative(float variance)
{
	float stdv = std::sqrtf(variance);

	return stdv;
}





__host__ float calculateMedian(device_vector<int> data)
{
	
	if(data.size() % 2)
	{
		return (float)(data[data.size() / 2] + data[data.size() / 2 + 1]) / (float)2;
	}
	
	return (float)data[data.size() / 2 + 1];
}


__host__ int calculateMode(device_vector<int> d_data)
{
	sort(d_data.begin(), d_data.end());

	size_t num_unique = inner_product(d_data.begin(), d_data.end() - 1,
		d_data.begin() + 1,
		0,
		thrust::plus<int>(),
		thrust::not_equal_to<int>()) + 1;

	thrust::device_vector<int> d_output_keys(num_unique);
	thrust::device_vector<int> d_output_counts(num_unique);
	thrust::reduce_by_key(d_data.begin(), d_data.end(),
		thrust::constant_iterator<int>(1),
		d_output_keys.begin(),
		d_output_counts.begin());

	thrust::device_vector<int>::iterator mode_iter;
	mode_iter = max_element(d_output_counts.begin(), d_output_counts.end());

	int mode = d_output_keys[mode_iter - d_output_counts.begin()];

	return mode;
}


__host__ void calculateStatistics(device_vector<int> d_data, int range)
{
	float mean = calculateMean(d_data);
	float variance = calculateVariance(d_data, mean);
	float std_dv = calculateStandardDerivative(variance);
	float median = calculateMedian(d_data);
	float mode = calculateMode(d_data);

	printf("Mean = %f Variance = %f Standard Deviation = %f Median = %f Mode = %f", mean, variance, std_dv, median, mode);
}

struct parallel_random_generator
{
	__host__ __device__
		unsigned int operator()(const unsigned int n) const
	{
		default_random_engine rng;
		rng.discard(n);
		return rng();
	}
};


__host__ thrust::device_vector<int> generateData(int length, int range)
{
	device_vector<int> numbers(length);

	counting_iterator<int> index_sequence_begin(0);

	transform(index_sequence_begin,
		index_sequence_begin + length,
		numbers.begin(),
		parallel_random_generator());

	return numbers;
}



__device__ float calculateMean(long long sum, int length)
{
	return (float)sum / (float)length;
}

__device__ float calculateVariance(float mean)
{


	//	return variance;
}

__device__ float calculateStandardDerivativeCustom(float variance)
{
	// standard dev is just a sqrt away
	float stdv = std::sqrtf(variance);

	return stdv;
}





__device__ float calculateMedian(int *data, int length)
{

	if (length % 2)
	{
		return (float)(data[length / 2] + data[length / 2 + 1]) / (float)2;
	}

	return (float)data[length / 2 + 1];
}

__device__ void calculateMode(int *data, int length, int *count_values, int range, int quantity, float *result)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = idx * quantity; i < (idx + 1) * quantity; i++)
		count_values[data[i]]++;

	__syncthreads();
	if (idx == 0)
	{
		int max = 0;
		for (int i = 0; i < range; i++)
			if (max < count_values[i])
			{
				max = count_values[i];
				*result = i;
			}
	}
}

__global__ void calculateStatisticsCustom(int *data, int length, int* count_values, int range, long long sum, float variance, int quantity, float* mode)
{
	float mean = calculateMean(sum, length);
	float std_dv = calculateStandardDerivativeCustom(variance);
	float median = calculateMedian(data, length);
	calculateMode(data, length, count_values, range, quantity, mode);
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx == 0)
		printf("Mean = %f Variance = %f Standard Deviation = %f Median = %f Mode = %f", mean, variance, std_dv, median, *mode);
}


__host__ void generateData(int *arr, int length, int range)
{
	srand(time(NULL));
	for (int i = 0; i < length; ++i)
		arr[i] = randomFloat(range);
}

int main()
{
	int length = 1000000;
	int range = 1000;

	{
		device_vector< int > d_data = generateData(length, range);

		clock_t start = clock();
		calculateStatistics(d_data, range);
		cudaDeviceSynchronize();
		clock_t stop = clock();
		printElapsed(start, stop);

	}

	cout << endl;

	{

		int *h_data = new int[length], *d_data;
		generateData(h_data, length, range);
		
		clock_t start = clock();

		thrust::sort(h_data, h_data + length);
		cudaMalloc(&d_data, length * sizeof(int));
		cudaMemcpy(d_data, h_data, length * sizeof(int), cudaMemcpyHostToDevice);

		int *d_count_values;
		cudaMalloc(&d_count_values, range * sizeof(int));
		cudaMemset(d_count_values, 0, range * sizeof(int));
		
		float *d_mode;
		cudaMalloc(&d_mode, sizeof(float));

		thrust::device_vector< int > iVec(h_data, h_data + length);
		long long sum = thrust::reduce(iVec.begin(), iVec.end(), 0, thrust::plus<int>());

		
		float variance = thrust::transform_reduce(
			iVec.cbegin(),
			iVec.cend(),
			varianceshifteop((float)sum / (float)length),
			0.0f,
			thrust::plus<float>()) / (iVec.size() - 1);

		int block = 1000;
		int threads = 500;
		calculateStatisticsCustom << <block, threads >> >(d_data, length, d_count_values, range, sum, variance, length / (block * threads), d_mode);
		cudaDeviceSynchronize();

		clock_t stop = clock();
		printElapsed(start, stop);

	}


	return 0;
}
