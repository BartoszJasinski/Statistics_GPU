#include "backup.h"
/*
struct parallel_random_generator
{
	__host__ __device__
		unsigned int operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;

		// discard n numbers to avoid correlation
		rng.discard(n);

		// return a random number
		return rng();
	}
};

/*

int main(void)
{
int N = 256;

// device storage for the random numbers
thrust::device_vector<int> numbers(N);

// a sequence counting up from 0
thrust::counting_iterator<int> index_sequence_begin(0);

// transform the range [0,1,2,...N]
// to a range of random numbers
thrust::transform(index_sequence_begin,
index_sequence_begin + N,
numbers.begin(),
parallel_random_generator());

// print out the random numbers
for (int i = 0; i < N; ++i)
{
std::cout << numbers[i] << " ";
}
std::cout << std::endl;

return 0;
}
#1#


__host__ thrust::device_vector<int> generateData(int *arr, int length, int range)
{
	thrust::device_vector<int> numbers(length);

	thrust::counting_iterator<int> index_sequence_begin(0);

	thrust::transform(index_sequence_begin,
		index_sequence_begin + length,
		numbers.begin(),
		parallel_random_generator());

	return numbers;
}


int main()
{
	int length = 1000000;
	int *h_data = new int[length], *d_data;
	int range = 1000;

	clock_t start = clock();

	generateData(h_data, length, range);
	clock_t stop = clock();
	print_elapsed(start, stop);


	thrust::sort(h_data, h_data + length);
	cudaMalloc(&d_data, length * sizeof(int));
	cudaMemcpy(d_data, h_data, length * sizeof(int), cudaMemcpyHostToDevice);

	int *d_count_values;
	cudaMalloc(&d_count_values, range * sizeof(int));
	cudaMemset(d_count_values, 0, range * sizeof(int));
	thrust::device_vector< int > iVec = generateData(h_data, length, range);
	long long sum = thrust::reduce(iVec.begin(), iVec.end(), 0, thrust::plus<int>());
	float variance = thrust::transform_reduce(
		iVec.cbegin(),
		iVec.cend(),
		varianceshifteop((float)sum / (float)length),
		0.0f,
		thrust::plus<float>()) / (iVec.size() - 1);
	calculateStatistics << <1, 1 >> >(d_data, length, d_count_values, range, sum, variance);
	cudaDeviceSynchronize();


	return 0;
}*/