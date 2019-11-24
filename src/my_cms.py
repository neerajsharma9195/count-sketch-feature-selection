import statistics
import hashlib
import random

kernel = '''
extern "C"
__inline__ __device__
int hash(int value, int range, int a, int b)
{
	int h = a * value + b;
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h % range;
}

extern "C"
__inline__ __device__
float minimum(float a, float b, float c)
{
	return fminf(fminf(a,b),c);
}

extern "C"
__inline__ __device__
float update_retrieve(float* mem,
	float* result,
	const int N,
	const int D,
	const long index,
	const float value)
{
    	int a = 994443;
    	int b = 609478;
        const int hash_idx = hash(index, N, a, b) * D + threadIdx.x;
		mem[hash_idx] += value; 
        return mem[hash_idx];
}

extern "C"
__inline__ __device__
float cms_update_retrieve(float* mem,
	float* result,
	const int N,
	const int W,
	const int D,
	const long index,
	const float value)
{
	float r[3];
	int a[3] = {994443, 4113759, 9171025};
	int b[3] = {609478, 2949676, 2171464};
	for(int idx = 0; idx < 3; ++idx)
	{
		const int hash_idx = idx*W + hash(index, N, a[idx], b[idx]) * D + threadIdx.x;
		mem[hash_idx] += value; 
		r[idx] = mem[hash_idx];
	}
	return minimum(r[0], r[1], r[2]);
}

extern "C"
__global__
void hash_update_retrieve(const long* indices,
	const float* values,
	float* mem,
	float* result,
	const int N,
	const int W,
    const int D)
{
	if(threadIdx.x < D)
	{
		const int idx = blockIdx.x * D + threadIdx.x;
		const float value = values[idx];
		const long index = indices[blockIdx.x];
		result[idx] = cms_update_retrieve(mem, result, N, W, D, index, value);
	}
}
'''


# SHA256
# BLAKE2B
# MD5: 128 bits
# mmh3:


class CountMinSketch(object):
    def __init__(self, w, h):
        self.h = h
        self.w = w
        self.cms = [[0 for i in range(w)] for j in range(h)]
        self.hash_func_list = [lambda x: hashlib.md5(str(x).encode()) % w,
                               lambda x: hashlib.sha3_256(str(x).encode()) % w,
                               lambda x: hashlib.blake2b(str(x).encode()) % w]
        self.sign_funcs = [1 if random.random() < 0.5 else -1, 1 if random.random() < 0.5 else -1,
                           1 if random.random() < 0.5 else -1]

    def query(self, x):
        output = [self.cms[i][h(x)] for i, h in enumerate(self.hash_func_list)]
        return statistics.median(output)

    def update(self, x, up_val):
        for i, h in enumerate(self.hash_func_list):
            val = h(x)
            self.cms[i][val] = self.cms[i][val] + (up_val * self.sign_funcs[i])


if __name__ == '__main__':
    cms = CountMinSketch(w=128, h=3)
    cms.update(5, 0.2)
