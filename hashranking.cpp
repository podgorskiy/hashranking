/*
 # Copyright 2018-2019 Stanislav Pidhorskyi
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #	http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 # ==============================================================================
 */

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifdef _WIN32
#include <intrin.h>
#define popcount32 __popcnt
#define popcount64 __popcnt64
#else
#define popcount32 __builtin_popcount
#define popcount64 __builtin_popcountll
#endif

#include <inttypes.h>

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style> ndarray_float;
typedef py::array_t<uint8_t, py::array::c_style> ndarray_uint8;
typedef py::array_t<uint32_t, py::array::c_style> ndarray_uint32;
typedef py::array_t<uint64_t, py::array::c_style> ndarray_uint64;

inline uint8_t hamming_distance32(uint32_t x, uint32_t y)
{
	uint32_t val = x ^ y;
	return (uint8_t)popcount32(val);
}

inline uint8_t hamming_distance64(uint64_t x, uint64_t y)
{
	uint64_t val = x ^ y;
	return (uint8_t)popcount64(val);
}

template<typename T>
uint8_t hamming_distance(T x, T y);

template<>
inline uint8_t hamming_distance(uint32_t x, uint32_t y)
{
	return hamming_distance32(x, y);
}

template<>
inline uint8_t hamming_distance(uint64_t x, uint64_t y)
{
	return hamming_distance64(x, y);
}

inline void to_int32_hashes(py::array_t<float, py::array::c_style> x, uint32_t* __restrict out)
{
	auto p = x.unchecked<2>();
	int w = (int)p.shape(1);
	int h = (int)p.shape(0);

	for (int i = 0; i < h; ++i)
	{
		uint32_t output = 0;
		uint32_t power = 1;
		const float* __restrict hash = p.data(i, 0);

		for (int y = 0; y < w; ++y)
		{
			output += (hash[y] > 0.0f ? power : 0);
			power *= 2;
		}
		out[i] = output;
	}
}

inline void to_int64_hashes(py::array_t<float, py::array::c_style> x, uint64_t* __restrict out)
{
	auto p = x.unchecked<2>();
	int w = (int)p.shape(1);
	int h = (int)p.shape(0);

	for (int i = 0; i < h; ++i)
	{
		uint64_t output = 0;
		uint64_t power = 1;
		const float* __restrict hash = p.data(i, 0);

		for (int y = 0; y < w; ++y)
		{
			output += (hash[y] > 0.0f ? power : 0);
			power *= 2;
		}
		out[i] = output;
	}
}

template<typename T>
void to_int_hashes(py::array_t<float, py::array::c_style> x, T* out);

template<>
inline void to_int_hashes(py::array_t<float, py::array::c_style> x, uint32_t* out)
{
	to_int32_hashes(x, out);
}

template<>
inline void to_int_hashes(py::array_t<float, py::array::c_style> x, uint64_t* out)
{
	to_int64_hashes(x, out);
}

template<typename T>
void _calc_hamming_dist(T* __restrict b2, ssize_t b2_size, T b1, uint8_t* out)
{
	ssize_t j = 0;
	for (; j + 3 < b2_size; j += 4)
	{
		out[j + 0] = hamming_distance<T>(b2[j + 0], b1);
		out[j + 1] = hamming_distance<T>(b2[j + 1], b1);
		out[j + 2] = hamming_distance<T>(b2[j + 2], b1);
		out[j + 3] = hamming_distance<T>(b2[j + 3], b1);
	}
	for (; j < b2_size; ++j)
	{
		out[j] = hamming_distance<T>(b2[j], b1);
	}
}

template<typename T>
ndarray_uint8 _calc_hamming_dist(ndarray_float b1, ndarray_float b2)
{
	py::buffer_info buf1 = b1.request(), buf2 = b2.request();
	
	py::gil_scoped_release release;
	
	ndarray_uint8 result = ndarray_uint8(std::vector<ssize_t>{buf1.shape[0], buf2.shape[0]});
	auto r = result.mutable_unchecked<2>();
	
	T* __restrict b1_int = (T*)(malloc(sizeof(T) * buf1.shape[0]));
	T* __restrict b2_int = (T*)(malloc(sizeof(T) * buf2.shape[0]));
	to_int_hashes<T>(b1, b1_int);
	to_int_hashes<T>(b2, b2_int);
	
	for (ssize_t i = 0, l = buf1.shape[0]; i < l; ++i)
	{
		uint8_t* __restrict ptr = r.mutable_data(i, 0);
		T b1_i = b1_int[i];
		_calc_hamming_dist<T>(b2_int, buf2.shape[0], b1_i, ptr);
	}
	free(b1_int);
	free(b2_int);
	
	return result;
}

ndarray_uint8 calc_hamming_dist(ndarray_float b1, ndarray_float b2)
{
	py::buffer_info buf1 = b1.request(), buf2 = b2.request();

	if (buf1.ndim != 2 || buf2.ndim != 2)
		throw std::runtime_error("Number of dimensions must be two");

	if (buf1.shape[1] != buf2.shape[1])
		throw std::runtime_error("Second dimension must match");
	
	if (buf1.shape[1] > 64)
		throw std::runtime_error("Supports only hashes up to 64b");
	
	bool hash32 = buf1.shape[1] <= 32;
	
	if (hash32)
	{
		return _calc_hamming_dist<uint32_t>(b1, b2);
	}
	else
	{
		return _calc_hamming_dist<uint64_t>(b1, b2);
	}
}

void argsort_1d(uint32_t* __restrict out_ptr, const uint8_t* __restrict d_ptr, ssize_t size)
{
	int32_t count[65];
	memset(count, 0, sizeof(int32_t) * 65);
	
	for (ssize_t y = 0; y < size; ++y)
		count[d_ptr[y]]++;	 
	
	for (int i = 1; i < 65; ++i)
	{
		count[i] += count[i - 1];	
	}
	
	for (ssize_t y = size -1; y >= 0; --y)
	{
		int8_t key = d_ptr[y];	 
		out_ptr[count[key] - 1] = (uint32_t)y;	 
		count[key] -= 1;   
	} 
}

ndarray_uint32 argsort(ndarray_uint8 distance)
{
	py::buffer_info d_info = distance.request();
	
	py::gil_scoped_release release;
	
	if (d_info.ndim != 2)
		throw std::runtime_error("Number of dimensions must be two");

	ssize_t l1 = d_info.shape[0];
	ssize_t l2 = d_info.shape[1];
	
	ndarray_uint32 result = ndarray_uint32(std::vector<ssize_t>{l1, l2});
	
	auto r = result.mutable_unchecked<2>();
	auto d = distance.mutable_unchecked<2>();
	
	for (ssize_t x = 0; x < l1; ++x)
	{
		uint32_t* __restrict out_ptr = r.mutable_data(x, 0);
		const uint8_t* __restrict d_ptr = d.data(x, 0);
		
		argsort_1d(out_ptr, d_ptr, l2);
	}
	
	return result;
}

PYBIND11_MODULE(_hashranking, m) {
	m.doc() = "";

	m.def("calc_hamming_dist", &calc_hamming_dist, "Compute hamming distance of all hash pairs from two arrays of hashes");
	m.def("argsort", &argsort, "Argsort of distance matrix along second dimention");

	//m.def("add_circle_filled", &AddCircleFilled, py::arg("centre"), py::arg("radius"), py::arg("col"), py::arg("num_segments") = 12);
}
