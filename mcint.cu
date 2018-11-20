//# define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP

#include <thrust/functional.h> // function objects & tools
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include "helper_cuda.h"

/**********************************************************/

#ifdef _WIN32

#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>

typedef LARGE_INTEGER app_timer_t;

static inline void timer(app_timer_t *t_ptr)
{
#ifdef __CUDACC__
  checkCudaErrors(cudaDeviceSynchronize());
#endif
  QueryPerformanceCounter(t_ptr);
}

double elapsed_time(app_timer_t start, app_timer_t stop)
{
  LARGE_INTEGER clk_freq;
  QueryPerformanceFrequency(&clk_freq);
  return (stop.QuadPart - start.QuadPart) /
         (double) clk_freq.QuadPart * 1e3;
}

#else

#include <time.h> /* requires linking with rt library
                     (command line option -lrt) */

typedef struct timespec app_timer_t;

static inline void timer(app_timer_t *t_ptr) {
#ifdef __CUDACC__
    checkCudaErrors(cudaDeviceSynchronize());
#endif
    clock_gettime(CLOCK_MONOTONIC, t_ptr);
}

double elapsed_time(app_timer_t start, app_timer_t stop) {
    return 1e+3 * (stop.tv_sec - start.tv_sec) +
           1e-6 * (stop.tv_nsec - start.tv_nsec);
}

#endif

/**********************************************************/

class randuni :
        public thrust::unary_function<unsigned long long, float> {
private:
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> uni;
public:
    randuni(unsigned int seed, float a = 0.0f, float b = 1.0f) :
            rng(seed), uni(a, b) {}

    __host__ __device__

    float operator()(unsigned long long i) {
        rng.discard(i); // odrzuæ liczby z "poprzednich" w¹tków
        return uni(rng);
    }
};

/**********************************************************/

typedef thrust::tuple<float, float, float> point3D;

struct fun : public thrust::unary_function<point3D, float> {
    __host__ __device__

    float operator()(const point3D &p) const {
        float x = thrust::get<0>(p);
        float y = thrust::get<1>(p);
        float z = thrust::get<2>(p);
        float s = x * x + y * y + z * z;
        if (s <= 1) return 1;
        else return 0;
    }
};

/**********************************************************/

int main() {
    app_timer_t t0, t1, t2, t3;
    float integral;
    timer(&t0); //--------------------------------------------
    thrust::device_vector<float> x(1000), y(x.size()), z(x.size());
    timer(&t1); //--------------------------------------------
    randuni gen_x(40, -1.0f, 1.0f);
    randuni gen_y(41, -1.0f, 1.0f);
    randuni gen_z(42, -1.0f, 1.0f);

    thrust::transform(thrust::make_counting_iterator<unsigned long long>(0),
                      thrust::make_counting_iterator<unsigned long long>(x.size()),
                      x.begin(), gen_x);

    thrust::transform(thrust::make_counting_iterator<unsigned long long>(0),
                      thrust::make_counting_iterator<unsigned long long>(y.size()),
                      y.begin(), gen_y);

    thrust::transform(thrust::make_counting_iterator<unsigned long long>(0),
                      thrust::make_counting_iterator<unsigned long long>(z.size()),
                      z.begin(), gen_z);

    timer(&t2); //--------------------------------------------

    integral = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end())),
            fun(),
            0.0f,
            thrust::plus<float>()
    ) * 8 / 1000;

    timer(&t3); //--------------------------------------------

    using std::cout;
    using std::endl;
    cout << "pi = " << 0.75f * integral << std::endl; // =pi?
    cout << "Inicjacja:  " << elapsed_time(t0, t1) << " ms" << endl;
    cout << "Generacja:  " << elapsed_time(t1, t2) << " ms" << endl;
    cout << "Integracja: " << elapsed_time(t2, t3) << " ms" << endl;
    cout << "R A Z E M : " << elapsed_time(t0, t3) << " ms" << endl;
#ifdef _WIN32
    if (IsDebuggerPresent()) getchar();
#endif
    return 0;
}