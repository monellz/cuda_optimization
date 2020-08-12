#pragma once

#include <stdint.h>
#include <sys/time.h>
#include <iostream>
#include <stdio.h>
using namespace std;


#define GETTIME(now)    \
    do {    \
        struct timeval t;   \
        gettimeofday(&t, NULL); \
        now = t.tv_sec + t.tv_usec; \
    } while (0)

#define CUDA_CALL(call)                                                \
  do {                                                                  \
    cudaError_t _status = call;                                         \
    if (_status != cudaSuccess) {                                     \
      const char *errstr;                                               \
      errstr = cudaGetErrorString(_status);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)



class Timer {
private:
    uint64_t begin;
    uint64_t total;
public:
    Timer(): begin(0), total(0) {}

    void start() {
        GETTIME(begin);
    }

    void stop() {
        uint64_t now;
        GETTIME(now);
        total += now - begin;
    }

    void show() {
        printf("Time: %lf sec / %lu usec\n", (double)total / 1000000.0, total);
    }

    void clear() {
        total = 0;
    }
};