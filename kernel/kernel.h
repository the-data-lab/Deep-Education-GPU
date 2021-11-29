#pragma once
#include <string>
#include <cstring>
#include "stdint.h"
#include "op.h"
#include <cassert>
#ifdef B64
typedef uint64_t vid_t;
#elif B32
typedef uint32_t vid_t;
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

typedef union __univeral_type {
    vid_t    value_int;
#ifdef B32   
    float    value_float;
#else     
    double   value_float;
    double   value_double;
#endif
} univ_t;


template <class T>
struct array1d_t {
    T* data_ptr;
    int64_t col_count;
    bool alloc;

    CUDA_CALLABLE_MEMBER T& operator[] (int64_t index) {//returns the element 
        return data_ptr[index];
    }
    CUDA_CALLABLE_MEMBER  array1d_t(T* ptr, int64_t a_col_count) {
        data_ptr = ptr;
        col_count = a_col_count;
    }
};



//2D tensor
template <class T>
struct array2d_t {
    T* data_ptr;
    int64_t row_count;
    int64_t col_count;
    CUDA_CALLABLE_MEMBER T* operator[] (int64_t index) {//returns a row
        return data_ptr + col_count*index;
    }
    array2d_t(T* a_ptr, int64_t a_row_count, int64_t a_col_count) {
        data_ptr = a_ptr;
        row_count = a_row_count;
        col_count = a_col_count;
    }
};

//3D tensor
template <class T>
struct array3d_t {
    T* data_ptr;
    int64_t matrix_count;
    int64_t row_count;
    int64_t col_count;
    CUDA_CALLABLE_MEMBER T* operator[] (int64_t index) {//returns a matrix
        return data_ptr + col_count * row_count * index;
    }
    array3d_t(T* a_ptr, int64_t a_matrix_count, int64_t a_row_count, int64_t a_col_count) {
        data_ptr = a_ptr;
        matrix_count = a_matrix_count;
        row_count = a_row_count;
        col_count = a_col_count;
    }
};

class edge_t {
    public:
        vid_t src;
        vid_t dst;
        //edge properties here if any.
};

class csr_t {
    public:
        vid_t * offset;
        char * Nebr;
        vid_t dst_size;
        vid_t v;
        vid_t  e_count;

    public:
        csr_t() {};

        CUDA_CALLABLE_MEMBER  void init(vid_t a_vcount,vid_t a_dstsize, void * a_offset, void * a_nebrs) {
            v        = a_vcount;
            dst_size = a_dstsize;
            offset   = (vid_t * ) a_offset;
            Nebr     = (char * ) a_nebrs;
            e_count  = offset[v];
        }

        CUDA_CALLABLE_MEMBER  vid_t get_degree(vid_t index) const {
            return offset[index+1]-offset[index];
        }

        vid_t get_vcount() const {
            return v;
        }

        CUDA_CALLABLE_MEMBER vid_t get_ecount() const {
            return e_count;
        }

        CUDA_CALLABLE_MEMBER  vid_t get_vid(vid_t* header, vid_t index) const {
            vid_t* h = (vid_t*)((char*)header + index*dst_size);
            return *h;
        }

        CUDA_CALLABLE_MEMBER  vid_t get_nebrs(vid_t v, vid_t*& ptr) const {
            vid_t degree = offset[v+1] - offset[v];
            ptr = (vid_t*)(Nebr + offset[v]*dst_size);
            return degree; 
        }  
};

class graph_t {
    public:
        csr_t* csr;
        csr_t* csc;

        graph_t() {
            csr = new csr_t;
            csc = new csr_t;
        }

        void init_cpu(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, void* a_offset1, void* a_nebrs1) {
            csr->init(a_vcount, a_dstsize, a_offset, a_nebrs);
            csc->init(a_vcount, a_dstsize, a_offset1, a_nebrs1);
        }

        CUDA_CALLABLE_MEMBER  void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, void* a_offset1, void* a_nebrs1) {
            csr->init(a_vcount, a_dstsize, a_offset, a_nebrs);
            csc->init(a_vcount, a_dstsize, a_offset1, a_nebrs1);
        }

        CUDA_CALLABLE_MEMBER vid_t get_vcount() {
            return csr->v;
        }

        CUDA_CALLABLE_MEMBER   vid_t get_edge_count() {
            return csr->e_count;
        }
};


