#include<stdio.h>
#include<cuda_runtime.h>
#include <time.h>
#define SIZE__ 4096
#define M SIZE__
#define N SIZE__
#define K SIZE__
#define THREAD_PER_BLOCK 8 


__global__ void gemm(int*a, int *b, int *c){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int res=0;
    __shared__ int Asub[THREAD_PER_BLOCK][THREAD_PER_BLOCK];
    __shared__ int Bsub[THREAD_PER_BLOCK][THREAD_PER_BLOCK];
    for(int start=0;start<K;start+=THREAD_PER_BLOCK){
	if(i<M)
            Asub[tx][ty]=a[i*M+start+ty];
	else
	    Asub[tx][ty]=0;
	if(j<N)
            Bsub[tx][ty]=b[(start+tx)*K+j];
	else
	    Bsub[tx][ty]=0;
	__syncthreads();
	if(i<M&&j<N)
        for(int k=0;k<THREAD_PER_BLOCK;++k){
            res+=Asub[tx][k]*Bsub[k][ty];
        }
	__syncthreads();
    }
    if(i<M&&j<N)
	c[i*M+j]=res;
}

int main(){
//  freopen("log.out","w",stdout);
    int *a, *b, *c, *c_cmp;
    a = (int*) malloc(M * K * sizeof(int));
    b = (int*) malloc(K * N * sizeof(int));
    c = (int*) malloc(M * N * sizeof(int));
    c_cmp = (int*) malloc(M * N * sizeof(int));
    srand((unsigned)time(NULL)); 
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            a[i * K + j] = rand() % 1024;
        }
    }
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            b[i * N + j] = rand() % 1024;
        }
        
    }
    int *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, M * K * sizeof(int));
    cudaMalloc(&b_d, K * N * sizeof(int));
    cudaMalloc(&c_d, M * N * sizeof(int));
    cudaMemcpy(a_d, a, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(THREAD_PER_BLOCK, THREAD_PER_BLOCK);    // block及thread的分配方式可以自己修改
    dim3 grid(M/THREAD_PER_BLOCK+1, N/THREAD_PER_BLOCK+1); //防止M,N不能被整除
    gemm<<<grid, block>>>(a_d, b_d, c_d);
    if(SIZE__<=1024){
        cudaMemcpy(c, c_d, M * N * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
                for(int k = 0; k < K; k++){
                    c_cmp[i * N + j] += a[i * K + k] * b[k * N + j]; 
                }
            }
        }
        bool flag = 1;
        for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
                if(c_cmp[i * N + j] != c[i * N + j]){
                    flag = 0;
                    break;
                }
            }
            if(flag==0)break;
        }
        if(flag){
            printf("result correct\n");
        }
        else{
            printf("result wrong\n");
        }
    }
    else {
    	printf("no verify\n");
    }
    free(a);
    free(b);
    free(c);
    free(c_cmp);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
