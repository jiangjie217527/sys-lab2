#include<stdio.h>
#include<cuda_runtime.h>
#include <time.h>

#define M 64
#define N 64
#define K 64
#define THREAD_PRE_BLOCK 32 


__global__ void gemm(int*a, int *b, int *c){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    c[i*M+j]=0;
    for(int k=0;k<N;k++){
        c[i*M+j]+=a[i*M+k]*b[k*M+j];
    }
}

int main(){
//	freopen("log.out","w",stdout);
    int *a, *b, *c, *c_cmp;
    a = (int*) malloc(M * K * sizeof(int));
    b = (int*) malloc(K * N * sizeof(int));
    c = (int*) malloc(M * N * sizeof(int));
    c_cmp = (int*) malloc(M * N * sizeof(int));
    srand((unsigned)time(NULL)); 
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            a[i * K + j] = rand() % 5;
        }
    }
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            b[i * N + j] = rand() % 5;
        }
        
    }
    int *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, M * K * sizeof(int));
    cudaMalloc(&b_d, K * N * sizeof(int));
    cudaMalloc(&c_d, M * N * sizeof(int));
    cudaMemcpy(a_d, a, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(THREAD_PRE_BLOCK, THREAD_PRE_BLOCK);    // block及thread的分配方式可以自己修改
    dim3 grid(M/THREAD_PRE_BLOCK, N/THREAD_PRE_BLOCK);
    gemm<<<grid, block>>>(a_d, b_d, c_d);

    cudaMemcpy(c, c_d, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < K; k++){
                c_cmp[i * N + j] += a[i * K + k] * b[k * N + j]; 
            }
        }
    }
    bool flag = 1;
    /*
    printf("c_cmp\n");
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
		printf("%d ",c_cmp[i*N+j]);
        }
	printf("\n");
    }
    printf("c\n");
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
		printf("%d ",c[i*N+j]);
        }
	printf("\n");
    }
    */
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
    free(a);
    free(b);
    free(c);
    free(c_cmp);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
