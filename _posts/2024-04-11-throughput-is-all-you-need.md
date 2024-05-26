---
title: Throughput is all you need
layout: post
---
<!-- markdownlint-disable MD033 MD036-->

Written by [Romit Jain](https://www.linkedin.com/in/r0m1t/)

## Throughput, why?

If we want to build efficient applications on top of current LLMs, there are currently two challenges:

1. Improving **Inference latency**: The speed with which the model returns the tokens per second
2. Improving **Inference throughput**: The total number of requests that the model can serve in parallel

Inferencing LLMs with lower latency comes down to working around the limitations of the GPU’s memory bandwidth <sup>[1]</sup>. FlashAttention, speculative decoding, and KV caching are ways in which one can improve the latency of the model.

Increasing inference throughput comes down to effectively managing the available VRAM of the GPU. Given a limited budget of GPU VRAM, there are various areas where improvements can be made:

1. Reducing the size of the model: By quantization or knowledge distillation eg: GPTQ
2. Batching<sup>[2]</sup>: Batching more requests in the same amount of GPU VRAM
3. Separating prefill and decoding stages of generation <sup>[8]</sup>

One can refer to [Nvidia’s blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)<sup>[5]</sup> for an overview of the above concepts.

For this blog, let's zoom into one specific aspect of improving throughput, i.e. batching. After the model is loaded in the GPU VRAM, whatever remaining memory is available to us is reserved for the KV cache and serving the requests. The only lever that we can control here apart from the model size is the KV cache. Efficiently managing this KV cache can help us dramatically increase throughput by enabling us to batch more requests. For certain use cases, it can increase the throughput by 20x compared to native HuggingFace implementation.

vLLM is one such library that helps us achieve very high throughout. vLLM deploys LLMs on GPUs and focuses on:

1. Allocating the KV cache in the most efficient way possible
2. This, in turn, allows us to increase the batch size and server more requests per minute

In this blog, we will learn about the intuition behind vLLM, and its inner workings and also simulate it for a real-world application to understand the nuances and limitations of the library.

## Setup

Taking real-world numbers around model sizes and GPU VRAM can help visualize and validate the workings of vLLM. Let us consider a case of deploying a Mistral 7B model on the highest-end consumer-grade GPU (Nvidia RTX 4090). If we choose to deploy the model at half-precision (FP 16, each parameter taking 2 bytes), the model would occupy ~14 GB of the VRAM from the available 24 GB VRAM on a 4090 GPU. Assuming an overhead of 3 GBs, the GPU would have 7 GB of VRAM available. This 7 GB of available VRAM will be reserved for the KV cache.

![alt_text](assets/images/post1/image1.png "image_tooltip")
Figure 1: Memory layout of the GPU

In our scenario, we would assume 8k as the context length to serve the model. Whenever a request arrives, the model computes the attention scores for all the prompt tokens and then generates one token at a time using autoregressive decoding. While decoding, it requires some VRAM on the GPU to store the token. A single token would take 0.125MB of VRAM to be stored in the KV cache.

```text
Token size calculation

For every token, we need to store its corresponding tokens for K and V matrices. We also need to store it for all the layers and all the attention heads.

The general formula is: 2*2*n*h*d, where the first 2 is for FP 16 weights (2 bytes), the second 2 is for the K/V matrix, n is for the number of layers, h is for the number of heads, d is the embedding dimension
For Mistral 7B, 2*2* 32*8*128 = 0.125 MB

The KV cache for a single request on the complete context length of the model would be 1 GB (8k * 0.125 MB).
```

**A case for a single GPU serving a single request**

If we decide to serve only a single request at a time with this GPU, we would be wasting a lot of resources. Given that 7 GB of VRAM is available for KV cache, the model can store cache for 56k tokens (7 GB/ 0.125 MB). Considering all of the VRAM to be reserved for a single request, the space for 48k tokens (56k-8k) would be wasted since the model has a context length of only 8k tokens. The throughput of the model would be very low (only a single request is being processed at a time) and it is not using all of the VRAM of the GPU available to it. It would be wasting 6 GB of memory for every request.

This is termed as external fragmentation. This is clearly not the best way to utilize the GPU for serving LLMs. Figure 2 shows the extreme version of external fragmentation.

![alt_text](assets/images/post1/image2.png "image_tooltip")

Figure 2: Inside the KV cache: Single request

**A case for a single GPU serving multiple requests**

How can we improve upon this? Enter batching. In batching, we serve multiple requests at the same time taking advantage of the parallelism of GPUs. Let’s consider a scenario where we are serving multiple requests at the same time of 8k context length each. GPU would need to pre-allocate the space for 8k tokens for every request. For every request, the GPU would need 1 GB of VRAM to store the KV cache. Hence, it would be able to serve 7 requests concurrently (7 GB/ 1 GB). This would avoid external fragmentation in our scenario, but it could lead to another problem.

One thing to note here is that every request might not generate 8k tokens. Request 1 may end up generating 4k tokens, Request 2 may end up just generating 2k tokens, and so on. But since we had already reserved space for all the 8k tokens, we are wasting the memory and not utilizing the complete memory. This is called internal fragmentation.

There can be another scenario where after allocating the memory for all the requests, the available VRAM of the GPU is less than the memory required for a single request. In this scenario, the memory for the request will not be allocated and the remaining memory will be wasted. This is again a case of external fragmentation.

![alt_text](assets/images/post1/image3.png "image_tooltip")

Figure 3: Inside the KV cache: Multiple requests

**A case for a single GPU serving multiple requests efficiently**

So, is there any improvement possible over the naive batching method we discussed earlier? Yes, indeed there is a way. Enter vLLM.

Let’s assume that the complete memory of the GPU is broken down into small chunks of memory called blocks. Each block is equivalent to the memory required for 16 tokens (i.e. in our example, 0.125 MB * 16 = 2 MB). Once we allocate memory for a block, even partially, it won't be available for any other allocation.

Since every request might not need 8k tokens, let’s assume that on average every request would require 5000 tokens. GPU will allocate 313 blocks (5000/16) of memory for the request. These blocks are not stored in a contiguous layout in the memory. Hence, we would need to maintain an address book that maps every request to its corresponding blocks. There’s another optimization in here. Since this memory is not stored in a contiguous memory, we don’t need to allocate all of the memory at once. We can allocate memory as and when required once the previous blocks are filled to the capacity. This is the core of how vLLM allocates memory.

![alt_text](assets/images/post1/image4.png "image_tooltip")

Figure 4: vLLM token to block mapping.
Source <sup>[7]</sup>

The above solves 2 problems:

1. The request only allocates memory required for its generation instead of pre-allocating for the complete context length of the model. The memory allocation happens at the block level, so technically memory is allocated for 16 tokens at a time. This reduces internal fragmentation significantly
    1. If the request uses 1.5k tokens, we need to allocate memory only for 94 blocks i.e. 94 * 2 MB = 184 MB, instead of 1 GB for the complete 8k context length of the model
    2. A single request’s tokens can be stored in multiple blocks
2. The complete memory is broken down into equally sized blocks, so even external fragmentation is minimized. The block size is chosen such that it fills the available GPU memory evenly.

The approaches defined above help in utilizing the GPU VRAM efficiently. Given the block size of 2 MB, vLLM can store a total of ~3500 blocks in the available memory of 7 GB. If each request needs 313 blocks (5k tokens on average) during its lifetime, the GPU would have memory to serve 11 requests in parallel. By using the KV cache more effectively and allocating memory in blocks instead of complete context length, vLLM has increased the throughput from 7 to 11 in our example.

This is how vLLM helps in increasing the batch size and throughput of any model. For computing attention over tokens distributed in non-contagious blocks, vLLM has introduced Paged Attention. Paged Attention are optimized CUDA kernels to access tokens from different blocks and compute attention scores over them.

## Inside the simulation

To understand the behavior of vLLM in production, let us simulate a real scenario of a chat application. This chat application uses an LLM and is being served by vLLM. For chat applications, we have another dimension where a single chat can have multiple turns of conversation alternating between user and assistant messages.

![alt_text](assets/images/post1/image5.png "image_tooltip")

Figure 5: A multi-turn conversation. From the perspective of an LLM, all of these messages are a part of a single request. As the conversation progresses, every new message from the user gets appended to the same request and is sent to the LLM again

Our objective is to predict the behavior of vLLMs and try to replicate them in the experiments. To start with, let's consider some simulation parameters (similar to our example in the previous section):

1. Block size (number of tokens stored together in one block): 16
2. The average number of turns in each chat: 10
3. Average input token length at each turn in the chat: 150
4. Average output token length at each turn in the chat: 350
5. Average latency for each turn in the chat: 10s <sup>[4]</sup>
6. Average number of tokens required for a single chat session: (150 + 350) * 10 = 5000
7. The average number of blocks required for a single chat session: is 313 (5000/16)

For serving an LLM, let’s take any flavor of the Mistral 7B model deployed at half precision. Taking the model parameters,

1. Model dimension: 128
2. Number of layers: 32
3. Number of KV heads: 32
4. Input sequence length: 8192

According to these parameters, we would require:

1. 0.125 MB of memory per token in KV cache
2. 2 MB of memory per block (assuming block size to be 16 tokens, 0.125 * 16 = 2 MB)

Assuming 7 GB of KV cache available for our use

1. We can store ~3500 blocks in GPU VRAM (7GB/2MB)
2. As calculated above, given an average of 313 blocks per chat session and 3500 blocks available, we can hold 11 (floor(3500/313)) conversations in a single GPU and serve them in parallel

Based on our simulation, we calculated that an LLM served by vLLM can serve 11 requests in parallel for our setup. If we were implementing a naive batching, it would have not been able to serve more than 7 requests parallelly (which we discussed in the previous sections). Let’s experiment with this simulation to test the calculation. I send _N_ number of requests at once to a model hosted using the vLLM backend. Note that these requests are long-running (each request has multiple turns).

Below you can find the results from the experiments, where you can see two things:

1. Scheduler State: Number of requests being served concurrently by vLLM
2. Cache Utilization: % of GPU memory being used. Note that this percentage is based on the KV cache space we calculated earlier (i.e. 7 GB is the total GPU memory for the KV cache in our setup. If the utilization is 50%, that would translate to 3.5 GB of KV cache being used)

N = 10, we can see that the GPU utilization never reached 100%.

![alt_text](assets/images/post1/image6.png "image_tooltip")

N = 12, we can see that the GPU utilization reached 100% utilization, and 1 of the requests is moved to a waiting queue for some time (where it is not processed). This indicates that the results we got are similar to what we got from the experiments.

![alt_text](assets/images/post1/image7.png "image_tooltip")

N = 14, we can see that the GPU utilization hits 100% and then approximately 2 requests are moved to the waiting queue

![alt_text](assets/images/post1/image8.png "image_tooltip")

We can notice two things here:

1. It takes some time for the GPU to reach 100% utilization. This is because currently we have deployed a chat application where each turn takes 10 seconds and we have a total of 10 turns. So, the KV cache keeps on getting larger and larger as time goes by. But once the chat conversation ends after 10 turns, we will notice a drop in the GPU utilization.
2. If we go above the calculated parallel limit of our chats, we will eventually see some requests being transferred to a waiting queue. That implies the GPU is completely utilized and it can not process all the requests in a single batch.

The complete experiment can be rerun and you can find the code used to run the experiments [here](https://github.com/cmeraki/vllm-simulation).

An overview of all the parameters we discussed is mentioned below for reference. You can make a copy of the following [sheet](https://docs.google.com/spreadsheets/d/1BsLg2zcqSgiEyssqH9Wt0qG-mKGJ9S7rHjuSt8iH3hA/edit?usp=sharing) and play with simulation parameters to understand the requirements. Yellow blocks can be updated, and green blocks are calculated ones.

<table>
  <tr>
   <td><strong>Model Parameters</strong>
   </td>
   <td><strong>Value</strong>
   </td>
   <td><strong>Units</strong>
   </td>
  </tr>
  <tr>
   <td>Model size
   </td>
   <td><p style="text-align: right">
7.00</p>

   </td>
   <td>B
   </td>
  </tr>
  <tr>
   <td>Model dim
   </td>
   <td><p style="text-align: right">
128</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Model layers
   </td>
   <td><p style="text-align: right">
32</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Model KV heads
   </td>
   <td><p style="text-align: right">
8</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Bytes per parameter
   </td>
   <td><p style="text-align: right">
2</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Input sequence length
   </td>
   <td><p style="text-align: right">
8192</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>vLLM Parameters</strong>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Block size
   </td>
   <td><p style="text-align: right">
16</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>GPU Parameters</strong>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Memory
   </td>
   <td><p style="text-align: right">
24</p>

   </td>
   <td>GB
   </td>
  </tr>
  <tr>
   <td>Utilization
   </td>
   <td><p style="text-align: right">
100%</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Buffer
   </td>
   <td><p style="text-align: right">
3</p>

   </td>
   <td>GB
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Simulation params</strong>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Total turns in a chat
   </td>
   <td><p style="text-align: right">
10</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Input tokens in a turn
   </td>
   <td><p style="text-align: right">
150</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Output tokens in a turn
   </td>
   <td><p style="text-align: right">
350</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Experimental results</strong>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Average latency per turn
   </td>
   <td><p style="text-align: right">
10</p>

   </td>
   <td>s
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Calculations</strong>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Memory per token
   </td>
   <td><p style="text-align: right">
0.125</p>

   </td>
   <td>MB
   </td>
  </tr>
  <tr>
   <td>Memory per block
   </td>
   <td><p style="text-align: right">
2</p>

   </td>
   <td>MB
   </td>
  </tr>
  <tr>
   <td>Memory remaining for KV cache
   </td>
   <td><p style="text-align: right">
7</p>

   </td>
   <td>GB
   </td>
  </tr>
  <tr>
   <td>Total token length of a chat
   </td>
   <td><p style="text-align: right">
5000</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Total blocks required for a chat
   </td>
   <td><p style="text-align: right">
313</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Blocks that can be stored in KV cache
   </td>
   <td><p style="text-align: right">
3584</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Total chats that can be served concurrently at full context length
   </td>
   <td><p style="text-align: right">
11</p>

   </td>
   <td>
   </td>
  </tr>
</table>

## Notes

vLLM does a few more things:

1. KV cache reuse: By reusing the KV cache for different requests, a new request can skip computing the attention scores for the common tokens. This translates to lower latency. However, this is not the contribution of this paper. KV caching is a common technique used during LLM serving
    1. Single prompt, multiple generations: vLLM can cache a common prompt or prefix and use that for multiple generations. This is similar to the above and helps in reducing latency
    2. Parallel sampling and beam search: Following on from the above, vLLM also implements KV cache reuse for parallel sampling and beam search.
2. Pause the world: Whenever a new request comes in between the decoding stage of ongoing requests in the batch, vLLM pauses the generation of requests in the batch and computes the KV cache for the new request. Once the KV cache is computed, it adds it to the batch and continues decoding the new batch
    3. This results in higher latency if too many requests are coming back to back
    4. vLLM is working to update this behavior
3. Queue: vLLM also provides a FastAPI server on top of its backend. It implements queues that store the request that vLLM can not serve if the GPU memory is full

## References

These are some of the references that I have linked throughout the blog and some general recommended reading for getting a better understanding of the concepts we discussed in the blog.

1. [Making Deep Learning go Brrrr From First Principles](https://horace.io/brrr_intro.html)
2. [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference#the-basics-of-llm-inference)
3. For a single token generation, the latency is usually bound by the memory bandwidth of the GPU. Considering Nvidia 4090 which has a memory bandwidth of 1008 GB/s and Mistral 7B which has 14 GB parameters, the ideal estimate of latency would be 72 tok/s (1008/14). In the real world, you can expect to get around 60 tok/s
    1. For 600 tokens, the total time comes around to be 10s (600/60)
    2. Refer to this blog for more explanation: [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/#latency-calculations)
4. [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
5. [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
6. [vLLM](https://blog.vllm.ai/2023/06/20/vllm.html)
7. [Throughput is Not All You Need: Maximizing Goodput in LLM Serving using Prefill-Decode Disaggregation](https://hao-ai-lab.github.io/blogs/distserve/)

