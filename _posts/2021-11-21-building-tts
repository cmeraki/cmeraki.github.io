## Indri TTS / ASR
######  Nov 21, 2024

Today, we are releasing Indri TTS model series, which are 124M/350M param, multilingual, fully autoregressive TTS models, that can produce hyper realistic human voices. You can try out the models here : https://indrivoice.ai . Or download and use it on your machine from github / hf. Currently the model supports English, Hindi and Kannada. New languages are easy to add using scripts provided in git repo.

Indri can generate hyper-realistic audio that is very hard to differentiate from real speech. It faithfully reproduces background noises, echoes, music and non-speech sounds alongwith speech. Here are a few examples of generations : 

### Data
We have used 20k hours of available English TTS data, alongwith 5k hours of per language data. 

We collected videos with clean audio from sharing websites and passed it through whisper-v3-turbo to generate transcriptions. These transcriptions are limited to 15s in length. We also post process the chunks and remove any silences longer than 250ms.

#### What to look for in data ?
1. Clearly spoken speech. You should be able to make out the words that are being spoken. E.g. podcasts, talks etc. make for great sources, whereas on-site news or action movie clips do not. 
2. No background music, sounds etc. Although we can remove the background using separation, it leaves artifacts which the model learns to replicate.


### Modelling
#### Audio Tokenizer
A lot about tokenizers has been covered in previous blog. If you haven't, go through the tokenizers blog to understand how to decide on an audio tokenizer. 

#### Impact of tokenizer
1. Small context length : Using a tokenizer which has low frequency, results in small sequences. This makes them easier to model. E.g. Hubert is 50Hz, and encoded 
2. Speed : 
3. Final model size : 

We use Mimi tokenizer (link), which produces 32 codebooks at 12.5Hz. We found 8 codebooks to be sufficient to faithfully reproduce audio under consideration. 

#### Handling audio tokens
Transformers are good at modelling 1-D sequences. Audio tokenizers convert audio into n-codebooks at kHz, giving a 2D sequence of tokens. 

We convert this to a 1D sequence by weaving codebooks together. Tokens of n-th codebook are offset by (n-1 x  n_tokens_per_codebook). Both semantic and audio tokens are weaved together in a single sequence. 

For n_codebooks = 2, tokens_per_codebook = 16 :

$$
\begin{bmatrix}
1 & 5 & 3 \\
12 & 8 & 9 \\
\end{bmatrix}
$$
converts to 
$$
\begin{bmatrix}
1 & 5 & 3 & 12 + 16 & 8 + 16 & 9 + 16 \\
\end{bmatrix}
$$

This results in an audio vocab of size n_codebooks x tokens_per_codebook.

We bring text and audio tokens into a common embedding space and train a small transformer (gpt2) over text+audio sequences. 

### Sequences
Indri is a multimodal decoder only transformer (gpt2 arch), that consumes and generates both audio and text tokens as part of same sequence. We convert different problems such as tts/asr/continuation into sequence to sequence problems, indicating tasks by special tokens. 

TTS systems such as spear-tts use a tiered approach where they train two models : 
1. text to semantic tokens : learns to read
2. semantic to acoustic tokens : learns to speak

This separates the speaker voice characteristics (e.g. pitch) from reading (e.g. speed, accent) etc. But first model has to complete its generation, for the next model to start producing output. Hence streaming output can only start when all semantic tokens are ready.

We use a single model to generate both semantic and acoustic tokens. Hence we can stream output from the moment first audio has been generated. 

#### Token Sequence

We use special tokens to indicate:
1. start of modality `<text>, <audio>`
2. a common stop token `<stop>`
3. speaker identifier `<speaker_idx>`
4. task `<tts>, <asr>`


### References
