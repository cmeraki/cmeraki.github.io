---
title: Journey of Indri
layout: post
---
<!-- markdownlint-disable MD033 MD036 MD053-->

Written by [Romit Jain](https://www.linkedin.com/in/r0m1t/)

## Journey of Indri

To build a conversational voice AI model, the easiest way to get started is, to take the audio from the user, transcribe it using ASR models like OpenAI’s Whisper, prompt an LLM with the text to get what it should speak next, and then pass the generated text to a text-to-speech model (like Suno-AI’s Bark).

This approach has a few problems, which are better described here. The crux is that creating a seamless conversation experience with this system is tough. For one, if audio is simply converted to text using ASR, the resulting text loses all the nuances like emotion, and pauses from the audio. Additional models are needed to embed all those nuances of speech. The other part is that ASR and TTS models work for speech but not for other sounds (construction noise, car engine noise, rain, etc.)

A better approach has a simpler system, get 1 model which can take in audio and generate audio while having the capability to think for the response. Building a model like this has its own set of challenges and is a field of its own called "textless NLP".

This article will cover the basics of this field and how to go about building such a conversational AI model.

### Tokenization

The first part of building an Audio model is tokenization. Text tokenization is relatively straightforward since text is a modality invented by humans so it is already tokenized (discrete) in some form. There are tokenizers like TikToken, and Sentence Piece that can tokenize the text by learning the patterns of subword occurrence in the data.

But tokenizing audio is slightly more challenging due to these reasons:

1. Audio is a continuous signal. So there are no obvious ways to cut it (discretize.
2. There are infinite possibilities for an audio signal
3. Tokens need to be such that they can represent all of the audio data that can be possible

The way to tokenize audio is to let a model learn the token embeddings. Two popular models to achieve this are:

1. [SoundStream](https://research.google/blog/soundstream-an-end-to-end-neural-audio-codec/) by Google
2. [Encodec](https://audiocraft.metademolab.com/encodec.html) by Meta, implementation on [GitHub](https://github.com/facebookresearch/encodec)

Both of them are based on quantizing the middle layer of a variational auto-encoder. These models are trained for audio compression and hence are really good for preserving traits of the audio. Tokens learned by such methods are called **Acoustic tokens**. The tokens extracted from these models preserve speaker information but do not preserve any coherent meaning of what has been said. Why? Because they are trained for audio reconstruction and every feature layer has a narrow view of the nearby audio. Hence, we can't use these tokens alone to model audio.

So, how to extract tokens from the audio that can preserve the semantic meaning of what is being said in the audio? Model the embeddings created from the audio using a transformer encoder model like BERT and force the model to learn a discretized representation of those embeddings (tokens). There are a couple of works in this direction, some of the notable ones being:

1. [Hubert](https://arxiv.org/abs/2106.07447)
2. [Wav2Vec Bert](https://arxiv.org/abs/2108.06209)

In both of these approaches, the embeddings are extracted from one of the middle layers and quantized either using K-means or residual vector quantization (RVQ). These tokens preserve the semantics of what is being said in the audio and are usable for long-term language modeling. But, they don't preserve characteristics of the speaker. Tokens learned by combining audio embeddings with BERT-style modeling are called **Semantic tokens**.

For building conversational voice AI models, both types of tokens are used.

### Bringing text to audio space

This step is optional, but we want a model to do either TTS or STT, one approach is to model text tokens to semantic tokens extracted from audio. These semantic tokens can be used to generate voice. For these models, we would require parallel data of audio and its corresponding transcripts.

A high-level design of how text to speech model would be trained:

1. (Text-to-semantic) A model trained to take text tokens and output semantic tokens (extracted from the corresponding audio)
2. (Semantic-to-acoustic) A model trained to take semantic tokens and generate their corresponding acoustic tokens
3. Detokenization of these audio tokens using the above models like Encodec or Soundstream to generate audio

This approach is taken by [Bark from Suno-AI](https://github.com/suno-ai/bark) and [SpearTTS](https://google-research.github.io/seanet/speartts/examples/).

For STT, the text-to-semantic model is replaced by the semantic-to-text model and applied towards the end of the pipeline. Note that semantic tokens are extracted from the audio while training.

### Building an Audio language model

Another challenge in building an intelligent conversation AI model is information density. Audio is very sparse in carrying the information. So we require a huge amount of audio data to pre-train a model that has built a world model. However, we can certainly train only on audio. The resulting model will be able to speak legible words clearly in a naturally sounding voice, but unless it is trained on a huge amount of clean data, the sentences won't make a lot of sense. This is the work of the paper [AudioLM](https://google-research.github.io/seanet/audiolm/examples/) by Google. It combines both acoustic and semantic tokens to generate speech. It's a pure audio model that takes in audio and generates the continuation of that audio.

Another approach is instead of pretraining a model on audio data, we can use a pre-trained text model and rely on its knowledge that it has learned from trillions of text tokens. We can finetune using a mix of audio and text data to build a truly audio-language model. For this, a high-level design of training would look like

1. A model finetuned on audio semantic tokens by expanding the vocabulary of an existing pre-trained text language model
   1. This model is trained in such a way that it can take either text or audio and generate either text tokens or audio semantic tokens
2. (Semantic-to-acoustic) A model trained to take semantic tokens and generate their corresponding acoustic tokens
3. Acoustic tokens are finally detokenized using models like Encodec or Soundstream to generate audio

This is the approach taken by Google in the paper [AudioPALM](https://google-research.github.io/seanet/audiopalm/examples/).

If we have a good dataset for finetuning the text language model and a good instruction dataset, we will have a model that can take in audio/text and generate audio/text. This model will have speech-to-speech translation, speech-to-text, and text-to-speech capabilities. This model will also be able to have conversations in audio or text or a mix of both.
