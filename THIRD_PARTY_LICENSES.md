# Third-party software and licenses used in E.L.I.A.

This project uses the following third-party components:

- **Picovoice Porcupine**  
  - Function: Wake-word detection  
  - License: SDK under Apache 2.0; wake-word engine proprietary (free for non-commercial use)  
  - Link: https://picovoice.ai/

- **WebRTC VAD (py-webrtcvad)**  
  - Function: Voice Activity Detection (start/end of speech, silence removal)  
  - License: BSD 3-Clause  
  - Link: https://github.com/wiseman/py-webrtcvad

- **Faster-Whisper**  
  - Function: Automatic Speech Recognition (ASR)  
  - License: MIT License  
  - Link: https://github.com/guillaumekln/faster-whisper

- **Gemma 3 (27B) via OpenRouter**  
  - Function: Large Language Model (LLM) for clarification and answer generation  
  - License: Apache 2.0 (model released by Google DeepMind)  
  - Accessed via: OpenRouter API (https://openrouter.ai)  
  - Link: https://huggingface.co/google/gemma-3-27b-it

- **Flask**  
  - Function: Web framework (server, API endpoints, output delivery)  
  - License: BSD 3-Clause  
  - Link: https://github.com/pallets/flask

- **spaCy**  
  - Function: NLP (intent detection, entity extraction)  
  - License: MIT License  
  - Link: https://github.com/explosion/spaCy

- **Edge-TTS**  
  - Function: Text-to-Speech (TTS) in Italian  
  - License: MIT License  
  - Link: https://github.com/rany2/edge-tts

- **ChromaDB**  
  - Function: Vector database for semantic search and memory persistence  
  - License: Apache 2.0  
  - Link: https://github.com/chroma-core/chroma

- **Sentence Transformers (SBERT)**  
  - Function: Embedding generation for semantic similarity  
  - License: Apache 2.0  
  - Link: https://github.com/UKPLab/sentence-transformers

- **neuraly/bert-base-italian-cased-sentiment**  
  - Function: Sentiment analysis (positive, neutral, negative classification)  
  - License: Apache 2.0  
  - Link: https://huggingface.co/neuraly/bert-base-italian-cased-sentiment

---

## Notes
- This repository itself is released under the MIT License (see LICENSE file).  
- Third-party software components retain their own licenses, which must be respected in any use or distribution of this project.  
