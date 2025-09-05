# E.L.I.A. – Educational Learning Intelligent Assistant

Un assistente vocale educativo in italiano, con riconoscimento del *wake word*, trascrizione (ASR), generazione di risposte via LLM e sintesi vocale (TTS) usando **Edge TTS**.  
Questo progetto nasce come parte di un **tirocinio e tesi universitaria** presso l'**Università degli Studi di Salerno**.

---

## 🚀 Caratteristiche principali

- Attivazione vocale tramite wake word (**Picovoice**)  
- Riconoscimento vocale con **Faster-Whisper**  
- Integrazione con LLM tramite API (es. **GEMMA**)  
- Sintesi vocale con **Edge TTS** (servizio Microsoft Edge)  
- Configurazione flessibile tramite **variabili d’ambiente**  

---

## 📦 Prerequisiti

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Anaconda o Miniconda)  
- Python 3.8+ (gestito da `environment.yml`)  
- API key per:
  - **Picovoice**
  - **Servizio LLM** (es. GEMMA)

---

## 🔧 Installazione

Clona la repo e crea l’ambiente Conda:

```bash
git clone https://github.com/GiuseppeGambardella/E.L.I.A.-Educational-Learning-Intelligent-Assistant-.git
cd E.L.I.A.-Educational-Learning-Intelligent-Assistant-

# Crea l’ambiente con tutte le dipendenze
conda env create -f environment.yml

# Attiva l’ambiente
conda activate elia-env

# Configura PYTHONPATH per permettere gli import del progetto
conda env config vars set PYTHONPATH=src
conda deactivate
conda activate elia-env

# Spostarsi in src/
# Copia e personalizza il file .env
cp .env.example .env
# Modifica .env con le tue credenziali
```

## ▶️ Utilizzo

Dopo aver installato e configurato l’ambiente, puoi avviare il progetto in due modi:

### 1. Avviare il server Flask
Spostarsi in src/elia e avviare app.py

```bash
python src/elia/app.py
```
Questo comando avvia il backend con le API disponibili sugli endpoint locali (`/ask`, `/attention`):

### 3. Avviare il client con wake word

Per usare l’attivazione vocale (wake word **“Ehi Elia”**):

```bash
python src/elia/client/wake.py
```
Resta in ascolto finché non pronunci il wake word.

Quando viene riconosciuto, viene attivata la registrazione audio.

L’audio viene trascritto con Faster-Whisper.

La risposta è generata dal LLM configurato nel .env.

Infine, la risposta viene letta tramite Edge TTS.

### 4. Controllo attenzione degli studenti

Per testare il modulo che richiama l’attenzione degli studenti esegui:

```bash
python src/elia/client/check_attention.py
```

Lo script chiama l’endpoint /attention.

Genera un messaggio breve e conciso (max 15 parole).

Serve a simulare il richiamo all’attenzione di uno studente distratto.

### 5. Test Report Emotivi

Per testare il sistema di analisi emotiva dello studente:

```bash
python src/elia/client/report.py
```

