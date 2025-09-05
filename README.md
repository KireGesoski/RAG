# RAG
RAG and CAG projects
INSTAL LLM
# 0) Prereqs
sudo apt update
sudo apt install -y git build-essential cmake

# (Optional but recommended for faster CPU math)
sudo apt install -y libopenblas-dev

# 1) Get the source (adjust the path to where you want it)
cd ~/Desktop/AI_projects/RAG
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 2) Configure CMake (enable OpenBLAS if installed)
cmake -S . -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS

# 3) Build
cmake --build build -j

# For Guard Rails
https://github.com/guardrails-ai/guardrails


PHOENIX
https://www.youtube.com/watch?v=AzkwfI4TyAY

1.pip install arize-phoenix - (pip install arize-phoenix[evals] openai langchain langchain-comunity langchainhub langchain-openai langchain-chroma bs4 gcsfs)
2.toouch serverr.py
# and inside server.py session = px.launch_app()
3.python -m phoenix.server.main serve

