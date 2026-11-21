@echo off
rem T4 (PCHONK-FTI-001): builds ref_logits.exe against the reference llama.cpp c7518dbe build (shared libraries),
rem with the same compiler and flags as build_capture_l0.bat.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined REF (echo Set REF to a llama.cpp c7518dbe checkout built for CUDA 13.0 with shared libraries. & exit /b 1)
cd /d "%~dp0" || exit /b 1
cl /nologo /O2 /EHsc /std:c++17 /wd4996 /DLLAMA_SHARED /DGGML_SHARED /I"%REF%\include" /I"%REF%\ggml\include" ref_logits.cpp /Fe:ref_logits.exe /link "%REF%\build\src\llama.lib" "%REF%\build\ggml\src\ggml.lib" "%REF%\build\ggml\src\ggml-base.lib" || exit /b 1
echo built ref_logits.exe
