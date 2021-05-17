# CudaMemoryAnalyzer

Библиотека для поиска ошибок выхода за пределы массива и некорректных обращений к памяти через `__restrict__` указатели в CUDA ядрах.

## Build

Перед сборкой библиотеки необходимо собрать clang из [llvm-project](https://github.com/llvm/llvm-project):
```sh
# cd ...
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-12.0.0
# Конфигурируем:
# Для ускорения сборки и уменьшения размера билда
# можно оставить только необходимые таргеты при помощи
# дополнительного флага -DLLVM_TARGETS_TO_BUILD="NVPTX;X86"
cmake -S llvm -B build -DLLVM_ENABLE_PROJECTS="clang"
# Собираем (Можно оставить только необходимый config)
cmake --build build --config Release
cmake --build build --config Debug
```

Затем необходимо добавить переменную окружения `LLVM_PROJ_DIR` ссылающуюся на корневую папку llvm-project.

Так же для работы анализатора необходимо установить [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

Теперь можно собирать этот проект обычным Ctrl+B из Visual Studio.

## Using
В solution-e 2 проекта: библиотека CudaMemoryAnalyzer и тесты.

Публичные хедеры библиотеки располагаются в [CudaMemoryAnalyzer/inc](CudaMemoryAnalyzer/inc). Кроме самой библиотеки необходимо линковаться с [lib/libz3.lib](lib/libz3.lib).

Для запуска анализатора необходимо вызвать `checkBufferOverflows()` или `checkRestrictViolations()`.