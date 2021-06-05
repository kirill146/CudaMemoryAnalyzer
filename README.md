# CudaMemoryAnalyzer

Библиотека для поиска ошибок выхода за пределы массива и некорректных обращений к памяти через `__restrict__` указатели в CUDA ядрах.

## Build

Перед сборкой библиотеки необходимо собрать clang из [llvm-project](https://github.com/llvm/llvm-project).
```bat
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-12.0.0
```
Конфигурируем (для ускорения сборки и уменьшения размера билда можно оставить только необходимые таргеты при помощи дополнительного флага `-DLLVM_TARGETS_TO_BUILD="NVPTX;X86"`):
```bat
cmake -S llvm -B build -DLLVM_ENABLE_PROJECTS="clang"
````
Собираем (Можно оставить только необходимый `--config`):
```bat
cmake --build build --config Release
cmake --build build --config Debug
```
Устанавливаем переменную окружения `LLVM_PROJ_DIR` на текущую директорию (т. е. на корневую папку llvm-project):
```bat
setx LLVM_PROJ_DIR %cd%
```

Так же для работы анализатора необходимо установить [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

Теперь можно собирать проекты из этого репозитория обычным Ctrl+B из Visual Studio.

## Using
В solution-e 3 проекта: библиотека `CudaMemoryAnalyzer`, stand-alone утилита `cma` для запуска библиотеки с параметрами из конфигурационного json файла и тесты.

Публичные хедеры библиотеки располагаются в [CudaMemoryAnalyzer/inc](CudaMemoryAnalyzer/inc). Кроме самой библиотеки необходимо линковаться с [lib/libz3.lib](lib/libz3.lib).

Для запуска анализатора необходимо вызвать `checkBufferOverflows()` или `checkRestrictViolations()`.
