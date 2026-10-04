# AMC — KNN, CNN e Linear SVM

Projeto experimental para Classificação Automática de Modulação (AMC), contendo experimentos com algoritmos de aprendizado de máquina e aprendizado profundo.

Este documento descreve a configuração do ambiente para execução local em **Windows utilizando WSL2 + Ubuntu**, incluindo suporte à GPU NVIDIA para TensorFlow.

---

## 1. Ambiente recomendado

A configuração utilizada neste projeto é:

```text
Windows
└── WSL2
    └── Ubuntu
        └── Python
            └── .venv
                ├── Dependências do requirements.txt
                ├── TensorFlow 2.21
                ├── CUDA
                ├── cuDNN
                └── Jupyter / ipykernel
```

Para execução dos experimentos TensorFlow utilizando GPU NVIDIA, o projeto deve ser executado dentro do WSL2.

---

# 2. Pré-requisitos no Windows

## 2.1. Verificar a GPU NVIDIA

Abra o PowerShell:

```powershell
nvidia-smi
```

A GPU deverá aparecer na listagem.

Exemplo:

```text
NVIDIA GeForce RTX 2060
```

---

## 2.2. Instalar/verificar o WSL2

No PowerShell:

```powershell
wsl --status
```

Verifique também as distribuições instaladas:

```powershell
wsl -l -v
```

A distribuição Ubuntu deverá aparecer utilizando a versão 2:

```text
NAME      STATE      VERSION
Ubuntu    Running    2
```

Caso o WSL ainda não esteja instalado:

```powershell
wsl --install
```

Após a instalação, reinicie o Windows caso seja solicitado.

---

# 3. Armazenamento do projeto

Recomenda-se manter o projeto dentro do sistema de arquivos Linux do WSL.

Exemplo:

```text
/home/<usuario>/Projetos/IFCE/AMC_KNN_CNN_LinearSVM
```

Exemplo utilizado:

```bash
~/Projetos/IFCE/AMC_KNN_CNN_LinearSVM
```

Evite executar o projeto diretamente em caminhos montados do Windows, como:

```text
/mnt/c/Users/...
```

principalmente para workloads que fazem muitas operações de leitura e escrita.

---

# 4. Abrir o projeto no WSL

Abra o Ubuntu e navegue até o projeto:

```bash
cd ~/Projetos/IFCE/AMC_KNN_CNN_LinearSVM
```

Execute:

```bash
code .
```

O VS Code deverá abrir conectado ao ambiente WSL.

No canto inferior esquerdo do VS Code deve aparecer algo semelhante a:

```text
WSL: Ubuntu
```

---

# 5. Configuração automática do ambiente

O projeto possui o script:

```text
setup_env.sh
```

Dê permissão de execução:

```bash
chmod +x setup_env.sh
```

Execute:

```bash
./setup_env.sh
```

O script realizará automaticamente:

```text
1. Verificação de execução no Linux/WSL
2. Verificação da GPU NVIDIA
3. Criação do ambiente virtual .venv
4. Atualização do pip
5. Instalação do requirements.txt
6. Instalação do ipykernel
7. Instalação do TensorFlow com suporte CUDA
8. Configuração das bibliotecas CUDA do TensorFlow
9. Correção do carregamento de libcusolver.so.11
10. Configuração do ptxas
11. Registro do kernel Jupyter
12. Teste de detecção da GPU
```

---

# 6. Ativar o ambiente manualmente

Após executar o setup, ative o ambiente com:

```bash
source .venv/bin/activate
```

O terminal deverá ficar semelhante a:

```text
(.venv) usuario@computador:~/Projetos/IFCE/AMC_KNN_CNN_LinearSVM$
```

Para sair do ambiente:

```bash
deactivate
```

---

# 7. Selecionar o ambiente no VS Code

Abra o notebook `.ipynb`.

No canto superior direito escolha:

```text
Select Kernel
```

Depois:

```text
Python Environments
```

Selecione:

```text
.venv
```

O interpretador deverá apontar para:

```text
<projeto>/.venv/bin/python
```

É possível confirmar executando uma célula:

```python
import sys

print(sys.executable)
```

O resultado deverá ser semelhante a:

```text
/home/<usuario>/Projetos/IFCE/AMC_KNN_CNN_LinearSVM/.venv/bin/python
```

---

# 8. Verificar TensorFlow e GPU

Execute no notebook:

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("CUDA build:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))
```

Resultado esperado:

```text
TensorFlow: 2.21.0
CUDA build: True
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

# 9. Configurar crescimento de memória da GPU

É recomendado permitir que o TensorFlow aumente gradualmente o uso de memória da GPU:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("GPUs:", gpus)
```

Isso evita que o TensorFlow tente reservar toda a VRAM disponível imediatamente.

---

# 10. Testar processamento na GPU

Execute:

```python
import tensorflow as tf

with tf.device("/GPU:0"):
    a = tf.random.normal([5000, 5000])
    b = tf.random.normal([5000, 5000])
    c = tf.matmul(a, b)

print(c.device)
```

O resultado deverá terminar com algo semelhante a:

```text
/device:GPU:0
```

---

# 11. Monitorar utilização da GPU

Em outro terminal WSL:

```bash
watch -n 1 nvidia-smi
```

Durante treinamento ou operações TensorFlow, o processo Python deverá aparecer na GPU.

Para sair:

```text
Ctrl + C
```

---

# 12. Dependências

As dependências gerais do projeto estão definidas em:

```text
requirements.txt
```

Para instalá-las manualmente:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

O ambiente GPU do TensorFlow pode ser instalado manualmente com:

```bash
python -m pip install "tensorflow[and-cuda]==2.21.0"
```

---

# 13. Problema conhecido — TensorFlow 2.21 e libcusolver

Neste ambiente, o TensorFlow 2.21 pode estar instalado corretamente e ainda apresentar:

```text
Cannot dlopen some GPU libraries.
Skipping registering GPU devices...
GPU: []
```

Mesmo quando:

```python
tf.test.is_built_with_cuda()
```

retorna:

```text
True
```

O projeto aplica durante o `setup_env.sh` um ajuste para disponibilizar:

```text
libcusolver.so.11
```

no caminho de bibliotecas utilizado pelo TensorFlow.

O equivalente manual é:

```bash
SP=$(python -c 'import site; print(site.getsitepackages()[0])')

ln -sf ../../cusolver/lib/libcusolver.so.11 \
"$SP/nvidia/cublas/lib/libcusolver.so.11"
```

---

# 14. Configuração do ptxas

O script também localiza o executável `ptxas` instalado pelas dependências NVIDIA e cria um link dentro do ambiente virtual.

Manualmente:

```bash
PTXAS=$(find "$VIRTUAL_ENV" -type f -name ptxas -print -quit)

ln -sf "$PTXAS" "$VIRTUAL_ENV/bin/ptxas"
```

Verifique:

```bash
which ptxas
```

e:

```bash
ptxas --version
```

---

# 15. Diagnóstico

## GPU disponível no WSL

```bash
nvidia-smi
```

---

## Python utilizado

```bash
which python
```

Deve retornar:

```text
<projeto>/.venv/bin/python
```

---

## TensorFlow

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## CUDA habilitado no TensorFlow

```bash
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
```

Esperado:

```text
True
```

---

## GPU detectada

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Esperado:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Pacotes NVIDIA instalados

```bash
python -m pip list | grep nvidia
```

---

## Localizar ptxas

```bash
find "$VIRTUAL_ENV" -type f -name ptxas
```

---

# 16. Recriar o ambiente

Caso seja necessário reconstruir completamente o ambiente:

```bash
deactivate 2>/dev/null || true

rm -rf .venv

./setup_env.sh
```

O ambiente será criado novamente a partir do:

```text
requirements.txt
```

---

# 17. Atualizar requirements.txt

Após adicionar uma nova dependência ao projeto, o arquivo pode ser atualizado com:

```bash
python -m pip freeze > requirements.txt
```

Antes de sobrescrever o arquivo, verifique se o projeto utiliza versões de dependências definidas manualmente.

---

# 18. Execução resumida

Em uma máquina já configurada com Windows + WSL2 + NVIDIA, normalmente basta:

```bash
cd ~/Projetos/IFCE/AMC_KNN_CNN_LinearSVM

chmod +x setup_env.sh

./setup_env.sh

source .venv/bin/activate

code .
```

Depois selecione o kernel `.venv` no notebook e execute os experimentos.