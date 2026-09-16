# Tech Challenge 03 — Medical Triage

Projeto desenvolvido pro Tech Challenge da pós-graduação em Machine Learning Engineering da FIAP.

A solução implementa um pipeline de Machine Learning para classificação de textos médicos, contemplando treinamento, disponibilização do modelo por meio de API REST, conteinerização, automação do treinamento, monitoramento da aplicação e otimização da inferência.


## Visão Geral

O projeto utiliza técnicas de processamento de linguagem natural (NLP) para classificar textos médicos.

O fluxo desenvolvido contempla:

1. Preparação e treinamento do modelo de Machine Learning.
2. Vetorização dos textos utilizando TF-IDF.
3. Disponibilização do modelo através de uma API REST com FastAPI.
4. Conteinerização da aplicação utilizando Docker.
5. Automação do treinamento utilizando Apache Airflow.
6. Exposição de métricas utilizando Prometheus.
7. Visualização das métricas através do Grafana.
8. Otimização da inferência utilizando ONNX.
9. Testes automatizados e validações de qualidade.


## Arquitetura

O fluxo principal dessa solução pode ser representado da seguinte forma:

```text
Dataset
   |
   v
Treinamento do modelo
   |
   v
TF-IDF + Classificador
   |
   +--------------------+
   |                    |
   v                    v
Modelo baseline      Conversão ONNX
   |                    |
   v                    v
FastAPI              Benchmark
   |
   v
/ predict
   |
   +------> /metrics
              |
              v
          Prometheus
              |
              v
            Grafana
```

O Apache Airflow é utilizado pra orquestrar o processo de treinamento do modelo.


## Tecnologias Utilizadas

- Python 3.11
- Scikit-learn
- FastAPI
- Uvicorn
- Docker
- Apache Airflow
- PostgreSQL
- Prometheus
- Grafana
- ONNX
- ONNX Runtime
- Pytest
- Ruff
- Joblib


## Estrutura do Projeto

```text
tech-challenge-03-medical-triage/
|
|-- app/
|   `-- main.py
|
|-- dags/
|   `-- medical_training_dag.py
|
|-- data/
|
|-- models/
|
|-- monitoring/
|   |-- prometheus.yml
|   `-- grafana-dashboard.json
|
|-- src/
|   |-- benchmark.py
|   |-- optimize.py
|   `-- train.py
|
|-- tests/
|   |-- test_api.py
|   `-- test_train.py
|
|-- Dockerfile
|-- Dockerfile.airflow
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```
```markdown
> **Observação:** os diretórios `data/` e `models/` armazenam, os dados utilizados
> no treinamento e os artefatos gerados localmente pelo pipeline. Esses arquivos
> não são versionados no repositório devido ao tamanho e à natureza dos artefatos gerados.


## Treinamento do Modelo

O treinamento pode ser executado através do módulo:

```bash
python -m src.train
```

O pipeline realiza a transformação dos textos utilizando TF-IDF e treina o classificador responsável pelas previsões.
Os artefatos resultantes são armazenados no diretório `models/`.


## API REST

A aplicação disponibiliza o modelo através de uma API desenvolvida com FastAPI.
Com os serviços em execução, a documentação Swagger pode ser acessada em:

```text
http://localhost:8000/docs
```


### Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/` | Verificação básica da aplicação |
| GET | `/health` | Health check da API |
| POST | `/predict` | Realiza a classificação de um texto médico |
| GET | `/metrics` | Expõe métricas no formato Prometheus |


### Exemplo de requisição

```json
{
  "text": "Patient presents with severe chest pain and shortness of breath."
}
```

Exemplo de resposta obtida durante os testes:

```json
{
  "class_id": 5,
  "class_name": "General pathological conditions",
  "latency_ms": 14.265
}
```


## Docker

Os componentes da solução são executados utilizando Docker Compose.
Para iniciar os serviços:

```bash
docker compose up -d
```

Para verificar os containers:

```bash
docker compose ps
```

Entre os serviços utilizados pela solução estão:

- API de inferência;
- Prometheus;
- PostgreSQL;
- Airflow Webserver;
- Airflow Scheduler.


## Apache Airflow

O Apache Airflow é utilizado pra orquestrar o processo de treinamento.
A DAG está definida em:

```text
dags/medical_training_dag.py
```

A interface do Airflow pode ser acessada em:

```text
http://localhost:8080
```

A utilização do Airflow permite automatizar e organizar o pipeline de treinamento, aproximando o projeto de um fluxo de MLOps.


## Monitoramento

A API expõe métricas no endpoint:

```text
http://localhost:8000/metrics
```

O Prometheus realiza a coleta dessas métricas periodicamente.
A interface do Prometheus pode ser acessada em:

```text
http://localhost:9090
```

O target configurado para a API pode ser validado através da página de targets do Prometheus.


### Métricas da aplicação

Entre as métricas implementadas, temos:

```text
medical_api_requests_total
medical_api_errors_total
medical_api_request_latency_seconds
```

Essas métricas permitem acompanhar volume de requisições, erros e latência das inferências.


## Grafana

O Grafana é utilizado para visualização das métricas coletadas pelo Prometheus.
A interface pode ser acessada em:

```text
http://localhost:3000
```

Foi criado o dashboard:

```text
Medical Triage - Monitoring
```

O dashboard possui os seguintes painéis:

- Total de Requisições
- Total de Erros
- Latência Média
- Evolução da Latência

A configuração exportada do dashboard está versionada em:

```text
monitoring/grafana-dashboard.json
```


## Otimização com ONNX

Além do modelo baseline, foi implementada uma versão otimizada utilizando ONNX.
A otimização pode ser executada através de:

```bash
python -m src.optimize
```

O processo gera:

```text
models/tfidf_vectorizer.joblib
models/medical_classifier.onnx
```

O modelo convertido mantém a mesma predição do modelo baseline nos testes de validação realizados.


## Benchmark

O benchmark pode ser executado utilizando:

```bash
python -m src.benchmark
```

Foram realizadas 500 execuções por abordagem.

| Métrica    | Baseline      | ONNX          |
|------------|--------------:|--------------:|
| Média      | 0.4300 ms     | 0.3277 ms     |
| P50        | 0.4027 ms     | 0.3120 ms     |
| P95        | 0.5578 ms     | 0.4286 ms     |
| Throughput | 2325.52 req/s | 3051.80 req/s |

Resultados observados:

- Speedup: **1.31x**
- Redução média de latência: **23.80%**
- Ganho de throughput: **31.23%**

Os resultados demonstram que a utilização do ONNX reduziu a latência da inferência e aumentou a capacidade de processamento.


## Testes Automatizados

Os testes podem ser executados através de:

```bash
pytest -v
```

Resultado obtido:

```text
8 passed
```

Os testes validam funcionalidades da API e componentes do pipeline de treinamento.


## Qualidade de Código

O projeto utiliza Ruff para análise estática e padronização do código.

```bash
ruff check src tests app
```


## Execução Rápida

### 1. Criar e ativar o ambiente virtual

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Treinar o modelo

```bash
python -m src.train
```

### 4. Executar os serviços

```bash
docker compose up -d
```

### 5. Validar a API

Acesse:

```text
http://localhost:8000/docs
```

### 6. Validar o Prometheus

Acesse:

```text
http://localhost:9090
```

### 7. Validar o Grafana

Acesse:

```text
http://localhost:3000
```


## Validações Realizadas

Durante o desenvolvimento foram validados:

- treinamento e inferência do modelo;
- API REST e endpoint `/predict`;
- health check;
- exposição das métricas;
- coleta das métricas pelo Prometheus;
- integração Prometheus + Grafana;
- dashboard de monitoramento;
- orquestração utilizando Airflow;
- conversão do modelo para ONNX;
- equivalência de predição entre baseline e ONNX;
- benchmark de desempenho;
- testes automatizados.


## Conclusão

O projeto implementa um fluxo de Machine Learning que vai além do treinamento do modelo, incorporamos práticas importantes de Machine Learning Engineering e MLOps.
A solução contempla disponibilização por API, conteinerização, orquestração, observabilidade, testes e otimização de inferência, permitindo acompanhar tanto o funcionamento da aplicação quanto o desempenho do modelo em execução.