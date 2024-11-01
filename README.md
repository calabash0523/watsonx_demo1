# WatsonX.AI RAG 애플리케이션 가이드

## 소개

이 애플리케이션은 IBM의 WatsonX.AI와 벡터 데이터베이스(FAISS 또는 Milvus)를 사용하여 RAG(Retrieval-Augmented Generation) 시스템을 쉽게 데모할 수 있는 Streamlit 기반 웹 애플리케이션입니다. 이 도구를 사용하면 문서를 업로드하고, 질문을 하면 관련 정보를 검색하여 AI가 답변을 생성합니다.

### 주요 특징:
- WatsonX.AI를 이용한 텍스트 생성
- FAISS 또는 Milvus를 이용한 벡터 검색 (FAISS는 milvus옵션을 선택하지 않은 경우 Default로 동작하며, 인메모리 형태의 라이브러리 벡터검색으로 이 앱과 함께 설치되어 별도의 서버로 구성이 필요하지 않습니다.)
- 다양한 문서 형식 지원 (PDF, Word, Excel, PowerPoint)
- 사용자 정의 가능한 프롬프트 템플릿
- 다양한 LLM 모델 선택 가능

## 설치 가이드

### 1. 필요 조건
- Python 3.8 이상
- pip (Python 패키지 관리자)
- Git (선택사항: 소스 코드를 다운로드하는 데 사용)

### 2. 프로젝트 다운로드
GitHub에서 프로젝트를 다운로드하거나, 제공받은 소스 코드 파일을 로컬 컴퓨터의 원하는 위치에 저장합니다.

### 3. 가상 환경 설정 (권장)
가상 환경을 사용하면 프로젝트별로 독립적인 Python 환경을 만들 수 있습니다.

Windows:
```
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

### 4. 필요한 라이브러리 설치
1. requirements.txt 파일 사용
* requirements.txt 파일 작성

프로젝트 루트 디렉토리에 `requirements.txt` 파일을 생성하고 다음과 같이 작성합니다: (테스트 환경의 `requirements.txt`를 다운로드 해서 사용하셔도 됩니다.)

```
streamlit==1.2.0
PyPDF2==1.26.0
python-docx==0.8.11
openpyxl==3.0.9
python-pptx==0.6.21
sentence-transformers==2.2.0
faiss-cpu==1.7.2
pymilvus==2.1.1
```

버전을 지정하지 않으려면:

```
streamlit
PyPDF2
python-docx
openpyxl
python-pptx
sentence-transformers
faiss-cpu
pymilvus
```
### 설치 명령어
```bash
pip install -r requirements.txt
```
2. pip install로 개별 설치
각 라이브러리를 개별적으로 설치하려면:
```bash
pip install streamlit
pip install PyPDF2
pip install python-docx
pip install openpyxl
pip install python-pptx
pip install sentence-transformers
pip install faiss-cpu
pip install pymilvus
```

특정 버전 설치:

```bash
pip install streamlit==1.2.0
```

### 5. IBM Cloud 계정 및 WatsonX.AI 설정
- IBM Cloud 계정을 생성합니다.
- WatsonX.AI 서비스를 활성화합니다.
- API 키와 프로젝트 ID를 얻습니다.

### 6. 애플리케이션 실행
프로젝트 디렉토리에서 다음 명령을 실행하여 애플리케이션을 시작합니다:

```
streamlit run app.py
```

웹 브라우저가 자동으로 열리고 애플리케이션이 표시됩니다.

## 사용 방법

1. **로그인**: IBM Cloud API 키와 프로젝트 ID를 입력합니다.

2. **문서 업로드**: 사이드바에서 PDF, Word, Excel, PowerPoint 파일을 업로드합니다.

3. **설정 구성**: 
   - 사용자 프롬프트 템플릿 설정
   - LLM 모델 선택 및 파라미터 조정
   - 벡터 검색 설정 (임베딩 모델, 청크 크기 등)

4. **대화 시작**: 채팅 인터페이스에서 질문을 입력합니다.

5. **결과 확인**: AI의 응답을 확인하고, 필요시 사용된 파일 및 프롬프트 정보를 확인합니다.

## 주의사항

- Milvus를 사용하려면 별도의 Milvus 서버 설정이 필요합니다.
- 대용량 파일 처리 시 시간이 걸릴 수 있습니다.
- API 사용량에 따라 비용이 발생할 수 있으므로 IBM Cloud 요금 정책을 확인하세요.

## 문제 해결

- 연결 오류 발생 시 네트워크 연결과 API 키를 확인하세요.
- 파일 업로드 실패 시 파일 형식과 크기를 확인하세요.
- 로그를 확인하여 자세한 오류 정보를 얻을 수 있습니다.

## 설정변경
`config/app.toml` 파일을 통해서 언어선택(영문/한글)의 설정부터, LLM Model, Embedding 모델 등을 수정하실 수 있습니다.

```toml
[general]
language = "ko" # supported "ko"(Korean), "en"(English)
language_file = "config/messages.json"
method = "/ml/v1/text/generation_stream?version=2023-05-29"

[model]
supported_models = [
    "mistralai/mixtral-8x7b-instruct-v01",
    "mistralai/mistral-large",
    "meta-llama/llama-3-1-70b-instruct",
    "meta-llama/llama-3-1-8b-instruct",
    "meta-llama/llama-3-405b-instruct"
]

embedding_models = [
    "all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-large"
]

[url_options]
Dallas = "https://us-south.ml.cloud.ibm.com"
London = "https://eu-gb.ml.cloud.ibm.com"
Frankfurt = "https://eu-de.ml.cloud.ibm.com"
Tokyo = "https://jp-tok.ml.cloud.ibm.com"
```


---

# WatsonX.AI RAG Application Guide

## Introduction

This application is a Streamlit-based web application that easily demonstrates a Retrieval-Augmented Generation (RAG) system using IBM's WatsonX.AI and a vector database (FAISS or Milvus). With this tool, you can upload documents, ask questions, and the AI will retrieve relevant information to generate responses.

### Key Features:
- Text generation using WatsonX.AI
- Vector search using FAISS or Milvus (FAISS operates as the default option if Milvus is not selected, functioning as an in-memory vector search library, and does not require a separate server configuration with this app.)
- Support for various document formats (PDF, Word, Excel, PowerPoint)
- Customizable prompt templates
- Multiple LLM model options available

## Installation Guide

### 1. Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (Optional: for downloading the source code)

### 2. Download the Project
Download the project from GitHub or save the provided source code files to a desired location on your local computer.

### 3. Setting Up a Virtual Environment (Recommended)
Using a virtual environment allows you to create an isolated Python environment for the project.

For Windows:
```
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

### 4. Installing Required Libraries
1. Using `requirements.txt` file
   - Create a `requirements.txt` file in the root directory of your project and add the following content (You can also use the `requirements.txt` from the test environment):

   ```
   streamlit==1.2.0
   PyPDF2==1.26.0
   python-docx==0.8.11
   openpyxl==3.0.9
   python-pptx==0.6.21
   sentence-transformers==2.2.0
   faiss-cpu==1.7.2
   pymilvus==2.1.1
   ```

   If you don't want to specify versions:

   ```
   streamlit
   PyPDF2
   python-docx
   openpyxl
   python-pptx
   sentence-transformers
   faiss-cpu
   pymilvus
   ```

   ### Installation Command:
   ```bash
   pip install -r requirements.txt
   ```

2. Installing individually with pip
   - To install each library individually:

   ```bash
   pip install streamlit
   pip install PyPDF2
   pip install python-docx
   pip install openpyxl
   pip install python-pptx
   pip install sentence-transformers
   pip install faiss-cpu
   pip install pymilvus
   ```

   To install a specific version:

   ```bash
   pip install streamlit==1.2.0
   ```

### 5. Setting Up IBM Cloud Account and WatsonX.AI
- Create an IBM Cloud account.
- Enable the WatsonX.AI service.
- Obtain your API key and project ID.

### 6. Running the Application
Start the application by running the following command in the project directory:

```
streamlit run app.py
```

A web browser will automatically open, displaying the application.

## Usage Instructions

1. **Login**: Enter your IBM Cloud API key and project ID.

2. **Upload Documents**: Use the sidebar to upload PDF, Word, Excel, or PowerPoint files.

3. **Configure Settings**: 
   - Set up custom prompt templates
   - Select the LLM model and adjust parameters
   - Configure vector search settings (embedding model, chunk size, etc.)

4. **Start Conversation**: Enter your question in the chat interface.

5. **Review Results**: View the AI's response, and if needed, check the files and prompt information used.

## Notes

- To use Milvus, a separate Milvus server setup is required.
- Processing large files may take time.
- API usage may incur costs, so review IBM Cloud's pricing policies.

## Troubleshooting

- If you encounter connection errors, check your network connection and API key.
- If file upload fails, verify the file format and size.
- Check logs for detailed error information.

## Configuration Changes

You can modify various settings through the `config/app.toml` file, including language selection (English/Korean), LLM Model, Embedding model, and more.

```toml
[general]
language = "ko" # supported "ko"(Korean), "en"(English)
language_file = "config/messages.json"
method = "/ml/v1/text/generation_stream?version=2023-05-29"

[model]
supported_models = [
    "mistralai/mixtral-8x7b-instruct-v01",
    "mistralai/mistral-large",
    "meta-llama/llama-3-1-70b-instruct",
    "meta-llama/llama-3-1-8b-instruct",
    "meta-llama/llama-3-405b-instruct"
]

embedding_models = [
    "all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-large"
]

[url_options]
Dallas = "https://us-south.ml.cloud.ibm.com"
London = "https://eu-gb.ml.cloud.ibm.com"
Frankfurt = "https://eu-de.ml.cloud.ibm.com"
Tokyo = "https://jp-tok.ml.cloud.ibm.com"
```
