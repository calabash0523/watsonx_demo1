import json
import time
import requests
import urllib3
import shlex

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def delete_all_documents(base_url, index, api_key):
    url = f"{base_url}/{index}/_delete_by_query"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"ApiKey {api_key}"
    }
    data = {
        "query": {
            "match_all": {}
        }
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
    print(f"{index} 인덱스의 모든 문서를 삭제합니다...")
    if response.status_code == 200:
        print("모든 문서가 삭제되었습니다.")
    else:
        print(f"문서 삭제 중 오류 발생: {response.text}")

def check_document_structure(base_url, index, api_key, content, max_retries=12, delay=10):
    url = f"{base_url}/{index}/_search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"ApiKey {api_key}"
    }
    query = {
        "query": {
            "match_phrase": {"content": content}
        }
    }

    for i in range(max_retries):
        response = requests.get(url, headers=headers, json=query, verify=False)
        result = response.json()
        if result['hits']['total']['value'] > 0:
            doc = result['hits']['hits'][0]
            if 'vector_embedding' in doc['_source'] or 'ml' in doc['_source']:
                print(f"벡터화 필드를 찾았습니다. (시도 {i+1}/{max_retries})")
                print("문서 내용:")
                print(json.dumps(doc['_source'], indent=2))
                return doc
        print(f"벡터화 필드 대기 중... (시도 {i+1}/{max_retries})")
        time.sleep(delay)

    print("벡터화 필드를 찾을 수 없습니다.")
    return None

def print_curl_command(url, headers, data):
    headers_str = ' '.join([f"-H '{k}: {v}'" for k, v in headers.items()])
    data_str = shlex.quote(data)
    curl_command = f"curl -X POST '{url}' {headers_str} -d '{data_str}' --insecure"
    print("\n생성된 curl 명령어:")
    print(curl_command)
    print("\n")

def test_watsonx_discovery_connection(config):
    print("watsonx Discovery 연결 테스트를 시작합니다...")
    
    try:
        base_url = config['server_url']
        api_key = config['api_key']
        index = config['index_name']

        # 모든 문서 삭제
        delete_all_documents(base_url, index, api_key)

        # 테스트 문서 준비
        test_docs = [
            {
                "file_name": f"test_file_{i}.txt", 
                "content": f"This is test document {i} for ML model analysis.",
                "_extract_binary_content": True,
                "_reduce_whitespace": True,
                "_run_ml_inference": True
            } 
            for i in range(1, 11)  # 10개의 테스트 문서 생성
        ]

        # Bulk ingest
        bulk_data = "\n".join(f"{json.dumps(action)}\n{json.dumps(doc)}" for action, doc in zip(
            [{"index": {"_index": index}} for _ in test_docs],
            test_docs
        )) + "\n"

        url = f"{base_url}/_bulk?pipeline={index}&pretty"
        headers = {
            "Content-Type": "application/x-ndjson",
            "Authorization": f"ApiKey {api_key}",
            "Accept": "application/json"
        }
        
        # curl 명령어 출력
        print_curl_command(url, headers, bulk_data)
        
        print("테스트 문서 bulk ingest 중...")
        response = requests.post(url, headers=headers, data=bulk_data.encode('utf-8'), verify=False)
        result = response.json()
        if 'errors' in result and not result['errors']:
            print(f"Bulk ingest 성공: {len(test_docs)}개 문서")
        else:
            print(f"Bulk ingest 중 오류 발생: {response.text}")

        # 문서 처리 확인
        print("문서 처리 결과 확인 중...")
        for doc in test_docs:
            result = check_document_structure(base_url, index, api_key, doc['content'])
            if result:
                print(f"문서 '{doc['file_name']}'가 성공적으로 처리되었습니다.")
                if 'vector_embedding' in result['_source']:
                    print("벡터 임베딩 필드가 존재합니다.")
                elif 'ml' in result['_source']:
                    print("ML 필드 내용:")
                    print(json.dumps(result['_source']['ml'], indent=2))
                else:
                    print("문서에 벡터화 필드가 없습니다.")
            else:
                print(f"문서 '{doc['file_name']}'의 벡터화가 완료되지 않았습니다.")

        print("연결 테스트가 완료되었습니다.")
        return True
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return False

# 설정 예시
config = {
    'server_url': 'https://1e9ad3b1-0489-40e6-a14c-d72b120ac9c7.blijtlfd05jdimoomdig.databases.appdomain.cloud:30632',
    'api_key': 'UWJlWkQ1SUJWY09tQ3B2b0FGeEM6Rzc4RDJVbHpTVUs2QzMzOHhGQWlPZw==',
    'index_name': 'search-index-test'
}

test_watsonx_discovery_connection(config)