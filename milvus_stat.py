from pymilvus import connections, Collection, utility

def connect_to_milvus(milvus_config):
    try:
        connections.connect(
            "default", 
            host=milvus_config["milvus_host"], 
            port=milvus_config["milvus_port"], 
            secure=True, 
            server_pem_path=milvus_config["server_pem_path"], 
            server_name=milvus_config["server_name"], 
            user=milvus_config["user"], 
            password=milvus_config["password"]
        )
        print("Milvus에 성공적으로 연결되었습니다.")
        return True
    except Exception as e:
        print(f"Milvus 연결 실패: {str(e)}")
        return False


def get_collection_details(collection_name):
    try:
        collection = Collection(collection_name)
        
        print(f"\n컬렉션 '{collection_name}'의 상세 정보:")
        
        # 1. 스키마 정보
        schema = collection.schema
        print("\n1. 스키마 정보:")
        print(f"   설명: {schema.description}")
        print("   필드:")
        for field in schema.fields:
            print(f"     - 이름: {field.name}, 타입: {field.dtype}, 주키 여부: {field.is_primary}")
        
        # 2. 기본 정보
        print("\n2. 기본 정보:")
        print(f"   이름: {collection.name}")
        print(f"   설명: {collection.description}")
        print(f"   비어 있음: {collection.is_empty}")
        print(f"   엔티티 수: {collection.num_entities}")
        
        # 3. 주키 정보
        primary_field = collection.primary_field
        print("\n3. 주키 정보:")
        if primary_field:
            print(f"   이름: {primary_field.name}, 타입: {primary_field.dtype}")
        else:
            print("   주키가 없습니다.")
        
        # 4. 파티션 정보
        partitions = collection.partitions
        print("\n4. 파티션 정보:")
        for partition in partitions:
            print(f"   - 이름: {partition.name}, 설명: {partition.description}")
        
        # 5. 인덱스 정보
        indexes = collection.indexes
        print("\n5. 인덱스 정보:")
        if indexes:
            for index in indexes:
                print(f"   필드 이름: {index.field_name}")
                print(f"   인덱스 이름: {index.index_name}")
                print(f"   인덱스 타입: {index.params.get('index_type', 'N/A')}")
                print(f"   메트릭 타입: {index.params.get('metric_type', 'N/A')}")
                print(f"   추가 파라미터: {index.params.get('params', {})}")
        else:
            print("   인덱스가 없습니다.")
        
        # 6. 로드 상태
        load_state = utility.load_state(collection_name)
        print(f"\n6. 컬렉션 로드 상태: {load_state}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    # Milvus 연결 설정
    milvus_config = {
        'milvus_host': '158.175.187.145',
        'milvus_port': '8080',
        'user': 'root',
        'server_name': 'localhost',
        'password': '4XYg2XK6sMU4UuBEjHq4EhYE8mSFO3Qq',
        "server_pem_path": "./temp/cert.pem",
    }


    # Milvus 연결
    if connect_to_milvus(milvus_config):
        # 컬렉션 이름 설정
        collection_name = "_4XYg2XK6sMU4UuBEjHq4EhYE8mSFO3Qq"
        
        # 컬렉션 상세 정보 조회
        get_collection_details(collection_name)
    
    # 연결 종료
    connections.disconnect("default")