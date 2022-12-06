GRPC_SOURCES = stt_service_pb2.py stt_service_pb2_grpc.py
GRPC_DIR = speaker_verification

all: $(GRPC_SOURCES)

$(GRPC_SOURCES): protobufs/speaker_verification.proto
	python -m grpc_tools.protoc -I protobufs --python_out=$(GRPC_DIR)/.  \
	--grpc_python_out=$(GRPC_DIR)/. protobufs/speaker_verification.proto

clean:
	rm $(GRPC_DIR)/$(GRPC_SOURCES)