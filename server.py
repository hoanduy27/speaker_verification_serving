import logging
import os
import sys
from concurrent import futures

import grpc
import yaml
from google.protobuf.json_format import MessageToDict
from pydub import AudioSegment

from speaker_verification import (speaker_verification_pb2,
                                  speaker_verification_pb2_grpc)
from verifier import SpeakerVerifier

SV_HOST =  os.environ.get('SV_HOST', 'localhost')
SV_PORT = os.environ.get('SV_PORT', '5001')

# CHANNEL_IP = f"{SV_HOST}:{SV_PORT}"

# channel = grpc.insecure_channel(CHANNEL_IP)
# stub = speaker_verification_pb2_grpc.SpeakerVerificationServiceStub()

class SpeakerVerifierServicer(speaker_verification_pb2_grpc.SpeakerVerificationService):
    def __init__(self, model_file, config):
        self.verifier = SpeakerVerifier(
            model_file, 
            config
        )
        self.sim_threshold = config.get('sim_threshold', 0.5)

    def verify(self, enroll_utterances, utterance):
        distance = self.verifier.verify(enroll_utterances, utterance)
        return distance, distance > self.sim_threshold

    def SpeakerVerify(self, request, context):
        logging.info("Verifying")
        distance, accept = self.verify(request.enroll_utterances, request.utterance)
        
        
        return speaker_verification_pb2.SpeakerVerificationResponse(similarity=distance, accept=accept)

def serve(model_file, conf_file):
    with open(conf_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    logging.info("Server starting...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    speaker_verification_pb2_grpc.add_SpeakerVerificationServiceServicer_to_server(
        SpeakerVerifierServicer(model_file, config), server
    )
    logging.info(f"Started server on {SV_HOST}:{SV_PORT}")
    server.add_insecure_port('{}:{}'.format(SV_HOST, SV_PORT))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) != 3:
        exit("Usage: python %s <model_file> <config_file>" % sys.argv[0])
    serve(sys.argv[1], sys.argv[2])